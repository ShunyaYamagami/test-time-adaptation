import logging
from easydict import EasyDict
import os
import numpy as np
import tqdm
import yaml
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from transformers import BertModel
import clip
from sinkhorn_knopp import sinkhorn_knopp as skp
from sklearn.cluster import SpectralBiclustering

from domainbed import networks
from methods.base import TTAMethod
from augmentations.transforms_cotta import get_tta_transforms
from augmentations.aug import get_transform_when_iteration
from models.my_transformer import MyTransformer
from models.mine import Mine, MineTrainer
from loss.nt_xent import NTXentLoss

logger = logging.getLogger(__name__)


class RMT(TTAMethod):
    def __init__(self, steps, episodic, window_length, dataset_name, arch_name, num_classes, src_loader, ckpt_dir, ckpt_path,
                 contrast_mode, temperature, projection_dim, lambda_ce_src, lambda_ce_trg, lambda_cont, m_teacher_momentum, num_samples_warm_up, save_dir):
        super().__init__(steps, episodic, window_length)
        logger.info(f"----- RMT __init__ -----")

        self.hparams = EasyDict({
            "clip_backbone": 'ViT-B/32',  # choice(['ViT-B/32', 'ViT-B/16', 'RN101']),
            # classnameのリストは，cifar10 = torcivision.datasets.CIFAR10 でインスタンスを作成し，cifar10.classesで取得できる.
            "class_names": sorted(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truc"]),
            "num_domain_tokens": 16,
            "sentence_prompt": True, ####### 独自に追加.
            'mlp_depth': 3,
            'mlp_width': 512,
            'mlp_dropout': 0.1,
            'lr': 1e-3,
            'momentum': 0.1,
            'weight_decay': 0.,
            "base_model": {
                "architecture": "mlp",
                # "architecture": "my_transformer",
                'pretrained': False,
            },
            "warmup": {
                "use": False,
                "load_model": False,
            },
            "prototypes": {
                "use": False,
                "load": False,
            },
            "domain_loss": {
                "method": "mine",  # choice([False, 'nt_xent', 'mine']),  ここがFalseなら, domain_lossは使わない事になる.
                # "method": False,  # choice([False, 'nt_xent', 'mine']),  ここがFalseなら, domain_lossは使わない事になる.
                "nt_xent_temperature": 0.5,
            },
            "cuda_visible_devices": os.environ.get('CUDA_VISIBLE_DEVICES'),
            "exec_num": os.environ.get('exec_num'),
        })
        ####################################################################################
        ####################################################################################
        ####################################################################################
        if self.hparams["cuda_visible_devices"] == 0:
            self.hparams["prototypes"]["use"] = False
            self.hparams["domain_loss"]["method"] = False
        elif self.hparams["cuda_visible_devices"] == 1:
            self.hparams["prototypes"]["use"] = False
            self.hparams["domain_loss"]["method"] = "mine"
        elif self.hparams["cuda_visible_devices"] == 2:
            self.hparams["prototypes"]["use"] = True
            self.hparams["domain_loss"]["method"] = False
        elif self.hparams["cuda_visible_devices"] == 3:
            self.hparams["prototypes"]["use"] = True
            self.hparams["domain_loss"]["method"] = "mine"
        ##################
        if self.hparams["exec_num"] == 2:
            self.hparams["base_model"]["architecture"] = "my_transformer"
        ####################################################################################
        ####################################################################################
        ####################################################################################

        assert self.hparams['base_model']['architecture'] in ['mlp', 'my_transformer'], f'base_model must be "mlp" or "my_transformer", but got {self.hparams["base_model"]}'
        assert self.hparams['domain_loss']['method'] in [False, 'nt_xent', 'mine'], f"loss method must be False, 'nt_xent' or 'mine', but got {self.hparams['domain_loss']['method']}"
        assert self.hparams['prototypes']['load'] and self.hparams['prototypes']['use'] or not self.hparams['prototypes']['load'], "if load is True, use must be True"
        assert self.hparams['warmup']['load_model'] and self.hparams['warmup']['use'] or not self.hparams['warmup']['load_model'], "if load_model is True, use must be True"
        assert self.hparams['warmup']['use'] and self.warmup_steps > 0 or not self.hparams['warmup']['use'], "warmup_steps must be set when warmup is used" 

        with open(os.path.join(save_dir, "hparams.yaml"), 'w') as file:
            yaml.dump(self.hparams, file, default_flow_style=False, sort_keys=False)

        ########## Set Parameters ##########
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.EMBEDDING_DIM = 512

        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.src_loader = src_loader
        self.src_loader_iter = iter(self.src_loader)
        self.contrast_mode = contrast_mode
        self.temperature = temperature
        self.base_temperature = self.temperature
        self.projection_dim = projection_dim
        self.lambda_ce_src = lambda_ce_src
        self.lambda_ce_trg = lambda_ce_trg
        self.lambda_cont = lambda_cont
        self.m_teacher_momentum = m_teacher_momentum
        self.save_dir = save_dir
        self.warmup_steps = num_samples_warm_up // self.src_loader.batch_size
        self.tta_transform = get_tta_transforms(self.dataset_name)
        self.normal_transform, self.clsdst_transform = get_transform_when_iteration(grid=3)

        ########## Set Models ##########
        self.set_clip_models()
        if self.hparams['base_model']['architecture'] == "mlp":
            self.base_model = networks.MLP(
                self.EMBEDDING_DIM,
                self.EMBEDDING_DIM * self.hparams['num_domain_tokens'],
                self.hparams
            ).to(device=self.device, dtype=self.clip_model.dtype)
        elif self.hparams['base_model']['architecture'] == "my_transformer":
            self.base_model = MyTransformer(
                width=self.EMBEDDING_DIM,
                out_width = self.EMBEDDING_DIM * self.hparams['num_domain_tokens'],
                layers=12,
                heads=8,
                context_length=src_loader.batch_size,
                # context_length=self.clip_model.context_length,
                attn_mask="build_attention_mask"
            ).to(device=self.device, dtype=self.clip_model.dtype)
            
        self.optimizer = torch.optim.SGD(
            self.base_model.parameters(),
            lr=self.hparams['lr'],
            momentum=self.hparams['momentum']
        )
        self._set_student_teacher_models() 

        self.final_lr = self.optimizer.param_groups[0]["lr"]

        ########## Prototypes ##########
        if self.hparams['prototypes']['use']:
            proto_dir_path = os.path.join(ckpt_dir, "prototypes")
            if self.dataset_name == "domainnet126":
                fname = f"protos_{self.dataset_name}_{ckpt_path.split(os.sep)[-1].split('_')[1]}.pth"
            else:
                fname = f"protos_{self.dataset_name}_{arch_name}.pth"
            fname = os.path.join(proto_dir_path, fname)

            # get source prototypes
            if self.hparams['prototypes']['load'] and os.path.exists(fname):
                logger.info(f"Loading class-wise source prototypes from {fname}...")
                self.prototypes_src = torch.load(fname)
            else:                
                os.makedirs(proto_dir_path, exist_ok=True)
                features_src = torch.tensor([])
                labels_src = torch.tensor([])
                logger.info("Extracting source prototypes...")
                with torch.no_grad():
                    for data in tqdm.tqdm(self.src_loader):
                        x, y = data[0], data[1]
                        tmp_features = self.clip_model.encode_image(x.to(self.device))  # (B, EMBEDDING_DIM)
                        # features_src = torch.cat([features_src, tmp_features.cpu()], dim=0)  # (画像数, EMBEDDING_DIM)
                        base_features = self.base_model(tmp_features)
                        student_features = self.model(base_features)  # (B, EMBEDDING_DIM)
                        features_src = torch.cat([features_src, student_features.cpu()], dim=0)  # (画像数, EMBEDDING_DIM)
                        labels_src = torch.cat([labels_src, y], dim=0)
                        if len(features_src) > 100000:
                            break
                # create class-wise source prototypes
                self.prototypes_src = torch.tensor([])
                for i in range(self.num_classes):
                    mask = labels_src == i
                    self.prototypes_src = torch.cat([self.prototypes_src, features_src[mask].mean(dim=0, keepdim=True)], dim=0)

                torch.save(self.prototypes_src, fname)

                self.prototypes_src = self.prototypes_src.to(self.device).unsqueeze(1)  # (クラス数，1, EMBEDDING_DIM)
                self.prototype_labels_src = torch.arange(start=0, end=self.num_classes, step=1).to(self.device).long()  # (クラス数)
                
            ########## Projector ##########
            if self.dataset_name == "domainnet126":
                # do not use a projector since the network already clusters the features and reduces the dimensions
                self.projector = nn.Identity()
            else:
                num_channels = self.prototypes_src.shape[-1]
                self.projector = nn.Sequential(nn.Linear(num_channels, self.projection_dim),
                                            nn.ReLU(),
                                            nn.Linear(self.projection_dim, self.projection_dim)).to(self.device)
                self.optimizer.add_param_group({'params': self.projector.parameters(), 'lr': self.optimizer.param_groups[0]["lr"]})


        ########## Domain Loss ##########
        if self.hparams["domain_loss"]["method"]:
            self.skp = skp.SinkhornKnopp()
            n_clusters=(4, 3)
            self.biclust_model = SpectralBiclustering(n_clusters=n_clusters, method="log", random_state=0)
            if self.hparams["domain_loss"]["method"] == "mine":
                self.mine_net = Mine().cuda()
                self.mine_trainer = MineTrainer(self.mine_net)
                self.optimizer.add_param_group({'params': self.mine_net.parameters(), 'lr': self.optimizer.param_groups[0]["lr"]})
                self.mine_trainer.mine_net_optim = self.optimizer
            elif self.hparams["domain_loss"]["method"] == "nt_xent":
                self.domain_criterion = NTXentLoss(self.device, self.src_loader.batch_size, self.hparams["domain_loss"]["nt_xent_temperature"])


        ########## Warm Up ##########
        # warm up the mean-teacher framework
        if self.hparams['warmup']['use'] and self.warmup_steps > 0:
            warmup_ckpt_path = os.path.join(ckpt_dir, "warmup")
            if self.dataset_name == "domainnet126":
                source_domain = ckpt_path.split(os.sep)[-1].split('_')[1]
                ckpt_path = f"ckpt_warmup_{self.dataset_name}_{source_domain}_{arch_name}_bs{self.src_loader.batch_size}_step{self.warmup_steps}.pth"
            else:
                ckpt_path = f"ckpt_warmup_{self.dataset_name}_{arch_name}_bs{self.src_loader.batch_size}_step{self.warmup_steps}.pth"
            ckpt_path = os.path.join(warmup_ckpt_path, ckpt_path)
            
            if self.hparams['warmup']['load_model'] and os.path.exists(ckpt_path):
                logger.info(f"Loading warmup checkpoint... from {ckpt_path}")
                checkpoint = torch.load(ckpt_path)
                self.base_model.load_state_dict(checkpoint["base_model"])
                self.model.load_state_dict(checkpoint["model"])
                self.model_ema.load_state_dict(checkpoint["model_ema"])
                if self.hparams["domain_loss"]["method"] == "mine":
                    self.mine_net.load_state_dict(checkpoint["mine_net"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                logger.info(f"Loaded from {ckpt_path}")
            else:
                os.makedirs(warmup_ckpt_path, exist_ok=True)
                self.warmup()
                model_dict = {
                    "base_model": self.base_model.state_dict(),
                    "model": self.model.state_dict(),
                    "model_ema": self.model_ema.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                }
                if self.hparams["domain_loss"]["method"] == "mine":
                    model_dict["mine_net"] = self.mine_net.state_dict()
                torch.save(model_dict, ckpt_path)

        self.models = [self.base_model, self.model, self.model_ema]
        if self.hparams['prototypes']["use"]:
            self.models.append(self.projector)
        if self.hparams["domain_loss"]["method"] == "mine":
            self.models.append(self.mine_net)
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()


    def set_clip_models(self):
        # embedding dim for image and text encoder.
        logger.info(f"----- Set Clip Models,  clip_backbone : {self.hparams['clip_backbone']}  -----")
        self.clip_model, preprocess = clip.load(self.hparams['clip_backbone'], device=self.device)
        self.clip_model = self.clip_model.float()

        # CLIPモデルのパラメータは更新させない
        logger.info('Set self.clip_model.parameters.reguires_grad = False!')
        for param in self.clip_model.parameters():
            param.requires_grad = False
        

        ##### Class Prompt用  refer DPLCLIP
        prompt_prefix = ' '.join(['X'] * self.hparams['num_domain_tokens'])
        
        if self.hparams['sentence_prompt']:
            logger.info('Using sentence_prompt in DPLCLIP...')
            classnames = [f"a photo of a {name.replace('_', ' ')}" for name in self.hparams['class_names']]
        else:
            classnames = [name.replace('_', ' ') for name in self.hparams['class_names']]

        ##### to get default token_prefix and token_suffix.
        class_prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]  # (クラス数)
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in class_prompts]).to(self.device)  # (クラス数, 77)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(self.tokenized_prompts).type(self.clip_model.dtype)  # (クラス数 (入力文の数), 77 (各文のトークン数（シーケンス長）), self.EMBEDDING_DIM)
        self.register_buffer('token_prefix', embedding[:, :1, :])  # SOS (クラス数, 1, self.EMBEDDING_DIM)  各クラスの埋め込みトークンの最初の次元がSOS. SOS (Start of Sentence): SOSトークンは、文の開始を示すために使用されます。通常、モデルへの入力テキストの最初に追加されます。SOSトークンは、モデルに対して文の生成を開始するよう指示します。
        self.register_buffer('token_suffix', embedding[:, self.hparams['num_domain_tokens'] + 1:, :])  # CLS, EOS (クラス数, 68, self.EMBEDDING_DIM)  68 := 77 - num_domain_tokens_tokens - 2.

        ##### Domain Prompt用  refer DPLCLIP
        if self.hparams["domain_loss"]["method"]:
            domain_prompts = prompt_prefix + 'a photo of a '
            self.domain_tokenized_prompts = clip.tokenize(domain_prompts).to(self.device)  # (1, 77)
            domain_embedding = self.clip_model.token_embedding(self.domain_tokenized_prompts).type(self.clip_model.dtype)  # (1, 77 (各文のトークン数（シーケンス長）), self.EMBEDDING_DIM)
            # self.register_buffer('domain_token_prefix', domain_embedding[:, :1, :])  # SOS (1, 1, self.EMBEDDING_DIM)
            self.register_buffer('domain_token_prefix', domain_embedding[:, :1:, :])  # CLS, EOS (1, 68, self.EMBEDDING_DIM)  68 := 77 - num_domain_tokens_tokens - 2.
            self.register_buffer('domain_token_suffix', domain_embedding[:, self.hparams['num_domain_tokens'] + 1:, :])  # CLS, EOS (1, 68, self.EMBEDDING_DIM)  68 := 77 - num_domain_tokens_tokens - 2.
    

    def _set_student_teacher_models(self):
        """ F() Network for extraction of domain feature.

            ここで, self.modelとしているのは F() Network. 元コードのpre-trained modelとは異なることに注意.
            既存コードを書きたくなかったからこうした．
            どちらも同じ "Student model"という意味では共通している.
        """
        logger.info(f"----- set_student_teacher_models  -----")
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        # Student model
        # self.model = networks.MLP(self.EMBEDDING_DIM, self.EMBEDDING_DIM * self.hparams['num_domain_tokens'], self.hparams).to(device=self.device, dtype=self.clip_model.dtype)
        self.model = networks.MLP(self.EMBEDDING_DIM * self.hparams['num_domain_tokens'], self.EMBEDDING_DIM * self.hparams['num_domain_tokens'], self.hparams).to(device=self.device, dtype=self.clip_model.dtype)
        self.model.apply(init_weights)
        # Setup EMA model (Teacherモデル)  重みの更新はしない
        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()
        
        self.optimizer.add_param_group({'params': self.model.parameters(), 'lr': self.optimizer.param_groups[0]["lr"]})
        self.optimizer.add_param_group({'params': self.model_ema.parameters(), 'lr': self.optimizer.param_groups[0]["lr"]})


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def warmup(self):
        logger.info(f"----- WarmUp()  -----")
        logger.info(f"Starting warm up...{self.warmup_steps} steps")
        for i in tqdm.tqdm(range(self.warmup_steps)):
            #linearly increase the learning rate
            for par in self.optimizer.param_groups:
                par["lr"] = self.final_lr * (i+1) / self.warmup_steps

            # sample source batch
            try:
                batch = next(self.src_loader_iter)
            except StopIteration:
                self.src_loader_iter = iter(self.src_loader)
                batch = next(self.src_loader_iter)

            imgs_src = self.normal_transform(batch[0]).to(self.device)
            labels_src = batch[1].long().to(self.device)
            imgs_src_clsdst = self.clsdst_transform(batch[0]).to(self.device)

            self.optimizer.zero_grad()
            
            ##### Domain Loss
            if self.hparams["domain_loss"]["method"]:
                domain_loss = self._calc_clip_domain_loss(imgs_src_clsdst, labels_src)
                domain_loss.backward()
                self.optimizer.step()
            
            ##### Class Loss
            _, logits_per_image, logits_per_image_ema = self._calc_clip_class_logits(imgs_src, labels_src)
            loss_sce = symmetric_cross_entropy(logits_per_image, logits_per_image_ema).mean(0)
            loss_sce.backward()
            self.optimizer.step()

            self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.m_teacher_momentum)

        logger.info(f"Finished warm up...")
        for par in self.optimizer.param_groups:
            par["lr"] = self.final_lr


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        imgs_test = self.normal_transform(x[0]).to(self.device)
        imgs_test_aug = self.tta_transform((imgs_test)).to(self.device)

        self.optimizer.zero_grad()

        ##### Domain Training
        if self.hparams["domain_loss"]["method"]:
            imgs_test_clsdst = self.clsdst_transform(x[0]).to(self.device).to(self.device)
            domain_loss = self._calc_clip_domain_loss(imgs_test_clsdst)
            domain_loss.backward()
            self.optimizer.step()
            
        ##### Self Training
        features_test, logits_per_image, logits_per_image_ema = self._calc_clip_class_logits(imgs_test)
        features_test_aug, logits_per_image_aug, _ = self._calc_clip_class_logits(imgs_test_aug)
        # 式(6)
        loss_entropy = self_training_loss(x=logits_per_image, x_aug=logits_per_image_aug, x_ema=logits_per_image_ema).mean(0)
        # 式(9)のうち, targetドメインに対するLoss (第1項)
        loss_trg = self.lambda_ce_trg * loss_entropy
        
        ##### contrastive_loss
        if self.hparams['prototypes']["use"]:
            ##### Contrastive Lossのために, prototypes_srcとstudentの特徴ベクトルの類似度を計算する．
            with torch.no_grad():
                # dist[:, i] contains the distance from every source sample to one test sample
                # self.prototypes_srcは、ソースプロトタイプの特徴ベクトルを表すテンソル. './ckpt/prototypes/protos_cifar10_c_Standard.pth'から読み込んだもの.
                dist = F.cosine_similarity(
                    x1=self.prototypes_src.repeat(1, features_test.shape[0], 1),
                    x2=features_test.view(1, features_test.shape[0], features_test.shape[1]).repeat(self.prototypes_src.shape[0], 1, 1),
                    dim=-1)  # (クラス数, B)

                # for every test feature, get the nearest source prototype and derive the label
                # 指定した次元でテンソル内の最大の要素とそのインデックスを取得する
                _, indices = dist.topk(1, largest=True, dim=0)  # (1, B)
                indices = indices.squeeze(0)  # (B,)

            features = torch.cat([self.prototypes_src[indices],
                                features_test.view(features_test.shape[0], 1, features_test.shape[1]),
                                features_test_aug.view(features_test.shape[0], 1, features_test.shape[1])], dim=1)  # (B, 3, EMBEDDING_DIM)
            # 式(7)
            loss_contrastive = self.contrastive_loss(features=features, labels=None)
            # 式(9)のうち, targetドメインに対するLoss (第2項)
            loss_trg += self.lambda_cont * loss_contrastive

        loss_trg.backward()
        self.optimizer.step()

        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.m_teacher_momentum)

        # create and return the ensemble prediction
        return logits_per_image + logits_per_image_ema


    @torch.no_grad()
    def forward_sliding_window(self, x):
        """
        Create the prediction for single sample test-time adaptation with a sliding window
        :param x: The buffered data created with a sliding window
        :return: Model predictions
        """
        imgs_test = x[0]
        outputs_test = self.model(imgs_test)
        outputs_ema = self.model_ema(imgs_test)
        return outputs_test + outputs_ema



    def _calc_clip_class_logits(self, imgs, labels=None):
        """
            imgs: (B, 3, 224, 224)
        """
        image_features = self.clip_model.encode_image(imgs)  # (B, EMBEDDING_DIM)
        # Student / Teacher Model
        # domain_features = self.model(image_features)  # (B, EMBEDDING_DIM * num_domain_tokens)
        base_features = self.base_model(image_features)  # (B, EMBEDDING_DIM)
        domain_features = self.model(base_features)  # (B, EMBEDDING_DIM * num_domain_tokens)
        mean_domain_features = domain_features.mean(dim=0, keepdim=True)  # バッチ平均をとる. (1, EMBEDDING_DIM * num_domain_tokens)
        # 各クラス特徴と結合するために，クラス数文複製する.
        _mean_domain_features = mean_domain_features.repeat_interleave(len(self.hparams['class_names']), dim=0)  # (Sourceクラス数, EMBEDDING_DIM * num_domain_tokens)

        # domain_features_ema = self.model_ema(image_features)  # (B, EMBEDDING_DIM * num_domain_tokens)
        domain_features_ema = self.model_ema(base_features)  # (B, EMBEDDING_DIM * num_domain_tokens)
        mean_domain_features_ema = domain_features_ema.mean(dim=0, keepdim=True)  # バッチ平均をとる. (1, EMBEDDING_DIM * num_domain_tokens)
        _mean_domain_features_ema = mean_domain_features_ema.repeat_interleave(len(self.hparams['class_names']), dim=0)  # (Sourceクラス数, EMBEDDING_DIM * num_domain_tokens)
        
        # Text Features
        # prefix + domain_features + suffix(=CLS, EOS) である Text Featuresを作成する.
        text_features = self._get_class_text_features(_mean_domain_features)  # (Sourceクラス数, EMBEDDING_DIM)
        text_features_ema = self._get_class_text_features(_mean_domain_features_ema)  # (Sourceクラス数, EMBEDDING_DIM)

        # Normalize Features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_ema = text_features_ema / text_features_ema.norm(dim=-1, keepdim=True)
        
        # calc logits  t()は転置, 
        # self.clip_model.logit_scale.exp()によるスケール変換は, 類似度スコアの非負化を行い、類似度の解釈や比較を容易にし，指数関数によるスケール変換は正規化や確率的な処理にも関連する．
        logits_per_image = self.clip_model.logit_scale.exp() * image_features @ text_features.t()  # (B, Sourceクラス数)
        logits_per_image_ema = self.clip_model.logit_scale.exp() * image_features @ text_features_ema.t()  # (B, Sourceクラス数)
        
        return image_features, logits_per_image, logits_per_image_ema


    def _calc_clip_domain_loss(self, imgs_clsdst, label=None):
        """
            imgs_clsdst: (B, 3, 224, 224)
        """
        image_clsdst_features = self.clip_model.encode_image(imgs_clsdst)  # (B, EMBEDDING_DIM)
        # domain_features = self.model(image_clsdst_features)  # (B, EMBEDDING_DIM * num_domain_tokens)
        domain_features = self.base_model(image_clsdst_features)  # (B, EMBEDDING_DIM)
        domain_text_features = self._get_domain_text_features(domain_features, batch_size=imgs_clsdst.shape[0])  # (B, EMBEDDING_DIM

        image_clsdst_features = image_clsdst_features / image_clsdst_features.norm(dim=-1, keepdim=True)
        domain_text_features = domain_text_features / domain_text_features.norm(dim=-1, keepdim=True)
        
        if self.hparams["domain_loss"]["method"] == "nt_xent":
            loss = self.domain_criterion(image_clsdst_features, domain_text_features)
        elif self.hparams["domain_loss"]["method"] == "mine":
            # 流石にデータセット単位で類似度計算を行うと，10,000*10,000の計算量となるので，バッチごとに行う．
            # そのため，バッチサイズは大きめでなくてはならない.
            sim_mtx = F.cosine_similarity(
                    x1=image_clsdst_features.unsqueeze(0),  # (B, B, EMBEDDING_DIM)
                    x2=domain_text_features.unsqueeze(1),  # (B, B, EMBEDDING_DIM)
                    dim=-1).detach().cpu().numpy()  # (B, B)

            bistochastic_mtx = self._get_bistochastic_mtx(sim_mtx)  # (B, B)
            clustered_mtx = self._biclustering(bistochastic_mtx)  # (B, B)

            diag = torch.diag(clustered_mtx).long()
            mean_feat_per_clust = [domain_text_features[diag == clust].mean(dim=0) for clust in sorted(torch.unique(diag))]

            for i in range(len(mean_feat_per_clust)):
                for j in range(i+1, len(mean_feat_per_clust)):
                    data = torch.stack([mean_feat_per_clust[i], mean_feat_per_clust[j]], dim=1)
                    loss, _ = self.mine_trainer.get_loss(data)
        return loss
    
            
    def _get_class_text_features(self, domain_feature):
        """ domain feature + text feature 
            domain_feature: (Sourceクラス数, EMBEDDING_DIM * num_domain_tokens)

            self.token_prefix.shape:  SOS (Sourceクラス数, 1, EMBEDDING_DIM)
            self.token_suffix.shape:  EOS (Sourceクラス数, 77 - 1 - self.hparams['num_domain_tokens'], EMBEDDING_DIM)
        """
        domain_feature = domain_feature.reshape(-1, self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)  # (Sourceクラス数, num_domain_tokens, EMBEDDING_DIM)
        prompt_domain_feat = torch.cat([self.token_prefix, domain_feature, self.token_suffix], dim=1)  # (Sourceクラス数, 77, EMBEDDING_DIM)
        
        # refer CoOp: CoOP github. https://github.com/KaiyangZhou/CoOp/blob/b0a058869cef00a4e4ea5256d40fd7681119c099/trainers/coop.py#L46
        # domain_featureに位置エンコーディングを加え，位置情報をembedding空間に反映する．
        x = prompt_domain_feat + self.clip_model.positional_embedding.type(self.clip_model.dtype)  # (Sourceクラス数, 77, EMBEDDING_DIM)
        x = x.permute(1, 0, 2)  # (77, Sourceクラス数, EMBEDDING_DIM)
        x = self.clip_model.transformer(x)  # (77, Sourceクラス数, EMBEDDING_DIM)
        x = x.permute(1, 0, 2)  # (Sourceクラス数, 77, EMBEDDING_DIM)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)  # (Sourceクラス数, 77, EMBEDDING_DIM)
        
        # mapping domain_features to text_features.
        # xがドメイン特徴(にtransformerを適用したもの)，self.tokenized_promptsがテキスト特徴．
        text_features = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection  # (Sourceクラス数, EMBEDDING_DIM))
        return text_features


    def _get_domain_text_features(self, domain_features, batch_size):
        """
            domain_features: (B, EMBEDDING_DIM * num_domain_tokens)
        return
            domain_text_features: (B, EMBEDDING_DIM)
        """
        domain_features = domain_features.reshape(domain_features.shape[0], self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)  # (B, EMBEDDING_DIM * num_domain_tokens)
        rep_token_prefix = self.domain_token_prefix.repeat_interleave(batch_size, dim=0)  # (B, 1, EMBEDDING_DIM)
        rep_token_suffix = self.domain_token_suffix.repeat_interleave(batch_size, dim=0)  # (B, 77-1-num_domain_tokens, EMBEDDING_DIM)
        domain_token_cat = torch.cat([rep_token_prefix, domain_features, rep_token_suffix], dim=1)  # (B, 77, EMBEDDING_DIM)

        x = domain_token_cat + self.clip_model.positional_embedding.type(self.clip_model.dtype)  # (B, 77, EMBEDDING_DIM)
        x = x.permute(1, 0, 2)  # (77, B, EMBEDDING_DIM)
        x = self.clip_model.transformer(x)  # (77, B, EMBEDDING_DIM)
        x = x.permute(1, 0, 2)  # (B, 77, EMBEDDING_DIM)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)  # (B, 77, EMBEDDING_DIM)
        
        domain_text_features = x[torch.arange(x.shape[0]), self.domain_tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection  # (B, EMBEDDING_DIM))

        return domain_text_features
        

    def _get_bistochastic_mtx(self, sim_mtx):
        """
            sim_mtx: Similarity Matrix (Square Matrix) ([-1, 1])
        """
        sim_mtx = (sim_mtx + 1) / 2  # [-1, 1]の類似度行列を[0, 1]に変換.
        bistochastic_mtx = self.skp.fit(sim_mtx)
        
        return bistochastic_mtx

    def _biclustering(self, scaled_sim_mtx):
        """
            scaled_sim_mtx: 正規化した Similarity Matrix (Square Matrix) ([0, 1])
        """
        self.biclust_model.fit(scaled_sim_mtx)
        clustered_mtx = torch.tensor(np.outer(self.biclust_model.row_labels_, self.biclust_model.column_labels_))

        return clustered_mtx
    

    # Integrated from: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    def contrastive_loss(self, features, labels=None, mask=None):
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).cuda()
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().cuda()
        else:
            mask = mask.float().cuda()

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = self.projector(contrast_feature)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


@torch.jit.script
def self_training_loss(x, x_aug, x_ema):# -> torch.Tensor:
    """ 式(6) """
    return - 0.25 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - 0.25 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1) \
           - 0.25 * (x_ema.softmax(1) * x_aug.log_softmax(1)).sum(1) - 0.25 * (x_aug.softmax(1) * x_ema.log_softmax(1)).sum(1)

@torch.jit.script
def symmetric_cross_entropy(x, x_ema):# -> torch.Tensor:
    return -0.5 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - 0.5 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)


def update_ema_variables(ema_model, model, alpha_teacher):
    """ Studentにした変更を, EMA(Exponential Moving Average) によりTeacherモデルへ反映する. 
    :param ema_model: Teacher model
    :param model: Student model
    :param alpha_teacher: weight
    """
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def to_numpy_array(lst):
    np_array = []
    for item in lst:
        if isinstance(item, torch.Tensor):
            np_array.append(item.cpu().detach().numpy())
        elif isinstance(item, list):
            np_array.append(to_numpy_array(item))
    return np.array(np_array)