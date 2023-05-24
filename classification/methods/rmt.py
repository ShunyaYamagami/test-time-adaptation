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
from torch.cuda.amp import GradScaler
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import clip
from sinkhorn_knopp import sinkhorn_knopp as skp
from sklearn.cluster import SpectralBiclustering
from pathlib import Path
from functools import partial

from conf import get_num_classes
from domainbed import networks
from methods.base import TTAMethod
from augmentations.transforms_cotta import get_tta_transforms
from augmentations.aug import get_transform_when_iteration
from models.model import get_model, split_up_model
from models.my_transformer import MyTransformer
from models.mine import Mine, MineTrainer
from loss.nt_xent import NTXentLoss
from methods.pretrain import Pretrain

logger = logging.getLogger(__name__)


class RMT(TTAMethod):
    def __init__(self, cfg, hparams, steps, episodic, window_length, dataset_name, arch_name, src_loader, ckpt_dir, ckpt_path,
                 contrast_mode, temperature, projection_dim, lambda_ce_src, lambda_ce_trg, lambda_cont, m_teacher_momentum, num_samples_warm_up, save_dir, model_weights):
        super().__init__(steps, episodic, window_length)
        logger.info(f"----- RMT __init__ -----")
        ####################################################################################
        ####################################################################################
        # if hparams["cuda_visible_devices"] == [0]:
        #     hparams["architecture"] = 'USE_BASE_MODEL'
        # elif hparams["cuda_visible_devices"] == [2]:
        #     hparams["architecture"] = 'DOMAIN_FIRST'
        ####################################################################################
        ####################################################################################
        self.cfg = cfg
        self.hparams = hparams
        assert isinstance(self.hparams['cuda_visible_devices'], list) and all(isinstance(x, int) for x in self.hparams['cuda_visible_devices']), "cuda_visible_devices must be list of int."
        assert isinstance(self.hparams['exec_num'], int)
        assert self.hparams['optimizer'] in ['Adam', 'SGD']
        assert self.hparams['architecture']['base_model'] in [False, 'mlp', 'my_transformer']
        assert self.hparams['architecture']['domain_embedding_pos'] in ['first', 'cat']
        assert self.hparams['domain_loss']['method'] in ['nt_xent', 'mine']
        assert self.hparams['architecture']['domain_embedding_pos'] in ['first', 'cat'] and self.hparams['domain_loss']['method'] in ['nt_xent', 'mine'] or not self.hparams['architecture']['domain_embedding_pos']
        assert self.hparams['pretrain']['prompt_net'] and self.hparams['architecture']['domain_embedding_pos'] and self.hparams['domain_loss']['method']  or not self.hparams['pretrain']['prompt_net'], "if prompt_net is True, domain_embedding_pos and domain_loss must be True."
        assert self.hparams['prototypes']['load'] and self.hparams['prototypes']['use'] or not self.hparams['prototypes']['load'], "if load is True, use must be True"
        assert self.hparams['warmup']['load_model'] and self.hparams['warmup']['use'] or not self.hparams['warmup']['load_model'], "if load_model is True, use must be True"
        assert self.hparams['warmup']['use'] and num_samples_warm_up > 0 or not self.hparams['warmup']['use'], "warmup_steps must be set when warmup is used" 

        ########## Set save_dir ##########
        if self.hparams['rename_save_dir']:
            old_save_dir = save_dir
            dirname = f"{'_'.join(Path(old_save_dir).name.split('_')[:3])}--base_{hparams['architecture']['base_model']}--warm_{hparams['warmup']['use']}--proto_{hparams['prototypes']['use']}--domain_{hparams['domain_loss']['method']}"
            dirname = dirname.replace('False', 'F').replace('True', 'T').replace('my_transformer', 'tf')
            save_dir = str(Path(old_save_dir).parent / dirname)
            os.rename(old_save_dir, save_dir)
            logger.info(f"----- Rename save_dir: {save_dir} -----")

        with open(os.path.join(save_dir, "hparams.yaml"), 'w') as file:
            yaml.dump(self.hparams, file, default_flow_style=False, sort_keys=False)

        ########## Set Parameters ##########
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.EMBEDDING_DIM = 512
        self.dataset_name = dataset_name
        self.arch_name = arch_name
        self.class_names = src_loader.dataset.classes
        self.num_classes = len(self.class_names)
        self.src_loader = src_loader
        self.src_loader_iter = iter(self.src_loader)
        self.ckpt_dir = ckpt_dir
        self.ckpt_path = ckpt_path
        self.contrast_mode = contrast_mode
        self.temperature = temperature
        self.base_temperature = self.temperature
        self.projection_dim = projection_dim
        self.lambda_ce_src = lambda_ce_src
        self.lambda_ce_trg = lambda_ce_trg
        self.lambda_cont = lambda_cont
        self.m_teacher_momentum = m_teacher_momentum
        self.save_dir = save_dir
        self.model_weights = model_weights
        self.src_batch_size = self.src_loader.batch_size
        self.warmup_steps = num_samples_warm_up // self.src_batch_size
        self.tta_transform = get_tta_transforms(self.dataset_name)
        self.normal_transform, self.clsdst_transform = get_transform_when_iteration(grid=4)
        # 自動的にレイヤ毎に最適なビット精度を選択してくれる（convはfp16, bnはfp32等）. ベストプラクティスを選択してくれるため、便利。use_amp=Falseではfp32を使用する。
        self.scaler = GradScaler(enabled=self.hparams['mixed_precision'])
                
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
                        image_fts = self._learning(x.to(self.device), source=True)
                        #####  TODO: WarmUpの後にPrototypesを作成した方が良いのでは?
                        features_src = torch.cat([features_src, image_fts.cpu()], dim=0)  # (画像数, EMBEDDING_DIM)
                        labels_src = torch.cat([labels_src, y], dim=0)
                        if len(features_src) > 100000:
                            break
                # create class-wise source prototypes
                self.prototypes_src = torch.tensor([])
                for i in range(self.num_classes):
                    mask = labels_src == i
                    self.prototypes_src = torch.cat([self.prototypes_src, features_src[mask].mean(dim=0, keepdim=True)], dim=0)

                torch.save(self.prototypes_src, fname)

                self.prototypes_src = self.prototypes_src.to(self.device).unsqueeze(1) 
                self.prototype_labels_src = torch.arange(start=0, end=self.num_classes, step=1).to(self.device).long()  # (クラス数)


        ########## Set Models ##########
        self.set_clip_models()
        num_channels = self.prototypes_src.shape[-1] if self.hparams['prototypes']['use'] else None
        self.models = set_models(hparams, self.device, self.EMBEDDING_DIM, self.clip_model.dtype, dataset_name,
                                    self.src_batch_size, projection_dim, num_channels)
        self.optimizers = set_optimizers(hparams, self.models)
        self.final_lr = self.optimizers['optimizer'].param_groups[0]["lr"]

        ########## Domain Loss ##########
        if self.hparams["domain_loss"]["method"]:
            self.skp = skp.SinkhornKnopp()
            n_clusters=(4, 3)
            self.biclust_model = SpectralBiclustering(n_clusters=n_clusters, method="log", random_state=0)
            assert self.hparams["domain_loss"]["method"] in [False, "mine"], "Now only support False or MINE for domain loss when _set_optimizer()"
            if self.hparams["domain_loss"]["method"] == "mine":
                self.mine_trainer = MineTrainer(self.models['mine'])
                self.mine_trainer.mine_optim = self.optimizers['domain_optimizer']
            elif self.hparams["domain_loss"]["method"] == "nt_xent":
                self.ntxent_criterion = NTXentLoss(self.device, self.src_loader.batch_size, self.hparams["domain_loss"]["nt_xent_temperature"])

                
                

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
                for model_name in self.models.keys():
                    self.models[model_name].load_state_dict(checkpoint[model_name])
                for optim_name in self.optimizers.keys():
                    self.optimizers[optim_name].load_state_dict(checkpoint[optim_name])
                logger.info(f"Loaded from {ckpt_path}")
            else:
                os.makedirs(warmup_ckpt_path, exist_ok=True)
                self.warmup()
                state_dicts = {}
                for model_name in self.models.keys(): 
                    state_dicts[model_name] = self.models[model_name].state_dict()
                for optim_name in self.optimizers.keys():
                    state_dicts[optim_name] = self.optimizers[optim_name].state_dict()
                torch.save(state_dicts, ckpt_path)


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
            classnames = [f"a photo of a {name.replace('_', ' ')}" for name in self.class_names]
        else:
            classnames = [name.replace('_', ' ') for name in self.class_names]

        ##### to get default token_prefix and token_suffix.
        class_prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]  # (クラス数)
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in class_prompts]).to(self.device)  # (クラス数, 77)
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


    def _learning(self, x, y=None, pretrain=False, source=False, warmup=False):
        """ 
            source も warmup も False の場合は, Forwardプロセスを指す.
            変数名の fts は features の fts.
        """
        assert (pretrain, source, warmup).count(True) <= 1
        x_cls = self.normal_transform(x)  # (B, 3, 224, 224)
        image_fts = self.clip_model.encode_image(x_cls)  # (B, EMBEDDING_DIM)

        ##### Return Source
        if source:
            return F.normalize(image_fts)

        ##### Domain Features, Loss
        if self.hparams["domain_loss"]["method"]:
            self._learning__backward_domain_loss(x)
        
        ##### Class Features, Loss, Concat Text Features.
        self.optimizers['optimizer'].zero_grad()

        logits_st, norm_image_fts, middle_fts = self._learning__get_sttc_logits(self.models['model_st'], image_fts)  # (B, num_classes)
        logits_ema, _, _ = self._learning__get_sttc_logits(self.models['model_ema'], image_fts, middle_fts)  # (B, num_classes)

        if warmup:
            ##### Warm Up
            loss = symmetric_cross_entropy(logits_st, logits_ema).mean(0)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizers['optimizer'])
            self.scaler.update()
            self.models['model_ema'] = update_ema_variables(ema_model=self.models['model_ema'], model=self.models['model_st'], alpha_teacher=self.m_teacher_momentum)
            return
        else:
            ##### Forward Process
            x_aug = self.tta_transform(x_cls)
            image_aug_fts = self.clip_model.encode_image(x_aug)  # (B, EMBEDDING_DIM)
            logits_aug, norm_image_aug_fts, _ = self._learning__get_sttc_logits(self.models['model_st'], image_aug_fts)  # (B, num_classes)

            if pretrain:
                """ Student Modelは, augとともにクラス正解誤差で逆伝播
                    Domain Projectorは, augとのcontrastive learningで行う. (後々RPLを応用しても良いかもしれない.)
                """
                if self.hparams["domain_loss"]["method"]:
                    self.optimizers['domain_optimizer'].zero_grad()
                    domain_fts = self._learning__backward_domain_loss(x_cls, pretrain=True)
                    domain_aug_fts = self._learning__backward_domain_loss(x_aug, pretrain=True)
                    pre_domain_loss = self.pre_domain_criterion(domain_fts, domain_aug_fts)
                    self.scaler.scale(pre_domain_loss).backward()
                    self.scaler.step(self.optimizers['domain_optimizer'])
                    self.scaler.update()

                pre_loss_st = self.pre_class_criterion(logits_st.argmax(1), y)
                pre_loss_aug = self.pre_class_criterion(logits_aug.argmax(1), y)
                pre_class_loss = pre_loss_st + pre_loss_aug
                self.scaler.scale(pre_class_loss).backward()
                self.scaler.step(self.optimizers['domain_optimizer'])
                self.scaler.update()

                return pre_loss_st
            else:
                # Loss
                # 式(6)
                loss_entropy = self_training_loss(x=logits_st, x_aug=logits_aug, x_ema=logits_ema).mean(0)
                # 式(9)のうち, targetドメインに対するLoss (第1項)
                loss_trg = self.lambda_ce_trg * loss_entropy

                ##### contrastive_loss
                if self.hparams['prototypes']["use"]:
                    ##### Contrastive Lossのために, prototypes_srcとstudentの特徴ベクトルの類似度を計算する．
                    with torch.no_grad():
                        # dist[:, i] contains the distance from every source sample to one test sample
                        # self.prototypes_srcは、ソースプロトタイプの特徴ベクトルを表すテンソル. './ckpt/prototypes/protos_cifar10_c_Standard.pth'から読み込んだもの.
                        # self.prototypes_src  (num_classes, EMBEDDING_DIM * num_domain_tokens)
                        dist = F.cosine_similarity(
                            x1=self.prototypes_src.repeat(1, norm_image_fts.shape[0], 1),
                            x2=norm_image_fts.view(1, norm_image_fts.shape[0], norm_image_fts.shape[1]).repeat(self.prototypes_src.shape[0], 1, 1),
                            dim=-1)  # (num_classes, B)
                        # for every test feature, get the nearest source prototype and derive the label
                        # 指定した次元でテンソル内の最大の要素とそのインデックスを取得する
                        _, indices = dist.topk(1, largest=True, dim=0)  # (1, B)
                        indices = indices.squeeze(0)  # (B,)

                    features = torch.cat([self.prototypes_src[indices],
                                        norm_image_fts.view(norm_image_fts.shape[0], 1, norm_image_fts.shape[1]),
                                        norm_image_aug_fts.view(norm_image_fts.shape[0], 1, norm_image_fts.shape[1])], dim=1)  # (B, 3, EMBEDDING_DIM)
                    # 式(7)
                    loss_contrastive = self.contrastive_loss(features=features, labels=None)
                    # 式(9)のうち, targetドメインに対するLoss (第2項)
                    loss_trg += self.lambda_cont * loss_contrastive
                
                # loss_trg.backward()
                # self.optimizer.step()
                self.scaler.scale(loss_trg).backward()
                self.scaler.step(self.optimizers['optimizer'])
                self.scaler.update()
                self.models['model_ema'] = update_ema_variables(ema_model=self.models['model_ema'], model=self.models['model_st'], alpha_teacher=self.m_teacher_momentum)
                
                # create and return the ensemble prediction
                return logits_st + logits_ema


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def warmup(self):
        logger.info(f"----- WarmUp()  -----")
        logger.info(f"Starting warm up...{self.warmup_steps} steps")
        for i in tqdm.tqdm(range(self.warmup_steps)):
            #linearly increase the learning rate
            for par in self.optimizers['optimizer'].param_groups:
                par["lr"] = self.final_lr * (i+1) / self.warmup_steps
            # sample source batch
            try:
                src_batch = next(self.src_loader_iter)
            except StopIteration:
                self.src_loader_iter = iter(self.src_loader)
                src_batch = next(self.src_loader_iter)

            self._learning(src_batch[0].to(self.device), warmup=True)

        logger.info(f"Finished warm up...")
        for par in self.optimizers['optimizer'].param_groups:
            par["lr"] = self.final_lr


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        ensembled_logits = self._learning(x[0])
        return ensembled_logits


    def _learning__get_sttc_logits(self, model, image_fts, middle_fts=None):
        """
            'sttc' stand for Student or Teacher.
            middle_fts: CLIPのimage_encoderからの特徴量, またはbase_modelを通した後の特徴量
                (B, EMBEDDING_DIM) or (B, EMBEDDING_DIM * num_domain_tokens)
        """
        ##### Get Middle Features
        if middle_fts is None:
            if self.hparams["architecture"]['base_model']:
                middle_fts = self.models['base_model'](image_fts)  # (B, EMBEDDING_DIM)
            elif self.hparams["architecture"]['domain_embedding_pos'] == 'first':
                middle_fts = self.models['domain_projector'](image_fts)  # (B, EMBEDDING_DIM * num_domain_tokens)
            else:
                middle_fts = image_fts  # (B, EMBEDDING_DIM)
        
        #####  Student/Text Features
        sttc_fts = model(middle_fts)  # (B, EMBEDDING_DIM * num_domain_tokens)
        mean_sttc_fts = sttc_fts.mean(dim=0, keepdim=True)  # バッチ平均をとる. (1, EMBEDDING_DIM * num_domain_tokens)
        repeated_sttc_fts = mean_sttc_fts.repeat_interleave(self.num_classes, dim=0)  # 各クラス特徴と結合するために，クラス数文複製する. (num_classes, EMBEDDING_DIM * num_domain_tokens)

        if self.hparams['architecture']['domain_embedding_pos'] == 'cat':
            norm_sttc_fts = F.normalize(repeated_sttc_fts)  # 正規化. (num_classes, EMBEDDING_DIM * num_domain_tokens)
            with torch.no_grad():
                domain_fts = self.models['domain_projector'](middle_fts)  # (B, EMBEDDING_DIM * num_domain_tokens)
                mean_domain_fts = domain_fts.mean(dim=0, keepdim=True)
                repeated_domain_fts = mean_domain_fts.repeat_interleave(self.num_classes, dim=0)
                norm_domain_fts = F.normalize(repeated_domain_fts)
            leanable_parameters = norm_sttc_fts + norm_domain_fts
        else:
            leanable_parameters = repeated_sttc_fts

        # Text Features
        text_fts = self._cat_class_domain_text_features(leanable_parameters, class_domain="class")  # (num_classes, EMBEDDING_DIM)
        norm_text_fts = F.normalize(text_fts)
        # Concat Student/Teacher Features and Text Features
        norm_image_fts = F.normalize(image_fts)
        logits = self.clip_model.logit_scale.exp() * norm_image_fts @ norm_text_fts.t()  # (B, num_classes)  # self.clip_model.logit_scale.exp()によるスケール変換は, 類似度スコアの非負化を行い、類似度の解釈や比較を容易にし，指数関数によるスケール変換は正規化や確率的な処理にも関連する．
        
        return logits, norm_image_fts, middle_fts


    def _learning__backward_domain_loss(self, x, pretrain=False):
        self.optimizers['domain_optimizer'].zero_grad()
        x_clsdst = self.clsdst_transform(x)  # (B, 3, 224, 224)
        # Features
        image_clsdst_fts = self.clip_model.encode_image(x_clsdst)  # (B, EMBEDDING_DIM)

        if self.hparams["architecture"]['base_model']:
            base_clsdst_fts = self.models['base_model'](image_clsdst_fts)  # (B, EMBEDDING_DIM)
            domain_fts = self.models['domain_projector'](base_clsdst_fts)  # (B, EMBEDDING_DIM * num_domain_tokens)
        else:
            domain_fts = self.models['domain_projector'](image_clsdst_fts)  # (B, EMBEDDING_DIM * num_domain_tokens)

        if pretrain:
            return F.normalize(domain_fts, dim=1)
            
        domain_text_fts = self._cat_class_domain_text_features(domain_fts, class_domain="domain")  # (B, EMBEDDING_DIM)

        image_clsdst_fts = F.normalize(image_clsdst_fts)
        domain_text_fts = F.normalize(domain_text_fts)
        # Loss
        domain_loss = self._get_domain_loss(image_clsdst_fts, domain_text_fts)
        self.scaler.scale(domain_loss).backward()
        self.scaler.step(self.optimizers['domain_optimizer'])
        self.scaler.update()
    
    def _get_domain_loss(self, image_clsdst_features, domain_text_features):
        """
            image_clsdst_features: (B, EMBEDDING_DIM)
            domain_text_features: (B, EMBEDDING_DIM)
        """
        if self.hparams["domain_loss"]["method"] == "nt_xent":
            domain_loss = self.ntxent_criterion(image_clsdst_features, domain_text_features)
        elif self.hparams["domain_loss"]["method"] == "mine":
            # 流石にデータセット単位で類似度計算を行うと，10,000*10,000の計算量となるので，バッチごとに行う．
            # そのため，バッチサイズは大きめでなくてはならない.
            sim_mtx = F.cosine_similarity(
                    x1=image_clsdst_features.unsqueeze(0),  # (1, B, EMBEDDING_DIM)
                    x2=domain_text_features.unsqueeze(1),  # (B, 1, EMBEDDING_DIM)
                    dim=-1).detach().cpu().numpy()  # (B, B)

            bistochastic_mtx = self._get_bistochastic_mtx(sim_mtx)  # (B, B)
            clustered_mtx = self._biclustering(bistochastic_mtx)  # (B, B)

            diag = torch.diag(clustered_mtx).long()
            mean_feat_per_clust = [domain_text_features[diag == clust].mean(dim=0) for clust in sorted(torch.unique(diag))]

            domain_loss = 0.
            for i in range(len(mean_feat_per_clust)):
                for j in range(i+1, len(mean_feat_per_clust)):
                    data = torch.stack([mean_feat_per_clust[i], mean_feat_per_clust[j]], dim=1)
                    mine_loss, _ = self.mine_trainer.get_loss(data)
                    ##### TODO: 足していくのが適切か？ 平均とるのが適切か?
                    domain_loss += mine_loss

        return domain_loss
    
            
    def _cat_class_domain_text_features(self, class_domain_fts, class_domain):
        """ class_domain_fts + text feature 
            L: classの場合は クラス数, domainの場合は batch size
            self.token_prefix.shape:  SOS (L, 1, EMBEDDING_DIM)
            self.token_suffix.shape:  EOS (L, 77 - 1 - self.hparams['num_domain_tokens'], EMBEDDING_DIM)
        args:
            class_domain_fts: (L, EMBEDDING_DIM * num_domain_tokens)
            class_domain: "class" or "domain"
        return:
            class_domain_fts + text feature 
        """
        assert class_domain in ["class", "domain"], f"Invalid class_domain: {class_domain}. class_domain must be 'class' or 'domain'"
        class_domain_fts = class_domain_fts.reshape(-1, self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)  # (L, num_domain_tokens, EMBEDDING_DIM)
        
        if class_domain == "class":
            param_fts = torch.cat([self.token_prefix, class_domain_fts, self.token_suffix], dim=1)  # (L, 77, EMBEDDING_DIM)
            tokenized_prompts = self.tokenized_prompts
        else:
            batch_size = class_domain_fts.shape[0]
            rep_token_prefix = self.domain_token_prefix.repeat_interleave(batch_size, dim=0)  # (L, 1, EMBEDDING_DIM)
            rep_token_suffix = self.domain_token_suffix.repeat_interleave(batch_size, dim=0)  # (L, 77-1-num_domain_tokens, EMBEDDING_DIM)
            param_fts = torch.cat([rep_token_prefix, class_domain_fts, rep_token_suffix], dim=1)  # (L, 77, EMBEDDING_DIM)
            tokenized_prompts = self.domain_tokenized_prompts  # (L, 77)
        
        # refer CoOp: CoOP github. https://github.com/KaiyangZhou/CoOp/blob/b0a058869cef00a4e4ea5256d40fd7681119c099/trainers/coop.py#L46
        # domain_featureに位置エンコーディングを加え，位置情報をembedding空間に反映する．
        x = param_fts + self.clip_model.positional_embedding.type(self.clip_model.dtype)  # (L, 77, EMBEDDING_DIM)
        x = x.permute(1, 0, 2)  # (77, L, EMBEDDING_DIM)
        x = self.clip_model.transformer(x)  # (77, L, EMBEDDING_DIM)
        x = x.permute(1, 0, 2)  # (L, 77, EMBEDDING_DIM)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)  # (L, 77, EMBEDDING_DIM)
        
        # mapping domain_features to class_domain_text_fts.
        class_domain_text_fts = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection  # (L, EMBEDDING_DIM))

        return class_domain_text_fts
        

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
        contrast_feature = self.models['projector'](contrast_feature)
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


def set_models(hparams, device, EMBEDDING_DIM, clip_model_dtype, dataset_name,
                src_batch_size, projection_dim, num_channels=None):
    """ F() Network for extraction of domain feature.

        ここで, self.modelとしているのは F() Network. 元コードのpre-trained modelとは異なることに注意.
        既存コードを書きたくなかったからこうした．
        どちらも同じ "Student model"という意味では共通している.
    """
    ##### Student Teacher Model
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
    # Student model
    input_dim = EMBEDDING_DIM * hparams['num_domain_tokens'] \
        if hparams['architecture']['domain_embedding_pos'] == 'first' else EMBEDDING_DIM
    model_st = networks.MLP(input_dim,
                            EMBEDDING_DIM * hparams['num_domain_tokens'],
                            hparams
                            ).to(device=device, dtype=clip_model_dtype)
    model_st.apply(init_weights)

    # Teacher model
    # model_ema = self.copy_model(model_st)
    model_ema = TTAMethod.copy_model(model_st)
    for param in model_ema.parameters():
        param.detach_()

    ##### return models
    return_models = {
        "model_st": model_st,
        "model_ema": model_ema
    }
    ##### Base Model
    if hparams['architecture']['base_model'] == "mlp":
        return_models['base_model'] = networks.MLP(EMBEDDING_DIM,
                                                    EMBEDDING_DIM,
                                                    hparams
                                                    ).to(device=device, dtype=clip_model_dtype)
    elif hparams['architecture']['base_model'] == "my_transformer":
        return_models['base_model'] = MyTransformer(width=EMBEDDING_DIM,
                                                    out_width=EMBEDDING_DIM,
                                                    layers=12,
                                                    heads=8,
                                                    context_length=src_batch_size,
                                                    attn_mask="build_attention_mask"
                                                    ).to(device=device, dtype=clip_model_dtype)
    ##### Domain Projector
    if hparams["domain_loss"]["method"]:
        return_models['domain_projector'] = \
            nn.Sequential(nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM * hparams['num_domain_tokens'])).to(device=device, dtype=clip_model_dtype)

    ##### Projector
    if hparams['prototypes']['use']:
        if dataset_name == "domainnet126":
            # do not use a projector since the network already clusters the features and reduces the dimensions
            projector = nn.Identity()
        else:
            projector = nn.Sequential(nn.Linear(num_channels, projection_dim),
                                        nn.ReLU(),
                                        nn.Linear(projection_dim, projection_dim)).to(device)
            return_models['projector'] = projector
    ##### MINE
    if hparams["domain_loss"]["method"] == "mine":
        return_models['mine'] = Mine().to(device)

    return return_models


def set_optimizers(hparams, models):
    assert hparams["optimizer"] in ["Adam", "SGD"], "optimizer must be Adam or SGD"
    class_args_params = []
    domain_args_params = []
    ###### optimizerのparamsに代入する各モデルのparameters
    for model_name, model in models.items():
        if model_name in ['model_st', 'model_ema', 'base_model', 'projector']:
            class_args_params.append({'params': model.parameters()})
        elif model_name in ['domain_projector', 'mine']:
            domain_args_params.append({'params': model.parameters()})
        else:
            raise NotImplementedError(model_name)
    optimizers = {}
    ###### optimizerのpartial関数
    if hparams["optimizer"] == "Adam":
        optimizers['optimizer'] = torch.optim.Adam(class_args_params, lr=hparams['lr'])
        if len(domain_args_params) > 0:
            optimizers['domain_optimizer'] = torch.optim.Adam(domain_args_params, lr=hparams['lr'])
    elif hparams["optimizer"] == "SGD":
        optimizers['optimizer'] = torch.optim.SGD(class_args_params, lr=hparams['lr'], momentum=hparams['momentum'])
        if len(domain_args_params) > 0:
            optimizers['domain_optimizer'] = torch.optim.SGD(domain_args_params, lr=hparams['lr'], momentum=hparams['momentum'])

    return optimizers


def to_numpy_array(lst):
    np_array = []
    for item in lst:
        if isinstance(item, torch.Tensor):
            np_array.append(item.cpu().detach().numpy())
        elif isinstance(item, list):
            np_array.append(to_numpy_array(item))
    return np.array(np_array)