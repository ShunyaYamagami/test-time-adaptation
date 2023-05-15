import logging

from easydict import EasyDict
import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
import torchvision
import clip

from domainbed import networks
from methods.base import TTAMethod
from models.model import split_up_model
from augmentations.transforms_cotta import get_tta_transforms

logger = logging.getLogger(__name__)


def update_ema_variables(ema_model, model, alpha_teacher):
    """ Studentにした変更を, EMA(Exponential Moving Average) によりTeacherモデルへ反映する. 
    :param ema_model: Teacher model
    :param model: Student model
    :param alpha_teacher: weight
    """
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class RMT(TTAMethod):
    def __init__(self, steps, episodic, window_length, dataset_name, arch_name, num_classes, src_loader, ckpt_dir, ckpt_path,
                 contrast_mode, temperature, projection_dim, lambda_ce_src, lambda_ce_trg, lambda_cont, m_teacher_momentum, num_samples_warm_up):
        super().__init__(steps, episodic, window_length)

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
            "load_warmup_model": False  ######### 今は毎回warmupしよう.
        })
        self.set_clip_models()

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
        # arguments neeeded for warm up
        self.warmup_steps = num_samples_warm_up // self.src_loader.batch_size
        self.final_lr = self.optimizer.param_groups[0]["lr"]

        self.tta_transform = get_tta_transforms(self.dataset_name)

        # Setup EMA model
        # Teacher モデル
        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        # split up the model
        # pre-trained modelを分割する. 間でsource prototypesにalignさせるためだな.
        # self.feature_extractor, self.classifier = split_up_model(model, arch_name, self.dataset_name)

        # define the prototype paths
        proto_dir_path = os.path.join(ckpt_dir, "prototypes")
        if self.dataset_name == "domainnet126":
            fname = f"protos_{self.dataset_name}_{ckpt_path.split(os.sep)[-1].split('_')[1]}.pth"
        else:
            fname = f"protos_{self.dataset_name}_{arch_name}.pth"
        fname = os.path.join(proto_dir_path, fname)

        # get source prototypes
        if os.path.exists(fname):
            logger.info(f"Loading class-wise source prototypes from {fname}...")
            self.prototypes_src = torch.load(fname)
        else:
            raise ValueError("----- 今の所こっちは使わないので，使わない事を分かりやすくするためだけにエラーを出しておく -----")
            os.makedirs(proto_dir_path, exist_ok=True)
            features_src = torch.tensor([])
            labels_src = torch.tensor([])
            logger.info("Extracting source prototypes...")
            with torch.no_grad():
                for data in tqdm.tqdm(self.src_loader):
                    x, y = data[0], data[1]
                    tmp_features = self.feature_extractor(x.cuda())
                    features_src = torch.cat([features_src, tmp_features.view(tmp_features.shape[:2]).cpu()], dim=0)
                    labels_src = torch.cat([labels_src, y], dim=0)
                    if len(features_src) > 100000:
                        break

            # create class-wise source prototypes
            self.prototypes_src = torch.tensor([])
            for i in range(self.num_classes):
                mask = labels_src == i
                self.prototypes_src = torch.cat([self.prototypes_src, features_src[mask].mean(dim=0, keepdim=True)], dim=0)

            torch.save(self.prototypes_src, fname)

        self.prototypes_src = self.prototypes_src.cuda().unsqueeze(1)
        self.prototype_labels_src = torch.arange(start=0, end=self.num_classes, step=1).cuda().long()

        # setup projector
        if self.dataset_name == "domainnet126":
            # do not use a projector since the network already clusters the features and reduces the dimensions
            self.projector = nn.Identity()
        else:
            num_channels = self.prototypes_src.shape[-1]
            self.projector = nn.Sequential(nn.Linear(num_channels, self.projection_dim), nn.ReLU(),
                                           nn.Linear(self.projection_dim, self.projection_dim)).cuda()
            self.optimizer.add_param_group({'params': self.projector.parameters(), 'lr': self.optimizer.param_groups[0]["lr"]})

        # warm up the mean-teacher framework
        if self.warmup_steps > 0:
            warmup_ckpt_path = os.path.join(ckpt_dir, "warmup")
            if self.dataset_name == "domainnet126":
                source_domain = ckpt_path.split(os.sep)[-1].split('_')[1]
                ckpt_path = f"ckpt_warmup_{self.dataset_name}_{source_domain}_{arch_name}_bs{self.src_loader.batch_size}.pth"
            else:
                ckpt_path = f"ckpt_warmup_{self.dataset_name}_{arch_name}_bs{self.src_loader.batch_size}.pth"
            ckpt_path = os.path.join(warmup_ckpt_path, ckpt_path)
            
            if self.hparams['load_warmup_model'] and os.path.exists(ckpt_path):
                raise ValueError("----- 今の所WarmUpは毎回したいので, ここでWarmUpモデルのロードはしない. 使わない事を分かりやすくするためだけにエラーを出しておく -----")
                logger.info("Loading warmup checkpoint...")
                checkpoint = torch.load(ckpt_path)
                self.model.load_state_dict(checkpoint["model"])
                self.model_ema.load_state_dict(checkpoint["model_ema"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                logger.info(f"Loaded from {ckpt_path}")
            else:
                os.makedirs(warmup_ckpt_path, exist_ok=True)
                self.warmup()
                torch.save({"model": self.model.state_dict(),
                            "model_ema": self.model_ema.state_dict(),
                            "optimizer": self.optimizer.state_dict()
                            }, ckpt_path)

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.model, self.model_ema, self.projector]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()


    def set_clip_models(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # embedding dim for image and text encoder.
        self.EMBEDDING_DIM = 512
        print(f"----- self.hparams['clip_backbone'] : {self.hparams['clip_backbone']}  -----")
        self.clip_model, preprocess = clip.load(self.hparams['clip_backbone'], device=self.device)
        self.clip_model = self.clip_model.float()

        # CLIPモデルのパラメータは更新させない
        print('Set self.clip_model.parameters.reguires_grad = False!')
        for param in self.clip_model.parameters():
            param.requires_grad = False
        

        ##### class DPLCLIP(CLIP): #####
        #  initial prompt.
        prompt_prefix = ' '.join(['X'] * self.hparams['num_domain_tokens'])
        
        if self.hparams['sentence_prompt']:
            print('Using sentence_prompt in DPLCLIP...')
            classnames = [f"a photo of a {name.replace('_', ' ')}" for name in self.hparams['class_names']]
        else:
            classnames = [name.replace('_', ' ') for name in self.hparams['class_names']]
        prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]  # (クラス数)  ['X X X X X X X X dog.', 'X X X X X X X X elephant.' ...]
        
        #####  to get default token_prefix and token_suffix.
        ##### プロンプト（テキスト）のトークン化、トークン埋め込み、およびトークンの前後の特別なトークン（SOS、CLS、EOSなど）の処理を行う．
        # プロンプトのトークン化. 各単語に一意のトークンIDが割り当てられる.
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)  # (クラス数, 77)  tokenized_prompts[0] = tensor([49406,   343,   343,   343,   343,   343,   343,   343,   343,  1929, 269, 49407, 0, 0, ...])
        with torch.no_grad():
            # トークン化されたテキストを入力し、トークンIDを埋め込みベクトルに変換する (トークン埋め込み).
            embedding = self.clip_model.token_embedding(self.tokenized_prompts).type(self.clip_model.dtype)  # (クラス数 (入力文の数), 77 (各文のトークン数（シーケンス長）), self.EMBEDDING_DIM)
        
        # SOS (Start of Sentence): SOSトークンは、文の開始を示すために使用されます。通常、モデルへの入力テキストの最初に追加されます。SOSトークンは、モデルに対して文の生成を開始するよう指示します。
        # 各クラスの埋め込みトークンの最初の次元がSOS.
        # SOS
        self.register_buffer('token_prefix', embedding[:, :1, :])  # (クラス数, 1, self.EMBEDDING_DIM)  [-0.0001,  0.0002, -0.0046,  ...,  0.0010,  0.0025,  0.0049]
        # CLS, EOS
        self.register_buffer('token_suffix', embedding[:, self.hparams['num_domain_tokens'] + 1:, :])  # (クラス数, 68, self.EMBEDDING_DIM)  68 := 77 - num_domain_tokens_tokens - 2. , [ 0.0013,  0.0046, -0.0115,  ...,  0.0112,  0.0147,  0.0040],...,.
        
        # F() Network for extraction of domain feature.
        ######################################################
        ######################################################
        ###### ここで, self.modelとしているのは F() Network. 元コードのpre-trained modelとは異なることに注意.
        ###### 既存コードを書きたくなかったからこうした．
        ###### どちらも同じ "Student model"という意味では共通している.
        ######################################################
        ######################################################
        self.model = networks.MLP(self.EMBEDDING_DIM, self.EMBEDDING_DIM * self.hparams['num_domain_tokens'], self.hparams).to(device=self.device, dtype=self.clip_model.dtype)
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        
        self.model.apply(init_weights)
        
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams['lr'],
            momentum=self.hparams['momentum']
        )

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def warmup(self):
        logger.info(f"Starting warm up...{self.warmup_steps} steps")
        for i in range(self.warmup_steps):
            if i % 100 == 0:
                logger.info(f"warm up step {i}")
            #linearly increase the learning rate
            for par in self.optimizer.param_groups:
                par["lr"] = self.final_lr * (i+1) / self.warmup_steps

            # sample source batch
            try:
                batch = next(self.src_loader_iter)
            except StopIteration:
                self.src_loader_iter = iter(self.src_loader)
                batch = next(self.src_loader_iter)

            imgs_src, labels_src = batch[0], batch[1]
            imgs_src, labels_src = imgs_src.cuda(), labels_src.cuda().long()

            logits_per_image, logits_per_image_ema = self._calc_clip_logits(imgs_src, labels_src)
            loss = symmetric_cross_entropy(logits_per_image, logits_per_image_ema).mean(0)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.m_teacher_momentum)

        logger.info(f"Finished warm up...")
        for par in self.optimizer.param_groups:
            par["lr"] = self.final_lr


    def _calc_clip_logits(self, imgs, labels=None):
        """
            imgs: (B, 3, 224, 224)
        """
        image_features = self.clip_model.encode_image(imgs)  # (B, EMBEDDING_DIM)

        # Student / Teacher Model
        domain_features = self.model(image_features)  # (B, EMBEDDING_DIM * num_domain_tokens)
        mean_domain_features = domain_features.mean(dim=0, keepdim=True)  # バッチ平均をとる. (1, EMBEDDING_DIM * num_domain_tokens)
        # 各クラス特徴と結合するために，クラス数文複製する.
        _mean_domain_features = mean_domain_features.repeat_interleave(len(self.hparams['class_names']), dim=0)  # (Sourceクラス数, EMBEDDING_DIM * num_domain_tokens)

        domain_features_ema = self.model_ema(image_features)  # (B, EMBEDDING_DIM * num_domain_tokens)
        mean_domain_features_ema = domain_features_ema.mean(dim=0, keepdim=True)  # バッチ平均をとる. (1, EMBEDDING_DIM * num_domain_tokens)
        _mean_domain_features_ema = mean_domain_features_ema.repeat_interleave(len(self.hparams['class_names']), dim=0)  # (Sourceクラス数, EMBEDDING_DIM * num_domain_tokens)
        
        # Text Features
        # prefix + domain_features + suffix(=CLS, EOS) である Text Featuresを作成する.
        text_features = self._get_text_features(_mean_domain_features)  # (Sourceクラス数, EMBEDDING_DIM)
        text_features_ema = self._get_text_features(_mean_domain_features_ema)  # (Sourceクラス数, EMBEDDING_DIM)

        # Normalize Features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_ema = text_features_ema / text_features_ema.norm(dim=-1, keepdim=True)
        
        # calc logits  t()は転置, 
        # self.clip_model.logit_scale.exp()によるスケール変換は, 類似度スコアの非負化を行い、類似度の解釈や比較を容易にし，指数関数によるスケール変換は正規化や確率的な処理にも関連する．
        logits_per_image = self.clip_model.logit_scale.exp() * image_features @ text_features.t()  # (B, Sourceクラス数)
        logits_per_image_ema = self.clip_model.logit_scale.exp() * image_features @ text_features_ema.t()  # (B, Sourceクラス数)
        
        return logits_per_image, logits_per_image_ema

            
    def _get_text_features(self, domain_feature, coop=False):
        """ domain feature + text feature 
            domain_feature: (Sourceクラス数, EMBEDDING_DIM * num_domain_tokens)
        """
        domain_feature = domain_feature.reshape(-1, self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)  # (Sourceクラス数, num_domain_tokens, EMBEDDING_DIM)
        # self.token_prefix.shape:  SOS (Sourceクラス数, 1, EMBEDDING_DIM)
        # domain_feature         :  CLS (Sourceクラス数, self.hparams['num_domain_tokens'], EMBEDDING_DIM))
        # self.token_suffix.shape:  EOS (Sourceクラス数, 77 - 1 - self.hparams['num_domain_tokens'], EMBEDDING_DIM)
        prompts_with_domain_features = torch.cat([self.token_prefix, domain_feature, self.token_suffix], dim=1)  # (Sourceクラス数, 77, EMBEDDING_DIM)
        
        # refer CoOp: CoOP github. https://github.com/KaiyangZhou/CoOp/blob/b0a058869cef00a4e4ea5256d40fd7681119c099/trainers/coop.py#L46
        # domain_featureに位置エンコーディングを加え，位置情報をembedding空間に反映する．
        x = prompts_with_domain_features + self.clip_model.positional_embedding.type(self.clip_model.dtype)  # (Sourceクラス数, 77, EMBEDDING_DIM)
        x = x.permute(1, 0, 2)  # (77, Sourceクラス数, EMBEDDING_DIM)
        x = self.clip_model.transformer(x)  # (77, Sourceクラス数, EMBEDDING_DIM)
        x = x.permute(1, 0, 2)  # (Sourceクラス数, 77, EMBEDDING_DIM)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)  # (Sourceクラス数, 77, EMBEDDING_DIM)
        
        # mapping domain_features to text_features.
        # xがドメイン特徴(にtransformerを適用したもの)，self.tokenized_promptsがテキスト特徴．
        text_features = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection  # (Sourceクラス数, EMBEDDING_DIM))
        return text_features


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

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        imgs_test = x[0]
        imgs_test_aug = self.tta_transform((imgs_test))

        self.optimizer.zero_grad()
        ###########################################################################################
        ###########################################################################################
        # # forward original test data
        # features_test = self.feature_extractor(imgs_test)
        # outputs_test = self.classifier(features_test)

        # # forward augmented test data
        # features_aug_test = self.feature_extractor(self.tta_transform((imgs_test)))
        # outputs_aug_test = self.classifier(features_aug_test)

        # # forward original test data through the ema model
        # outputs_ema = self.model_ema(imgs_test)

        ##### Contrastive Lossのために, prototypes_srcとstudentの特徴ベクトルの類似度を計算する．
        # with torch.no_grad():
        #     # dist[:, i] contains the distance from every source sample to one test sample
        #     # self.prototypes_srcは、ソースプロトタイプの特徴ベクトルを表すテンソル. './ckpt/prototypes/protos_cifar10_c_Standard.pth'から読み込んだもの.
        #     dist = F.cosine_similarity(
        #         x1=self.prototypes_src.repeat(1, features_test.shape[0], 1),
        #         x2=features_test.view(1, features_test.shape[0], features_test.shape[1]).repeat(self.prototypes_src.shape[0], 1, 1),
        #         dim=-1)

        #     # for every test feature, get the nearest source prototype and derive the label
        #     # 指定した次元でテンソル内の最大の要素とそのインデックスを取得する
        #     _, indices = dist.topk(1, largest=True, dim=0)
        #     indices = indices.squeeze(0)

        # features = torch.cat([self.prototypes_src[indices],
        #                       features_test.view(features_test.shape[0], 1, features_test.shape[1]),
        #                       features_aug_test.view(features_test.shape[0], 1, features_test.shape[1])], dim=1)
        # # 式(7)
        # loss_contrastive = self.contrastive_loss(features=features, labels=None)
        # # 式(6)
        # loss_entropy = self_training(x=outputs_test, x_aug=outputs_aug_test, x_ema=outputs_ema).mean(0)
        # # 式(9)のうち, targetドメインに対するLoss (第1,2項)
        # loss_trg = self.lambda_ce_trg * loss_entropy + self.lambda_cont * loss_contrastive
        # loss_trg.backward()
        #########################################################################################
        #########################################################################################
        logits_per_image, logits_per_image_ema = self._calc_clip_logits(imgs_test)
        logits_per_image_aug, _ = self._calc_clip_logits(imgs_test_aug)

        loss_entropy = self_training(x=logits_per_image, x_aug=logits_per_image_aug, x_ema=logits_per_image_ema).mean(0)
        # loss_trg = self.lambda_ce_trg * loss_entropy + self.lambda_cont * loss_contrastive
        loss_trg = loss_entropy
        loss_trg.backward()

        if self.lambda_ce_src > 0:
            ############ Train on labeled source data -> 今回は完全Source-Freeとしたい. ############
            raise ValueError('----- Training on labeled source data -----')
            # logger.info('----- Training on labeled source data -----')
            # # sample source batch
            # try:
            #     batch = next(self.src_loader_iter)
            # except StopIteration:
            #     self.src_loader_iter = iter(self.src_loader)
            #     batch = next(self.src_loader_iter)

            # # train on labeled source data
            # imgs_src, labels_src = batch[0], batch[1]
            # features_src = self.feature_extractor(imgs_src.cuda())
            # outputs_src = self.classifier(features_src)
            # loss_ce_src = F.cross_entropy(outputs_src, labels_src.cuda().long())
            # loss_ce_src *= self.lambda_ce_src
            # loss_ce_src.backward()

        self.optimizer.step()

        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.m_teacher_momentum)

        # create and return the ensemble prediction
        # return outputs_test + outputs_ema
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

    # pre-trained model(e.g. ViT)の中でパラメータ更新する箇所を指定する
    # @staticmethod
    # def configure_model(model):
    #     """Configure model"""
    #     # model.train()
    #     model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
    #     # disable grad, to (re-)enable only what we update
    #     model.requires_grad_(False)
    #     # enable all trainable
    #     for m in model.modules():
    #         if isinstance(m, nn.BatchNorm2d):
    #             m.requires_grad_(True)
    #             # force use of batch stats in train and eval modes
    #             m.track_running_stats = False
    #             m.running_mean = None
    #             m.running_var = None
    #         elif isinstance(m, nn.BatchNorm1d):
    #             m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
    #             m.requires_grad_(True)
    #         else:
    #             m.requires_grad_(True)
    #     return model

@torch.jit.script
def self_training(x, x_aug, x_ema):# -> torch.Tensor:
    """ 式(6) """
    return - 0.25 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - 0.25 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1) \
           - 0.25 * (x_ema.softmax(1) * x_aug.log_softmax(1)).sum(1) - 0.25 * (x_aug.softmax(1) * x_ema.log_softmax(1)).sum(1)

@torch.jit.script
def symmetric_cross_entropy(x, x_ema):# -> torch.Tensor:
    return -0.5 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - 0.5 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)

def to_numpy_array(lst):
    np_array = []
    for item in lst:
        if isinstance(item, torch.Tensor):
            np_array.append(item.cpu().detach().numpy())
        elif isinstance(item, list):
            np_array.append(to_numpy_array(item))
    return np.array(np_array)