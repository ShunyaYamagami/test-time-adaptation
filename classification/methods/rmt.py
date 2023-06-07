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

logger = logging.getLogger(__name__)


class RMT(TTAMethod):
    def __init__(self, cfg, hparams, steps, episodic, window_length, dataset_name, arch_name, src_loader, ckpt_dir, ckpt_path,
                 contrast_mode, temperature, projection_dim, lambda_ce_src, lambda_ce_trg, lambda_cont, m_teacher_momentum, num_samples_warm_up, save_dir, model_weights):
        super().__init__(steps, episodic, window_length)
        logger.info(f"----- RMT __init__ -----")
        ####################################################################################
        ####################################################################################
        if hparams['cuda_visible_devices'] == [0]:
            hparams['architecture']['domain_learning'] = True
            hparams['architecture']['domain_token_dim'] = 0
            hparams['pretrain']['load'] = True
        elif hparams['cuda_visible_devices'] == [2]:
            hparams['architecture']['domain_learning'] = True
            hparams['architecture']['domain_token_dim'] = 8
            hparams['pretrain']['load'] = False
        ####################################################################################
        ####################################################################################
        self.cfg = cfg
        self.hparams = hparams
        assert isinstance(self.hparams['cuda_visible_devices'], list) and all(isinstance(x, int) for x in self.hparams['cuda_visible_devices']), "cuda_visible_devices must be list of int."
        assert isinstance(self.hparams['exec_num'], int)
        assert self.hparams['domain_loss']['method'] in ['nt_xent', 'mine']
        assert self.hparams['pretrain']['load'] and not self.hparams['warmup']['load_model'] or not self.hparams['pretrain']['load'], 'cannot be warmup.load_model == True when pretrain.load is True'
        assert self.hparams['warmup']['load_model'] and self.hparams['warmup']['use'] or not self.hparams['warmup']['load_model'], "if load_model is True, use must be True"
        assert self.hparams['warmup']['use'] and num_samples_warm_up > 0 or not self.hparams['warmup']['use'], "warmup_steps must be set when warmup is used" 

        ########## Set save_dir ##########
        if self.hparams['rename_save_dir']:
            old_save_dir = save_dir
            dirname = f"{'_'.join(Path(old_save_dir).name.split('_')[:3])}--warm_{hparams['warmup']['use']}--proto_{hparams['prototypes']['use']}--domain_{hparams['domain_loss']['method']}"
            dirname = dirname.replace('False', 'F').replace('True', 'T').replace('my_transformer', 'tf')
            save_dir = str(Path(old_save_dir).parent / dirname)
            os.rename(old_save_dir, save_dir)
            logger.info(f"----- Rename save_dir: {save_dir} -----")

        with open(os.path.join(save_dir, "hparams.yaml"), 'w') as file:
            yaml.dump(self.hparams, file, default_flow_style=False, sort_keys=False)

        ########## Set Parameters ##########
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.EMBEDDING_DIM = 512
        class_names = src_loader.dataset.classes
        num_classes = len(class_names)
        self.src_loader = src_loader
        self.src_loader_iter = iter(self.src_loader)
        self.pretrained_dir = os.path.join(ckpt_dir, dataset_name.replace('_c', ''), 'pretrained')
        # self.lambda_ce_src = lambda_ce_src
        # self.lambda_ce_trg = lambda_ce_trg
        # self.lambda_cont = lambda_cont
        # self.m_teacher_momentum = m_teacher_momentum
        self.save_dir = save_dir
        self.m_teacher_momentum = m_teacher_momentum
        self.normal_transform, self.clsdst_transform = get_transform_when_iteration(grid=4)
        # 自動的にレイヤ毎に最適なビット精度を選択してくれる（convはfp16, bnはfp32等）. ベストプラクティスを選択してくれるため、便利。use_amp=Falseではfp32を使用する。
        self.scaler = GradScaler(enabled=self.hparams['mixed_precision'])
                
        self.clip_model, self.tokens = set_clip_models(self.hparams, device, class_names)
        if self.hparams['warmup']['use']:
            self.warmup_trainer = WarmUpTrainer(arch_name, ckpt_dir, ckpt_path, dataset_name, num_samples_warm_up)
            if self.hparams['warmup']['load_model']:
                self.warmup_trainer.load_models()
            else:
                self.warmup_trainer.warmup()
        self.models = set_models(hparams, self.EMBEDDING_DIM, self.clip_model.dtype, device)
        self.optimizers = set_optimizers(hparams, self.models)
        self.self_trainer = SelfTrainer(hparams, self.clip_model, self.tokens, self.EMBEDDING_DIM, self.normal_transform, num_classes, dataset_name, device)
        self.domain_trainer = DomainTrainer(hparams, self.clip_model, self.tokens, self.models['mine'], self.EMBEDDING_DIM, self.clsdst_transform, self.src_loader.batch_size, device)
        self.final_lr = self.optimizers.param_groups[0]['lr']

    def learning(self, x):
        self.optimizers.zero_grad()
        self_train_loss, logits_st, logits_ema = self.self_trainer(x, self.models)
        domain_loss = self.domain_trainer(x, self.models)

        loss = self_train_loss + domain_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizers)
        self.scaler.update()
        self.update_ema_variables(self.m_teacher_momentum)

        return logits_st + logits_ema
    
    def update_ema_variables(self, alpha_teacher):
        """ Studentにした変更を, EMA(Exponential Moving Average) によりTeacherモデルへ反映する. 
        :param ema_model: Teacher model
        :param model: Student model
        :param alpha_teacher: weight
        """
        for ema_param, param in zip(self.models['model_ema'].parameters(), self.models['model_st'].parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
        # return self.models['model_ema']

    def not_prompt_tuning_learning(self, x):
        x_cls = self.normal_transform(x)  # (B, 3, 224, 224)
        with torch.no_grad():
            image_fts = self.clip_model.encode_image(x_cls)  # (B, EMBEDDING_DIM)
            text_fts = self.clip_model.encode_text(self.tokens['tokenized_prompts'])
        norm_image_fts = F.normalize(image_fts)
        norm_text_fts = F.normalize(text_fts)
        logits = self.clip_model.logit_scale.exp() * norm_image_fts @ norm_text_fts.t()  # (B, num_classes)
        return logits

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        if self.hparams['architecture']['prompt_tuning']:
            ensembled_logits = self.learning(x[0])
            return ensembled_logits
        else:
            logits = self.not_prompt_tuning_learning(x[0])
            return logits


class WarmUpTrainer(nn.Module):
    def __init__(self, arch_name, ckpt_dir, ckpt_path, dataset_name, num_samples_warm_up):
        super().__init__()
        # if self.hparams['warmup']['use'] and self.warmup_steps > 0:
        src_batch_size = self.src_loader.batch_size
        self.warmup_steps = num_samples_warm_up // src_batch_size
        self.warmup_ckpt_path = os.path.join(ckpt_dir, "warmup")
        if dataset_name == "domainnet126":
            source_domain = ckpt_path.split(os.sep)[-1].split('_')[1]
            ckpt_path = f"ckpt_warmup_{dataset_name}_{source_domain}_{arch_name}_bs{src_batch_size}_step{self.warmup_steps}.pth"
        else:
            ckpt_path = f"ckpt_warmup_{dataset_name}_{arch_name}_bs{src_batch_size}_step{self.warmup_steps}.pth"
        self.ckpt_path = os.path.join(self.warmup_ckpt_path, ckpt_path)
                
    def load_models(self):
        assert os.path.exists(self.ckpt_path)
        assert not self.hparams['pretrain']['load'], 'せっかくのpretrained-modelが上書きされてしまう.'
        logger.info(f"Loading warmup checkpoint... from {self.ckpt_path}")
        checkpoint = torch.load(self.ckpt_path)
        for model_name in self.models.keys():
            self.models[model_name].load_state_dict(checkpoint[model_name])
        for optim_name in self.optimizers.keys():
            self.optimizers[optim_name].load_state_dict(checkpoint[optim_name])
        logger.info(f"Loaded from {self.ckpt_path}")

    def warmup(self):
        os.makedirs(self.warmup_ckpt_path, exist_ok=True)
        logger.info(f"----- WarmUp()  -----")
        logger.info(f"Starting warm up...{self.warmup_steps} steps")
        for i in tqdm.tqdm(range(self.warmup_steps)):
            for par in self.optimizers.param_groups:
                par['lr'] = self.final_lr * (i+1) / self.warmup_steps
            try:
                src_batch = next(self.src_loader_iter)
            except StopIteration:
                self.src_loader_iter = iter(self.src_loader)
                src_batch = next(self.src_loader_iter)
            self.learning(src_batch[0].to(self.device), warmup=True)  ###################################
            raise ValueError("これあっている？")

        logger.info(f"Finished warm up...")
        for par in self.optimizers.param_groups:
            par['lr'] = self.final_lr
            state_dicts = {}
            for model_name in self.models.keys(): 
                state_dicts[model_name] = self.models[model_name].state_dict()
            for optim_name in self.optimizers.keys():
                state_dicts[optim_name] = self.optimizers[optim_name].state_dict()
            torch.save(state_dicts, self.ckpt_path)


class SelfTrainer(nn.Module):
    def __init__(self, hparams, clip_model, tokens, EMBEDDING_DIM, normal_transform, num_classes, dataset_name, device):
        super().__init__()
        self.hparams = hparams
        self.clip_model = clip_model
        self.tokens = tokens
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.normal_transform = normal_transform
        self.tta_transform = get_tta_transforms(dataset_name)
        self.num_classes = num_classes
        self.device = device

    def forward(self, x, models):
        x_trans = self.normal_transform(x)  # (B, 3, 224, 224)
        x_aug = self.tta_transform(x_trans)
        image_fts = self.clip_model.encode_image(x_trans)  # (B, EMBEDDING_DIM)
        image_aug_fts = self.clip_model.encode_image(x_aug)  # (B, EMBEDDING_DIM)
        logits_st = self.get_logits(models['model_st'], image_fts)  # (B, num_classes)
        logits_ema = self.get_logits(models['model_ema'], image_fts)  # (B, num_classes)
        logits_aug = self.get_logits(models['model_st'], image_aug_fts)  # (B, num_classes)
        
        # Loss 式(6)
        loss_entropy = self_training_loss(x=logits_st, x_aug=logits_aug, x_ema=logits_ema).mean(0)
        # Loss 式(9)のうち, targetドメインに対するLoss (第1項)
        loss_trg = loss_entropy

        return loss_trg, logits_st, logits_ema
        
    def get_logits(self, model, image_fts):
        sttc_fts = model(image_fts)  # (B, EMBEDDING_DIM * num_domain_tokens)
        mean_sttc_fts = sttc_fts.mean(dim=0, keepdim=True)  # バッチ平均をとる. (1, EMBEDDING_DIM * num_domain_tokens)
        repeated_sttc_fts = mean_sttc_fts.repeat_interleave(self.num_classes, dim=0)  # 各クラス特徴と結合するために，クラス数文複製する. (num_classes, EMBEDDING_DIM * num_domain_tokens)

        text_fts = self._cat_text_features(repeated_sttc_fts)  # (num_classes, EMBEDDING_DIM)
        norm_text_fts = F.normalize(text_fts)
        norm_image_fts = F.normalize(image_fts)
        logits = self.clip_model.logit_scale.exp() * norm_image_fts @ norm_text_fts.t()  # (B, num_classes)  # self.clip_model.logit_scale.exp()によるスケール変換は, 類似度スコアの非負化を行い、類似度の解釈や比較を容易にし，指数関数によるスケール変換は正規化や確率的な処理にも関連する．
        return logits

    def _cat_text_features(self, sttc_fts):
        sttc_fts = sttc_fts.reshape(-1, self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)  # (L, num_domain_tokens, EMBEDDING_DIM)
        param_fts = torch.cat([self.tokens['token_prefix'], sttc_fts, self.tokens['token_suffix']], dim=1)  # (L, 77, EMBEDDING_DIM)

        # refer CoOp: CoOP github. https://github.com/KaiyangZhou/CoOp/blob/b0a058869cef00a4e4ea5256d40fd7681119c099/trainers/coop.py#L46
        # domain_featureに位置エンコーディングを加え，位置情報をembedding空間に反映する．
        x = param_fts + self.clip_model.positional_embedding.type(self.clip_model.dtype)  # (L, 77, EMBEDDING_DIM)
        x = x.permute(1, 0, 2)  # (77, L, EMBEDDING_DIM)
        x = self.clip_model.transformer(x)  # (77, L, EMBEDDING_DIM)
        x = x.permute(1, 0, 2)  # (L, 77, EMBEDDING_DIM)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)  # (L, 77, EMBEDDING_DIM)
        
        sttc_text_fts = x[torch.arange(x.shape[0]), self.tokens['tokenized_prompts'].argmax(dim=-1)] @ self.clip_model.text_projection  # (L, EMBEDDING_DIM))
        return sttc_text_fts
    


class DomainTrainer(nn.Module):
    def __init__(self, hparams, clip_model, tokens, mine, EMBEDDING_DIM, clsdst_transform, batch_size, device):
        super().__init__()
        assert hparams['domain_loss']['method'] in ['mine']
        self.hparams = hparams
        self.clip_model = clip_model
        self.tokens = tokens
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.clsdst_transform = clsdst_transform
        self.device = device
        
        self.skp = skp.SinkhornKnopp()
        n_clusters=(2, 2)
        self.biclust_model = SpectralBiclustering(n_clusters=n_clusters, method="log", random_state=0)
        if self.hparams['domain_loss']['method'] == 'mine':
            self.mine_trainer = MineTrainer(mine)
        elif self.hparams['domain_loss']['method'] == 'nt_xent':
            self.ntxent_criterion = NTXentLoss(self.device, batch_size, self.hparams['domain_loss']['nt_xent_temperature'])

    def forward(self, x, models):
        x_clsdst = self.clsdst_transform(x)  # (B, 3, 224, 224)
        image_clsdst_fts = self.clip_model.encode_image(x_clsdst)  # (B, EMBEDDING_DIM)
        st_clsdst_fts = models['model_st'](image_clsdst_fts)  # (B, EMBEDDING_DIM * num_domain_tokens)
        ### TODO: Domain Projectorいらないかも．てか多分いらない．
        domain_fts = models['domain_projector'](st_clsdst_fts)  # (B, EMBEDDING_DIM * num_domain_tokens)
        domain_text_fts = self._cat_text_features(domain_fts)  # (B, EMBEDDING_DIM)

        image_clsdst_fts = F.normalize(image_clsdst_fts)
        domain_text_fts = F.normalize(domain_text_fts)
        
        if self.hparams['domain_loss']['method'] == "nt_xent":
            domain_loss = self.ntxent_criterion(image_clsdst_fts, domain_text_fts)
        elif self.hparams['domain_loss']['method'] == 'mine':
            # 流石にデータセット単位で類似度計算を行うと，10,000*10,000の計算量となるので，バッチごとに行う. そのため，バッチサイズは大きめでなくてはならない.
            sim_mtx = F.cosine_similarity(
                    x1=image_clsdst_fts.unsqueeze(0),  # (1, B, EMBEDDING_DIM)
                    x2=domain_text_fts.unsqueeze(1),  # (B, 1, EMBEDDING_DIM)
                    dim=-1).detach().cpu().numpy()  # (B, B)

            bistochastic_mtx = self._get_bistochastic_mtx(sim_mtx)  # (B, B)
            clustered_mtx = self._biclustering(bistochastic_mtx)  # (B, B)

            diag = torch.diag(clustered_mtx).long()
            mean_feat_per_clust = [domain_text_fts[diag == clust].mean(dim=0) for clust in sorted(torch.unique(diag))]

            domain_loss = 0.
            for i in range(len(mean_feat_per_clust)):
                for j in range(i+1, len(mean_feat_per_clust)):
                    data = torch.stack([mean_feat_per_clust[i], mean_feat_per_clust[j]], dim=1)
                    mine_loss, _ = self.mine_trainer.get_loss(data)
                    domain_loss += mine_loss

        return domain_loss
    

    def _cat_text_features(self, domain_fts):
        domain_fts = domain_fts.reshape(-1, self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)  # (L, num_domain_tokens, EMBEDDING_DIM)

        batch_size = domain_fts.shape[0]
        rep_token_prefix = self.tokens['domain_token_prefix'].repeat_interleave(batch_size, dim=0)  # (L, 1, EMBEDDING_DIM)
        rep_token_suffix = self.tokens['domain_token_suffix'].repeat_interleave(batch_size, dim=0)  # (L, 77-1-num_domain_tokens, EMBEDDING_DIM)
        param_fts = torch.cat([rep_token_prefix, domain_fts, rep_token_suffix], dim=1)  # (L, 77, EMBEDDING_DIM)
        tokenized_prompts = self.tokens['domain_tokenized_prompts']  # (L, 77)
    
        x = param_fts + self.clip_model.positional_embedding.type(self.clip_model.dtype)  # (L, 77, EMBEDDING_DIM)
        x = x.permute(1, 0, 2)  # (77, L, EMBEDDING_DIM)
        x = self.clip_model.transformer(x)  # (77, L, EMBEDDING_DIM)
        x = x.permute(1, 0, 2)  # (L, 77, EMBEDDING_DIM)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)  # (L, 77, EMBEDDING_DIM)
        
        domain_text_fts = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection  # (L, EMBEDDING_DIM))
        return domain_text_fts


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
        
        

@torch.jit.script
def self_training_loss(x, x_aug, x_ema):# -> torch.Tensor:
    """ 式(6) """
    return - 0.25 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - 0.25 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1) \
           - 0.25 * (x_ema.softmax(1) * x_aug.log_softmax(1)).sum(1) - 0.25 * (x_aug.softmax(1) * x_ema.log_softmax(1)).sum(1)

@torch.jit.script
def symmetric_cross_entropy(x, x_ema):# -> torch.Tensor:
    return -0.5 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - 0.5 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)


def set_clip_models(hparams, device, class_names):
    # embedding dim for image and text encoder.
    logger.info(f"----- Set Clip Models,  clip_backbone : {hparams['clip_backbone']}  -----")
    clip_model, preprocess = clip.load(hparams['clip_backbone'], device=device)
    clip_model = clip_model.float()

    # CLIPモデルのパラメータは更新させない
    logger.info('Set self.clip_model.parameters.reguires_grad = False!')
    for param in clip_model.parameters():
        param.requires_grad = False

    ##### Class Prompt用  refer DPLCLIP
    prompt_prefix = ' '.join(['X'] * hparams['num_domain_tokens'])
    
    if hparams['sentence_prompt']:
        logger.info('Using sentence_prompt in DPLCLIP...')
        classnames = [f"a photo of a {name.replace('_', ' ')}" for name in class_names]
    else:
        classnames = [name.replace('_', ' ') for name in class_names]

    ##### to get default token_prefix and token_suffix.
    tokens = {}
    class_prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]  # (クラス数)
    tokens['tokenized_prompts'] = torch.cat([clip.tokenize(p) for p in class_prompts]).to(device)  # (クラス数, 77)
    embedding = clip_model.token_embedding(tokens['tokenized_prompts']).type(clip_model.dtype)  # (クラス数 (入力文の数), 77 (各文のトークン数（シーケンス長）), self.EMBEDDING_DIM)
    tokens['token_prefix'] = embedding[:, :1, :]  # SOS (クラス数, 1, self.EMBEDDING_DIM)  各クラスの埋め込みトークンの最初の次元がSOS. SOS (Start of Sentence): SOSトークンは、文の開始を示すために使用されます。通常、モデルへの入力テキストの最初に追加されます。SOSトークンは、モデルに対して文の生成を開始するよう指示します。
    tokens['token_suffix'] = embedding[:, hparams['num_domain_tokens'] + 1:, :]  # CLS, EOS (クラス数, 68, EMBEDDING_DIM)  68 := 77 - num_domain_tokens_tokens - 2.
    
    ##### Domain Prompt用  refer DPLCLIP
    if hparams['architecture']['domain_learning']:
        domain_prompts = prompt_prefix + 'a photo of a '
        tokens['domain_tokenized_prompts'] = clip.tokenize(domain_prompts).to(device)  # (1, 77)
        domain_embedding = clip_model.token_embedding(tokens['domain_tokenized_prompts']).type(clip_model.dtype)  # (1, 77 (各文のトークン数（シーケンス長）), EMBEDDING_DIM)
        tokens['domain_token_prefix'] = domain_embedding[:, :1:, :]  # SOS (1, 1, EMBEDDING_DIM)
        tokens['domain_token_suffix'] = domain_embedding[:, hparams['num_domain_tokens'] + 1:, :]  # CLS, EOS (1, 68, EMBEDDING_DIM)  68 := 77 - num_domain_tokens_tokens - 2.
        
    return clip_model, tokens


def set_models(hparams, EMBEDDING_DIM, clip_model_dtype, device):
    rtn_models = {}
    ##### Self Training
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    rtn_models['model_st'] = networks.MLP(EMBEDDING_DIM,
                                EMBEDDING_DIM * hparams['num_domain_tokens'],
                                hparams
                                ).to(device=device, dtype=clip_model_dtype)
    rtn_models['model_st'].apply(init_weights)
    rtn_models['model_ema'] = TTAMethod.copy_model(rtn_models['model_st'])
    for param in rtn_models['model_ema'].parameters():
        param.detach_()

    ##### Domain Learning
    if hparams['architecture']['domain_learning']:
        rtn_models['domain_projector'] = nn.Sequential(nn.Linear(EMBEDDING_DIM * hparams['num_domain_tokens'], EMBEDDING_DIM * hparams['num_domain_tokens'])).to(device=device, dtype=clip_model_dtype)
    if hparams['domain_loss']['method'] == 'mine':
        rtn_models['mine'] = Mine().to(device)
    
    return rtn_models


def set_optimizers(hparams, models):
    assert hparams['optimizer'] in ["Adam", "SGD"]
    class_args_params = []
    for model_name, model in models.items():
        class_args_params.append({'params': model.parameters()})
    if hparams['optimizer'] == "Adam":
        optimizers = torch.optim.Adam(class_args_params, lr=hparams['lr'])
    elif hparams['optimizer'] == "SGD":
        optimizers = torch.optim.SGD(class_args_params, lr=hparams['lr'], momentum=hparams['momentum'])

    return optimizers


def to_numpy_array(lst):
    np_array = []
    for item in lst:
        if isinstance(item, torch.Tensor):
            np_array.append(item.cpu().detach().numpy())
        elif isinstance(item, list):
            np_array.append(to_numpy_array(item))
    return np.array(np_array)