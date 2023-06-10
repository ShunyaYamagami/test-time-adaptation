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
from clip.model import CLIP
from sinkhorn_knopp import sinkhorn_knopp as skp
# https://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_coclustering.html#sphx-glr-auto-examples-bicluster-plot-spectral-coclustering-py
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering
from pathlib import Path
from functools import partial
from matplotlib import pyplot as plt

from conf import get_num_classes
from domainbed import networks
from methods.base import TTAMethod
from augmentations.transforms_cotta import get_tta_transforms
from augmentations.aug import get_transform_when_iteration
from models.model import get_model, split_up_model
from models.mine import Mine, MineTrainer
from loss.nt_xent import NTXentLoss
from methods.my_modules import ImageEncoder, TextEncoder, PrototypeRunner
logger = logging.getLogger(__name__)


class RMT(TTAMethod):
    def __init__(self, cfg, hparams, steps, episodic, window_length, dataset_name, arch_name, src_loader, ckpt_dir, ckpt_path,
                 contrast_mode, temperature, projection_dim, lambda_ce_src, lambda_ce_trg, lambda_cont, m_teacher_momentum, num_samples_warm_up, save_dir, model_weights):
        super().__init__(steps, episodic, window_length)
        ####################################################################################
        ####################################################################################
        # if hparams['cuda_visible_devices'] == [0]:
        #     hparams['architecture']['domain_learning'] = True
        #     hparams['warmup']['use'] = True
        #     hparams['warmup']['load'] = True
        # if hparams['cuda_visible_devices'] == [2]:
        #     hparams['architecture']['domain_learning'] = True
        #     hparams['warmup']['use'] = True
        #     hparams['warmup']['load'] = True
        ####################################################################################
        ####################################################################################
        self.cfg = cfg
        self.hparams = hparams
        assert isinstance(self.hparams['cuda_visible_devices'], list) and all(isinstance(x, int) for x in self.hparams['cuda_visible_devices']), "cuda_visible_devices must be list of int."
        assert isinstance(self.hparams['exec_num'], int)
        assert self.hparams['sttc'] in ['linear', 'mlp']
        assert self.hparams['domain_loss']['method'] in ['nt_xent', 'mine']
        assert self.hparams['domain_loss']['prompt'] in [False, 'classname']
        assert self.hparams['warmup']['load'] and self.hparams['warmup']['use'] or not self.hparams['warmup']['load'], "if load_model is True, use must be True"
        assert self.hparams['warmup']['use'] and num_samples_warm_up > 0 or not self.hparams['warmup']['use'], "warmup_steps must be set when warmup is used" 

        ########## Set save_dir ##########
        if self.hparams['rename_save_dir']:
            old_save_dir = save_dir
            dirname = f"{'_'.join(Path(old_save_dir).name.split('_')[:3])}--warm_{hparams['warmup']['use']}--proto_{hparams['prototypes']['use']}--domain_{hparams['domain_loss']['method']}"
            dirname = dirname.replace('False', 'F').replace('True', 'T')
            save_dir = str(Path(old_save_dir).parent / dirname)
            os.rename(old_save_dir, save_dir)
            logger.info(f"----- Rename save_dir: {save_dir} -----")

        with open(os.path.join(save_dir, "hparams.yaml"), 'w') as file:
            yaml.dump(self.hparams, file, default_flow_style=False, sort_keys=False)

        ########## Set Parameters ##########
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.EMBEDDING_DIM = 512
        class_names = src_loader.dataset.classes
        num_classes = len(class_names)
        self.src_loader = src_loader
        self.src_loader_iter = iter(self.src_loader)
        self.save_dir = save_dir
        self.m_teacher_momentum = m_teacher_momentum
        self.normal_transform, self.clsdst_transform = get_transform_when_iteration(grid=4)
        self.scaler = GradScaler(enabled=self.hparams['mixed_precision'])  # 自動的にレイヤ毎に最適なビット精度を選択してくれる（convはfp16, bnはfp32等）. ベストプラクティスを選択してくれるため、便利。use_amp=Falseではfp32を使用する。
                
        self.clip_model, preprocess = clip.load(hparams['clip_backbone'], device=self.device)
        self.clip_model = self.clip_model.float()
        logger.info('Set self.clip_model.parameters.reguires_grad = False!')
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.prompt_learner = PromptLearner(hparams, self.clip_model, class_names, num_classes, self.device)
        
        if not self.hparams['architecture']['clip_only']:
            self.models = set_models(hparams, self.EMBEDDING_DIM, self.clip_model.dtype, self.device)
            if self.hparams['architecture']['learnable_parameters']:
                self.models['prompt_learner'] = self.prompt_learner
            self.optimizers = set_optimizers(hparams, self.models)
            self.prototypes_src = None
            if self.hparams['prototypes']['use']:
                prototype_runner = PrototypeRunner(hparams, self.clip_model, src_loader, self.normal_transform, self.clsdst_transform, num_classes, arch_name, ckpt_dir, ckpt_path, dataset_name, self.device)
                if self.hparams['prototypes']['load']:
                    self.prototypes_src = prototype_runner.load()
                else:
                    self.prototypes_src = prototype_runner()
                
            self.image_encoder = ImageEncoder(hparams, self.clip_model, self.normal_transform, self.clsdst_transform, dataset_name)
            self.text_encoder = TextEncoder(hparams, self.clip_model)

            assert self.hparams['architecture']['sttc_backward'] and self.hparams['architecture']['self_training'] or not self.hparams['architecture']['sttc_backward']
            assert self.hparams['architecture']['self_training_use_aug'] and self.hparams['architecture']['self_training'] or not self.hparams['architecture']['self_training_use_aug']
            if self.hparams['architecture']['self_training']:
                self.self_trainer = SelfTrainer(hparams, self.clip_model, lambda_ce_trg)
            if self.hparams['architecture']['domain_learning']:
                self.domain_trainer = DomainTrainer(hparams, self.clip_model, self.prototypes_src, self.models['mine'], self.src_loader.batch_size, save_dir, self.device)

            if self.hparams['warmup']['use']:
                self.warmup_criterion = nn.CrossEntropyLoss()
                src_batch_size = self.src_loader.batch_size
                self.warmup_steps = num_samples_warm_up // src_batch_size
                self.final_lr = self.optimizers.param_groups[0]['lr']
                ckpt_path = f"ckpt_warmup_{dataset_name}_{arch_name}_bs{src_batch_size}_step{self.warmup_steps}.pth"
                self.warmup_ckpt_path = os.path.join(ckpt_dir, "warmup")
                self.ckpt_path = os.path.join(self.warmup_ckpt_path, ckpt_path)
                self.warmup()


    def learning(self, x, y=None, warmup=False):
        assert (y is not None and warmup) or (y is None)
        loss = torch.tensor(0.0, requires_grad=True).cuda()
        self.optimizers.zero_grad()

        ### Domain Learning
        if self.hparams['architecture']['domain_learning'] and not warmup:  # warmupでここも学習すると時間がかかりすぎる.
            # iter_num = int(1e+1)
            # for it in tqdm.tqdm(range(iter_num)):
            # text
            domain_prompts, domain_tokenized_prompts = self.prompt_learner(class_domain='domain')
            domain_text_fts = self.text_encoder(domain_prompts, domain_tokenized_prompts)  # (num_classes, EMBEDDING_DIM)
            norm_domain_text_fts = F.normalize(domain_text_fts)
            # image
            image_clsdst_fts, _ = self.image_encoder(x, class_domain='domain')
            if self.hparams['architecture']['self_training']:
                image_clsdst_fts = self.models['model_st'](image_clsdst_fts)  # (B, EMBEDDING_DIM * num_domain_tokens)
            # Domain Training
            domain_loss = self.domain_trainer(image_clsdst_fts, norm_domain_text_fts)

            # loss += domain_loss
            loss -= domain_loss
            # self.scaler.scale(domain_loss).backward()
            # self.scaler.step(self.optimizers)
            # self.scaler.update()
            # self.optimizers.zero_grad()

        ### Self Training
        # text
        prompts, tokenized_prompts = self.prompt_learner(class_domain='class')
        text_fts = self.text_encoder(prompts, tokenized_prompts)  # (num_classes, EMBEDDING_DIM)
        norm_text_fts = F.normalize(text_fts)
        # image
        image_fts, image_aug_fts = self.image_encoder(x, class_domain='class')
        if self.hparams['architecture']['self_training']:
            ### Self Training
            self_train_loss, logits_st, logits_ema = self.self_trainer(self.models, image_fts, image_aug_fts, norm_text_fts)
            loss += self_train_loss
            logits = logits_st + logits_ema  # Ensembled Logits
        else:
            norm_image_fts = F.normalize(image_fts)
            logits = self.clip_model.logit_scale.exp() * norm_image_fts @ norm_text_fts.t()  # (B, num_classes)
                
        ### Cross Entropy Loss with Ground Truth
        if warmup:
            ##### TODO: あれ，これ過学習起きるんじゃない？
            if self.hparams['architecture']['self_training']:
                ce_loss = self.warmup_criterion(logits_st, y)
            else:
                ce_loss = self.warmup_criterion(logits, y)
            loss += ce_loss
        
        if loss == 0.0:
            logger.warning('[Warning]: loss is 0.0.')
        if (self.hparams['architecture']['self_training'] and self.hparams['architecture']['sttc_backward']) \
            or (self.hparams['architecture']['domain_learning'] and not warmup) \
            or warmup:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizers)
            self.scaler.update()
            if self.hparams['architecture']['self_training']:
                self.update_ema_variables(self.m_teacher_momentum)

        print(f"loss: {loss.item():.4f}\tdomain_loss: {domain_loss.item():.4f}\tself_train_loss: {self_train_loss.item():.4f}")
        return logits
    
    def update_ema_variables(self, alpha_teacher):
        """ Studentにした変更を, EMA(Exponential Moving Average) によりTeacherモデルへ反映する. 
        :param ema_model: Teacher model
        :param model: Student model
        :param alpha_teacher: weight
        """
        for ema_param, param in zip(self.models['model_ema'].parameters(), self.models['model_st'].parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    
    def warmup(self):
        if self.hparams['warmup']['load'] and os.path.exists(self.ckpt_path):
            logger.info(f"Loading warmup checkpoint... from {self.ckpt_path}")
            if self.hparams['warmup']['load_model_fname'] is not None:
                self.ckpt_path = os.path.join(Path(self.ckpt_path).parent, self.hparams['warmup']['load_model_fname'])
            checkpoint = torch.load(self.ckpt_path)
            for model_name, cp in checkpoint.items():
                if model_name != 'optimizer':
                    self.models[model_name].load_state_dict(cp)
                # else:
                #     self.optimizers.load_state_dict(cp)
                if model_name not in self.models.keys():
                    logger.warning(f"----- model_name: {model_name} is not in self.models.keys(). -----")
        else:
            os.makedirs(self.warmup_ckpt_path, exist_ok=True)
            logger.info(f"Starting warm up...{self.warmup_steps} steps")
            for i in tqdm.tqdm(range(self.warmup_steps)):
                for par in self.optimizers.param_groups:
                    par['lr'] = self.final_lr * (i+1) / self.warmup_steps
                try:
                    src_batch = next(self.src_loader_iter)
                except StopIteration:
                    self.src_loader_iter = iter(self.src_loader)
                    src_batch = next(self.src_loader_iter)
                x = src_batch[0].to(self.device)
                y = src_batch[1].to(self.device)
                logits = self.learning(x, y, warmup=True)
                if i % 50 == 0 or i == self.warmup_steps-1:
                    acc = logits.argmax(dim=1).eq(y).sum().item() / y.size(0)
                    logger.info(f"step: {i}\tacc: {acc*100:.2f}%")

            logger.info(f"Finished warm up...")
            for par in self.optimizers.param_groups:
                par['lr'] = self.final_lr
                state_dicts = {}
                for model_name in self.models.keys(): 
                    state_dicts[model_name] = self.models[model_name].state_dict()
                state_dicts['optimizer'] = self.optimizers.state_dict()
                torch.save(state_dicts, self.ckpt_path)

    def clip_only_learning(self, x):
        x_cls = self.normal_transform(x)  # (B, 3, 224, 224)
        with torch.no_grad():
            tokenized_prompts = self.prompt_learner.tokens['tokenized_prompts']
            image_fts = self.clip_model.encode_image(x_cls)  # (B, EMBEDDING_DIM)
            text_fts = self.clip_model.encode_text(tokenized_prompts)
        norm_image_fts = F.normalize(image_fts)
        norm_text_fts = F.normalize(text_fts)
        logits = self.clip_model.logit_scale.exp() * norm_image_fts @ norm_text_fts.t()  # (B, num_classes)
        return logits

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        if self.hparams['architecture']['clip_only']:
            logits = self.clip_only_learning(x[0])
            return logits
        else:
            ensembled_logits = self.learning(x[0])
            return ensembled_logits
        

class SelfTrainer(nn.Module):
    def __init__(self, hparams, clip_model:CLIP, lambda_ce_trg):
        super().__init__()
        self.hparams = hparams
        self.clip_model = clip_model
        self.lambda_ce_trg = lambda_ce_trg

    def forward(self, models, image_fts, image_aug_fts, norm_text_fts):
        logits_st = self._get_logits(models['model_st'], image_fts, norm_text_fts)  # (B, num_classes)
        logits_ema = self._get_logits(models['model_ema'], image_fts, norm_text_fts)  # (B, num_classes)
        
        if self.hparams['architecture']['self_training_use_aug']:
            logits_aug = self._get_logits(models['model_st'], image_aug_fts, norm_text_fts)  # (B, num_classes)
            loss_entropy = self_training_loss(x=logits_st, x_aug=logits_aug, x_ema=logits_ema).mean(0)  # Loss 式(6)
        else:
            loss_entropy = symmetric_cross_entropy(x=logits_st, x_ema=logits_ema).mean(0)  # Loss 式(6)

        loss_trg = self.lambda_ce_trg * loss_entropy  # Loss 式(9)のうち, targetドメインに対するLoss (第1項)

        return loss_trg, logits_st, logits_ema
        
    def _get_logits(self, model, image_fts, norm_text_fts):
        sttc_fts = model(image_fts)  # (B, EMBEDDING_DIM * num_domain_tokens)
        norm_sttc_fts = F.normalize(sttc_fts)
        logits = self.clip_model.logit_scale.exp() * norm_sttc_fts @ norm_text_fts.t()  # (B, num_classes)  # self.clip_model.logit_scale.exp()によるスケール変換は, 類似度スコアの非負化を行い、類似度の解釈や比較を容易にし，指数関数によるスケール変換は正規化や確率的な処理にも関連する．
        return logits


class DomainTrainer(nn.Module):
    def __init__(self, hparams, clip_model:CLIP, prototypes_src, mine, batch_size, save_dir, device):
        super().__init__()
        assert hparams['domain_loss']['method'] in ['mine']
        self.hparams = hparams
        self.clip_model = clip_model
        self.prototypes_src = prototypes_src
        self.save_dir = save_dir
        self.device = device
        self.iter_num = 0
        
        self.skp = skp.SinkhornKnopp()
        if hparams['domain_loss']['clustering_method'] == 'SpectralBiclustering':
            self.biclust_model = SpectralBiclustering(n_clusters=hparams['domain_loss']['n_clusters'], method="log", random_state=0)
        elif hparams['domain_loss']['clustering_method'] == 'SpectralCoclustering':
            self.biclust_model = SpectralCoclustering(n_clusters=hparams['domain_loss']['n_clusters'][0], random_state=0)

        if self.hparams['domain_loss']['method'] == 'mine':
            self.mine_trainer = MineTrainer(mine)
        elif self.hparams['domain_loss']['method'] == 'nt_xent':
            self.ntxent_criterion = NTXentLoss(self.device, batch_size, self.hparams['domain_loss']['nt_xent_temperature'])

    def forward(self, image_clsdst_fts, text_fts):
        image_clsdst_fts = F.normalize(image_clsdst_fts)
        text_fts = F.normalize(text_fts)
        if self.hparams['prototypes']['use']:
            indices = torch.randperm(self.prototypes_src.shape[0])[:image_clsdst_fts.shape[0]]
            sampled_prototypes = self.prototypes_src[indices].cuda()
            sampled_prototypes = F.normalize(sampled_prototypes)
            image_clsdst_fts = torch.cat([sampled_prototypes, image_clsdst_fts], dim=0)
            # if self.hparams['domain_loss']['prompt'] == 'classname':
            #     if self.hparams['domain_loss']['to_square'] == 'padding':
            #         # 正方行列にするためにpaddingを追加.
            #         text_padding = torch.zeros(image_clsdst_fts.shape[0] - text_fts.shape[0], text_fts.shape[1]).cuda()
            #         text_fts = torch.cat([text_fts, text_padding], dim=0)
            #     elif self.hparams['domain_loss']['to_square'] == 'duplicate':
            #         text_fts = torch.cat([text_fts, text_fts], dim=0)
        if self.hparams['domain_loss']['method'] == "nt_xent":
            domain_loss = self.ntxent_criterion(image_clsdst_fts, text_fts)
        elif self.hparams['domain_loss']['method'] == 'mine':
            sim_mtx = F.cosine_similarity(
                    x1=image_clsdst_fts.unsqueeze(0),  # (1, B, EMBEDDING_DIM)
                    x2=text_fts.unsqueeze(1),  # (B, 1, EMBEDDING_DIM)
                    dim=-1).detach().cpu().numpy()  # (B, B)

            # torch.save(image_clsdst_fts.cpu().detach(), 'prc_image_clsdst_fts.pt')
            # torch.save(text_fts.cpu().detach(), 'prc_text_fts.pt')
            # torch.save(sim_mtx, 'prc_sim_mtx.pt')
            # raise ValueError()

            ##### Doubly Stochastic Matrix
            ################################################################
            ################################################################
            ################################################################
            # bistochastic_mtx = self._get_bistochastic_mtx(sim_mtx)  # (B, B)
            bistochastic_mtx = sim_mtx
            ################################################################
            ################################################################
            ################################################################
            ##### Biclustering
            # clustered_mtx = self._biclustering(bistochastic_mtx)  # (B, B)
            # diag = torch.diag(clustered_mtx).long()
            self._biclustering(bistochastic_mtx)
            diag = torch.tensor(self.biclust_model.row_labels_)
            self._biclustering_log(bistochastic_mtx)

            ##### Minimize Mutual Information
            mean_feat_per_clust = [text_fts[diag == clust].mean(dim=0) for clust in sorted(torch.unique(diag))]
            if len(mean_feat_per_clust) == 1:
                logger.warning('[Warning] The Number of Clusters is 1')

            domain_loss = torch.tensor(0.0, requires_grad=True).cuda()
            for i in range(len(mean_feat_per_clust)):
                for j in range(i+1, len(mean_feat_per_clust)):
                    data = torch.stack([mean_feat_per_clust[i], mean_feat_per_clust[j]], dim=1)
                    mine_loss, _ = self.mine_trainer.get_loss(data)
                    domain_loss += mine_loss
        return domain_loss

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

    def _biclustering_log(self, sim_mtx):
        if self.iter_num % int(1e+3 * 50) == 0:
            os.makedirs(os.path.join(self.save_dir, 'biclustering'), exist_ok=True)
            sorted_sim_mtx = sim_mtx[np.argsort(self.biclust_model.row_labels_)]
            sorted_sim_mtx = sorted_sim_mtx[:, np.argsort(self.biclust_model.column_labels_)]
            plt.matshow(sorted_sim_mtx, cmap=plt.cm.Blues)
            plt.title(f"iter num : {self.iter_num}")
            plt.savefig(os.path.join(self.save_dir, 'biclustering', f"sim_mtx_{self.iter_num}.png"))
            self.iter_num += 1

class PromptLearner(nn.Module):
    def __init__(self, hparams, clip_model, class_names, num_classes, device):
        super().__init__()
        self.hparams = hparams
        self.num_classes = num_classes
        template = 'a photo of a'
        n_ctx = hparams['num_domain_tokens']
        ctx_dim = clip_model.ln_final.weight.shape[0]
        ctx_init = template
        if hparams['architecture']['learnable_parameters'] and not hparams['architecture']['clip_only']:
            if ctx_init:  ##### TODO: ランダムに初期化する方が良い？
                ctx_init = ctx_init.replace(" {}.", "")
                ctx_init = ctx_init.replace("_", " ")
                
                prompt = clip.tokenize(ctx_init).cuda()
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(clip_model.dtype)
                ctx_vectors = embedding[0, 1:1 + n_ctx, :]
                prompt_prefix = " ".join(["X"] * n_ctx)
                prompt_prefix = f"{prompt_prefix} {ctx_init}"  # X X X X X X X X X X X X X X X X 'a' 'photo' 'of' 'a'
            else:
                # random initialization
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=clip_model.dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)

            self.ctx = nn.Parameter(ctx_vectors)
            logger.info(f'Initial context: "{prompt_prefix}"')
            logger.info(f"Number of context words (tokens): {n_ctx}")
        else:
            prompt_prefix = ctx_init
            n_ctx = len(prompt_prefix.split()) + 1  # a photo of a [CLS]

        ##### Class Prompt用  refer DPLCLIP
        tokens = {}
        class_prompts = [prompt_prefix + ' ' + name + '.' for name in class_names]  # (クラス数)
        tokens['tokenized_prompts'] = torch.cat([clip.tokenize(p) for p in class_prompts]).to(device)  # (クラス数, 77)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokens['tokenized_prompts']).type(clip_model.dtype)  # (クラス数 (入力文の数), 77 (各文のトークン数（シーケンス長）), self.EMBEDDING_DIM)
        tokens['token_prefix'] = embedding[:, :1, :]  # SOS (クラス数, 1, self.EMBEDDING_DIM)  各クラスの埋め込みトークンの最初の次元がSOS. SOS (Start of Sentence): SOSトークンは、文の開始を示すために使用されます。通常、モデルへの入力テキストの最初に追加されます。SOSトークンは、モデルに対して文の生成を開始するよう指示します。
        tokens['token_suffix'] = embedding[:, 1 + n_ctx:, :]  # CLS, EOS (クラス数, 68, EMBEDDING_DIM)  68 := 77 - num_domain_tokens_tokens - 2.
        if not hparams['architecture']['learnable_parameters']:
            tokens['template_embedding'] = embedding[:, 1:1 + n_ctx, :]

        ##### Domain Prompt用
        if hparams['architecture']['domain_learning']:
            ##### TODO: ここをどう設計する？Prompt Distribution Learningぽくクラス名を付加させない？その場合類似度行列の列が単一になり，biclusteringもクソもない．
            if hparams['domain_loss']['prompt'] == 'classname':
                domain_prompts = [prompt_prefix + ' ' + name + '.' for name in class_names]  # (クラス数)
            else:
                domain_prompts = prompt_prefix
            tokens['domain_tokenized_prompts'] = clip.tokenize(domain_prompts).to(device)  # (1, 77)
            with torch.no_grad():
                domain_embedding = clip_model.token_embedding(tokens['domain_tokenized_prompts']).type(clip_model.dtype)  # (1, 77 (各文のトークン数（シーケンス長）), EMBEDDING_DIM)
            tokens['domain_token_prefix'] = domain_embedding[:, :1, :]  # SOS (1, 1, EMBEDDING_DIM)
            tokens['domain_token_suffix'] = domain_embedding[:, 1 + n_ctx:, :]  # CLS, EOS (1, 68, EMBEDDING_DIM)  68 := 77 - num_domain_tokens_tokens - 2.
            if tokens['domain_token_prefix'].shape[0] == 1:
                tokens['domain_token_prefix'] = tokens['domain_token_prefix'].repeat_interleave(self.num_classes, dim=0)
                tokens['domain_token_suffix'] = tokens['domain_token_suffix'].repeat_interleave(self.num_classes, dim=0)
                
        self.tokens = tokens

    def forward(self, class_domain='class'):
        if class_domain == 'class':
            prefix = self.tokens['token_prefix']
            suffix = self.tokens['token_suffix']
            tokenized_prompts = self.tokens['tokenized_prompts']
        else:
            prefix = self.tokens['domain_token_prefix']
            suffix = self.tokens['domain_token_suffix']
            tokenized_prompts = self.tokens['domain_tokenized_prompts']
            
        if self.hparams['architecture']['learnable_parameters']:
            ctx = self.ctx.unsqueeze(0).repeat_interleave(self.num_classes, dim=0)
        else:
            ctx = self.tokens['template_embedding']

        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        
        return prompts, tokenized_prompts
    

@torch.jit.script
def self_training_loss(x, x_aug, x_ema):# -> torch.Tensor:
    """ 式(6) """
    return - 0.25 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - 0.25 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1) \
           - 0.25 * (x_ema.softmax(1) * x_aug.log_softmax(1)).sum(1) - 0.25 * (x_aug.softmax(1) * x_ema.log_softmax(1)).sum(1)

@torch.jit.script
def symmetric_cross_entropy(x, x_ema):# -> torch.Tensor:
    return -0.5 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - 0.5 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)


def set_models(hparams, EMBEDDING_DIM, clip_model_dtype, device):
    rtn_models = {}
    ##### Self Training
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    if hparams['architecture']['self_training']:
        if hparams['sttc'] == 'linear':
            rtn_models['model_st'] = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM ).to(device=device, dtype=clip_model_dtype)
        elif hparams['sttc'] == 'mlp':
            rtn_models['model_st'] = networks.MLP(EMBEDDING_DIM, EMBEDDING_DIM, hparams).to(device=device, dtype=clip_model_dtype)
        rtn_models['model_st'].apply(init_weights)
        rtn_models['model_ema'] = TTAMethod.copy_model(rtn_models['model_st'])
        for param in rtn_models['model_ema'].parameters():
            param.detach_()

        if not hparams['architecture']['sttc_backward']:
            for param in rtn_models['model_st'].parameters():
                param.requires_grad = False

    ##### Domain Learning
    if hparams['architecture']['domain_learning']:
        if hparams['domain_loss']['use_domain_projector']:
            rtn_models['domain_projector'] = nn.Sequential(nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)).to(device=device, dtype=clip_model_dtype)
        if hparams['domain_loss']['method'] == 'mine':
            rtn_models['mine'] = Mine().to(device)
        
    return rtn_models


def set_optimizers(hparams, models):
    assert hparams['optimizer'] in ["Adam", "SGD"]
    class_args_params = []
    for model_name, model in models.items():
        if model_name != 'model_ema':
            class_args_params.append({'params': model.parameters()})
    if hparams['optimizer'] == "Adam":
        optimizers = torch.optim.Adam(class_args_params, lr=hparams['lr'])
    elif hparams['optimizer'] == "SGD":
        optimizers = torch.optim.SGD(class_args_params, lr=hparams['lr'], momentum=hparams['momentum'])

    return optimizers
