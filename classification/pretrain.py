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
from pathlib import Path

from domainbed import networks
from methods.base import TTAMethod
from methods.rmt import set_clip_models, set_optimizers
from datasets.data_loading import get_source_loader
from loss.nt_xent import NTXentLoss
from augmentations.transforms_cotta import get_tta_transforms
from augmentations.aug import get_transform_when_iteration
from conf import cfg, load_cfg_fom_args, set_hparams

logger = logging.getLogger(__name__)

class Pretrain():
    def __init__(self, cfg, hparams) -> None:
        super().__init__()
        self.cfg = cfg
        self.hparams = hparams
        assert not self.hparams['architecture']['base_model'], '現在対応していません.'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_name = self.cfg.CORRUPTION.DATASET
        # self.src_batch_size = self.cfg.TEST.BATCH_SIZE if self.cfg.TEST.BATCH_SIZE > 1 else self.cfg.TEST.WINDOW_LENGTH
        self.src_batch_size = 120
        self.projection_dim = self.cfg.CONTRAST.PROJECTION_DIM
        self.pretrained_dir = os.path.join(cfg.CKPT_DIR, self.dataset_name.replace('_c', ''), 'pretrained')
        os.makedirs(self.pretrained_dir, exist_ok=True)
        self.EMBEDDING_DIM = 512
        self.tta_transform = get_tta_transforms(self.dataset_name)
        self.normal_transform, self.clsdst_transform = get_transform_when_iteration(grid=4)
        self.scaler = GradScaler(enabled=self.hparams['mixed_precision'])

        train_dataset, test_dataset, self.train_loader, self.test_loader = self.dataloader()
        class_names = train_dataset.classes
        self.num_classes = len(class_names)
        ########## Set CLIP Models ##########
        self.clip_model, self.tokens = set_clip_models(self.hparams, self.device, class_names)
        self.clip_model = self.clip_model.cuda()
        ########## Set Models ##########
        self.models = _set_models(hparams, self.device, self.EMBEDDING_DIM, self.clip_model.dtype, self.dataset_name, self.src_batch_size)
        self.optimizers = set_optimizers(hparams, self.models)
        self.final_lr = self.optimizers['optimizer'].param_groups[0]["lr"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizers['optimizer'], T_max=len(self.train_loader), eta_min=0, last_epoch=-1)
        self.pre_class_criterion = nn.CrossEntropyLoss()
        self.pre_domain_criterion = NTXentLoss(self.device, self.src_batch_size, self.hparams["domain_loss"]["nt_xent_temperature"])


    def dataloader(self):
        args = {
            'dataset_name': self.dataset_name,
            'root_dir': self.cfg.DATA_DIR,
            'adaptation': self.cfg.MODEL.ADAPTATION,
            'batch_size': self.src_batch_size,
            'ckpt_path': self.cfg.CKPT_PATH,
            'percentage': self.cfg.SOURCE.PERCENTAGE,
            'workers': min(self.cfg.SOURCE.NUM_WORKERS, os.cpu_count()),
        }
        train_dataset, train_loader = get_source_loader(**args, train_split=True)
        test_dataset, test_loader = get_source_loader(**args, train_split=False)

        return train_dataset, test_dataset, train_loader, test_loader
        
        
    # def pre_embedding(self):
    #     def _get_save_feats(dataloader, save_name, use_tta_transform=False, train=True):
    #         if train:
    #             self.clip_model.train()
    #         else:
    #             self.clip_model.eval()
    #         ###
    #         with torch.no_grad():
    #             cat_features = torch.Tensor([])
    #             cat_labels = torch.Tensor([])
    #             for (x, y) in tqdm.tqdm(dataloader):
    #                 x_cls = self.normal_transform(x)
    #                 if use_tta_transform:
    #                     x_aug = self.tta_transform(x_cls)
    #                     image_fts = self.clip_model.encode_image(x_aug.to(self.device))
    #                 else:
    #                     image_fts = self.clip_model.encode_image(x_cls.to(self.device))

    #                 cat_features = torch.cat([cat_features, image_fts.cpu()], dim=0)
    #                 cat_labels = torch.cat([cat_labels, y], dim=0)
    #             torch.save(cat_features, os.path.join(self.pretrained_dir, save_name))
    #             torch.save(cat_labels, os.path.join(self.pretrained_dir, save_name.replace('feats', 'labels')))

    #     if self.hparams['exec_num'] == -1:
    #         _get_save_feats(self.train_loader, 'train_feats.pt', train=True)
    #         _get_save_feats(self.test_loader, 'test_feats.pt', train=False)
    #     else:
    #         _get_save_feats(self.train_loader, 'train_aug_feats_{}.pt'.format(self.hparams['exec_num']))

        
    def cls_train(self):
        self.best = 0.
        for epoch in tqdm.tqdm(range(self.hparams['pretrain_epochs'])):
            self.models['model_st'].train()
            for batch_idx, (x, y) in tqdm.tqdm(enumerate(self.train_loader)):
                x, y = x.to(self.device), y.to(self.device)
                x_cls = self.normal_transform(x)
                
                with torch.no_grad():
                    image_fts = self.clip_model.encode_image(x_cls)

                self.optimizers['optimizer'].zero_grad()
                logits_st, _, _ = self._learning__get_sttc_logits(self.models['model_st'], image_fts, for_domain=False)  # (B, num_classes)

                loss_st = self.pre_class_criterion(logits_st, y)
                loss_st_total = loss_st

                ##### 下手なaugmentationなので, これで学習すると精度が著しく下がる #######
                # x_aug = self.tta_transform(x_cls)
                # with torch.no_grad():
                    # image_fts_aug = self.clip_model.encode_image(x_aug)
                # logits_st_aug, _, _ = self._learning__get_sttc_logits(self.models['model_st'], image_fts_aug, for_domain=False)  # (B, num_classes)
                # loss_st_aug = self.pre_class_criterion(logits_st_aug, y)
                # loss_st_total = loss_st + loss_st_aug
                ###################################################################

                self.scaler.scale(loss_st_total).backward()
                self.scaler.step(self.optimizers['optimizer'])
                self.scaler.update()
                # self.scheduler.step()

            # valid
            self.models['model_st'].eval()
            corrects = torch.Tensor([])
            for batch_idx, (x, y) in tqdm.tqdm(enumerate(self.test_loader)):
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    x_cls = self.normal_transform(x)  # (B, 3, 224, 224)
                    with torch.no_grad():
                        image_fts = self.clip_model.encode_image(x_cls)  # (B, EMBEDDING_DIM)
                    logits_st, _, _ = self._learning__get_sttc_logits(self.models['model_st'], image_fts, for_domain=False)  # (B, num_classes)
                    correct = (logits_st.argmax(1) == y).float()
                    corrects = torch.cat([corrects, correct.cpu()], dim=0)

            accuracy = corrects.sum() / len(corrects)
            logger.info(f"Epoch[{epoch}] : {accuracy}")
            if accuracy > self.best:
                logger.info(f"------------------------Best Accuracy : {accuracy}------------------------")
                self.best = accuracy
                torch.save(self.models['model_st'].state_dict(), os.path.join(self.pretrained_dir, f"best_model.pth"))
                # torch.save(self.models['domain_projector'].state_dict(), os.path.join(self.pretrained_dir, f"best_domain_projector.pth"))
                with open(os.path.join(self.pretrained_dir, f"best_accuracy.txt"), "w") as f:
                    f.write(f"Epoch[{epoch}] : {self.best}")
        
    def domain_train(self):
        """ Domain Features, Loss,  Contrastive Loss """
        assert self.hparams["domain_loss"]["method"]
        for epoch in tqdm.tqdm(range(self.hparams['pretrain_epochs'])):
            for model_name in self.models.keys():
                self.models[model_name].train()
            for batch_idx, (x, y) in tqdm.tqdm(enumerate(self.train_loader)):
                x, y = x.to(self.device), y.to(self.device)
                x_cls = self.normal_transform(x)
                x_aug = self.tta_transform(x_cls)

                self.optimizers['domain_optimizer'].zero_grad()
                domain_fts = self._learning__backward_domain_loss(x)
                domain_aug_fts = self._learning__backward_domain_loss(x_aug)

                norm_domain_fts = F.normalize(domain_fts)
                norm_domain_aug_fts = F.normalize(domain_aug_fts)

                loss_domain = self.pre_domain_criterion(norm_domain_fts, norm_domain_aug_fts)

                loss_domain.backward()
                self.optimizers['domain_optimizer'].step()
                # self.scaler.scale(loss_domain).backward()
                # self.scaler.step(self.optimizers['domain_optimizer'])
                # self.scaler.update()

                torch.save(self.models['domain_projector'].state_dict(), os.path.join(self.pretrained_dir, f"domain_projector.pth"))

            logger.info(f"Epoch[epoch]\tloss_domain : {loss_domain}")


            
    def _learning__get_sttc_logits(self, model, image_fts, for_domain=False):
        if for_domain:
            if self.hparams["architecture"]['base_model']:
                middle_fts = self.models['base_model'](image_fts)  # (B, EMBEDDING_DIM)
            elif self.hparams["architecture"]['domain_embedding_pos'] == 'first':
                middle_fts = self.models['domain_projector'](image_fts)  # (B, EMBEDDING_DIM * num_domain_tokens)
            else:
                middle_fts = image_fts  # (B, EMBEDDING_DIM)
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

        
    def _cat_class_domain_text_features(self, class_domain_fts, class_domain):
        """ class_domain_fts + text feature 
            L: classの場合は クラス数, domainの場合は batch size
            self.tokens['token_prefix'].shape:  SOS (L, 1, EMBEDDING_DIM)
            self.tokens['token_suffix'].shape:  EOS (L, 77 - 1 - self.hparams['num_domain_tokens'], EMBEDDING_DIM)
        args:
            class_domain_fts: (L, EMBEDDING_DIM * num_domain_tokens)
            class_domain: "class" or "domain"
        return:
            class_domain_fts + text feature 
        """
        assert class_domain in ["class", "domain"], f"Invalid class_domain: {class_domain}. class_domain must be 'class' or 'domain'"
        class_domain_fts = class_domain_fts.reshape(-1, self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)  # (L, num_domain_tokens, EMBEDDING_DIM)
        
        if class_domain == "class":
            param_fts = torch.cat([self.tokens['token_prefix'], class_domain_fts, self.tokens['token_suffix']], dim=1)  # (L, 77, EMBEDDING_DIM)
            tokenized_prompts = self.tokens['tokenized_prompts']
        else:
            batch_size = class_domain_fts.shape[0]
            rep_token_prefix = self.tokens['domain_token_prefix'].repeat_interleave(batch_size, dim=0)  # (L, 1, EMBEDDING_DIM)
            rep_token_suffix = self.tokens['domain_token_suffix'].repeat_interleave(batch_size, dim=0)  # (L, 77-1-num_domain_tokens, EMBEDDING_DIM)
            param_fts = torch.cat([rep_token_prefix, class_domain_fts, rep_token_suffix], dim=1)  # (L, 77, EMBEDDING_DIM)
            tokenized_prompts = self.tokens['domain_tokenized_prompts']  # (L, 77)
        
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

    
    def _learning__backward_domain_loss(self, x):
        x_clsdst = self.clsdst_transform(x)  # (B, 3, 224, 224)
        with torch.no_grad():
            image_clsdst_fts = self.clip_model.encode_image(x_clsdst)  # (B, EMBEDDING_DIM)

        if self.hparams["architecture"]['base_model']:
            base_clsdst_fts = self.models['base_model'](image_clsdst_fts)  # (B, EMBEDDING_DIM)
            domain_fts = self.models['domain_projector'](base_clsdst_fts)  # (B, EMBEDDING_DIM * num_domain_tokens)
        else:
            domain_fts = self.models['domain_projector'](image_clsdst_fts)  # (B, EMBEDDING_DIM * num_domain_tokens)
        
        return domain_fts


def _set_models(hparams, device, EMBEDDING_DIM, clip_model_dtype, dataset_name,
                src_batch_size, projection_dim=None, num_channels=None):
    """ F() Network for extraction of domain feature.

        ここで, self.modelとしているのは F() Network. 元コードのpre-trained modelとは異なることに注意.
        既存コードを書きたくなかったからこうした．
        どちらも同じ "Student model"という意味では共通している.
    """
    assert (hparams['prototypes']['use'] and projection_dim is not None) or (not hparams['prototypes']['use'])
    assert (hparams['prototypes']['use'] and num_channels is not None) or (not hparams['prototypes']['use'])

    ##### Student Teacher Model
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
    # Student model
    # input_dim = EMBEDDING_DIM * hparams['num_domain_tokens'] \
        # if hparams['architecture']['domain_embedding_pos'] == 'first' else EMBEDDING_DIM
    #################################################################
    #################################################################
    # prompt tuning の効果を見るために，一時的にこうする.
    input_dim = EMBEDDING_DIM
    #################################################################
    #################################################################
    model_st = networks.MLP(input_dim,
                            EMBEDDING_DIM * hparams['num_domain_tokens'],
                            hparams
                            ).to(device=device, dtype=clip_model_dtype)
    model_st.apply(init_weights)

    model_st = model_st.cuda()
    ##### return models
    return_models = {"model_st": model_st}
    ##### Domain Projector
    if hparams["domain_loss"]["method"]:
        # domain_projector = nn.Sequential(nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM * hparams['num_domain_tokens'])).to(device=f"cuda:{CUDA_OTHERS}", dtype=clip_model_dtype)
        domain_projector = nn.Sequential(nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM * hparams['num_domain_tokens'])).to(device=device, dtype=clip_model_dtype)
        return_models['domain_projector'] = domain_projector

    return return_models


if __name__ == '__main__':
    load_cfg_fom_args("Evaluation")
    assert cfg.SETTING in ["reset_each_shift",           # reset the model state after the adaptation to a domain
                           "continual",                  # train on sequence of domain shifts without knowing when shift occurs
                           "gradual",                    # sequence of gradually increasing / decreasing domain shifts
                           "mixed_domains",              # consecutive test samples are likely to originate from different domains
                           "correlated",                 # sorted by class label
                           "mixed_domains_correlated",   # mixed domains + sorted by class label
                           "gradual_correlated",         # gradual domain shifts + sorted by class label
                           "reset_each_shift_correlated"
                           ]
    hparams = set_hparams()
    ###############################################################################################################
    ###############################################################################################################
    hparams['optimizer'] = 'Adam'
    hparams['lr'] = 1e-2
    hparams['pretrain_epochs'] = 1000
    ###############################################################################################################
    ###############################################################################################################

    pretrain = Pretrain(cfg, hparams)
    # pretrain.pre_embedding()
    if hparams['cuda_visible_devices'] == [1]:
        logger.info("---------------  cls_train  ---------------")
        pretrain.cls_train()
    elif hparams['cuda_visible_devices'] == [2]:
        logger.info("---------------  domain_train  ---------------")
        pretrain.domain_train()
    else:
        raise NotImplementedError
