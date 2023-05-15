import logging

import os
import tqdm
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F

from methods.base import TTAMethod
from models.model import split_up_model
from augmentations.transforms_cotta import get_tta_transforms
import clip
from easydict import EasyDict
import numpy as np


logger = logging.getLogger(__name__)


class SourceOnlyCLIP(nn.Module):
    def __init__(self):
        super().__init__()

        self.hparams = EasyDict({
            "clip_backbone": 'ViT-B/32',  # choice(['ViT-B/32', 'ViT-B/16', 'RN101']),
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

    def forward(self, x):
        logits_per_image = self._calc_clip_logits(x)
        return logits_per_image


    def set_clip_models(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"----- self.hparams['clip_backbone'] : {self.hparams['clip_backbone']}  -----")
        self.clip_model, preprocess = clip.load(self.hparams['clip_backbone'], device=self.device)
        self.clip_model = self.clip_model.float()

        # CLIPモデルのパラメータは更新させない
        print('Set self.clip_model.parameters.reguires_grad = False!')
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        ##### class DPLCLIP(CLIP): #####
        if self.hparams['sentence_prompt']:
            print('Using sentence_prompt in DPLCLIP...')
            classnames = [f"a photo of a {name.replace('_', ' ')}" for name in self.hparams['class_names']]
        else:
            classnames = [name.replace('_', ' ') for name in self.hparams['class_names']]

        self.tokenized_prompts = torch.cat([clip.tokenize(c) for c in classnames]).to(self.device)
        self.text_features = self.clip_model.encode_text(self.tokenized_prompts)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)


    def _calc_clip_logits(self, imgs):
        """
            imgs: (B, 3, 224, 224)
        """
        image_features = self.clip_model.encode_image(imgs)
        
        # calc logits  t()は転置, 
        # self.clip_model.logit_scale.exp()によるスケール変換は, 類似度スコアの非負化を行い、類似度の解釈や比較を容易にし，指数関数によるスケール変換は正規化や確率的な処理にも関連する．
        logits_per_image = self.clip_model.logit_scale.exp() * image_features @ self.text_features.t()  # (B, Sourceクラス数)
        
        return logits_per_image

            
def to_numpy_array(lst):
    np_array = []
    for item in lst:
        if isinstance(item, torch.Tensor):
            np_array.append(item.cpu().detach().numpy())
        elif isinstance(item, list):
            np_array.append(to_numpy_array(item))
    return np.array(np_array)