import logging
from easydict import EasyDict
import os
import numpy as np
import tqdm
import yaml
from time import time, sleep
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
from sklearn.cluster import SpectralBiclustering
from pathlib import Path
from functools import partial

from conf import get_num_classes
from domainbed import networks
from methods.base import TTAMethod
from augmentations.transforms_cotta import get_tta_transforms
from augmentations.aug import get_transform_when_iteration
from models.model import get_model, split_up_model
from models.mine import Mine, MineTrainer
from loss.nt_xent import NTXentLoss

logger = logging.getLogger(__name__)


class ImageEncoder(nn.Module):
    def __init__(self, hparams, clip_model:CLIP, normal_transform, clsdst_transform, dataset_name):
        super().__init__()
        self.hparams = hparams
        self.clip_model = clip_model
        self.normal_transform = normal_transform
        self.clsdst_transform = clsdst_transform
        self.tta_transform = get_tta_transforms(dataset_name)
        
    def forward(self, x, class_domain):
        if class_domain == "class":
            x = self.normal_transform(x)  # (B, 3, 224, 224)
        else:
            x = self.clsdst_transform(x)  # (B, 3, 224, 224)

        with torch.no_grad():
            image_fts = self.clip_model.encode_image(x)  # (B, EMBEDDING_DIM)

        if class_domain == 'class' and self.hparams['architecture']['self_training_use_aug']:
            x_aug = self.tta_transform(x)
            with torch.no_grad():
                image_aug_fts = self.clip_model.encode_image(x_aug)  # (B, EMBEDDING_DIM)
            return image_fts, image_aug_fts
        else:
            return image_fts, None

class TextEncoder(nn.Module):
    """ refer CoCoOp """
    def __init__(self, hparams, clip_model:CLIP):
        super().__init__()
        self.hparams = hparams
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompt, tokenized_prompts):
        x = prompt + self.positional_embedding.type(self.dtype)  # (L, 77, EMBEDDING_DIM)
        x = x.permute(1, 0, 2)  # (77, L, EMBEDDING_DIM)
        x = self.transformer(x)  # (77, L, EMBEDDING_DIM)
        x = x.permute(1, 0, 2)  # (L, 77, EMBEDDING_DIM)
        x = self.ln_final(x).type(self.dtype)  # (L, 77, EMBEDDING_DIM)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection  # (L, EMBEDDING_DIM))

        return x
    


class PrototypeRunner(nn.Module):
    def __init__(self, hparams, clip_model:CLIP, src_loader, normal_transform, clsdst_transform, num_classes, arch_name, ckpt_dir, ckpt_path, dataset_name, device):
        super().__init__()
        self.clip_model = clip_model
        self.src_loader = src_loader
        self.clsdst_transform = clsdst_transform
        self.num_classes = num_classes
        self.device = device
        self.proto_dir_path = os.path.join(ckpt_dir, "prototypes")
        if dataset_name == "domainnet126":
            fname = f"protos_{dataset_name}_{ckpt_path.split(os.sep)[-1].split('_')[1]}.pth"
        else:
            fname = f"protos_{dataset_name}_{arch_name}.pth"
        self.fname = os.path.join(self.proto_dir_path, fname)

    def load(self):
        assert os.path.exists(self.fname)
        logger.info(f"Loading class-wise source prototypes from {self.fname}...")
        prototypes_src = torch.load(self.fname)
        return prototypes_src

    def forward(self):
        os.makedirs(self.proto_dir_path, exist_ok=True)
        features_src = torch.tensor([])
        logger.info("Extracting source prototypes...")
        with torch.no_grad():
            for data in tqdm.tqdm(self.src_loader):
                if len(features_src) > 10000:
                    break
                x = data[0].cuda()
                x_clsdst = self.clsdst_transform(x)  # (B, 3, 224, 224)
                image_clsdst_fts = self.clip_model.encode_image(x_clsdst)  # (B, EMBEDDING_DIM)
                image_clsdst_fts = F.normalize(image_clsdst_fts)
                features_src = torch.cat([features_src, image_clsdst_fts.cpu()], dim=0)  # (画像数, EMBEDDING_DIM)

        torch.save(features_src, self.fname)
        features_src = features_src.to(self.device).unsqueeze(1) 
        return features_src


def decorator_timer(some_function):
    """ 時間計測用のデコレータ """
    def wrapper(*args, **kwargs):
        t1 = time()
        result = some_function(*args, **kwargs)
        end = time()-t1
        print(f'\t{end:.2f}s')
        return result
    return wrapper