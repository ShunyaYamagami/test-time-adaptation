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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" task_specific """
assert not self.hparams['clip_model']['task_specific'], 'task_spevific must be False. 今の所, task_specificにしても精度が上がるわけがない.'
####################################################################################################################
if self.hparams["clip_model"]["task_specific"]:
    cfg = EasyDict({
        'CORRUPTION': {
            'DATASET': self.dataset_name
        },
        'CKPT_DIR': self.ckpt_dir,
        'CKPT_PATH': self.ckpt_path,
        'MODEL': {
            'ARCH': self.arch_name,
            'WEIGHTS': self.model_weights
        }
    })
    self.vit_task_specific = get_model(cfg, self.num_classes)
    self.vit_task_specific.eval()
    for param in self.vit_task_specific.parameters():
        param.requires_grad = False
    self.vit_feature_extractor, _ = split_up_model(self.vit_task_specific, self.arch_name, self.dataset_name)
    self.vit_feature_extractor.new_layer = nn.Sequential(
        nn.Linear(640, self.EMBEDDING_DIM, bias=False),
        nn.BatchNorm1d(self.EMBEDDING_DIM),
        nn.ReLU()
    ).to(device=self.device, dtype=self.clip_model.dtype)
    self.vit_feature_extractor = nn.DataParallel(self.vit_feature_extractor)

####################################################################################################################
def _learning()
    if self.hparams["clip_model"]["task_specific"]:
        image_encoder = self.vit_feature_extractor
    else:
        image_encoder = self.clip_model.encode_image
        
####################################################################################################################
def get_tta_transforms(dataset, gaussian_std: float=0.005, soft=False, padding_mode='edge', cotta_augs=True, clip_task_specific=False):
    if "cifar" in dataset and clip_task_specific:
        img_shape = (32, 32, 3) ############################
    else:
        img_shape = (224, 224, 3) ############################
    n_pixels = img_shape[0]

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
