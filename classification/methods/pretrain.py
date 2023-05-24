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

from methods.rmt import set_models, set_optimizers
from datasets.data_loading import get_source_loader
from loss.nt_xent import NTXentLoss

logger = logging.getLogger(__name__)


class Pretrain():
    def __init__(self, cfg, hparams) -> None:
        super().__init__()
        self.cfg = cfg
        self.hparams = hparams
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_name = self.cfg.CORRUPTION.DATASET
        self.batch_size_src = self.cfg.TEST.BATCH_SIZE if self.cfg.TEST.BATCH_SIZE > 1 else self.cfg.TEST.WINDOW_LENGTH
        self.projection_dim = self.cfg.CONTRAST.PROJECTION_DIM
        self.pretrained_dir = os.path.join(cfg.CKPT_DIR, self.dataset_name, 'pretrained')

        self.models = set_models()
        self.domain_projectors = set_optimizers()
        self.pre_class_criterion = nn.CrossEntropyLoss()
        self.pre_domain_criterion = NTXentLoss(self.device, self.batch_size_src, self.hparams["domain_loss"]["nt_xent_temperature"])

    def dataloader(self):
        args = {
            'dataset_name': self.dataset_name,
            'root_dir': self.cfg.DATA_DIR,
            'adaptation': self.cfg.MODEL.ADAPTATION,
            'batch_size': self.batch_size_src,
            'ckpt_path': self.cfg.CKPT_PATH,
            'percentage': self.cfg.SOURCE.PERCENTAGE,
            'workers': min(self.cfg.SOURCE.NUM_WORKERS, os.cpu_count()),
        }
        _, self.train_loader = get_source_loader(**args, train_split=True)
        _, self.test_loader = get_source_loader(**args, train_split=False)
        
        
    def train(self):
        best = 0.
        for epoch in tqdm.tqdm(range(self.hparams['pretrain']['epochs'])):
            for batch_idx, (x, y) in enumerate(self.train_loader):
                # train
                for m in self.models:
                    m.train()
                # self._learning(x.cuda(), y.cuda(), pretrain=True)

            # valid
            for batch_idx, (x, y) in enumerate(self.test_loader):
                with torch.no_grad():
                    self.model.eval()
                    self.domain_projector.eval()
                    x_cls = self.normal_transform(x)  # (B, 3, 224, 224)
                    image_fts = self.clip_model.encode_image(x_cls)  # (B, EMBEDDING_DIM)
                    logits_st, _, _ = self._learning__get_sttc_logits(self.model, image_fts)  # (B, num_classes)
                    correct = (logits_st.argmax(1) == y).float() 
                    accuracy = correct.sum() / len(correct)
                    if accuracy > best:
                        logger.info(f"Epoch[{epoch}] : {best}")
                        best = accuracy
                        torch.save(self.model.state_dict(), os.path.join(self.pretrained_dir, f"best_model.pth"))
                        torch.save(self.domain_projector.state_dict(), os.path.join(self.pretrained_dir, f"best_domain_projector.pth"))
                        with open(os.path.join(self.pretrained_dir, f"best_accuracy.txt"), "w") as f:
                            f.write(f"Epoch[{epoch}] : {best}")
        pass

    def load(self):
        pass

    