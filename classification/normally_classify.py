# TTAを行わずに，単純にCLIPとViTのクラス分類性能を比較する
from copy import deepcopy
from easydict import EasyDict
import numpy as np
from tqdm import tqdm
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from models.model import split_up_model
import clip


class NormalCLIP(nn.Module):
    def __init__(self, class_names):
        super().__init__()
        self.hparams = EasyDict({
            "clip_backbone": 'ViT-B/32',  # choice(['ViT-B/32', 'ViT-B/16', 'RN101']),
            "class_names": class_names,
            "num_domain_tokens": 16,
            'mlp_depth': 3,
            'mlp_width': 512,
            'mlp_dropout': 0.1,
            'lr': 1e-3,
            'momentum': 0.1,
            'weight_decay': 0.,
            "load_warmup_model": False  ######### 今は毎回warmupしよう.
        })
        self.set_clip_models()


    def set_clip_models(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"----- self.hparams['clip_backbone'] : {self.hparams['clip_backbone']}  -----")
        self.clip_model, preprocess = clip.load(self.hparams['clip_backbone'], device=self.device)
        # CLIPモデルのパラメータは更新させない
        print('Set self.clip_model.parameters.reguires_grad = False!')
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        ##### class DPLCLIP(CLIP): #####
        classnames = [f"a photo of a {name.replace('_', ' ')}" for name in self.hparams['class_names']]
        self.tokenized_prompts = torch.cat([clip.tokenize(c) for c in classnames]).to(self.device)
        self.text_features = self.clip_model.encode_text(self.tokenized_prompts)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)


    def forward(self, x):
        image_features = self.clip_model.encode_image(x)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.clip_model.logit_scale.exp() * image_features @ self.text_features.t()  # (B, Sourceクラス数)
        return logits_per_image
    

def get_vit_model(dataset_name="cifar10"):
    """ ベースとなるpre-trained model(ViTなど)をロード. """
    print("----- loading base_model from model_zoo -----")
    if dataset_name == "cifar10":
        arch = "Standard"
    elif dataset_name == "cifar100":
        arch = "Hendrycks2020AugMix_ResNeXt"
    else:
        raise NotImplementedError(f"dataset_name: {dataset_name} is not implemented.")
    base_model = load_model(arch, "./ckpt", dataset_name, ThreatModel.corruptions)

    return base_model


def get_dataloader(dataset_name="cifar10", does_resize = True):
    dparams = EasyDict({
        "root_dir": "/nas/data/syamagami/tta/rmt",
        "train_split": False,
        "batch_size": 200,
        "num_workers": 4,
    })
    if does_resize:
        transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    
    if dataset_name == "cifar10":
        source_dataset = torchvision.datasets.CIFAR10(root=dparams.root_dir, train=dparams.train_split, download=True, transform=transform)
    elif dataset_name == "cifar100":
        source_dataset = torchvision.datasets.CIFAR100(root=dparams.root_dir, train=dparams.train_split, download=True, transform=transform)
    else:
        raise NotImplementedError(f"dataset_name: {dataset_name} is not implemented.")

    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=dparams.batch_size, shuffle=True, num_workers=dparams.num_workers, drop_last=False)
    return source_dataset, source_loader

    
def evaluation(model, dataloader, model_name="clip", cuda_i=0):
    correct = 0.
    print("***************************************************")
    print(model_name)
    len_dataloader = len(dataloader)
    print(f"dataloader: {len_dataloader}")
    for i, (imgs, labels) in tqdm(enumerate(dataloader)):
        imgs, labels = imgs.to(f"cuda:{cuda_i}"), labels.to(f"cuda:{cuda_i}")
        output = model(imgs)
        pred = output.argmax(dim=1)
        correct += (pred == labels).float().sum()
    accuracy = correct.item() / len(dataloader.dataset) * 100.
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    dataset_name = "cifar10"
    print(f"----- dataset: {dataset_name} -----")
    clip_dataset, clip_loader = get_dataloader(dataset_name, does_resize = True)
    vit_dataset, vit_loader   = get_dataloader(dataset_name, does_resize = False)
    class_names = clip_dataset.classes

    clip_model = NormalCLIP(class_names)
    clip_model.eval()
    vit_model = get_vit_model(dataset_name)
    vit_model = vit_model.to("cuda:1")
    vit_model.eval()
    # feature_extractor, classifier = split_up_model(vit_model, "Standard", "cifar10")

    evaluation(clip_model, clip_loader, model_name="clip", cuda_i=0)
    evaluation(vit_model, vit_loader, model_name="vit", cuda_i=1)