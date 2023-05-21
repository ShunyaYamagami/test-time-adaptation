# TTAを行わずに，単純にCLIPとViTのクラス分類性能を比較する
from copy import deepcopy
from easydict import EasyDict
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn

import clip

from prc_functions.dataloader import get_dataloader
from prc_functions.vit_model import get_vit_model

class NormalCLIP(nn.Module):
    def __init__(self, class_names):
        super().__init__()
        self.hparams = EasyDict({
            "clip_backbone": 'ViT-B/32',  # choice(['ViT-B/32', 'ViT-B/16', 'RN101']),
            "class_names": class_names,
        })
        self.set_clip_models()

    def set_clip_models(self):
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        print(f"----- self.hparams['clip_backbone'] : {self.hparams['clip_backbone']}  -----")
        self.clip_model, preprocess = clip.load(self.hparams['clip_backbone'], device=self.device)
        # CLIPモデルのパラメータは更新させない
        print('Set self.clip_model.parameters.reguires_grad = False!')
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        classnames = [f"a photo of a {name.replace('_', ' ')}" for name in self.hparams['class_names']]
        self.tokenized_prompts = torch.cat([clip.tokenize(c) for c in classnames]).to(self.device)
        self.text_features = self.clip_model.encode_text(self.tokenized_prompts)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def forward(self, x):
        image_features = self.clip_model.encode_image(x)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.clip_model.logit_scale.exp() * image_features @ self.text_features.t()  # (B, Sourceクラス数)
        return logits_per_image
    
    
def evaluation(model, dataloader, model_name="clip", cuda_i=0):
    correct = 0.
    print("***************************************************")
    print(model_name)
    len_dataloader = len(dataloader)
    print(f"dataloader: {len_dataloader}")
    for i, (imgs, labels) in tqdm(enumerate(dataloader)):
        imgs, labels = imgs.to(f"cpu"), labels.to(f"cpu")
        output = model(imgs)
        pred = output.argmax(dim=1)
        correct += (pred == labels).float().sum()
    accuracy = correct.item() / len(dataloader.dataset) * 100.
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    dataset_name = "cifar10"
    print(f"----- dataset: {dataset_name} -----")
    print(f"----- CLIP -----")
    clip_dataset, clip_loader = get_dataloader(dataset_name, does_resize = True)
    class_names = clip_dataset.classes
    clip_model = NormalCLIP(class_names)
    clip_model.eval()
    evaluation(clip_model, clip_loader, model_name="clip", cuda_i=0)

    print(f"----- ViT -----")
    vit_dataset, vit_loader = get_dataloader(dataset_name, does_resize = False)
    vit_model = get_vit_model(dataset_name)
    vit_model = vit_model.to(f"cpu")
    vit_model.eval()
    # feature_extractor, classifier = split_up_model(vit_model, "Standard", "cifar10")
    evaluation(vit_model, vit_loader, model_name="vit", cuda_i=1)
