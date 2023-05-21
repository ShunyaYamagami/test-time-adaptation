from easydict import EasyDict
import numpy as np
from tqdm import tqdm
import os

import torch
import torchvision
import torchvision.transforms as transforms
from augmentations.aug import get_jigsaw
from datasets.data_loading import get_clsdst_transform



def get_dataloader(dataset_name="cifar10", does_resize = True):
    dparams = EasyDict({
        "root_dir": "/nas/data/syamagami/tta/rmt",
        "train_split": False,
        "batch_size": 200,
        "num_workers": 4,
    })
    if does_resize:
        # refer clip.load preprocess
        transform = transforms.Compose([
            # transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        # transform = None
    
    if dataset_name == "cifar10":
        source_dataset = torchvision.datasets.CIFAR10(root=dparams.root_dir, train=dparams.train_split, download=True, transform=transform)
    elif dataset_name == "cifar100":
        source_dataset = torchvision.datasets.CIFAR100(root=dparams.root_dir, train=dparams.train_split, download=True, transform=transform)
    else:
        raise NotImplementedError(f"dataset_name: {dataset_name} is not implemented.")

    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=dparams.batch_size, shuffle=True, num_workers=dparams.num_workers, drop_last=False)
    return source_dataset, source_loader
