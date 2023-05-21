from copy import deepcopy
from easydict import EasyDict
import numpy as np
from tqdm import tqdm
import os


from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

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