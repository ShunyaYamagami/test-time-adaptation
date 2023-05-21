import os
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


def get_jigsaw(im:Image, grid=3) -> Image:
    im = im.copy()
    im_edge = im.size[0]
    s = int(im_edge / grid)
    tile = [im.crop(np.array([s * (n % grid), s * int(n / grid), s * (n % grid + 1), s * (int(n / grid) + 1)]).astype(int)) for n in range(grid**2)]
    random.shuffle(tile)
    dst = Image.new('RGB', (int(s * grid), int(s * grid)))
    for i, t in enumerate(tile):
        dst.paste(t, (i % grid * s, int(i / grid) * s))
    im = dst

    return im


def shuffle_image_tensor(image_tensor:torch.Tensor, grid:int=4, C=3) -> torch.Tensor:
    """ テンソル型に変換した後の画像をパッチに分割して Block Shuffle する.
    args:
        image_tensor (torch.Tensor): [B, C, H, W]
        grid (int): 1辺のパッチ数
        C (int): チャンネル数
    return:
        shuffled_image_tensor (torch.Tensor): [B, C, H, W]
    """
    patch_num = grid ** 2
    image_edge = image_tensor.shape[-1]

    assert grid == 4, """
            gridが3の場合, shuffle_image_tensorの出力サイズが224でなくなるので, リサイズし直す必要がある. 
            よって, 基本的にgrid=4にする. grid==3にしたい場合は, get_transform_when_iterationにてもう一度resizeし直してくれ.
            """
    
    # 最も近いgridの倍数にリサイズ
    while image_edge % grid != 0:
        image_edge += 1

    h, w = image_edge, image_edge
    patch_h, patch_w = h // grid, w // grid
    patch_i = torch.randperm(patch_num)

    unfold = image_tensor.unfold(2, patch_h, patch_h).unfold(3, patch_h, patch_w)  # [B, C, grid, grid, patch_h, patch_w]
    shuffled = unfold.reshape(-1, C, patch_num, patch_h, patch_w)[:, :, patch_i, :]  # [B, C, patch_num, patch_h, patch_w]
    shuffled_image_tensor = shuffled.reshape(-1, C, grid, grid, patch_h, patch_w).permute(0, 1, 2, 4, 3, 5).reshape(-1, C, h, w)  # [B, C, H, W]

    return shuffled_image_tensor


def get_transform_when_iteration(grid=4):
    # refer clip.load preprocess
    transform = transforms.Compose([
        # transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        # transforms.Resize(224),
        # transforms.CenterCrop(224),
        # transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    clsdst_transform = transforms.Compose([
        # transforms.Lambda(lambda im: shuffle_image_tensor(im, grid=grid)),
        transforms.Lambda(shuffle_image_tensor),
        # transforms.Resize(224),  # gridが3の場合，shuffle_image_tensorの出力サイズが224でなくなるので，リサイズし直す必要がある.よって，基本的にgrid=4にしようと思う.
        *transform.transforms,
    ])
    
    return transform, clsdst_transform
    