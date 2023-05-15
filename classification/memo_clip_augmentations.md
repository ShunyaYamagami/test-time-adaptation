import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
"""
前処理を変えたら精度が急激に上がった．
前処理方法は clpi.load preprocess を参照.
\_convert_image_to_rgb のみ行わない

# CIFAR10 実験

以下, normally_classify.py より, CIFAR10 での実験結果
ViT
Accuracy: 94.77%

以下の結果から，
`transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))`
が重要であることがわかる．

"""

```
transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
Accuracy: 70.10%
```

```
transform = transforms.Compose([
    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
Accuracy: 88.77%
```

```
transform = transforms.Compose([
    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
Accuracy: 69.54%
```

```
transform = transforms.Compose([
    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
Accuracy: 88.77%
```

```
transform = transforms.Compose([
    transforms.Resize(224),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
Accuracy: 87.80%
```
