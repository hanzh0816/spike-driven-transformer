import os
import shutil
import torch
import numpy as np
import random
from torchvision import datasets, transforms
import torch.distributed as dist
from timm.data import Mixup
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import to_2tuple
import torch.utils.data.dataloader


def _build_transform(is_train, config):
    if is_train:
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
        )
        return transform

    t = []

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    t.append(transforms.Resize(size=to_2tuple(config.DATA.IMG_SIZE)))
    return transforms.Compose(t)


def _build_imagenet_dataset(is_train, config):
    transform = _build_transform(is_train, config)
    prefix = "train" if is_train else "val"
    root = os.path.join(config.DATA.DATA_PATH, prefix)
    dataset = datasets.ImageFolder(root, transform=transform)  # type: ignore
    return dataset
