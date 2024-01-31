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
    return transforms.Compose(t)


def _build_dataset(is_train, config):
    transform = _build_transform(is_train, config)
    prefix = "train" if is_train else "val"
    root = os.path.join(config.DATA.DATA_PATH, prefix)
    dataset = datasets.ImageFolder(root, transform=transform)  # type: ignore
    return dataset


def build_imagenet_loader(config):
    dataset_train = _build_dataset(is_train=True, config=config)
    dataset_val = _build_dataset(is_train=False, config=config)

    dataloader_train = torch.utils.data.dataloader.DataLoader(
        dataset=dataset_train,
        shuffle=True,
        drop_last=True,
    )

    dataloader_val = torch.utils.data.dataloader.DataLoader(
        dataset=dataset_val,
        shuffle=True,
        drop_last=True,
    )

    return dataset_train, dataset_val, dataloader_train, dataloader_val
