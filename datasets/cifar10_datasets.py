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
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    if is_train:
        transform = transforms.Compose(
            [
                transforms.Resize(size=to_2tuple(config.DATA.IMG_SIZE)),
                # 对原始32*32图像四周各填充4个0像素（40*40），然后随机裁剪成32*32
                transforms.RandomCrop(32, padding=4),
                # 按0.5的概率水平翻转图片
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
            ]
        )
        return transform

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=to_2tuple(config.DATA.IMG_SIZE)),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
        ]
    )
    return transform


def _build_cifar_dataset(is_train, config):
    
    # todo: implement load tiny cifar dataset
    transform = _build_transform(is_train, config)
    prefix = "train" if is_train else "val"
    root = os.path.join(config.DATA.DATA_PATH, prefix)
    dataset = datasets.CIFAR10(
        root=root, train=is_train, download=True, transform=transform
    )
    return dataset



