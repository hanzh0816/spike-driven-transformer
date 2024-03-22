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
    CIFAR10_TRAIN_MEAN = (0.485, 0.456, 0.406)
    CIFAR10_TRAIN_STD = (0.229, 0.224, 0.225)

    if is_train:
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=is_train,
            color_jitter=config.DATA.COLOR_JITTER,
            auto_augment=config.DATA.AUTO_AUG,
            interpolation="bicubic",
            re_prob=config.DATA.RE_PROB,
            re_mode=config.DATA.RE_MODE,
            re_count=config.DATA.RE_COUNT,
            mean=CIFAR10_TRAIN_MEAN,
            std=CIFAR10_TRAIN_STD,
        )
        return transform

    transform = transforms.Compose(
        [
            transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD),
        ]
    )
    return transform


def _build_cifar_dataset(is_train, config):

    transform = _build_transform(is_train, config)
    prefix = "train" if is_train else "val"
    root = os.path.join(config.DATA.DATA_PATH, prefix)
    dataset = datasets.CIFAR10(
        root=root, train=is_train, download=True, transform=transform  # type: ignore
    )
    return dataset
