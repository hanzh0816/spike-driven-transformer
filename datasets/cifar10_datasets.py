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


def _build_dataset(is_train, config):
    transform = _build_transform(is_train, config)
    prefix = "train" if is_train else "val"
    root = os.path.join(config.DATA.DATA_PATH, prefix)
    dataset = datasets.CIFAR10(
        root=root, train=is_train, download=True, transform=transform
    )
    return dataset


def build_cifar_loader(config):
    dataset_train = _build_dataset(is_train=True, config=config)
    dataset_val = _build_dataset(is_train=False, config=config)
    
    sampler_train = torch.utils.data.Sampler(dataset_train)
    sampler_val 

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


def build_cifar_tiny_loader(config):
    # todo: implement load tiny cifar dataset
    pass


if __name__ == "__main__":

    from yacs.config import CfgNode as CN
    import matplotlib.pyplot as plt
    import numpy as np

    _C = CN()
    _C.DATA = CN()
    _C.DATA.DATA_PATH = r"/data1/hzh/cifar10"
    _C.DATA.IMG_SIZE = 32
    config = _C.clone()
    dataset_train, dataset_val, dataloader_train, dataloader_val = build_cifar_loader(
        config=config
    )

    batch = next(iter(dataloader_train))
    images, labels = batch
    image = images[0]
    label = labels[0]
    print(
        f"image type: {type(image)} , image size:{images.shape} , label type:{type(label)}"
    )

    image = image.numpy()
    # plt.imshow(
    #     np.transpose(image, (1, 2, 0))
    # )  # 如果数据格式为（C，H，W），需要转换为（H，W，C）
    # plt.axis("off")
    # plt.show()

    print("Label:", label)
