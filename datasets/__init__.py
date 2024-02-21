from .imagenet_dataset import build_imagenet_loader
from .cifar10_datasets import build_cifar_loader


def build_loader(config):
    if config.DATA.DATASET == "imagenet":
        return build_imagenet_loader(config)
    elif config.DATA.DATASET == "cifar10" or config.DATA.DATASET == "cifar100":
        return build_cifar_loader(config)
    else:
        raise NotImplementedError
