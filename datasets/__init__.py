from .imagenet_dataset import _build_imagenet_dataset
from .cifar10_datasets import _build_cifar_dataset

import torch
import torch.utils.data


def build_loader(config):
    if config.DATA.DATASET == "imagenet":
        dataset_train = _build_imagenet_dataset(is_train=True, config=config)
        dataset_val = _build_imagenet_dataset(is_train=False, config=config)
    elif (
        config.DATA.DATASET == "cifar10"
        or config.DATA.DATASET == "cifar100"
        or config.DATA.DATASET == "cifar_tiny"
    ):
        dataset_train = _build_cifar_dataset(is_train=True, config=config)
        dataset_val = _build_cifar_dataset(is_train=False, config=config)

    else:
        raise NotImplementedError

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=config.WORLD_SIZE,
        rank=config.LOCAL_RANK,
        shuffle=True,
    )
    if config.DIST_EVAL:
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val,
            num_replicas=config.WORLD_SIZE,
            rank=config.LOCAL_RANK,
            shuffle=False,
        )
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset=dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.VAL_BATCH_SIZE,
        drop_last=False,
    )

    return dataset_train, dataset_val, data_loader_train, data_loader_val
