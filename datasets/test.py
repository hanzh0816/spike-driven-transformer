from imagenet_dataset import _build_imagenet_dataset
from cifar10_datasets import _build_cifar_dataset

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
        batch_size=config.DATA.BATCH_SIZE,
        drop_last=False,
    )

    return dataset_train, dataset_val, data_loader_train, data_loader_val




if __name__ == "__main__":

    from yacs.config import CfgNode as CN
    import matplotlib.pyplot as plt
    import numpy as np

    _C = CN()
    _C.DATA = CN()
    _C.DATA.DATA_PATH = r"./cifar10"
    _C.DATA.IMG_SIZE = 32
    _C.DATA.DATASET = 'cifar10'
    _C.DATA.BATCH_SIZE = 10
    _C.WORLD_SIZE = 4
    _C.LOCAL_RANK =0

    _C.DIST_EVAL = False
    config = _C.clone()
    dataset_train, dataset_val, dataloader_train, dataloader_val = build_loader(
        config=config
    )

    batch = next(iter(dataloader_train))
    images, labels = batch
    image = images[0]
    label = labels[0]
    print(
        f"image type: {type(image)} , image size:{images.shape} , label type:{type(label)} , batch size: {len(batch[0])}"
    )

    image = image.numpy()
    # plt.imshow(
    #     np.transpose(image, (1, 2, 0))
    # )  # 如果数据格式为（C，H，W），需要转换为（H，W，C）
    # plt.axis("off")
    # plt.show()

    print("Label:", label)
