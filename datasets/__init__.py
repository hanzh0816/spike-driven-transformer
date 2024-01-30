from .imagenet_dataset import build_imagenet_loader


def build_loader(config):
    if config.DATA.DATASET == "imagenet":
        return build_imagenet_loader(config)
    else:
        raise NotImplementedError
