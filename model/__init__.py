from pyexpat import model
from .spikeformer import SpikeDrivenTransformer


def build_model(config):
    model = SpikeDrivenTransformer(
        img_size_h=config.DATA.IMG_SIZE,
        img_size_w=config.DATA.IMG_SIZE,
        patch_size=config.MODEL.PATCH_SIZE,
        in_channels=config.DATA.CHANNELS,
        embed_dims=config.MODEL.EMBED_DIMS,
        mlp_ratio=config.MODEL.MLP_RATIO,
        num_heads=config.MODEL.NUM_HEADS,
        num_classes=config.MODEL.NUM_CLASSES,
        spike_T=config.MODEL.SPIKE_T,
        TET=config.TRAIN.TET,
        pooling_stat=config.MODEL.POOL_STAT,
        spike_mode=config.MODEL.SPIKE_MODE,
        backend=config.MODEL.BACKEND,
        depths=config.MODEL.NUM_LAYERS,
    )
    return model
