import torch.nn as nn
from .spikeformer import SpikeDrivenTransformer
from .resnet import ResNet50

from .spikeformer_origin import SpikeDrivenTransformer as sdt_origin
from timm.models.vision_transformer import _cfg


def build_model(config):
    if config.MODEL.NAME.startswith("sdt"):
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
    elif config.MODEL.NAME.startswith("ResNet"):
        model = ResNet50(config.MODEL.NUM_CLASSES)
    elif config.MODEL.NAME == "sdt-origin":
        model = sdt_origin(
            img_size_h=config.DATA.IMG_SIZE,
            img_size_w=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.PATCH_SIZE,
            in_channels=config.DATA.CHANNELS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dims=config.MODEL.EMBED_DIMS,
            num_heads=config.MODEL.NUM_HEADS,
            mlp_ratios=config.MODEL.MLP_RATIO,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            depths=config.MODEL.NUM_LAYERS,
            sr_ratios=1,
            T=config.MODEL.SPIKE_T,
            pooling_stat=config.MODEL.POOL_STAT,
            attn_mode="direct_xor",
            spike_mode="lif",
            dvs_mode=False,
            TET=False,
        )
        model.default_cfg = _cfg()
    else:
        raise NotImplementedError
    return model
