import torch
import torch.nn as nn
from spikingjelly.clock_driven import MultiStepLIFNode, MultiStepParametricLIFNode

from timm.models.layers import DropPath, to_2tuple


class MS_SPS(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        in_channels=2,
        embed_dims=256,
        pooling_stat=[1, 1, 1, 1],
        spike_mode="lif",
    ) -> None:
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = ()
        self.patch_size = patch_size
        self.pooling_stat = pooling_stat

        self.C = in_channels
        self.H, self.W = (
            self.image_size[0] // patch_size[0],
            self.image_size[1] // patch_size[1],
        )  # SPS result : H*W*embed_dims
