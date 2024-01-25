import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)

from module import *


class SpikeDrivenTransformer(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,  # ViT default patch size
        in_channels=3,
        embed_dims=512,  # embeding dimension(SPS module output dims)
        num_heads=8,
        pooling_stat=[1, 1, 1, 1],
    ):
        """
        initialize the SpikeDrivenTransformer
        """
        patch_embed = MS_SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            in_channels=in_channels,
            embed_dims=embed_dims,
        )

    def forward(self, x):
        pass
