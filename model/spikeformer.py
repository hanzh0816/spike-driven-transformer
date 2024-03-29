import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)

from module import MS_SPS, Encoder, Decoder

from timm.models.layers import trunc_normal_


class SpikeDrivenTransformer(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,  # ViT default patch size
        in_channels=3,
        embed_dims=512,  # embeding dimension(SPS module output dims)
        mlp_ratio=1.0,  # MLP ratio
        num_heads=8,
        num_classes=1000,  # cls num classes
        spike_T=4,  # spike train length
        TET=False,
        pooling_stat="1111",
        spike_mode="lif",  # spike neuron mode
        backend="torch",
        depths=4,  # number of encoder block
    ):
        super(SpikeDrivenTransformer, self).__init__()
        self.spike_T = spike_T
        self.TET = TET

        self.patch_embed = MS_SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            in_channels=in_channels,
            embed_dims=embed_dims,
            patch_size=patch_size,
            pooling_stat=pooling_stat,
            spike_mode=spike_mode,
            backend=backend,
        )

        self.encoder = Encoder(
            embed_dims=embed_dims,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            spike_mode=spike_mode,
            backend=backend,
            depths=depths,
        )

        self.decoder = Decoder(
            embed_dims=embed_dims,
            num_classes=num_classes,
            spike_mode=spike_mode,
            backend=backend,
        )

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, hook=None):
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.spike_T, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()

        x = self.patch_embed(x)
        x = self.encoder(x)

        x = x.flatten(3).mean(3)  # T,B,C,H,W -> T,B,C
        x = self.decoder(x)

        if not self.TET:
            x = x.mean(0)  # T,B,num_classes -> B,num_classes
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
