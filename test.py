import os
import torch
import torch.nn as nn
from model import SpikeDrivenTransformer

os.chdir(os.path.dirname(__file__))
T, B, C, H, W = (4, 32, 3, 128, 128)
x = torch.rand([B, T, C, H, W], requires_grad=True)
sdt = SpikeDrivenTransformer(
    img_size_h=128,
    img_size_w=128,
    patch_size=16,
    in_channels=3,
    embed_dims=512,
    mlp_ratio=0.8,
    num_heads=8,
    num_classes=1000,
    spike_T=4,
    TET=False,
)
hook = None
x, hook = sdt(x, hook)
print(x.shape)
