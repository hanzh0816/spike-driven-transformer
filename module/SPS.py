from copy import deepcopy
from sympy import false
import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)

from timm.models.layers import DropPath, to_2tuple


def get_lif_neuron(tau, mode, backend="torch"):
    lif = None
    if mode == "lif":
        lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)
    elif mode == "plif":
        lif = MultiStepParametricLIFNode(
            init_tau=tau, detach_reset=True, backend=backend
        )
    else:
        raise ValueError("Invalid lif mode")
    return lif


class PSModule(nn.Module):
    def __init__(
        self,
        img_size_h,
        img_size_w,
        in_channels,
        embed_dims,
        pooling_stat="1111",
        spike_mode="lif",
        backend="torch",
    ):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        self.pooling_stat = pooling_stat

        # layer 1
        self.proj_conv1 = nn.Conv2d(
            in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif1 = get_lif_neuron(tau=2.0, mode=spike_mode, backend=backend)

        # layer 2
        self.proj_conv2 = nn.Conv2d(
            embed_dims // 8,
            embed_dims // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif2 = get_lif_neuron(tau=2.0, mode=spike_mode, backend=backend)

        # layer 3
        self.proj_conv3 = nn.Conv2d(
            embed_dims // 4,
            embed_dims // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn3 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif3 = get_lif_neuron(tau=2.0, mode=spike_mode, backend=backend)

        # layer 4 w/o lif neuron
        self.proj_conv4 = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn4 = nn.BatchNorm2d(embed_dims)

        # max pool layer, MP is unlearnable
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, hook: dict = None) -> torch.Tensor:
        T, B, _, H, W = x.shape
        ratio = 1

        # layer 1
        x = self.proj_conv1(x.flatten(0, 1))
        x = self.proj_bn1(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif1(x)
        if hook is not None:
            hook[self._get_name() + "_lif1"] = x.detach()

        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat["0"] == "1":
            x = self.max_pool(x)
            ratio *= 2

        # layer 2
        x = self.proj_conv2(x.flatten(0, 1))
        x = self.proj_bn2(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif2(x)
        if hook is not None:
            hook[self._get_name() + "_lif2"] = x.detach()

        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat["1"] == "1":
            x = self.max_pool(x)
            ratio *= 2

        # layer 3
        x = self.proj_conv3(x.flatten(0, 1))
        x = self.proj_bn3(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif3(x)
        if hook is not None:
            hook[self._get_name() + "_lif3"] = x.detach()

        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat["2"] == "1":
            x = self.max_pool(x)
            ratio *= 2

        # layer 4
        x = self.proj_conv4(x.flatten(0, 1))
        x = self.proj_bn4(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()

        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat["3"] == "1":
            x = self.max_pool(x)
            ratio *= 2

        x = x.reshape(T, B, -1, H // ratio, W // ratio).contiguous()

        return x, hook


class RPEModule(nn.Module):
    def __init__(self, embed_dims, spike_mode, backend="torch"):
        self.rpe_proj_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = get_lif_neuron(tau=2.0, mode=spike_mode, backend=backend)

    def forward(self, x: torch.Tensor, hook: dict = None) -> torch.Tensor:
        T, B, _, H, W = x.shape

        # MS shortcut
        x_feat = deepcopy(x)
        x_feat = x_feat.flatten(0, 1).contiguous()
        # early lif neuron
        x = self.rpe_lif(x)
        if hook is not None:
            hook[self._get_name() + "_lif"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        x = self.rpe_proj_conv(x)
        x = self.rpe_bn(x)
        x = (x + x_feat).reshape(T, B, -1, H, W).contiguous()

        return x, hook


class MS_SPS(nn.Module):
    def __init__(
        self,
        img_size_h,
        img_size_w,
        in_channels,
        embed_dims,
        patch_size,
        pooling_stat="1111",
        spike_mode="lif",
        backend="torch",
    ) -> None:
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        self.patch_size = to_2tuple(patch_size)
        self.pooling_stat = pooling_stat

        # assert pooling_stat equivalent to patch size
        if 2 ** self.pooling_stat.count("1") != patch_size:
            raise Exception("num of pooling times does not match patch size")

        self.C = in_channels
        self.H, self.W = (
            self.image_size[0] // patch_size[0],
            self.image_size[1] // patch_size[1],
        )  # SPS result : H*W*embed_dims

        # PSM module
        self.psm = PSModule(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            in_channels=in_channels,
            embed_dims=embed_dims,
            patch_size=patch_size,
            pooling_stat=pooling_stat,
            spike_mode=spike_mode,
        )

        self.rpe = RPEModule(
            embed_dims=embed_dims, spike_mode=spike_mode, backend=backend
        )

    def forward(self, x: torch.Tensor, hook: dict = None) -> torch.Tensor:
        x, hook = self.psm(x, hook)
        x, hook = self.rpe(x, hook)
        return x, hook


if __name__ == "__main__":
    a = get_lif_neuron(2.0, "lif")
    b = get_lif_neuron(2.0, "alif")
    print(id(a))
    print(id(b))
