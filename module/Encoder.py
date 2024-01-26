import torch
import torch.nn as nn


from timm.models.layers import DropPath
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)


def get_lif_neuron(tau=2.0, mode="lif", backend="torch"):
    lif = None
    if mode == "lif":
        lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)
    elif mode == "plif":
        lif = MultiStepParametricLIFNode(init_tau=tau, detach_reset=True, backend=backend)
    else:
        raise ValueError("Invalid lif mode")
    return lif


class SDSABlock(nn.Module):
    def __init__(
        self,
        in_dims,
        hidden_dims,
        num_heads,
        spike_mode="lif",
        backend="torch",
        layer=0,
    ):
        super(SDSABlock, self).__init__()
        assert hidden_dims % num_heads == 0, f"dim {hidden_dims} should be divided by num_heads {num_heads}."

        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.spike_mode = spike_mode
        self.layer = layer

        # MS shortcut require early SN before block
        self.shortcut_lif = get_lif_neuron(tau=2.0, mode=spike_mode, backend=backend)

        self.q_proj_conv = nn.Conv2d(self.in_dims, self.hidden_dims, kernel_size=1, stride=1, bias=False)
        self.q_proj_bn = nn.BatchNorm2d(self.hidden_dims)
        self.q_proj_lif = get_lif_neuron(tau=2.0, mode=spike_mode, backend=backend)

        self.k_proj_conv = nn.Conv2d(self.in_dims, self.hidden_dims, kernel_size=1, stride=1, bias=False)
        self.k_proj_bn = nn.BatchNorm2d(self.hidden_dims)
        self.k_proj_lif = get_lif_neuron(tau=2.0, mode=spike_mode, backend=backend)

        self.v_proj_conv = nn.Conv2d(self.in_dims, self.hidden_dims, kernel_size=1, stride=1, bias=False)
        self.v_proj_bn = nn.BatchNorm2d(self.hidden_dims)
        self.v_proj_lif = get_lif_neuron(tau=2.0, mode=spike_mode, backend=backend)

        # talking-heads
        self.talking_heads_proj = nn.Conv1d(num_heads, num_heads, kernel_size=1, stride=1, bias=False)
        self.talking_heads_proj_lif = get_lif_neuron(tau=2.0, mode=spike_mode, backend=backend)

        # output project
        self.out_proj_lif = get_lif_neuron(tau=2.0, mode=spike_mode, backend=backend)
        self.out_proj_conv = nn.Conv2d(self.hidden_dims, self.hidden_dims, kernel_size=1, stride=1)
        self.out_proj_bn = nn.BatchNorm2d(self.hidden_dims)

    def forward(self, x: torch.Tensor, hook: dict = None):
        T, B, C, H, W = x.shape

        assert C % self.num_heads == 0, f"dim {C} should be divided by num_heads {self.num_heads}."

        identity = x
        N = H * W
        x = self.shortcut_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_shortcut_lif"] = x.detach()

        x_for_qkv = x.flatten(0, 1).contiguous()
        q_conv_out = self.q_proj_conv(x_for_qkv)
        q_conv_out = self.q_proj_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        q_conv_out = self.q_proj_lif(q_conv_out)  # type: torch.Tensor

        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_q_lif"] = q_conv_out.detach()

        q = (
            q_conv_out.flatten(3)  # H*W->N
            .transpose(-1, -2)  # T,B,C,N -> T,B,N,C
            .reshape(T, B, N, self.num_heads, C // self.num_heads)  # T,B,N,H,d
            .permute(0, 1, 3, 2, 4)  # T,B,H,N,d
            .contiguous()
        )

        k_conv_out = self.k_proj_conv(x_for_qkv)
        k_conv_out = self.k_proj_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        k_conv_out = self.k_proj_lif(k_conv_out)  # type: torch.Tensor

        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_k_lif"] = k_conv_out.detach()

        k = (
            k_conv_out.flatten(3)  # H*W->N
            .transpose(-1, -2)  # T,B,C,N -> T,B,N,C
            .reshape(T, B, N, self.num_heads, C // self.num_heads)  # T,B,N,H,d
            .permute(0, 1, 3, 2, 4)  # T,B,H,N,d
            .contiguous()
        )

        v_conv_out = self.v_proj_conv(x_for_qkv)
        v_conv_out = self.v_proj_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        v_conv_out = self.v_proj_lif(v_conv_out)  # type: torch.Tensor

        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_v_lif"] = v_conv_out.detach()

        v = (
            v_conv_out.flatten(3)  # H*W->N
            .transpose(-1, -2)  # T,B,C,N -> T,B,N,C
            .reshape(T, B, N, self.num_heads, C // self.num_heads)  # T,B,N,H,d
            .permute(0, 1, 3, 2, 4)  # T,B,H,N,d
            .contiguous()
        )

        # use hadamard product, head-wise calculation
        kv = k.mul(v)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv_before"] = kv.detach()

        # T,B,H,N,d -> T,B,H,1,d, sum of every column
        kv = kv.sum(dim=-2, keepdim=True)
        # todo: implement talking-heads

        kv = self.talking_heads_proj_lif(kv)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv"] = kv.detach()

        # broadcast hadamard product
        x = q.mul(kv)

        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_x_after_qkv"] = x.detach()

        # T,B,H,N,d -> T,B,C,H,W
        x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
        # todo: add out_proj_lif
        x = self.out_proj_conv(x.flatten(0, 1))
        x = self.out_proj_bn(x)
        x = x.reshape(T, B, C, H, W).contiguous()

        print(x)
        print(x.equal(identity))
        x = x + identity
        return x, v, hook


class MLPBlock(nn.Module):
    def __init__(
        self, in_features, out_features=None, hidden_features=None, spike_mode="lif", backend="torch", layer=0
    ):
        super(MLPBlock, self).__init__()
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features

        self.fc1_conv = nn.Conv2d(in_features, self.hidden_features, kernel_size=1, stride=1, bias=False)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = get_lif_neuron(tau=2.0, mode=spike_mode, backend=backend)

        self.fc2_conv = nn.Conv2d(
            self.hidden_features, self.out_features, kernel_size=1, stride=1, bias=False
        )
        self.fc2_bn = nn.BatchNorm2d(self.out_features)
        self.fc2_lif = get_lif_neuron(tau=2.0, mode=spike_mode, backend=backend)

        self.layer = layer

    def forward(self, x: torch.Tensor, hook: dict = None):
        T, B, C, H, W = x.shape
        identity = x

        x = self.fc1_lif(x)
        if hook is not None:   
            hook[self._get_name+]


class EncoderBlock(nn.Module):
    def __init__(
        self,
    ):
        pass


class Encoder(nn.Module):
    def __init__(self):
        pass


if __name__ == "__main__":
    sdsa = SDSABlock(512, 512, 8, layer=0)
    T, B, C, H, W = (4, 32, 512, 8, 8)
    x = torch.rand([T, B, C, H, W], requires_grad=True)
