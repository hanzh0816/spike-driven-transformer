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


class Decoder(nn.Module):
    def __init__(self, embed_dims, num_classes, spike_mode="lif", backend="torch"):
        super(Decoder, self).__init__()
        self.head_lif = get_lif_neuron(tau=2.0, mode=spike_mode, backend=backend)
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, hook: dict = None):
        # x.shape [T,B,C]
        x = self.head_lif(x)
        if hook is not None:
            hook["head_lif"] = x.detach()
        x = self.head(x)
        return x, hook
