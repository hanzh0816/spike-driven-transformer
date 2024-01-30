# build lr_scheduler

import numpy as np
import torch


def init_seed(config):
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
