import os
import time
import datetime
from numpy import dtype, int64
from tqdm import tqdm
import torch.nn.functional as F

import torch
from accelerate import Accelerator
import torch.nn as nn
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import accuracy, AverageMeter

from datasets import build_loader
from model import build_model
from config import parse_option
from utils import set_logger, init_seed
import random
import wandb
import model

from timm.loss import (
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
    JsdCrossEntropy,
    BinaryCrossEntropy,
)

if __name__ == "__main__":

    from yacs.config import CfgNode as CN
    import matplotlib.pyplot as plt
    import numpy as np

    _C = CN()
    _C.DATA = CN()
    _C.DATA.DATA_PATH = r"/data1/hzh/cifar10"
    _C.DATA.IMG_SIZE = 32
    _C.MODEL = CN()
    _C.MODEL.NUM_CLASSES = 10
    _C.MODEL.NAME = "ResNet"
    config = _C.clone()

    model = model.build_model(config)
    print(model)
    y = model(torch.randn(1, 3, 32, 32))
    print(y.size())
    
    # train_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
    # x = torch.ones((1, 10))

    # label = [3]
    # label = torch.tensor(label, dtype=torch.int64)
    # print(x.shape)
    # print(f"Training loss:{train_loss_fn(x,label)}")
