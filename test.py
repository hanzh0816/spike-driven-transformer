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
import utils

from timm.loss import (
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
    JsdCrossEntropy,
    BinaryCrossEntropy,
)

from matplotlib import pyplot as plt

def get_lr_per_epoch(scheduler, num_epoch):
    lr_per_epoch = []
    for epoch in range(num_epoch):
        lr_per_epoch.append(scheduler._get_lr(epoch))
    return lr_per_epoch

def print_dist(config):
    print(os.environ)
    print("|| MASTER_ADDR:",os.environ["MASTER_ADDR"],
        "|| MASTER_PORT:",os.environ["MASTER_PORT"],
        "|| LOCAL_RANK:",os.environ["LOCAL_RANK"],
        "|| RANK:",os.environ["RANK"], 
        "|| WORLD_SIZE:",os.environ["WORLD_SIZE"])
    print()
    
    print(f"world_size : {config.WORLD_SIZE} , local_rank : {config.LOCAL_RANK}")

if __name__ == "__main__":
    # os.chdir(os.path.dirname(__file__))
    args, config = parse_option()
    config = utils.init_distributed_mode(config)
    device = torch.device("cuda", config.LOCAL_RANK)
    
    utils.init_seed(config)
    _, _, data_loader_train, data_loader_val = build_loader(config)

    model = build_model(config)
    model.to(device)

    # correct actual lr
    config = utils.actual_lr(config)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[config.LOCAL_RANK],
        find_unused_parameters=False,
    )
    model_without_ddp = model.module
    # create optimizer
    optimizer = utils.get_optimizer(config, model)

    # create lr scheduler
    lr_scheduler, _ = utils.get_lr_scheduler(config, optimizer)

    num_epoch = 200
    lr_per_epoch = get_lr_per_epoch(lr_scheduler, num_epoch)
    plt.plot([i for i in range(num_epoch)], lr_per_epoch)
    plt.show()



# if __name__ == "__main__":

# from yacs.config import CfgNode as CN
# import matplotlib.pyplot as plt
# import numpy as np

# _C = CN()
# _C.DATA = CN()
# _C.DATA.DATA_PATH = r"/data1/hzh/cifar10"
# _C.DATA.IMG_SIZE = 32
# _C.MODEL = CN()
# _C.MODEL.NUM_CLASSES = 10
# _C.MODEL.NAME = "ResNet"
# config = _C.clone()

# model = model.build_model(config)
# print(model)
# y = model(torch.randn(1, 3, 32, 32))
# print(y.size())

# train_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
# x = torch.ones((1, 10))

# label = [3]
# label = torch.tensor(label, dtype=torch.int64)
# print(x.shape)
# print(f"Training loss:{train_loss_fn(x,label)}")
