import os
import time
import datetime
from numpy import int64
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

from timm.loss import (
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
    JsdCrossEntropy,
    BinaryCrossEntropy,
)

if __name__ == "__main__":
    train_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
    x = torch.ones(10).unsqueeze(0)
    y = torch.zeros(10).unsqueeze(0)
    print(f"Training loss:{train_loss_fn(x,y)}")
