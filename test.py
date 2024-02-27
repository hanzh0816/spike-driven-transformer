import os
import time
import datetime
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


if __name__ == "__main__":
    # os.chdir(os.path.dirname(__file__))
    args, config = parse_option()
    print(config)
    # print(config.__dict__)
    # main(accelerator, args, config, logger)
