import os
import time
import wandb
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional
import torch.backends.cudnn as cudnn

from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import accuracy, AverageMeter
import timm.optim.optim_factory as optim_factory

from model import build_model
from config import parse_option
from datasets import build_loader
import utils
from utils.train_utils import is_main_process


def main(config, logger):
    _, _, data_loader_train, data_loader_val = build_loader(config)

    # todo: add mixup here

    logger.info(f"Creating model:{config.MODEL.NAME}")
    model = build_model(config)
    model.to(device)

    eff_batch_size = (
        config.DATA.BATCH_SIZE * config.TRAIN.ACCUM_ITER * config.WORLD_SIZE
    )
    actual_lr = config.LR_SCHEDULER.LR * eff_batch_size / 256
    if utils.is_main_process():
        print("base lr: %.2e" % config.LR_SCHEDULER.LR)
        print("actual lr: %.2e" % actual_lr)
        print("accumulate grad iterations: %d" % config.TRAIN.ACCUM_ITER)
        print("effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[config.LOCAL_RANK],
        find_unused_parameters=False,
    )

    


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    args, config = parse_option()
    config = utils.init_distributed_mode(config)

    logger = utils.set_logger(config=config)

    device = torch.device("cuda", config.LOCAL_RANK)

    utils.init_seed(config)

    cudnn.benchmark = True

    main(config, logger)
