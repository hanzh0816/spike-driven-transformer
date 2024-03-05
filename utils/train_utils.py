# build lr_scheduler
import os
import numpy as np
import torch
import random
import torch.nn as nn
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

import torch.distributed as dist
from timm.loss import (
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
    JsdCrossEntropy,
    BinaryCrossEntropy,
)

from yacs.config import CfgNode as CN
import wandb


class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)


def init_seed(config):
    seed = config.SEED + config.LOCAL_RANK
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loss_fn(config):
    if config.TRAIN.LABEL_SMOOTH != 0.0:
        if config.TRAIN.BCE_LOSS:
            train_loss_fn = BinaryCrossEntropy(smoothing=config.TRAIN.LABEL_SMOOTH)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(
                smoothing=config.TRAIN.LABEL_SMOOTH
            )
    else:
        if config.TRAIN.BCE_LOSS:
            train_loss_fn = BinaryCrossEntropy()
        else:
            train_loss_fn = nn.CrossEntropyLoss()

    return train_loss_fn


def get_optimizer(config, model):
    optimizer = create_optimizer(args=get_optimizer_args(config), model=model)
    return optimizer


def get_mixup_fn(config):
    # todo: create mixup fn
    pass


def get_optimizer_args(config):
    opt_dict = {
        "opt": config.OPTIMIZER.OPT,
        "lr": config.LR_SCHEDULER.LR,
        "weight_decay": config.OPTIMIZER.WEIGHT_DECAY,
        "momentum": config.OPTIMIZER.MOMENTUM,
        "opt_eps": config.OPTIMIZER.EPS,
        "opt_betas": config.OPTIMIZER.BETAS,
    }
    cfg = DictToObject(opt_dict)
    return cfg


def get_lr_scheduler_args(config):
    sched_dict = {
        "eval_metric": config.EVAL_METRIC,
        "sched": config.LR_SCHEDULER.SCHED,
        "epochs": config.TRAIN.EPOCHS,
        "decay_epochs": config.LR_SCHEDULER.DECAY_EPOCHS,
        "warmup_epochs": config.LR_SCHEDULER.WARMUP_EPOCHS,
        "cooldown_epochs": config.LR_SCHEDULER.COOLDOWN_EPOCHS,
        "warmup_lr": config.LR_SCHEDULER.WARMUP_LR,
        "min_lr": config.LR_SCHEDULER.MIN_LR,
    }
    cfg = DictToObject(sched_dict)
    return cfg


def init_distributed_mode(config: CN):
    config.defrost()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        config.RANK = int(os.environ["RANK"])
        config.WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        config.LOCAL_RANK = int(os.environ["LOCAL_RANK"])

    else:
        # print("NOT using distributed mode")
        raise EnvironmentError("NOT using distributed mode")
        # return

    # 这里需要设定使用的GPU
    torch.cuda.set_device(config.LOCAL_RANK)
    # 这里是GPU之间的通信方式，有好几种的，nccl比较快也比较推荐使用。
    config.DIS_BACKEND = "nccl"
    config.DIS_URL = "env://"

    assert (
        config.RANK
        and config.WORLD_SIZE
        and config.DIS_BACKEND
        and config.DIS_URL is not None
    )

    # 启动多GPU
    dist.init_process_group(
        backend=config.DIS_BACKEND,
        init_method=config.DIS_URL,
        world_size=config.WORLD_SIZE,
        rank=config.RANK,
    )
    dist.barrier()

    config.freeze()
    return config


def wandb_init(config, device):

    wandb.init(
        project="spike-driven transformer",
        config=config,
        entity="snn-training",
        job_type="training",
        reinit=True,
        dir=config.OUTPUT,
        name=config.TAG + str(device),
    )


def is_main_process():
    if dist.get_rank() == 0:
        return True
    else:
        return False
