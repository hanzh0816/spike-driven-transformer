# build lr_scheduler
import os
import numpy as np
from collections import defaultdict, deque
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


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.

    References:
    Spike-driven Transformer V2:https://github.com/BICLab/Spike-Driven-Transformer-V2
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{max:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricMeter(object):
    """
    References:
    Spike-driven Transformer V2:https://github.com/BICLab/Spike-Driven-Transformer-V2
    """

    def __init__(self):
        self.meters = defaultdict(SmoothedValue)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return "\t".join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter


def get_world_size():
    return dist.get_world_size()


def get_rank():
    return dist.get_rank()


def is_main_process():
    if dist.get_rank() == 0:
        return True
    else:
        return False


def init_distributed_mode(config: CN):
    config.defrost()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        config.WORLD_ANK = int(os.environ["RANK"])
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
        config.WORLD_ANK
        and config.WORLD_SIZE
        and config.DIS_BACKEND
        and config.DIS_URL is not None
    )

    # 启动多GPU
    dist.init_process_group(
        backend=config.DIS_BACKEND,
        init_method=config.DIS_URL,
        world_size=config.WORLD_SIZE,
        rank=config.WORLD_ANK,
    )
    dist.barrier()

    config.freeze()
    return config


def init_seed(config):
    seed = config.SEED + config.LOCAL_RANK
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def wandb_init(config):

    wandb.init(
        project="spike-driven transformer",
        config=config,
        entity="snn-training",
        job_type="training",
        reinit=True,
        dir=config.OUTPUT,
        name=config.TAG,
    )


def actual_lr(config: CN):
    config.defrost()
    eff_batch_size = (
        config.DATA.BATCH_SIZE * config.TRAIN.ACCUM_ITER * config.WORLD_SIZE
    )
    ratio = eff_batch_size / 256

    actual_lr = config.LR_SCHEDULER.LR * ratio
    if is_main_process():
        print("base lr: %.2e" % config.LR_SCHEDULER.LR)
        print("actual lr: %.2e" % actual_lr)
        print("accumulate grad iterations: %d" % config.TRAIN.ACCUM_ITER)
        print("effective batch size: %d" % eff_batch_size)
    config.LR_SCHEDULER.LR = actual_lr
    config.LR_SCHEDULER.MIN_LR = config.LR_SCHEDULER.MIN_LR * ratio
    config.LR_SCHEDULER.WARMUP_LR = config.LR_SCHEDULER.WARMUP_LR * ratio

    config.freeze()
    return config


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


def get_lr_scheduler(config, optimizer):
    lr_scheduler, num_epochs = create_scheduler(
        args=get_lr_scheduler_args(config), optimizer=optimizer
    )
    return lr_scheduler, num_epochs


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [
        correct[: min(k, maxk)].reshape(-1).float().sum(0) * 100.0 / batch_size
        for k in topk
    ]


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


if __name__ == "__main__":
    train_meter = MetricMeter()
    train_meter.update(a=1)
    train_meter.meters["b"].update([1, 2, 3], n=3)

    print(train_meter.b.global_avg)
