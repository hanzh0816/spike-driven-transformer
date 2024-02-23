from email.policy import default
import os
from numpy import add
import yaml
import logging
import torch.distributed as dist
from yacs.config import CfgNode as CN

_C = CN()


def _add_BASE_NODE():
    """basic settings"""
    _C.BASE = [""]
    _C.LOG_LEVEL = logging.INFO
    # Tag of experiment, (cli, not required)
    _C.TAG = "default"
    # Frequency to save checkpoint (cli, not required)
    _C.SAVE_FREQ = 1
    # Frequency to logging info (cli, not vrequired)
    _C.PRINT_FREQ = 10
    # Path to output folder (cli, required)
    _C.OUTPUT = ""
    # Fixed random seed
    _C.SEED = 0
    # eval mode (cli, not required)
    _C.EVAL_MODE = False
    _C.EVAL_METRIC = "top1"  # best metric(top1,top5,loss)
    _C.EXPERIMENT = ""


def _add_DATA_NODE():
    """data settings"""
    _C.DATA = CN()
    # image settings (config, not required)
    _C.DATA.IMG_SIZE = 224
    _C.DATA.CHANNELS = 3

    # (cli & config, required)
    _C.DATA.BATCH_SIZE = 128
    # (cli & config, required)
    _C.DATA.VAL_BATCH_SIZE = 32

    # (cli, required)
    _C.DATA.DATA_PATH = ""
    # (config, required)
    _C.DATA.DATASET = ""

    # todo: data augmentation settings


def _add_MODEL_NODE():
    """model settings"""
    _C.MODEL = CN()
    # model name (config, required)
    _C.MODEL.NAME = "sdt"
    # num of classesconfig (config, not required)
    _C.MODEL.NUM_CLASSES = 1000

    # transformer settings, (config , not required)
    _C.MODEL.PATCH_SIZE = 16
    _C.MODEL.NUM_HEADS = 8
    # embedding dims
    _C.MODEL.EMBED_DIMS = 512
    # mlp block hidden_dims ratio
    _C.MODEL.MLP_RATIO = 1.0
    _C.MODEL.NUM_LAYERS = 4
    _C.MODEL.POOL_STAT = "1111"

    # LIF settings (config , required)
    _C.MODEL.SPIKE_MODE = "lif"
    _C.MODEL.SPIKE_T = 4
    _C.MODEL.BACKEND = "torch"

    # Resume full model and optimizer state from checkpoint (cli argument, not required)
    _C.MODEL.RESUME = False
    _C.MODEL.RESUME_PATH = ""

    # Initialize model from this checkpoint (cli argument, not required)
    _C.MODEL.INIT_CHECKPOINT = False
    _C.MODEL.CHECKPOINT_PATH = ""

    # Using pretrained model  (config, not required)
    _C.MODEL.PRETRAINED = False
    _C.MODEL.PRETRAINED_PATH = ""


def _add_TRAIN_NODE():
    """training settings"""
    _C.TRAIN = CN()

    # Training method
    _C.TRAIN.TET = False

    # training parameters
    _C.TRAIN.START_EPOCH = 0
    _C.TRAIN.EPOCHS = 200
    _C.TRAIN.BCE_LOSS = False
    _C.TRAIN.JSD_LOSS = False
    _C.TRAIN.LABEL_SMOOTH = 0.1


def _add_OPTIMIZER_NODE():
    """optimizer settings"""
    _C.OPTIMIZER = CN()
    # (cli argument, not required)

    # opt method
    _C.OPTIMIZER.OPT = "sgd"
    # Optimizer params (config , not required)
    _C.OPTIMIZER.EPS = None
    _C.OPTIMIZER.BETAS = None
    _C.OPTIMIZER.MOMENTUM = 0.9
    _C.OPTIMIZER.WEIGHT_DECAY = 0.0001
    _C.OPTIMIZER.CLIP_GRAD = None  # Clip gradient norm (default: None, no clipping)
    _C.OPTIMIZER.CLIP_MODE = "norm"  # Gradient clipping mode.("norm", "value", "agc")


def _add_LR_SCHEDULER_NODE():
    _C.LR_SCHEDULER = CN()
    # (cli argument, not required)
    _C.LR_SCHEDULER.SCHED = "step"
    _C.LR_SCHEDULER.LR = 0.01

    _C.LR_SCHEDULER.WARMUP_LR = 0.0001  # warmup learning rate (default: 0.0001)
    _C.LR_SCHEDULER.DECAY_EPOCHS = 30  # epoch interval to decay LR
    _C.LR_SCHEDULER.WARMUP_EPOCHS = 3  # epochs to warmup LR, if scheduler supports
    _C.LR_SCHEDULER.DECAY_RATE = 0.1  # LR decay rate (default: 0.1)


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault("BASE", [""]):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print("=> merge config from {}".format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def _update_config(config, args):

    _update_config_from_file(config, args.cfg)
    
    config.defrost()

    config.EXPERIMENT = args.exp

    # output
    if args.tag:
        config.TAG = args.tag
    if args.save_freq:
        config.SAVE_FREQ = args.save_freq
    if args.print_freq:
        config.PRINT_FREQ = args.print_freq

    config.OUTPUT = args.output
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    # data
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.vb_size:
        config.DATA.VAL_BATCH_SIZE = args.vb_size
    config.DATA.DATA_PATH = args.data_path

    # resume
    if args.resume:
        assert args.resume_path is not None, "resume path must be specified"
        config.MODEL.RESUME = args.resume
        config.MODEL.RESUME_PATH = args.resume_path

    # init checkpoint
    if args.init_checkpoint:
        assert args.checkpoint_path is not None, "checkpoint path must be specified"
        config.MODEL.INIT_CHECKPOINT = args.init_checkpoint
        config.MODEL.CHECKPOINT_PATH = args.checkpoint_path

    # model
    if args.eval:
        config.EVAL_MODE = args.eval
    if args.eval_metric:
        config.EVAL_METRIC = args.eval_metric

    # training
    if args.start_epoch:
        config.TRAIN.START_EPOCH = args.start_epoch
    if args.epochs:
        config.TRAIN.EPOCHS = args.epochs

    # optimizer
    if args.opt:
        config.OPTIMIZER.OPT = args.opt
    if args.opt_eps:
        config.OPTIMIZER.EPS = args.opt_eps
    if args.opt_betas:
        config.OPTIMIZER.BETAS = args.opt_betas
    if args.momentum:
        config.OPTIMIZER.MOMENTUM = args.momentum
    if args.weight_decay:
        config.OPTIMIZER.WEIGHT_DECAY = args.weight_decay
    if args.clip_grad:
        config.OPTIMIZER.CLIP_GRAD = args.clip_grad
    if args.clip_mode:
        config.OPTIMIZER.CLIP_MODE = args.clip_mode

    # lr scheduler
    if args.sched:
        config.LR_SCHEDULER.SCHED = args.sched
    if args.lr:
        config.LR_SCHEDULER.LR = args.lr
    if args.warmup_lr:
        config.LR_SCHEDULER.WARMUP_LR = args.warmup_lr
    if args.decay_epochs:
        config.LR_SCHEDULER.DECAY_EPOCHS = args.decay_epochs
    if args.warmup_epochs:
        config.LR_SCHEDULER.WARMUP_EPOCHS = args.warmup_epochs
    if args.decay_rate:
        config.LR_SCHEDULER.DECAY_RATE = args.decay_rate

    return config


def init_config():
    _add_BASE_NODE()
    _add_DATA_NODE()
    _add_MODEL_NODE()
    _add_TRAIN_NODE()
    _add_OPTIMIZER_NODE()
    _add_LR_SCHEDULER_NODE()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    init_config()
    config = _C.clone()
    config = _update_config(config, args)
    config.freeze()
    return config
