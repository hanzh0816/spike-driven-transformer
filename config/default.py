import os
import yaml
import logging
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
    _C.DIST_EVAL = False

    # DDP config
    _C.WORLD_RANK = None
    _C.LOCAL_RANK = None
    _C.WORLD_SIZE = None
    _C.DIS_BACKEND = None


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

    # data augmentation settings
    _C.DATA.COLOR_JITTER = None
    _C.DATA.AUTO_AUG = "rand-m9-mstd0.5-inc1"

    # random erase settings
    _C.DATA.RE_PROB = 0.25  # random erase prob
    _C.DATA.RE_MODE = "pixel"  #  random erase mode
    _C.DATA.RE_COUNT = 1  # random erase count
    _C.DATA.RE_SPLIT = False  # Do not random erase first (clean) augmentation split

    # mixup settings
    _C.DATA.MIXUP = 0.0  # mixup alpha, mixup enabled if > 0.
    _C.DATA.CUTMIX = 0.0  # cutmix alpha, cutmix enabled if > 0.
    _C.DATA.CUTMIX_MINMAX = None  # cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)
    _C.DATA.MIXUP_PROB = (
        1.0  # Probability of performing mixup or cutmix when either/both is enabled
    )
    _C.DATA.MIXUP_SWITCH_PROB = (
        0.5  # Probability of switching to cutmix when both mixup and cutmix enabled
    )
    _C.DATA.MIXUP_MODE = (
        "batch"  # How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
    )


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
    _C.MODEL.POOL_STAT = "_1111"

    # LIF settings (config , required)
    _C.MODEL.SPIKE_MODE = "lif"
    _C.MODEL.SPIKE_T = 4
    _C.MODEL.BACKEND = "torch"

    # Resume full model and optimizer state from checkpoint (cli argument, not required)
    _C.MODEL.RESUME = False
    _C.MODEL.RESUME_PATH = ""


def _add_TRAIN_NODE():
    """training settings"""
    _C.TRAIN = CN()

    # accumulate grad iterations
    _C.TRAIN.ACCUM_ITER = 1

    # Training method
    _C.TRAIN.TET = False

    # training parameters
    _C.TRAIN.START_EPOCH = 0
    _C.TRAIN.EPOCHS = 200
    _C.TRAIN.BCE_LOSS = False
    _C.TRAIN.JSD_LOSS = False
    _C.TRAIN.LABEL_SMOOTH = 0.0


def _add_OPTIMIZER_NODE():
    """optimizer settings"""
    _C.OPTIMIZER = CN()
    # (config, not required)

    # opt method
    _C.OPTIMIZER.OPT = "sgd"
    # Optimizer params (config , not required)
    _C.OPTIMIZER.EPS = None
    _C.OPTIMIZER.BETAS = None
    _C.OPTIMIZER.MOMENTUM = 0.9
    _C.OPTIMIZER.WEIGHT_DECAY = 1e-3


def _add_LR_SCHEDULER_NODE():
    _C.LR_SCHEDULER = CN()
    # (config, not required)
    _C.LR_SCHEDULER.SCHED = "step"
    _C.LR_SCHEDULER.LR = 0.01
    _C.LR_SCHEDULER.MIN_LR = 0.0
    _C.LR_SCHEDULER.CYCLE_EPOCHS = 300  # epoch per cycle

    # cycle config
    _C.LR_SCHEDULER.CYCLE_LIMIT = 1  # max cycle num
    _C.LR_SCHEDULER.CYCLE_DECAY = 1.0  # cycle decay
    _C.LR_SCHEDULER.CYCLE_MULT = 1.0  # periodic update multiple

    # warm-up config
    _C.LR_SCHEDULER.WARMUP_LR = 1e-5  # warmup learning rate (default: 0.0001)
    _C.LR_SCHEDULER.WARMUP_EPOCHS = 20  # epochs to warmup LR, if scheduler supports
    _C.LR_SCHEDULER.WARMUP_PREFIX = False  #  If set to True, then every new epoch number equals epoch = epoch - warmup_t.

    # noise config
    _C.LR_SCHEDULER.NOISE_RANGE_T = None  # range to add noise(list tuple or int)
    _C.LR_SCHEDULER.NOISE_PCT = 0.67  # The amount of noise to be added.
    _C.LR_SCHEDULER.NOISE_STD = 1.0
    _C.LR_SCHEDULER.NOISE_SEED = 42

    # other config
    _C.LR_SCHEDULER.DECAY_RATE = 0.01  # (plateau only) decay cycle init-learning-rate
    _C.LR_SCHEDULER.DECAY_EPOCHS = 30  # （StepLR only） epoch interval to decay LR
    _C.LR_SCHEDULER.COOLDOWN_EPOCHS = 0  # (plateau only)


def _update_config_from_file(config: CN, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)

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
