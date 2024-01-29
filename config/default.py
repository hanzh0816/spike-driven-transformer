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


def _add_DATA_NODE():
    """data settings"""
    _C.DATA = CN()
    # image settings (cli, not required)
    _C.DATA.IMG_SIZE = 224
    _C.DATA.CHANNELS = 3

    # (cli & config, required)
    _C.DATA.BATCH_SIZE = 128
    # (cli, required)
    _C.DATA.DATA_PATH = ""
    # (config, required)
    _C.DATA.DATASET = ""


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

    # Checkpoint to resume (cli argument, not required)
    _C.MODEL.RESUME = False
    _C.MODEL.RESUME_PATH = ""
    # Using pretrained model  (config, not required)
    _C.MODEL.PRETRAINED = False
    _C.MODEL.PRETRAINED_PATH = ""


def _add_TRAIN_NODE():
    """training settings"""
    _C.TRAIN = CN()
    _C.TRAIN.TET = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault("BASE", [""]):
        if cfg:
            _update_config_from_file(config, os.path.join(os.path.dirname(cfg_file), cfg))
    print("=> merge config from {}".format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def _update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    


def init_config():
    _add_BASE_NODE()
    _add_DATA_NODE()
    _add_MODEL_NODE()
    _add_TRAIN_NODE()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    init_config()
    config = _C.clone()
    config = _update_config(config, args)
    return config
