import os
import logging
import logging.handlers
import functools
from .train_utils import is_main_process

# from logging.handlers import RotatingFileHandler


def set_logger(config):
    # 日志设置
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(
        output_dir=config.OUTPUT, default_level=config.LOG_LEVEL, name=f"{config.MODEL.NAME}"
    )

    # print config
    if is_main_process():
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    return logger


@functools.lru_cache()
def create_logger(output_dir, default_level=logging.INFO, name=""):
    logger = logging.getLogger(name)
    logger.setLevel(default_level)
    logger.propagate = False
    log_handler = logging.handlers.RotatingFileHandler(
        os.path.join(output_dir, "log.txt"), maxBytes=1024 * 1024, backupCount=5, encoding="utf-8", mode="a"
    )

    # create formatter
    fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): [%(levelname)s] - %(message)s"

    # create handler
    log_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    log_handler.setLevel(default_level)
    logger.addHandler(log_handler)

    return logger
