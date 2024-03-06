import os
import wandb

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from model import build_model
from config import parse_option
from datasets import build_loader
import utils
from engine import train_one_epoch, evaluate


def main(config, logger):

    _, _, data_loader_train, data_loader_val = build_loader(config)

    # todo: add mixup here

    logger.info(f"Creating model:{config.MODEL.NAME}")
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

    # resume checkpoint
    if config.MODEL.RESUME:
        utils.load_model(config, model_without_ddp, optimizer)

    start_epoch = config.TRAIN.START_EPOCH
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    train_loss_fn = utils.get_loss_fn(config)
    validate_loss_fn = nn.CrossEntropyLoss()

    logger.info("Start training")

    max_accuracy = 0.0

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # 设置sampler epoch
        data_loader_train.sampler.set_epoch(epoch)  # type: ignore
        train_metrics = train_one_epoch(
            config=config,
            logger=logger,
            model=model,
            accum_iter=config.TRAIN.ACCUM_ITER,
            criterion=train_loss_fn,
            data_loader=data_loader_train,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
        )

        if config.DIST_EVAL:
            eval_metrics = evaluate(
                config=config,
                logger=logger,
                model=model,
                criterion=validate_loss_fn,
                data_loader=data_loader_val,
                device=device,
            )

        if utils.is_main_process():
            if not config.DIST_EVAL:
                eval_metrics = evaluate(
                    config=config,
                    logger=logger,
                    model=model,
                    criterion=validate_loss_fn,
                    data_loader=data_loader_val,
                    device=device,
                )
            acc1 = eval_metrics["acc1"]
            if epoch % config.SAVE_FREQ == 0 or (acc1 > max_accuracy):
                unwrapped_model = model.module
                utils.save_model(config, epoch, unwrapped_model, optimizer)

            max_accuracy = max(max_accuracy, acc1)

            print(f"Accuracy of the network on the valid images: {acc1:.1f}%")
            print(f"Max accuracy: {max_accuracy:.2f}%")
            logger.info(f"Accuracy of the network on the valid images: {acc1:.1f}%")
            logger.info(f"Max accuracy: {max_accuracy:.2f}%")

            metric = {**train_metrics, **eval_metrics}
            # todo: 添加wandb.watch 监视模型参数
            wandb.log(metric)

        # update lr
        if lr_scheduler is not None:
            # step LR for next epoch
            lr_scheduler.step(epoch + 1)

    wandb.finish()


if __name__ == "__main__":
    # os.chdir(os.path.dirname(__file__))
    args, config = parse_option()
    config = utils.init_distributed_mode(config)
    device = torch.device("cuda", config.LOCAL_RANK)

    # initiate setting
    logger = utils.set_logger(config=config)
    if utils.is_main_process():
        utils.wandb_init(config)
    utils.init_seed(config)

    cudnn.benchmark = True

    main(config, logger)
