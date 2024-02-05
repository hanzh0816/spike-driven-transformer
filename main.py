import os
import time
import datetime
from tqdm import tqdm
import torch.nn.functional as F
from spikingjelly.clock_driven import functional

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


def main(accelerator: Accelerator, args, config, logger):
    _, _, data_loader_train, data_loader_val = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.NAME}")
    model = build_model(config)

    # use timm.optim to create specific optimizer
    optimizer = create_optimizer(args=args, model=model)

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    # prepare
    data_loader_train, model, optimizer = accelerator.prepare(data_loader_train, model, optimizer)

    # resume checkpoint
    if config.MODEL.RESUME:
        # todo: 添加resume checkpoint
        pass

    criterion = nn.CrossEntropyLoss()
    logger.info("Start training")
    start_time = time.time()

    max_accuracy = 0.0

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # 设置sampler epoch
        train_one_epoch(
            accelerator=accelerator,
            config=config,
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            epoch=epoch,
        )

        if accelerator.is_main_process:
            acc1 = validate(config=config, data_loader=data_loader_val, model=model)

            if epoch % config.SAVE_FREQ == 0 and (acc1 > max_accuracy):
                unwrapped_model = accelerator.unwrap_model(model)
                # todo implement save checkpoints
                # save_model(unwrapped_model, config)

            print(f"Accuracy of the network on the valid images: {acc1:.1f}%")
            logger.info(f"Accuracy of the network on the valid images: {acc1:.1f}%")
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f"Max accuracy: {max_accuracy:.2f}%")

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            logger.info("Training time {}".format(total_time_str))


def train_one_epoch(accelerator, config, model, criterion, data_loader, optimizer, epoch):
    model.train()
    optimizer.zero_grad()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    num_steps = len(data_loader)
    start = time.time()
    end = time.time()

    t = tqdm(data_loader, disable=not accelerator.is_main_process)
    t.set_description("Processing:")

    for idx, (inputs, labels) in enumerate(t):

        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels.long())
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()

        functional.reset_net(model)

        loss_meter.update(loss.item(), labels.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if accelerator.is_main_process and (idx % config.PRINT_FREQ == 0):
            lr = optimizer.param_groups[0]["lr"]
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )

    if accelerator.is_main_process:
        epoch_time = time.time() - start
        logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, model, data_loader):
    model.eval()

    batch_time = AverageMeter()
    acc1_meter = AverageMeter()

    end = time.time()
    for idx, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)

        labels = labels.to(device)

        outputs = model(inputs)
        functional.reset_net(model)

        [acc1] = accuracy(outputs, labels)
        acc1_meter.update(acc1.item(), labels.size(0))
        # acc5_meter.update(acc5.item(), target.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
                f"Mem {memory_used:.0f}MB"
            )
    logger.info(f" * Acc@1 {acc1_meter.avg:.3f}")
    return acc1_meter.avg


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    accelerator = Accelerator()
    device = accelerator.device

    args, config = parse_option()
    logger = set_logger(config=config)
    init_seed(config)

    main(accelerator, args, config, logger)
