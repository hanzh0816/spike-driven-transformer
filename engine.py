import os
import time
import datetime
from tqdm import tqdm

import torch
from spikingjelly.clock_driven import functional

import utils
from utils.train_utils import is_main_process


def train_one_epoch(
    config, logger, model, accum_iter, criterion, data_loader, optimizer, epoch, device
):
    model.train()
    optimizer.zero_grad()

    train_meter = utils.MetricMeter()
    num_steps = len(data_loader)
    start = time.time()

    t = tqdm(data_loader, disable=not utils.is_main_process())
    t.set_description("Processing:")

    for idx, (inputs, labels) in enumerate(t):

        inputs = inputs.to(device)
        labels = labels.to(device)
        batch_size = inputs.shape[0]
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss_value = loss.item()
        loss = loss / accum_iter
        loss.backward()

        if (idx + 1) % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()
        functional.reset_net(model)

        loss_value = utils.all_reduce_mean(loss_value)
        train_meter.update(train_loss=loss_value)
        train_meter.update(batch_time=time.time() - start)

        start = time.time()
        if utils.is_main_process() and (idx % config.PRINT_FREQ == 0):
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            train_meter.synchronize_between_processes()
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                # f"time {train_meter.batch_time.value:.4f} ({train_meter.batch_time.avg:.4f})\t"
                f"loss {train_meter.train_loss.value:.4f} ({train_meter.train_loss.avg:.4f})\t"
                f"mem {memory_used:.0f}MB -- batch size {batch_size}"
            )

    optimizer.step()
    optimizer.zero_grad()
    lr = optimizer.param_groups[0]["lr"]
    train_meter.synchronize_between_processes()
    train_metrics = {"train_loss": train_meter.train_loss.global_avg, "lr": lr}
    return train_metrics


def evaluate(config, logger, model, criterion, data_loader, device):
    model.eval()

    eval_meter = utils.MetricMeter()

    for idx, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        acc1, acc5 = utils.accuracy(outputs, labels, topk=(1, 5))

        functional.reset_net(model)
        batch_size = inputs.shape[0]
        eval_meter.update(val_loss=loss.item())
        eval_meter.meters["acc1"].update(acc1.item(), n=batch_size)
        eval_meter.meters["acc5"].update(acc5.item(), n=batch_size)

        if utils.is_main_process() and (idx % config.PRINT_FREQ == 0):
            logger.info(
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Acc@1 {eval_meter.acc1.value:.3f} ({eval_meter.acc1.avg:.3f})\t"
                f"Loss {eval_meter.val_loss.value:.3f} ({eval_meter.val_loss.avg:.3f})\t"
            )

    eval_meter.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in eval_meter.meters.items()}
