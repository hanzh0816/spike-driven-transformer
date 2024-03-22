# save & load checkpoints
import os
import torch


def save_model(config, save_name, model_without_ddp, optimizer):
    epoch_name = str(save_name)
    output = os.path.join(config.OUTPUT, ("checkpoint-%s.pth" % epoch_name))
    to_save = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": save_name,
    }

    torch.save(to_save, output)
    print(f"Saved checkpoint-{epoch_name}.pth")


def load_model(config, model_without_ddp, optimizer):
    checkpoint = torch.load(config.MODEL.RESUME, map_location=torch.device("cpu"))
    model_without_ddp.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint and "epoch" in checkpoint:
        config.defrost()
        optimizer.load_state_dict(checkpoint["optimizer"])
        config.TRAIN.START_EPOCH = int(checkpoint["epoch"]) + 1
        config.freeze()
    print(f"Resume checkpoint {config.MODEL.RESUME}")
