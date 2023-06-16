import os
import wandb
import random
import numpy as np
import argparse
from pathlib import Path
from functools import reduce

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from ema_pytorch import EMA

import torch
from torch.optim import AdamW
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision import utils

from gaussian_ddpm import GaussianDiffusion
from models import MaskedUViT, MaskedDWTUViT

from datasets import VggFace, CelebAHQ, CelebA
from datasets import ( RandomMaskingGenerator, RandomBlockMaskingGenerator,
                      CropMaskingGenerator, DWTMaskingGenerator,)

from utils.config import parse_yml, combine
from utils.lr_schedule import prepare_lr_schedule
from utils.helper import cycle, exists, prepare_state_dict
from train_step import train


def parse_terminal_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="path to the config file")
    parser.add_argument("--overwrite", default="command-line", type=str, help="overwrite config/command-line arguments when conflicts occur")

    parser.add_argument("--name", type=str, default="temp", help="name of experiment")
    parser.add_argument("--seed", type=int, default=1232, help="seed for random number generator")

    parser.add_argument("--save_and_sample_every", type=int, default=10000, help="ckpt and sampling frequency, default is 10k")
    parser.add_argument("--train_steps", type=int, default=2000000, help="training steps, default is 2M")
    parser.add_argument("--num_samples", type=int, default=1, help="number of samples generated every 'save_and_sample_every' steps, default is 1")
    parser.add_argument("--gradient_accumulate_every", type=int, default=1, help="number of gradient accumulation, default is 1")

    parser.add_argument("--pretrained_model_ckpt", type=str, default="", help="path to pretrained weight, default is empty")

    parser.add_argument("--debug", action="store_true", help="if true, no online Wandb logging")
    parser.add_argument("--resume_from", type=int, default=None, help="resume from which step, e.g. if equals N, then `model-N.pt` will be loaded from the experiment result directory")

    parser.add_argument("--wandb_id", type=str, default=None, help="resume from which wandb experiment")
    parser.add_argument("--wandb_project", type=str, default="diffusion-model", help="which wandb project to submit to")

    return parser.parse_args()


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def build_model(args):
    
    name = args.network["name"]
    if name == "maskdm":
        return MaskedUViT(**args.network)
    elif name == "maskdwt":
        return MaskedDWTUViT(**args.network)
    else:
        raise NotImplementedError("only support mask uvit training")


def main():

    mp.set_start_method("spawn")
    accelerator = Accelerator(
        split_batches = True, # if True, then actual batch size equals args.batch_size
        dispatch_batches = False,
        mixed_precision = 'fp16',
        kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    accelerator.native_amp = True

    setup_for_distributed(accelerator.is_main_process)

    args = parse_terminal_args()
    config = parse_yml(args.config)
    if config is not None:
        args = combine(args, config)

    # set seed
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if accelerator.is_main_process and not args.debug:
        # init wandb logging
        if args.wandb_id is not None:
            wandb.init(project=args.wandb_project, config=args, id=args.wandb_id, resume="must")
        else:
            wandb.init(project=args.wandb_project, config=args)

    print(args)

    # prepare model, diffusion model, EMA and optimizer
    model = build_model(args)

    pretrained_model_ckpt = getattr(args, "pretrained_model_ckpt", "")
    if pretrained_model_ckpt != "":
        state_dict = prepare_state_dict(pretrained_model_ckpt)
        missing_key, unexpected_key = model.load_state_dict(state_dict, strict=False)
        print("missing keys: ",missing_key)
        print("unexpected keys: ",unexpected_key)
    else:
        print("No pretrained model ckpt is provided, train from scratch..")

    timesteps = 1000
    beta_schedule = getattr(args, "beta_schedule", "cosine")
    shift = getattr(args, "shift", False)
    loss_weight = getattr(args, "loss_weight", "p2")
    p2_loss_weight_gamma = getattr(args, "p2_loss_weight_gamma", 0)

    print(f"beta schedule:{beta_schedule}, loss_weight:{loss_weight} gamma:{p2_loss_weight_gamma}, shift:{shift}")

    diffusion_model = GaussianDiffusion(
        model,
        image_size = args.network["img_size"],
        timesteps = timesteps,       # number of steps
        sampling_timesteps = 500,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = getattr(args, "loss_type", "l2"),            # L1 or L2

        normalization = getattr(args.dataset, "NORMALIZATION", True), 
        loss_weight = loss_weight,
        p2_loss_weight_gamma = p2_loss_weight_gamma,

        clip_denoised = getattr(args, "clip_denoised", True), 
        clip_max = getattr(args, "clip_max", 1), 
        clip_min = getattr(args, "clip_min", -1), 

        channels = args.network.get("in_chans", 3),
        beta_schedule = beta_schedule,
        shift = shift,
    )

    n_parameters = sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad)
    print(f"Total Parameters: {n_parameters}")

    # two variables: ema, results_folder, will not be used in subprocesses
    ema = None
    results_folder = Path(os.path.join(args.results_folder, args.name))
    results_folder.mkdir(exist_ok = True)
    if accelerator.is_main_process:
        ema = EMA(diffusion_model, beta = args.ema_decay, update_every = args.ema_update_every)

    opt = AdamW(diffusion_model.parameters(), 
                lr = args.lr, betas = args.adam_betas,
                weight_decay=args.weight_decay)

    model, opt = accelerator.prepare(diffusion_model, opt)
    start_step = 0 # by default, start from first iteration. else start from iteration given by "resume_from"
    if args.resume_from:
        print(f"Resume training from checkpoint: model-{args.resume_from}.pt")
        start_step, model, ema, opt = load_training_state(model, ema, accelerator, opt, args.resume_from, results_folder=results_folder)

    # prepare dataset

    window_size = args.network["img_size"] // args.network["patch_size"]
    mask_ratio = getattr(args.dataset, "MASK_RATIO", 0.9)
    mask_type = getattr(args.dataset, "MASK_TYPE", "random")
    mask_crop_size = getattr(args.dataset, "MASK_CROP_SIZE", None)
    mask_block_size  = getattr(args.dataset, "MASK_BLOCK_SIZE", None)

    use_dwt = getattr(args.dataset, "USE_DWT", False)
    dwt_level = args.network.get("level", 1)
    print(f"Mask type:{mask_type}  Mask ratio:{mask_ratio} Mask crop size:{mask_crop_size}")

    if use_dwt:
        # NOTE: mask_ratio should be a list
        mask_generator = DWTMaskingGenerator(
            window_size, dwt_level, *mask_ratio, 
            scale = args.network["scale"],
            mask_type=mask_type, block_size=mask_block_size,
        )

    elif mask_type == "patch":
        mask_generator = RandomMaskingGenerator(
            window_size, mask_ratio
        )
    elif mask_type == "block":
        mask_generator = RandomBlockMaskingGenerator(
            window_size, mask_ratio, mask_block_size,
        )
    elif mask_type == "crop":
        mask_generator = CropMaskingGenerator(
            window_size, mask_crop_size
        )
    else:
        raise NotImplementedError(f"Unsupported mask type:{mask_type}")

    print(mask_generator)

    available_datasets= {
        "vggface": VggFace,
        "celebahq": CelebAHQ,
        "celeba": CelebA,
    }
    dataset = available_datasets[args.dataset.NAME](
        cfg = args.dataset, mode="train", mask_generator=mask_generator, verbose = accelerator.is_main_process
    )
    loader = DataLoader(dataset, batch_size = args.batch_size, 
                        shuffle = True, pin_memory = True,
                        num_workers = getattr(args, "num_workers", 16))
    loader = accelerator.prepare(loader)
    loader = cycle(loader)

    lr_schedule_value = prepare_lr_schedule(
                                base_value = args.lr, 
                                final_value = args.lr, 
                                total_iters = args.train_steps,
                                warmup_iters = getattr(args, "warmup_steps", 0),
                                start_warmup_value = getattr(args, "warmup_lr", None),
                                schedule="cosine"
                            )

    print(f"Start training from step:{start_step}")
    print(f"Using gradient accumulation: {args.gradient_accumulate_every}")

    train(
        diffusion_model,
        accelerator = accelerator,
        loader = loader,

        opt = opt,
        ema = ema,
        start_step = start_step,
        train_num_steps = args.train_steps,         # total training steps
        gradient_accumulate_every = args.gradient_accumulate_every,    # gradient accumulation steps
        save_and_sample_every = args.save_and_sample_every,
        num_samples = args.num_samples,
        batch_size=args.batch_size,
        results_folder = results_folder,
        clip_grad = getattr(args, "clip_grad", 1.0),

        lr_schedule_value = lr_schedule_value, # lr scheduler
    )


def load_training_state(model, ema, accelerator, opt, milestone, results_folder="./results"):

    device = accelerator.device
    data = torch.load(os.path.join(results_folder, f'model-{milestone}.pt'), map_location=device)
    model = accelerator.unwrap_model(model)
    model.load_state_dict(data['model'])
    step = data['step']
    opt.load_state_dict(data['opt'])

    if ema is not None:
        ema.load_state_dict(data['ema'])
    if exists(accelerator.scaler) and exists(data['scaler']):
        accelerator.scaler.load_state_dict(data['scaler'])

    return step, model, ema, opt


if __name__ == "__main__":
    main()