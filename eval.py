import os
import argparse
import random
import numpy as np


from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.utils import save_image

from gaussian_ddpm import GaussianDiffusion

from models import MaskedUViT

from utils.config import parse_yml, combine
from utils.helper import maybe_unnormalize_to_zero_to_one


def parse_terminal_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="path to config file")
    parser.add_argument("--overwrite", default="command-line", type=str, help="overwrite config/command-line arguments when conflicts occur")
    
    parser.add_argument("--bs", type=int, help="batch size used in evaluation")
    parser.add_argument("--total_samples", type=int, default=3000, help="samples to generate")
    parser.add_argument("--sampling_steps", type=int, default=250, help="DDIM sampling steps")
    parser.add_argument("--ddim_sampling_eta", type=float, default=1.0, help="DDIM sampling eta coefficient")

    parser.add_argument("--output", nargs="+", help="list of output path to save images")
    parser.add_argument("--ckpt", nargs="+",  help="list of path to model checkpoint")

    return parser.parse_args()


class Sampler(torch.nn.Module):

    def __init__(self, model):

        super().__init__()

        self.model = model

    def forward(self, *args, **kwargs):

        return self.model.sample(
            *args,
            **kwargs
        )


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
    else:
        raise NotImplementedError(f"Unsupported network type: {name}")


def save_img_onebyone(images_tensor, output_path, name):
    for i in range(images_tensor.shape[0]):    
        save_image(images_tensor[i], os.path.join(output_path, name+f"_{i}.png"))


@torch.no_grad()
def evaluation():

    mp.set_start_method("spawn")
    accelerator = Accelerator(
        split_batches = True, # if True, then actual batch size equals args.batch_size
        mixed_precision = 'fp16',
        kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    accelerator.native_amp = True

    setup_for_distributed(accelerator.is_main_process)

    # parse terimanl/config file arguments
    args = parse_terminal_args()
    config = parse_yml(args.config)
    if config is not None:
        args = combine(args, config)

    device = accelerator.device
    timesteps = 1000    # trained model time steps
    evalutation_batch_size = args.bs # size:bs 224:14
    subprocess = str(accelerator.process_index)
    # prepare model
    model = build_model(args)

    print(f"normalizataion: {getattr(args.dataset, 'NORMALIZATION', True)}")
    diffusion_model = GaussianDiffusion(
        model,
        image_size = args.network["img_size"],
        timesteps = timesteps,           # number of steps
        sampling_timesteps = getattr(args, "sampling_steps", 250),   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        ddim_sampling_eta = getattr(args, "ddim_sampling_eta", 1), # eta coefficient of DDIM sampling

        clip_denoised = getattr(args, "clip_denoised", True), 
        clip_max = getattr(args, "clip_max", 1), 
        clip_min = getattr(args, "clip_min", -1), 

        normalization = getattr(args.dataset, "NORMALIZATION", True), 

        loss_type = 'l2',            # L1 or L2
        channels = args.network.get("in_chans", 3),
    )
    diffusion_model.to(device)
    diffusion_model.eval()
    dm_sampler = Sampler(diffusion_model)
    # speedup sampling with mixed precision
    dm_sampler = accelerator.prepare(dm_sampler)

    # load model weights sequentially
    for i, ckpt in enumerate(args.ckpt):
        total_samples = args.total_samples
        exist_samples = len(os.listdir(args.output[i]))
        total_samples -= exist_samples
        num_processes = accelerator.num_processes
        while total_samples % num_processes != 0:
            total_samples += 1

        print(f"Total samples: {total_samples}")
        current_samples = total_samples // num_processes // evalutation_batch_size
        last = (total_samples// num_processes) % evalutation_batch_size

        print(f"Loading pretrained EMA weights: {ckpt}")
        if ckpt != "":
            raw_state_dict = torch.load(ckpt, map_location="cpu")["ema"]
            state_dict = {}
            for k,v in raw_state_dict.items():
                # in checkpoint[ema], there are two sets of paramters that start with:
                # oneline_model.
                # ema_model.
                if k.startswith("ema_model"):
                    k = k[10:] # remove prefix: ema_model.
                    state_dict[k] = v

            missing_key, unexpected_key = accelerator.unwrap_model(dm_sampler).model.load_state_dict(state_dict, strict=False)
            print("missing keys: ",missing_key)
            print("unexpected keys: ",unexpected_key)
        else:
            print("empty checkpoint")

        output_path = args.output[i]
        batch_size_lst = [evalutation_batch_size for i in range(current_samples)]
        if last:
            batch_size_lst += [last]

        # start sampling
        for j, bs in enumerate(batch_size_lst):
            samples = diffusion_model.sample(batch_size=bs)
            save_img_onebyone(samples, output_path, f"sample_{subprocess}_{random.random()}")

        accelerator.wait_for_everyone()

if __name__ == "__main__":
    evaluation()