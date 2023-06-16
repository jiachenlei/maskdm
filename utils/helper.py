import torch

from einops import rearrange
import math

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def maybe_normalize_to_neg_one_to_one(img, enable=True):
    return img * 2 - 1 if enable else img

def maybe_unnormalize_to_zero_to_one(t, enable=True):
    return (t + 1) * 0.5 if enable else t

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def prepare_state_dict(pretrained_model_ckpt):

    print(f"Loading pretrained weights: {pretrained_model_ckpt}")
    raw_state_dict = torch.load(pretrained_model_ckpt, map_location="cpu")["model"]
    state_dict = {}
    for k,v in raw_state_dict.items():
        if k.startswith("model"):
            state_dict[k[6:]] = v

    return state_dict