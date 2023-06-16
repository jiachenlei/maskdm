import math
import numpy as np


def cosine_scheduler(base_value, final_value, total_iters, warmup_iters=0,
                     start_warmup_value=0):

    warmup_schedule = np.array([])
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(total_iters - warmup_iters)
    if base_value != final_value:
        schedule = np.array(
            [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])
    else:
        print("No LR Decay after warmup")
        schedule = np.array([base_value for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    # if len(schedule) == 0: return None
    return schedule

def prepare_lr_schedule(base_value, final_value, total_iters, warmup_iters,
                     start_warmup_value=0, schedule="cosine"):

    if schedule == "cosine":
        return cosine_scheduler(base_value, final_value, total_iters, warmup_iters, start_warmup_value)
    else:
        raise NotImplementedError(f"unsupported LR schedule:{schedule}")
    