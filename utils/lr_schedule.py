import math
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

def prepare_lr_schedule(optimizer, warmup_steps):

    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1

    return LambdaLR(optimizer, fn)
    