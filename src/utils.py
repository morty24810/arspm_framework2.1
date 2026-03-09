import random, numpy as np, torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def piecewise_value(t: float, schedule):
    # schedule: list of (t0, t1, value, ...)
    for seg in schedule:
        t0, t1 = seg[0], seg[1]
        if t0 <= t < t1:
            return seg
    return schedule[-1]
