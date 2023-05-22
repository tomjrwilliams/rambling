
import torch

def pad(t, n, left = True, value = 0):
    if left:
        padding = (n, 0)
    else:
        padding = (0, n)
    # if also 2nd to last: (last_l, last_r, 2nd_l, 2nd_r), etc.
    return torch.nn.functional.pad(
        t, padding, "constant", value
    )

def rolling_windows(t, n):
    try:
        t_pad = pad(t, n - 1, value = 0)
        return t_pad.unfold(0, n, 1)
    except:
        assert False, t_pad

def calc_deltas(t):
    return torch.cat([
        torch.Tensor([0.]),
        torch.sub(t[1:], t[:-1])
    ])

def gaussian_walk(shape, mu, std):
    return torch.distributions.Normal(
        mu, std
    ).sample(shape).cumsum(dim=-1)