
import math

import numpy
import pandas

import torch

from . import graphs

def softplus(x, k):
    return math.log(1 + math.exp(k * x))

def softplus_torch(x, k):
    return torch.log(1 + torch.exp(k * x))

def exponential(x, offset):
    return 1 + math.exp(-1 * (x - offset))

def exponential_torch(x, offset):
    offset_expand = offset.expand(len(x), 1)[:, 0]
    return 1 + torch.exp(-1 * (x - offset_expand))

# domain 0 1
def sigmoid(x, offset):
    return 1 / (
        1 + math.exp(-1 * (x - offset))
    )

def sigmoid_torch(x, offset):
    offset_expand = offset.expand(len(x), 1)[:, 0]
    return 1 / (
        1 + torch.exp(-1 * (x - offset_expand))
    )

def sigmoid_biunit(x, offset):
    return (2 / (
        1 + math.exp(-1 * (x - offset))
    )) - 1

def sigmoid_biunit_torch(x, offset):
    offset_expand = offset.expand(len(x), 1)[:, 0]
    return (2 / (
        1 + torch.exp(-1 * (x - offset_expand))
    )) - 1

# domain 0, 1
def cubic_sigmoid(x, k):
    return 1 / (
        1 + math.exp((x ** 3) * -1 * k)
    )

def cubic_sigmoid_torch(x, k):
    # k_expand = k.expand(len(x), 1)[:, 0]
    return 1 / (
        1 + torch.exp(torch.pow(x, 3) * -1 * k)
    )

def cubic_sigmoid_biunit(x, k):
    return (2 / (
        1 + math.exp((x ** 3) * -1 * k)
    )) - 1

def cubic_sigmoid_biunit_torch(x, k):
    # k_expand = k.expand(len(x), 1)[:, 0]
    return (2 / (
        1 + torch.exp(torch.pow(x, 3) * -1 * k)
    )) - 1


def hump(x, offset, k):
    return 1 - (2 / (1 + math.exp(((x - offset) ** 2) * k)))

def hump_torch(x, offset, k):
    offset_expand = offset.expand(len(x), 1)[:, 0]
    # k_expand = k.expand(len(x), 1)[:, 0]
    return 1 - (2 / (1 + torch.exp(((x - offset) ** 2) * k)))

def hump_pair(x, offset, k):
    return x * math.exp(k * -1 * ((x - offset) ** 2))

def hump_pair_torch(x, offset, k):
    offset_expand = offset.expand(len(x), 1)[:, 0]
    return x * torch.exp(k * -1 * ((x - offset_expand) ** 2))

def hump_pair_root(offset, k):
    return (offset * k - ((
        k * (((offset ** 2) * k) + 2)
    ) ** (1/2))) / ( 2 * k)

def hump_pair_max_abs(offset, k):
    root = hump_pair_root(offset, k)
    return abs(hump_pair(root, offset, k))

def hump_pair_max_abs_torch(offset, k):
    root = hump_pair_root(offset, k)
    return torch.abs(hump_pair(root, offset, k))

def hump_pair_biunit(x, offset, k):
    scale = hump_pair_max_abs(offset, k)
    v = hump_pair(x, offset, k)
    return v / scale

def hump_pair_biunit_torch(x, offset, k):
    scale = hump_pair_max_abs_torch(offset, k)
    v = hump_pair_torch(x, offset, k)
    return v / scale

def sigmoid_hump_pair(x, offset1, offset2, k):
    v = sigmoid_biunit(x, offset1)
    return hump_pair_biunit(v, offset2, k)

def sigmoid_hump_pair_torch(x, offset1, offset2, k):
    v = sigmoid_biunit_torch(x, offset1)
    return hump_pair_biunit_torch(v, offset2, k)

def graph_transform(
    f,
    x,
    kwargs,
    labels,
):
    if isinstance(labels, str):
        labels = [
            "{}-{}".format(labels, kws)
            for kws in kwargs
        ]
    gs = [
        dict(
            f=graphs.line_graph,
            args = (
                x,
                [f(_x, **kws) for _x in x]
            ),
            kwargs=dict(label = label)
        )
        for kws, label in zip(kwargs, labels)
    ]
    graphs.render_graphs(gs)