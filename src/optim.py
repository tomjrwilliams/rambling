
import numpy
import pandas

import torch
import torch.optim

import tqdm.notebook

from . import graphs

def fit_parameters(
    params,
    loss_function,
    epochs = 1,
    iters = 100,
):
    num_iter = epochs * iters

    losses = []

    optimizer = torch.optim.SGD(list(params.values()), 0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.9
    )

    iterator = tqdm.tqdm(range(num_iter))
    for i in iterator:
        optimizer.zero_grad()

        loss = loss_function(*params.values())

        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()

        iterator.set_postfix(iter = i, loss=loss.item())

        if i > 0 and i % iters == 0:
            scheduler.step()

    return losses, params
