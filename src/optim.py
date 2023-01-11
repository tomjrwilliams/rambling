
import numpy
import pandas

import torch
import torch.optim

import tqdm.notebook

from . import graphs

def fit_parameters(
    params,
    f_loss,
    epochs = 1,
    iters = 100,
    f_converge = None,
    f_optimiser = lambda params: torch.optim.SGD(
        list(params.values()), 0.1,
    ),
    f_scheduler = lambda opt: torch.optim.lr_scheduler.ExponentialLR(
        opt, gamma=0.9
    ),
    progress_bar = False,
):
    num_iter = epochs * iters

    losses = []

    optimiser = f_optimiser(params)
    scheduler = f_scheduler(optimiser)

    iterator = tqdm.tqdm(range(num_iter), disable=not progress_bar)

    for i in iterator:
        optimiser.zero_grad()

        loss = f_loss(*params.values())
        losses.append(loss.item())
        
        loss.backward()
        optimiser.step()

        iterator.set_postfix(iter = i, loss=loss.item())

        if i > 0 and i % iters == 0:
            scheduler.step()

        if f_converge is not None and f_converge(losses, params):
            break

    if not progress_bar:
        iterator = tqdm.tqdm(range(num_iter))
        iterator.set_postfix(iter = i, loss=losses[-1], refresh = False)
        iterator.update(i)

    return losses, params
