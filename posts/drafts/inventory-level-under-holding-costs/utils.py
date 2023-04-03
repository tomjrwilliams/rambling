import numpy
import torch

import src.graphs
import src.optim

def holding_cost_constant(v):
    def f(q):
        return v * q
    return f

def profit_constant(v):
    def f(q):
        return v * q
    return f

def demand_gaussian(mu, sigma):
    dist = torch.distributions.Normal(mu, sigma)
    return dist.sample

def step_model(
    f_holding_cost,
    f_profit,
    f_demand,
    inventory,
):
    holding_cost = f_holding_cost(inventory)
    demand = f_demand().clamp(min = 0)
    shortfall = (demand - inventory).clamp(min=0)
    sales = demand - shortfall
    profit = f_profit(sales) - holding_cost
    return dict(
        inventory=inventory,
        holding_cost=holding_cost,
        demand=demand,
        shortfall=shortfall,
        sales=sales,
        profit=profit,
    )

def fit_optimal_inventory(
    f_holding_cost,
    f_profit,
    f_demand,
):
    inventory_optimal = torch.nn.Parameter(
        torch.Tensor([1.]), requires_grad = True
    )

    def f_converge(losses, *params, n_samples = 100):
        if len(losses) < n_samples:
            return False
        rng = (numpy.max(losses[-n_samples:]) - numpy.min(losses[-n_samples:]))
        mu = abs(numpy.mean(losses[-n_samples:]))
        return (rng / mu) < 0.001

    def f_loss(inventory):
        state = step_model(
            f_holding_cost,
            f_profit,
            f_demand,
            inventory,
        )
        return -1 * state["profit"]

    return src.optim.fit_parameters(
        {
            "inventory_optimal": inventory_optimal,
        },
        f_loss,
        epochs = 10,
        iters = 100,
        f_converge = f_converge,
    )

def f_sample_graph(samples):

    inventory = samples[0]["inventory"].item()
    kws = {
        "demand": dict(hlines=[0, inventory]),
        "shortfall": dict(hlines=[0]),
        "sales": dict(hlines=[0, inventory]),
        "profit": dict(hlines=[0]),
    }

    def sample_graph(k, f_filter):
        filtered_samples = [s for s in samples if f_filter(s)]
        return dict(
            f=src.graphs.scatter_graph,
            args = (
                list(range(len(filtered_samples))),
                [s[k].item() for s in filtered_samples],
            ),
            kwargs = dict(label = k, s= 0.3, **kws[k]),
        )

    return sample_graph