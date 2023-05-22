
import torch

from . import tensors


def smooth_ts_rolling_mean(ts, ns = [], dropout = 0.):
    dropout = torch.nn.Dropout(p=dropout)
    ts_rolling = torch.stack([
        tensors.rolling_windows(ts, n)
        for n in ns
    ])
    ts_rolling_mean = ts_rolling.mean(dim=-1)
    return dropout(torch.sub(
        ts_rolling_mean, ts
    )).square().mean()

def smooth_ts_rolling_coastline(ts, ns = [], dropout = 0.):
    dropout = torch.nn.Dropout(p=dropout)
    ts_deltas = tensors.calc_deltas(ts)
    ts_rolling = torch.stack([
        tensors.pad(
            tensors.rolling_windows(ts_deltas, n),
            max(ns) - n,
            value = 0
        )
        for n in ns
    ])
    return dropout(
        torch.sub(
            ts_rolling.abs().sum(dim=-1),
            ts_rolling.sum(dim=-1).abs(),
        )
    ).mean()

def smooth_ts_2nd_derivative(ts, dropout = 0.):
    dropout = torch.nn.Dropout(p=dropout)
    ts_deltas = tensors.calc_deltas(ts)
    ts_deltas_deltas = tensors.calc_deltas(ts_deltas)
    return dropout(ts_deltas_deltas).square().mean()


def smooth_ts_rolling_mean_corr(ts, ns = []):
    ts_deltas = tensors.calc_deltas(ts)
    ts_rolling_mean_corr = torch.stack([
        torch.cov(torch.stack([
            ts_deltas,
            tensors.pad(
                tensors.rolling_windows(ts_deltas, n).mean(dim=-1),
                n = 1,
                value= 0
            )[:-1]
        ]))[0][1] / ts_deltas.var()
        for n in ns
    ])
    ts_corr_pairs = (
        ts_rolling_mean_corr.abs()
        .unfold(0, 2, 1)
        .permute(1, 0)
    )
    return torch.sub(
        ts_corr_pairs[1],
        ts_corr_pairs[0],
    ).sum()

# def smooth_ts_brownian_variance(ts, ns = [], dropout = 0.):
#     dropout = torch.nn.Dropout(p=dropout)
#     ts_deltas = tensors.calc_deltas(ts)
#     ts_rolling_var = torch.stack([
#         tensors.rolling_windows(ts_deltas, n).var(dim=-1)
#         for n in ns
#     ])
#     brownian_var = torch.Tensor(ns)
#     return dropout(
#         torch.sub(ts_rolling_var.T, brownian_var)
#     ).square().mean()


# def smooth_ts_rolling_auto_corr(ts, ns = []):
#     ts_deltas = tensors.calc_deltas(ts)
#     ts_deltas_offset = ts_deltas.unfold(0, len(ts_deltas) - 1, 1)
#     # 2, n_windows, len_window
#     ts_rolling_auto_corr = torch.stack([
#         torch.stack([
#             torch.corrcoef(w)[0][1]
#             for w in ts_deltas_offset.unfold(1, n, 1).permute(1, 0, 2)
#         ])[-(len(ts) - max(ns)):]
#         for n in ns
#     ]).abs()
#     return torch.stack([
#         torch.sub(
#             ts_rolling_auto_corr[i + 1],
#             ts_rolling_auto_corr[i],
#         ).sum()
#         for i in range(len(ns) - 1)
#     ]).sum()