
import matplotlib.pyplot
import seaborn

import numpy
import pandas
import torch

def add_hzero(ax, xs):
    xs_not_none = [x for x in xs if x is not None]
    ax.hlines(
        y = 0, 
        xmin = xs_not_none[0], 
        xmax = xs_not_none[-1], 
        colors = "grey", 
        linestyles = "--",
        #
    )

def heatmap_graph(ax, *args, **kwargs):
    seaborn.heatmap(*args, ax = ax, **kwargs)

def bar_graph(ax, xs, ys, *args, with_hzero = True, **kwargs):
    ax.bar(xs, ys, *args, **kwargs)
    if with_hzero:
        add_hzero(ax, xs)

def hbar_graph(ax, xs, ys, *args, with_hzero = True, **kwargs):
    ax.bar(xs, ys, *args, **kwargs)
    if with_hzero:
        add_hzero(ax, xs)

def line_graph(ax, xs, ys, *args, with_hzero = True, **kwargs):
    ax.plot(xs, ys, *args, **kwargs)
    if with_hzero:
        add_hzero(ax, xs)

def scatter_graph(ax, xs, ys, *args, with_hzero = True, **kwargs):
    ax.scatter(xs, ys, **kwargs)
    if with_hzero:
        add_hzero(ax, xs)

def render_graphs(graphs, height = 1.5, width = 15, with_legend = True):
    
    nrows = len(graphs)
    f, axs = matplotlib.pyplot.subplots(
        nrows=nrows, 
        ncols=1,
        figsize=(width, nrows * height),
        tight_layout=True,
    )
    if nrows == 1:
        axs = [axs]
    
    for graph, ax in zip(graphs, axs):
        graph["f"](
            ax,
            *graph.get("args", tuple([])), 
            **graph.get("kwargs", {}),
            # 
        )

        if with_legend:
        
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
    return

def auto_correlation_plot(xs, n):
    corr = {
        i: numpy.correlate(xs[:-i], xs[i:])[0]
        for i in range(1, n)
    }
    return dict(
        f = bar_graph,
        args = (
            list(corr.keys()),
            list(corr.values()),
        ),
        kwargs = dict(
            label = "auto correlation"
        )
    )
