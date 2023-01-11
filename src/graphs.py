
import matplotlib.pyplot
import matplotlib.gridspec
import seaborn

import numpy
import pandas
import torch

def add_hlines(ax, xs, ys, colors = "grey", linestyles = "--"):
    xs_not_none = [x for x in xs if x is not None]
    for y in ys:
        ax.hlines(
            y = y, 
            xmin = min([min(xs_not_none), 0]),
            xmax = max([max(xs_not_none), 0]),
            colors=colors, 
            linestyles=linestyles,
            #
        )

def add_vlines(ax, ys, xs, colors = "grey", linestyles = "--"):
    ys_not_none = [y for y in ys if y is not None]
    for x in xs:
        ax.vlines(
            x = x, 
            ymin = min([min(ys_not_none), 0]),
            ymax = max([max(ys_not_none), 0]), 
            colors=colors, 
            linestyles=linestyles,
            #
        )

def heatmap_graph(ax, *args, **kwargs):
    seaborn.heatmap(*args, ax = ax, **kwargs)

def bar_graph(ax, xs, ys, *args, hlines = [], vlines = [], **kwargs):
    ax.bar(xs, ys, *args, **kwargs)
    if len(hlines):
        add_hlines(ax, xs, hlines)
    if len(vlines):
        add_vlines(ax, ys, vlines)

def hbar_graph(ax, xs, ys, *args, hlines = [], vlines = [], **kwargs):
    ax.bar(xs, ys, *args, **kwargs)
    if len(hlines):
        add_hlines(ax, xs, hlines)
    if len(vlines):
        add_vlines(ax, ys, vlines)

def line_graph(ax, xs, ys, *args, hlines = [], vlines = [], **kwargs):
    ax.plot(xs, ys, *args, **kwargs)
    if len(hlines):
        add_hlines(ax, xs, hlines)
    if len(vlines):
        add_vlines(ax, ys, vlines)

def scatter_graph(ax, xs, ys, *args, hlines = [], vlines = [], **kwargs):
    ax.scatter(xs, ys, **kwargs)
    if len(hlines):
        add_hlines(ax, xs, hlines)
    if len(vlines):
        add_vlines(ax, ys, vlines)

def render_graphs(graphs, height = 1.5, width = 8, with_legend = True):
    if isinstance(graphs[0], dict):
        return render_graph_rows(
            graphs, 
            height=height, 
            width=width, 
            with_legend=with_legend,
        )
    else:
        return render_graph_grid(
            graphs, 
            height=height, 
            width=width, 
            with_legend=with_legend,
        )

def render_graph_grid(graphs, height = 1.5, width = 8, with_legend = True):
    
    nrows = len(graphs)
    ncols = max([len(row) for row in graphs])

    fig = matplotlib.pyplot.figure(
        figsize=(width, nrows * height),
        tight_layout=True,
    )
    grid = matplotlib.gridspec.GridSpec(
        nrows, 
        ncols,
    )
    for r, row in enumerate(graphs):
        row_cols_ratio = len(row) / ncols

        if len(row) == 1:
            ax = fig.add_subplot(grid[r, :])
            render_graph(row[0], ax, with_legend=with_legend)

        elif int(row_cols_ratio) == row_cols_ratio:
            row_cols_ratio = int(row_cols_ratio)
            for g, graph in enumerate(row):
                ax = fig.add_subplot(grid[
                    r, 
                    int(g * row_cols_ratio):int((g+1) * row_cols_ratio)
                ])
                render_graph(graph, ax, with_legend=with_legend)
        else:
            for g, graph in enumerate(row):
                ax = fig.add_subplot(grid[r, g:(g+1)])
                render_graph(graph, ax, with_legend=with_legend)

    return

def render_graph_rows(graphs, height = 1.5, width = 8, with_legend = True):

    nrows = len(graphs)

    fig, axs = matplotlib.pyplot.subplots(
        nrows=nrows, 
        ncols=1,
        figsize=(width, nrows * height),
        tight_layout=True,
    )
    if nrows == 1:
        axs = [axs]
    
    for graph, ax in zip(graphs, axs):
        render_graph(graph, ax, with_legend=with_legend)

    return

def render_graph(
    graph,
    ax,
    with_legend = True,
):
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
