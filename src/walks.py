
import functools
import torch

from . import optim
from . import graphs

def plot_constrained_walks(
    n, 
    f_walk, 
    f_loss,
    iters = 500, 
    epochs = 5, 
    **kws
):

    gs = []

    for i in range(n):

        path = f_walk()

        losses, res = optim.fit_parameters(
            dict(
                path=torch.nn.Parameter(
                    torch.Tensor(path.tolist())
                ),
            ),
            functools.partial(f_loss, **kws),
            iters=iters,
            epochs=epochs,
            with_tqdm=False,
            # progress_bar=False,
        )
        res = res["path"]
        gs.append([
            dict(
                f=graphs.line_graph,
                kwargs = dict(
                    xs=list(range(len(losses))),
                    ys=losses,
                    label="loss",
                ),
            ),
            dict(
                f=graphs.line_graph,
                kwargs=dict(
                    xs=list(range(len(path))),
                    ys=path.tolist(),
                    label="path",
                ),
            ),
            dict(
                f=graphs.line_graph,
                kwargs=dict(
                    xs=list(range(len(res))),
                    ys=res.tolist(),
                    label="res",
                )
            )
        ])

    graphs.render_graphs(gs)