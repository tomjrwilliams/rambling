def in_universe(ticker, df, threshold):
    if ticker not in df.columns:
        return False
    return (
        df[ticker].sum() / len(df.index)
    ) >= threshold

# pca, _ = fit_pca(d_start, d_end, n)
# dfs = dfs_indices
# dfs = dfs_sectors
# universe = bt.algos.universe.configs.INDICES
# universe = bt.algos.universe.configs.GICS_SECTORS
def f_ticker_map(dfs, f_model, f_tickers, threshold = 1.):
    def f(*args, **kwargs):
        model, data = f_model(*args, **kwargs)
        tickers = f_tickers(model)
        return {
            i: tickers.filter(
                in_universe, df = dfs[i], threshold=threshold
            ) for i in sorted(dfs.keys())
        }
    return f