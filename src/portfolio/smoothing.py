import pandas as pd


def smooth_weights(weights_df: pd.DataFrame, eta: float) -> pd.DataFrame:
    """
    Exponential smoothing per symbol:
      w_t = (1-eta)*w_raw_t + eta*w_{t-1}

    weights_df columns: date, symbol, weight
    """
    w = weights_df.sort_values(["symbol", "date"]).copy()
    w["weight_smooth"] = 0.0

    for sym, g in w.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        prev = 0.0
        out = []
        for x in g["weight"].values:
            prev = (1.0 - eta) * float(x) + eta * float(prev)
            out.append(prev)
        w.loc[g.index, "weight_smooth"] = out

    w["weight"] = w["weight_smooth"]
    return w.drop(columns=["weight_smooth"])