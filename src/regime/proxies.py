from __future__ import annotations

import numpy as np
import pandas as pd


def corr_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """
    Convert correlation to a metric-like distance.
    d_ij = sqrt(0.5 * (1 - corr_ij))
    """
    c = corr.clip(-1.0, 1.0).fillna(0.0)
    dist = np.sqrt(0.5 * (1.0 - c.values))
    return pd.DataFrame(dist, index=c.index, columns=c.columns)


def average_offdiag_corr(corr: pd.DataFrame) -> float:
    c = corr.values.astype(float)
    n = c.shape[0]
    if n <= 1:
        return 0.0
    # exclude diagonal
    off = c[~np.eye(n, dtype=bool)]
    return float(np.nanmean(off))


def mst_total_length(distance: pd.DataFrame) -> float:
    """
    Prim's algorithm MST total length for a dense distance matrix.
    O(n^2), fine for n ~ 500 when done ~250 times.
    """
    D = distance.values.astype(float)
    n = D.shape[0]
    if n <= 1:
        return 0.0

    in_mst = np.zeros(n, dtype=bool)
    min_edge = np.full(n, np.inf, dtype=float)

    # start at 0
    min_edge[0] = 0.0
    total = 0.0

    for _ in range(n):
        u = int(np.argmin(np.where(in_mst, np.inf, min_edge)))
        in_mst[u] = True
        total += float(min_edge[u])

        # update frontier
        du = D[u, :]
        # only update nodes not in MST
        mask = ~in_mst
        min_edge[mask] = np.minimum(min_edge[mask], du[mask])

    return float(total)


def zscore_series(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - mu) / sd


def build_trade_flag(
    df: pd.DataFrame,
    corr_col: str = "avg_corr",
    mst_col: str = "mst_length",
    corr_quantile: float = 0.50,
    mst_quantile: float = 0.50
) -> pd.DataFrame:
    """
    Trade when:
      avg_corr <= median (or chosen quantile) AND mst_length >= median (or chosen quantile)

    This creates a binary gate that removes high-correlation / low-dispersion regimes.
    """
    out = df.copy()
    corr_thr = out[corr_col].quantile(corr_quantile)
    mst_thr = out[mst_col].quantile(mst_quantile)

    out["trade_flag"] = ((out[corr_col] <= corr_thr) & (out[mst_col] >= mst_thr)).astype(int)

    # Optional: add z-scores for inspection
    out[corr_col + "_z"] = zscore_series(out[corr_col])
    out[mst_col + "_z"] = zscore_series(out[mst_col])

    return out