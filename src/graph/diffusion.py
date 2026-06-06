import numpy as np
import pandas as pd
from typing import Tuple


def rolling_sum_signal(
    returns_mat: pd.DataFrame,
    end_date: pd.Timestamp,
    lookback: int = 5
) -> pd.Series:
    """
    Node signal x(t): 5-day cumulative log return ending at end_date.
    returns_mat: index=date, columns=symbol, values=ret_1d

    Returns a Series indexed by symbol.
    """
    # Ensure end_date exists
    if end_date not in returns_mat.index:
        raise KeyError(f"end_date {end_date} not in returns matrix index")

    end_idx = returns_mat.index.get_loc(end_date)
    start_idx = end_idx - lookback + 1
    if start_idx < 0:
        raise ValueError("Not enough history for rolling sum signal")

    window_slice = returns_mat.iloc[start_idx:end_idx + 1]
    x = window_slice.sum(axis=0)  # sum over dates -> per symbol
    return x


def diffuse_signal(
    x: pd.Series,
    L_sym: pd.DataFrame,
    alpha: float = 10.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Diffuse (smooth) a node signal x over a graph using:
      x_smooth = (I + alpha * L_sym)^(-1) x

    Uses a stable linear solve. If numerical issues appear, falls back to lstsq.
    """
    symbols = L_sym.index
    x_vec = x.reindex(symbols).fillna(0.0).values.astype(float)

    n = len(symbols)
    I = np.eye(n)
    M = I + alpha * L_sym.values

    try:
        x_smooth_vec = np.linalg.solve(M, x_vec)
    except np.linalg.LinAlgError:
        x_smooth_vec = np.linalg.lstsq(M, x_vec, rcond=None)[0]

    x_smooth = pd.Series(x_smooth_vec, index=symbols, name="x_smooth")
    residual = pd.Series(x_vec - x_smooth_vec, index=symbols, name="residual")
    return x_smooth, residual