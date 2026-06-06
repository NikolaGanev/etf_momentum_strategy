import pandas as pd
from typing import Iterator, Tuple, Optional


def make_returns_matrix(df: pd.DataFrame, returns_col: str) -> pd.DataFrame:
    """
    Long -> wide returns matrix:
    index=date, columns=symbol, values=returns_col
    """
    mat = df.pivot(index="date", columns="symbol", values=returns_col).sort_index()
    return mat


def iter_rolling_correlation_matrices(
    returns_mat: pd.DataFrame,
    window: int,
    max_dates: int = 250,
    min_periods: Optional[int] = None
) -> Iterator[Tuple[pd.Timestamp, pd.DataFrame]]:
    """
    Yield (end_date, corr_matrix) for the last `max_dates` rolling windows.

    This is streaming (generator) to avoid storing all matrices in memory.
    """
    if min_periods is None:
        min_periods = max(2, window // 2)

    idx = returns_mat.index
    if len(idx) < window:
        return

    # Eligible end indices: window-1 .. end
    end_indices = list(range(window - 1, len(idx)))

    # Keep only last max_dates
    if len(end_indices) > max_dates:
        end_indices = end_indices[-max_dates:]

    for end_idx in end_indices:
        end_date = idx[end_idx]
        window_slice = returns_mat.iloc[end_idx - window + 1: end_idx + 1]
        corr = window_slice.corr(min_periods=min_periods)
        yield end_date, corr