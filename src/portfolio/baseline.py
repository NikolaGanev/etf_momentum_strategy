import numpy as np
import pandas as pd


def zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def build_market_neutral_weights(
    scores: pd.Series,
    gross_exposure: float = 1.0,
    max_abs_weight: float = 0.25,
    long_short_frac: float = 0.30
) -> pd.Series:
    """
    Baseline market-neutral portfolio constructor.

    Steps:
    1) z-score the cross-sectional signal
    2) take top/bottom tails (long_short_frac each)
    3) neutralize (sum w = 0)
    4) scale to target gross exposure (sum |w| = gross_exposure)
    5) cap max position size, then re-neutralize & rescale

    Returns: Series indexed by symbol with weights.
    """
    s = scores.dropna().copy()
    if len(s) < 2:
        return pd.Series(0.0, index=scores.index)

    z = zscore(s)

    n = len(z)
    k = max(1, int(np.floor(long_short_frac * n)))

    long_names = z.sort_values(ascending=False).head(k).index
    short_names = z.sort_values(ascending=True).head(k).index

    w = pd.Series(0.0, index=z.index)
    w.loc[long_names] = z.loc[long_names]
    w.loc[short_names] = z.loc[short_names]

    if np.allclose(w.values, 0.0):
        out = pd.Series(0.0, index=scores.index)
        return out

    # Dollar neutral
    w = w - w.mean()

    # Scale to gross exposure
    gross = np.abs(w).sum()
    if gross > 0:
        w = w * (gross_exposure / gross)

    # Cap and re-normalize
    w = w.clip(lower=-max_abs_weight, upper=max_abs_weight)
    w = w - w.mean()

    gross = np.abs(w).sum()
    if gross > 0:
        w = w * (gross_exposure / gross)

    out = pd.Series(0.0, index=scores.index)
    out.loc[w.index] = w
    return out