from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.portfolio.smoothing import smooth_weights


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def compute_forecast_portfolio_vol(
    returns_window: pd.DataFrame,
    weights: pd.Series
) -> float:
    """
    Forecast daily portfolio volatility using sample covariance:
        vol = sqrt(w' Σ w)
    returns_window: rows=dates, cols=symbols, values=daily returns (log returns ok)
    weights: index=symbols
    """
    cols = [c for c in returns_window.columns if c in weights.index]
    if len(cols) < 2:
        return np.nan

    R = returns_window[cols].dropna(how="any")
    if len(R) < 20:
        return np.nan

    cov = R.cov().to_numpy(dtype=float)
    w = weights.loc[cols].to_numpy(dtype=float).reshape(-1, 1)

    if cov.shape[0] != w.shape[0]:
        return np.nan

    var_mat = (w.T @ cov @ w)  # (1,1)
    try:
        var = float(var_mat.item())
    except Exception:
        return np.nan

    if not np.isfinite(var) or var <= 0:
        return np.nan

    return float(np.sqrt(var))


def build_regime_scalar(regime_df: pd.DataFrame, date: pd.Timestamp) -> float:
    """
    Map regime proxies -> exposure scalar g in [0,1].
    High avg_abs_corr => more coupled market => risk-off => smaller g.

    We compute a rolling z-score of avg_abs_corr (last ~252 obs) and apply a sigmoid.
    """
    sub = regime_df[regime_df["date"] <= date].tail(252)
    if len(sub) < 60:
        return 1.0

    x = sub["avg_abs_corr"].astype(float)
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if sd <= 1e-8 or (not np.isfinite(sd)):
        return 1.0

    z = (float(x.iloc[-1]) - mu) / sd

    # sigmoid: higher z => lower g
    g = 1.0 / (1.0 + np.exp(1.25 * z))
    return float(np.clip(g, 0.0, 1.0))


def build_ts_mom_weights_vol_targeted(
    market_df: pd.DataFrame,
    regime_df: pd.DataFrame | None = None,
    lookback: int = 126,
    vol_window: int = 20,
    rebalance_every: int = 5,
    max_abs_weight: float = 0.35,
    eta: float = 0.85,
    mode: str = "long_short",          # "long_short" or "long_flat"
    vol_target_ann: float = 0.10,      # annualized vol target
    vol_target_lookback: int = 60,     # covariance lookback (days)
    max_leverage: float = 2.0,         # cap scaling
) -> pd.DataFrame:
    """
    ETF Time-Series Momentum weights with:
      - log-price trend signal (lookback)
      - inverse-vol instrument scaling
      - unit-gross normalization then portfolio vol targeting
      - optional regime exposure scaling from topology proxies
      - smoothing
    """

    df = market_df.sort_values(["symbol", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["adj_close"] = df["adj_close"].astype(float)

    # log price + daily log returns
    df["logp"] = np.log(df["adj_close"].replace(0.0, np.nan))
    df["logret_1d"] = df.groupby("symbol")["logp"].diff()

    # trend and per-instrument vol
    df["trend_L"] = df.groupby("symbol")["logp"].diff(lookback)
    df["vol_i"] = (
        df.groupby("symbol")["logret_1d"]
          .rolling(vol_window, min_periods=max(10, int(0.5 * vol_window)))
          .std()
          .reset_index(level=0, drop=True)
    )

    # Return matrix for covariance forecasts
    ret_mat = df.pivot(index="date", columns="symbol", values="logret_1d").sort_index()

    dates = sorted(df["date"].unique())
    prev_w = None
    rows = []

    target_daily_vol = vol_target_ann / np.sqrt(252.0)

    # Ensure regime df is clean
    if regime_df is not None:
        regime_df = regime_df.copy()
        regime_df["date"] = pd.to_datetime(regime_df["date"])
        regime_df = regime_df.sort_values("date")

    for i, d in enumerate(dates):
        g = df[df["date"] == d].set_index("symbol")
        do_rebalance = (i % rebalance_every == 0)

        # Hold weights between rebalances
        if (not do_rebalance) and (prev_w is not None):
            w = prev_w.reindex(g.index).fillna(0.0)
            rows.append(pd.DataFrame({"date": d, "symbol": w.index, "weight": w.values}))
            continue
        elif (not do_rebalance) and (prev_w is None):
            rows.append(pd.DataFrame({"date": d, "symbol": g.index, "weight": np.zeros(len(g), dtype=float)}))
            continue

        # --- Rebalance day: build raw signal ---
        trend = g["trend_L"]
        vol_i = g["vol_i"].replace(0.0, np.nan)

        if mode == "long_short":
            sig = np.sign(trend).replace(0.0, np.nan)
        elif mode == "long_flat":
            sig = (trend > 0).astype(float)
        else:
            raise ValueError("mode must be 'long_short' or 'long_flat'")

        raw = (sig / (vol_i + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Normalize to unit gross
        gross = float(np.abs(raw).sum())
        if gross > 0:
            w = raw / gross
        else:
            w = raw * 0.0

        # Clip and renormalize to unit gross
        w = w.clip(-max_abs_weight, max_abs_weight)
        gross2 = float(np.abs(w).sum())
        if gross2 > 0:
            w = w / gross2

        # --- Portfolio vol targeting ---
        if d in ret_mat.index:
            idx = ret_mat.index.get_loc(d)
            start = max(0, idx - vol_target_lookback)
            window = ret_mat.iloc[start:idx]  # excludes today's row
        else:
            window = ret_mat.iloc[-vol_target_lookback:]

        fvol = compute_forecast_portfolio_vol(window, w)

        if np.isnan(fvol) or fvol <= 1e-8:
            k = 1.0
        else:
            k = float(target_daily_vol / fvol)

        k = max(0.0, min(max_leverage, k))
        w = w * k

        # --- Regime scaler overlay (topology proxy) ---
        if regime_df is not None and len(regime_df) > 0:
            g_scale = build_regime_scalar(regime_df, d)
            w = w * g_scale

        prev_w = w
        rows.append(pd.DataFrame({"date": d, "symbol": w.index, "weight": w.values}))

    wdf = pd.concat(rows, ignore_index=True)

    # Smooth weights
    wdf = smooth_weights(wdf, eta=eta)
    return wdf


def main():
    root = project_root()

    market_path = root / "data" / "processed" / "market_data_live.parquet"
    market_df = pd.read_parquet(market_path).copy()
    market_df["date"] = pd.to_datetime(market_df["date"])

    regime_path = root / "data" / "processed" / "etf_regime_proxies.parquet"
    regime_df = pd.read_parquet(regime_path).copy()
    regime_df["date"] = pd.to_datetime(regime_df["date"])

    # ===== CONFIG =====
    lookback = 126
    vol_window = 20
    rebalance_every = 5

    mode = "long_short"
    eta = 0.85

    vol_target_ann = 0.10
    vol_target_lookback = 60
    max_leverage = 2.0

    max_abs_weight = 0.35
    # ==================

    print(f"RUN CONFIG CHECK: mode={mode}, lookback={lookback}, vol_target_ann={vol_target_ann}")

    wdf = build_ts_mom_weights_vol_targeted(
        market_df=market_df,
        regime_df=regime_df,
        lookback=lookback,
        vol_window=vol_window,
        rebalance_every=rebalance_every,
        max_abs_weight=max_abs_weight,
        eta=eta,
        mode=mode,
        vol_target_ann=vol_target_ann,
        vol_target_lookback=vol_target_lookback,
        max_leverage=max_leverage,
    )

    out_path = root / "data" / "processed" / "weights_ts_mom.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wdf.to_parquet(out_path)

    last_day = wdf["date"].max()
    last = wdf[wdf["date"] == last_day]

    print("\n=== STEP TS-7 WEIGHTS REPORT (TS MOM + VOL TARGET + REGIME) ===")
    print(f"Saved: {out_path.resolve()}")
    print(f"Last day gross: {float(np.abs(last['weight']).sum()):.6f}")
    print("===============================================================\n")


if __name__ == "__main__":
    main()