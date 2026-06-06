from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    cost_bps: float = 10.0          # cost per unit turnover (e.g., 10 bps = 0.10%)
    annualization: int = 252


def compute_next_day_returns(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    market_df must contain: date, symbol, ret_1d
    ret_1d at date t is (t-1 -> t).
    We need next-day return aligned with weights at date t:
      ret_fwd1(t) = ret_1d(t+1)
    """
    df = market_df.sort_values(["symbol", "date"]).copy()
    df["ret_fwd1"] = df.groupby("symbol")["ret_1d"].shift(-1)
    return df


def compute_turnover(weights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Turnover_t = sum_i |w_{t,i} - w_{t-1,i}|
    First date turnover uses w_prev=0 (entering positions).
    """
    w = weights_df.sort_values(["symbol", "date"]).copy()
    w["w_prev"] = w.groupby("symbol")["weight"].shift(1).fillna(0.0)
    w["abs_change"] = (w["weight"] - w["w_prev"]).abs()
    turnover = w.groupby("date", as_index=False)["abs_change"].sum().rename(columns={"abs_change": "turnover"})
    return turnover


def compute_max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def compute_metrics(daily: pd.DataFrame, cfg: BacktestConfig) -> Dict[str, Any]:
    r = daily["net_ret"].astype(float)
    gross = daily["gross_ret"].astype(float)

    ann = cfg.annualization
    mean = float(r.mean())
    std = float(r.std(ddof=0))

    sharpe = float(np.sqrt(ann) * mean / std) if std > 0 else 0.0
    cagr = float(daily["equity"].iloc[-1] ** (ann / len(daily)) - 1.0) if len(daily) > 0 else 0.0
    max_dd = compute_max_drawdown(daily["equity"])

    return {
        "days": int(len(daily)),
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "avg_daily_net_ret": mean,
        "avg_daily_gross_ret": float(gross.mean()),
        "avg_turnover": float(daily["turnover"].mean()),
        "total_cost": float(daily["cost"].sum()),
        "final_equity": float(daily["equity"].iloc[-1]) if len(daily) else 1.0,
    }


def run_backtest(
    market_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    cost_bps: float = 10.0,
    annualization: int = 252
) -> Dict[str, Any]:
    """
    Backtest a daily-rebalanced portfolio.

    Inputs:
      market_df: must contain columns [date, symbol, ret_1d]
      weights_df: must contain columns [date, symbol, weight]
      cost_bps: transaction cost in bps per unit turnover

    Returns dict with:
      - metrics fields
      - daily (DataFrame with date, gross_ret, turnover, cost, net_ret, equity)
    """
    cfg = BacktestConfig(cost_bps=float(cost_bps), annualization=int(annualization))

    # Compute forward 1-day returns aligned to weights date
    mkt = compute_next_day_returns(market_df)

    # Join weights(t) with ret_fwd1(t)
    merged = weights_df.merge(
        mkt[["date", "symbol", "ret_fwd1"]],
        on=["date", "symbol"],
        how="left"
    ).copy()

    # Drop dates where next-day return not available (typically the last date)
    merged = merged.dropna(subset=["ret_fwd1"])

    # Portfolio gross return per day
    merged["contrib"] = merged["weight"].astype(float) * merged["ret_fwd1"].astype(float)
    daily_gross = merged.groupby("date", as_index=False)["contrib"].sum().rename(columns={"contrib": "gross_ret"})

    # Turnover and costs
    turnover = compute_turnover(weights_df)
    daily = daily_gross.merge(turnover, on="date", how="left").fillna({"turnover": 0.0})

    cost_rate = cfg.cost_bps / 10000.0
    daily["cost"] = cost_rate * daily["turnover"].astype(float)
    daily["net_ret"] = daily["gross_ret"].astype(float) - daily["cost"].astype(float)

    daily = daily.sort_values("date").reset_index(drop=True)
    daily["equity"] = (1.0 + daily["net_ret"]).cumprod()

    metrics = compute_metrics(daily, cfg)
    metrics["daily"] = daily
    return metrics