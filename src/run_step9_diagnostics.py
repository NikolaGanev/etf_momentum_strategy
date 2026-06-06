from pathlib import Path
import numpy as np
import pandas as pd

from src.config.settings import Settings
from src.data.loader import load_processed_market_data
from src.graph.correlation import make_returns_matrix, rolling_correlation_matrices
from src.graph.graph_matrices import corr_to_adjacency, adjacency_to_laplacian
from src.graph.diffusion import rolling_sum_signal, diffuse_signal
from src.tda.persistent_homology import corr_to_distance, compute_persistent_homology_features
from src.portfolio.baseline import build_market_neutral_weights
from src.backtest.engine import BacktestConfig, run_backtest


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def smooth_weights(weights_df: pd.DataFrame, eta: float) -> pd.DataFrame:
    """
    w_t = (1-eta)*w_raw_t + eta*w_{t-1}
    Applied per symbol across time.
    """
    w = weights_df.sort_values(["symbol", "date"]).copy()
    w["w_smooth"] = 0.0

    for sym, g in w.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        prev = 0.0
        ws = []
        for x in g["weight"].values:
            prev = (1.0 - eta) * float(x) + eta * float(prev)
            ws.append(prev)
        w.loc[g.index, "w_smooth"] = ws

    w["weight"] = w["w_smooth"]
    w = w.drop(columns=["w_smooth"])
    return w


def build_dataset_for_alpha(
    market_df: pd.DataFrame,
    returns_mat: pd.DataFrame,
    corr_by_date: dict,
    corr_window: int,
    alpha: float,
    signal_lookback: int,
    tda_by_date: dict,
    max_dates: int = 250
) -> pd.DataFrame:
    """
    Build a minimal modeling dataset for diagnostics:
    date, symbol, ret_1d, fwd_ret_10, diff_resid, plus TDA regime features.
    """
    # Choose last max_dates dates
    dates = sorted(corr_by_date.keys())
    if len(dates) > max_dates:
        dates = dates[-max_dates:]

    rows = []
    for d in dates:
        corr = corr_by_date[d]
        A = corr_to_adjacency(corr, positive_only=True)
        _, Lsym = adjacency_to_laplacian(A)

        x = rolling_sum_signal(returns_mat, end_date=d, lookback=signal_lookback)
        _, resid = diffuse_signal(x, Lsym, alpha=alpha)

        # market labels for that date
        day = market_df[market_df["date"] == d][["date", "symbol", "ret_1d", "fwd_ret_10"]].copy()
        if day.empty:
            continue

        day = day.merge(
            resid.rename("diff_resid").reset_index().rename(columns={"index": "symbol"}),
            on="symbol",
            how="inner"
        )

        # add TDA features (same for all symbols on date)
        feats = tda_by_date[d]
        for k, v in feats.items():
            day[k] = v

        rows.append(day)

    ds = pd.concat(rows, ignore_index=True).dropna().sort_values(["date", "symbol"]).reset_index(drop=True)
    return ds


def build_weights_from_dataset(ds: pd.DataFrame) -> pd.DataFrame:
    """
    Daily cross-sectional baseline weights from diff_resid.
    """
    weights_rows = []
    for d, g in ds.groupby("date"):
        scores = g.set_index("symbol")["diff_resid"]
        w = build_market_neutral_weights(
            scores=scores,
            gross_exposure=1.0,
            max_abs_weight=0.25,      # prototype
            long_short_frac=0.30
        )
        out = pd.DataFrame({"date": d, "symbol": w.index, "weight": w.values})
        weights_rows.append(out)
    return pd.concat(weights_rows, ignore_index=True)


def main():
    cfg = Settings()
    root = project_root()

    # Load market (Step 2 output)
    market_path = root / "data" / "processed" / "market_data.parquet"
    market_df = load_processed_market_data(str(market_path))
    market_df = market_df.sort_values(["symbol", "date"]).copy()

    # Returns matrix
    returns_mat = make_returns_matrix(market_df, cfg.returns_col)

    # Correlations for chosen window (120)
    corr_window = 120
    corr_by_date = rolling_correlation_matrices(returns_mat, window=corr_window)

    # Precompute TDA-by-date for the same dates (so we don’t recompute per alpha)
    print("\nPrecomputing TDA features by date (window=120)...")
    dates = sorted(corr_by_date.keys())
    if len(dates) > 250:
        dates = dates[-250:]

    tda_by_date = {}
    for d in dates:
        dist = corr_to_distance(corr_by_date[d])
        feats = compute_persistent_homology_features(distance=dist, max_dim=1, max_edge_length=2.0)
        tda_by_date[d] = feats
    print(f"Computed TDA for {len(tda_by_date)} dates.\n")

    alphas = [0.1, 0.3, 1.0, 3.0, 10.0]
    etas = [0.0, 0.5, 0.8]
    cost_bps_list = [0.0, 10.0]
    signal_lookback = 5

    results = []

    for alpha in alphas:
        print(f"=== Building dataset + weights for alpha={alpha} ===")
        ds = build_dataset_for_alpha(
            market_df=market_df,
            returns_mat=returns_mat,
            corr_by_date=corr_by_date,
            corr_window=corr_window,
            alpha=alpha,
            signal_lookback=signal_lookback,
            tda_by_date=tda_by_date,
            max_dates=250
        )

        weights_df = build_weights_from_dataset(ds)

        for eta in etas:
            w_use = weights_df.copy()
            if eta > 0:
                w_use = smooth_weights(w_use, eta=eta)

            for cost_bps in cost_bps_list:
                daily, metrics = run_backtest(
                    weights_df=w_use,
                    market_df=ds,  # ds contains ret_1d; engine computes fwd1 internally
                    cfg=BacktestConfig(cost_bps=cost_bps)
                )

                results.append({
                    "alpha": alpha,
                    "eta": eta,
                    "cost_bps": cost_bps,
                    "days": metrics["days"],
                    "cagr": metrics["cagr"],
                    "sharpe": metrics["sharpe"],
                    "max_dd": metrics["max_drawdown"],
                    "avg_turnover": metrics["avg_turnover"],
                    "final_equity": metrics["final_equity"],
                })

        print(f"Done alpha={alpha}\n")

    res = pd.DataFrame(results).sort_values(["cost_bps", "eta", "alpha"]).reset_index(drop=True)

    out_path = root / "data" / "processed" / "diagnostics_step9_1.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out_path)

    print("\n=== STEP 9.1 DIAGNOSTICS SUMMARY ===")
    print(res.to_string(index=False))
    print(f"\nSaved diagnostics table to: {out_path.resolve()}")
    print("===================================\n")


if __name__ == "__main__":
    main()