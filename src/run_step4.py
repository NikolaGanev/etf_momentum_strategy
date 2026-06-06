from pathlib import Path
import pandas as pd

from src.config.settings import Settings
from src.data.loader import load_processed_market_data
from src.graph.correlation import make_returns_matrix, iter_rolling_correlation_matrices
from src.graph.graph_matrices import corr_to_adjacency, adjacency_to_laplacian
from src.graph.diffusion import rolling_sum_signal, diffuse_signal


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main():
    cfg = Settings()
    root = project_root()

    # Locked parameters from diagnostics
    signal_lookback = 5
    alpha = 10.0

    # Load market data
    df = load_processed_market_data(str(root / "data" / "processed" / "market_data.parquet"))
    returns_mat = make_returns_matrix(df, cfg.returns_col)

    all_rows = []

    for w in cfg.corr_windows:
        print(f"\n=== Diffusion signals for window={w} (all dates) ===")

        for d, corr in iter_rolling_correlation_matrices(returns_mat, window=w, max_dates=1000000):
            A = corr_to_adjacency(corr, positive_only=cfg.use_positive_correlations_only)
            _, L_sym = adjacency_to_laplacian(A)

            x = rolling_sum_signal(returns_mat, end_date=d, lookback=signal_lookback)
            x_smooth, resid = diffuse_signal(x, L_sym, alpha=alpha)

            tmp = pd.DataFrame({
                "date": d,
                "symbol": resid.index,
                "signal_5d": x.values,
                "signal_5d_smooth": x_smooth.values,
                "diff_resid": resid.values,
                "corr_window": w,
                "alpha": alpha,
                "signal_lookback": signal_lookback
            })
            all_rows.append(tmp)

        print(f"Computed residual signals for all dates (window={w}).")

    res_df = pd.concat(all_rows, ignore_index=True)
    res_df["date"] = pd.to_datetime(res_df["date"])

    out_path = root / "data" / "processed" / "diffusion_signals.parquet"
    res_df.to_parquet(out_path)
    print(f"\nSaved diffusion signals to: {out_path.resolve()}")


if __name__ == "__main__":
    main()