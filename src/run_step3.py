from pathlib import Path
import pandas as pd

from src.config.settings import Settings
from src.data.loader import load_processed_market_data
from src.graph.correlation import make_returns_matrix, iter_rolling_correlation_matrices
from src.graph.graph_matrices import corr_to_adjacency, adjacency_to_laplacian


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main():
    cfg = Settings()
    root = project_root()

    # Load Step 2 output (make sure you moved it to data/processed/market_data.parquet)
    df = load_processed_market_data(str(root / "data" / "processed" / "market_data.parquet"))
    returns_mat = make_returns_matrix(df, cfg.returns_col)

    out_dir = root / "data" / "processed" / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded returns matrix: {returns_mat.shape[0]} dates × {returns_mat.shape[1]} symbols")

    for w in cfg.corr_windows:
        print(f"\n--- Building graphs for window={w} (last 250 dates) ---")

        last_date = None
        last_corr = None

        for d, corr in iter_rolling_correlation_matrices(returns_mat, window=w, max_dates=250):
            last_date = d
            last_corr = corr

        if last_corr is None:
            print(f"Window={w}: not enough data.")
            continue

        A_last = corr_to_adjacency(last_corr, positive_only=cfg.use_positive_correlations_only)
        _, Lsym_last = adjacency_to_laplacian(A_last)

        # Persist last snapshot (easy to inspect)
        last_corr.to_parquet(out_dir / f"corr_last_w{w}.parquet")
        A_last.to_parquet(out_dir / f"adj_last_w{w}.parquet")
        Lsym_last.to_parquet(out_dir / f"laplacian_sym_last_w{w}.parquet")

        nnz = int((A_last.values > 0).sum())
        print(f"Window={w}: stored last-date matrices for {last_date.date()}")
        print(f"  Corr shape: {last_corr.shape}, Adjacency nnz: {nnz}")

    print(f"\nSaved graph artifacts to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()