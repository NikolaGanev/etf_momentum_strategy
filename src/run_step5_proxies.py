from pathlib import Path
import pandas as pd

from src.config.settings import Settings
from src.data.loader import load_processed_market_data
from src.graph.correlation import make_returns_matrix, iter_rolling_correlation_matrices
from src.regime.proxies import corr_to_distance, average_offdiag_corr, mst_total_length, build_trade_flag


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main():
    cfg = Settings()
    root = project_root()

    # We gate using the same primary window as the strategy
    corr_window = 120

    df = load_processed_market_data(str(root / "data" / "processed" / "market_data.parquet"))
    returns_mat = make_returns_matrix(df, cfg.returns_col)

    rows = []
    for d, corr in iter_rolling_correlation_matrices(returns_mat, window=corr_window, max_dates=10000000):
        avg_corr = average_offdiag_corr(corr)
        dist = corr_to_distance(corr)
        mst_len = mst_total_length(dist)

        rows.append({
            "date": d,
            "corr_window": corr_window,
            "avg_corr": avg_corr,
            "mst_length": mst_len
        })

    feat = pd.DataFrame(rows)
    feat["date"] = pd.to_datetime(feat["date"])

    # Build trade flag (median/median by default)
    feat = build_trade_flag(
        feat,
        corr_col="avg_corr",
        mst_col="mst_length",
        corr_quantile=0.50,
        mst_quantile=0.50
    )

    out_path = root / "data" / "processed" / "regime_proxy_features_w120.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(out_path)

    print("\n=== STEP 5P PROXY REGIME FEATURES REPORT ===")
    print(f"Saved: {out_path.resolve()}")
    print(f"Rows (dates): {len(feat)}")
    print(f"Trade days: {int(feat['trade_flag'].sum())} / {len(feat)}")
    print("===========================================\n")


if __name__ == "__main__":
    main()