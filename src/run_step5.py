from pathlib import Path
import pandas as pd

from src.config.settings import Settings
from src.data.loader import load_processed_market_data
from src.graph.correlation import make_returns_matrix, iter_rolling_correlation_matrices
from src.tda.persistent_homology import corr_to_distance, compute_ph_features_from_points
from src.tda.embedding import classical_mds


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main():
    cfg = Settings()
    root = project_root()

    df = load_processed_market_data(str(root / "data" / "processed" / "market_data.parquet"))
    returns_mat = make_returns_matrix(df, cfg.returns_col)

    out_path = root / "data" / "processed" / "tda_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []

    mds_dim = 10
    max_edge_length = 2.0

    for w in cfg.corr_windows:
        print(f"\n=== TDA (MDS+PH) features for window={w} (last 250 dates) ===")

        for d, corr in iter_rolling_correlation_matrices(returns_mat, window=w, max_dates=1000000):
            dist = corr_to_distance(corr)

            # Embed distance matrix into R^mds_dim
            pts = classical_mds(dist, dim=mds_dim)

            feats = compute_ph_features_from_points(
                points=pts,
                max_dim=1,
                max_edge_length=max_edge_length
            )

            feats["date"] = d
            feats["corr_window"] = w
            feats["mds_dim"] = mds_dim
            feats["max_edge_length"] = max_edge_length
            all_rows.append(feats)

        print(f"Computed TDA features for 250 dates (window={w}).")

    tda_df = pd.DataFrame(all_rows)
    tda_df["date"] = pd.to_datetime(tda_df["date"])
    tda_df.to_parquet(out_path)

    print(f"\nSaved TDA features to: {out_path.resolve()}")


if __name__ == "__main__":
    main()