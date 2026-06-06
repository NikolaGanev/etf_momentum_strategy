from pathlib import Path
import pandas as pd

from src.backtest.engine import run_backtest


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main():
    root = project_root()

    dataset_path = root / "data" / "processed" / "model_dataset_w120.parquet"
    weights_path = root / "data/processed/weights_w120_mom12_1.parquet"

    market_df = pd.read_parquet(dataset_path).copy()
    weights_df = pd.read_parquet(weights_path).copy()

    market_df["date"] = pd.to_datetime(market_df["date"])
    weights_df["date"] = pd.to_datetime(weights_df["date"])

    results = run_backtest(
        market_df=market_df,
        weights_df=weights_df,
        cost_bps=10.0,
        annualization=252
    )

    print("\n=== STEP 9 BACKTEST REPORT (MOMENTUM, w=120) ===")
    for k, v in results.items():
        if k == "daily":
            continue
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    out_path = root / "data" / "processed" / "backtest_daily_w120_momentum.parquet"
    results["daily"].to_parquet(out_path)

    print(f"\nSaved daily backtest series to: {out_path.resolve()}")
    print("==============================================\n")


if __name__ == "__main__":
    main()