from pathlib import Path
import pandas as pd

from src.backtest.engine import run_backtest


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main():
    root = project_root()

    market_path = root / "data" / "processed" / "market_data.parquet"
    weights_path = root / "data" / "processed" / "weights_ts_mom.parquet"

    market_df = pd.read_parquet(market_path).copy()
    weights_df = pd.read_parquet(weights_path).copy()

    market_df["date"] = pd.to_datetime(market_df["date"])
    weights_df["date"] = pd.to_datetime(weights_df["date"])

    results = run_backtest(
        market_df=market_df,
        weights_df=weights_df,
        cost_bps=10.0,
        annualization=252
    )

    print("\n=== STEP TS-9 BACKTEST REPORT (TIME-SERIES MOM) ===")
    for k, v in results.items():
        if k == "daily":
            continue
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    out_path = root / "data" / "processed" / "backtest_daily_ts_mom.parquet"
    results["daily"].to_parquet(out_path)
    print(f"\nSaved daily backtest series to: {out_path.resolve()}")
    print("===================================================\n")


if __name__ == "__main__":
    main()