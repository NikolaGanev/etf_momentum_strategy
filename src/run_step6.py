from pathlib import Path
import pandas as pd
import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def add_controls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).copy()

    df["vol_20d"] = (
        df.groupby("symbol")["ret_1d"]
        .rolling(20, min_periods=15)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["log_dollar_volume"] = np.log1p(df["dollar_volume"])
    return df


def add_12_1_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).copy()

    df["ret_252d"] = (
        df.groupby("symbol")["ret_1d"]
        .rolling(252, min_periods=200)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["ret_21d"] = (
        df.groupby("symbol")["ret_1d"]
        .rolling(21, min_periods=15)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["mom_12_1"] = df["ret_252d"] - df["ret_21d"]
    return df


def main():
    root = project_root()

    market = pd.read_parquet(root / "data/processed/market_data.parquet")
    market["date"] = pd.to_datetime(market["date"])

    market = add_controls(market)
    market = add_12_1_momentum(market)

    diffusion = pd.read_parquet(root / "data/processed/diffusion_signals.parquet")
    diffusion["date"] = pd.to_datetime(diffusion["date"])
    diffusion = diffusion[diffusion["corr_window"] == 120]

    regime = pd.read_parquet(root / "data/processed/regime_proxy_features_w120.parquet")
    regime["date"] = pd.to_datetime(regime["date"])

    keep_market = [
        "date", "symbol", "ret_1d", "adj_close", "dollar_volume",
        "vol_20d", "log_dollar_volume",
        "mom_12_1", "fwd_ret_5", "fwd_ret_10", "fwd_ret_20"
    ]

    keep_diff = ["date", "symbol", "diff_resid"]

    ds = (
        market[keep_market]
        .merge(diffusion[keep_diff], on=["date", "symbol"], how="inner")
        .merge(regime.drop(columns=["corr_window"]), on="date", how="inner")
        .dropna()
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )

    out = root / "data/processed/model_dataset_w120.parquet"
    ds.to_parquet(out)

    print("\n=== STEP 6 DATASET REPORT (12–1 MOMENTUM) ===")
    print(f"Saved: {out}")
    print(f"Dates: {ds['date'].nunique()}")
    print(f"Symbols: {ds['symbol'].nunique()}")
    print("============================================\n")


if __name__ == "__main__":
    main()