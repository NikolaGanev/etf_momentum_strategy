from pathlib import Path
import pandas as pd
import numpy as np

from src.portfolio.baseline import build_market_neutral_weights


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def zscore(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - mu) / sd


def main():
    root = project_root()
    df = pd.read_parquet(root / "data/processed/model_dataset_w120.parquet")
    df["date"] = pd.to_datetime(df["date"])

    gross = 1.0
    max_w = 0.02
    ls_frac = 0.30
    diffusion_beta = 0.3  # conservative start
    rebalance_every = 5

    dates = sorted(df["date"].unique())
    prev_w = None
    rows = []

    for i, d in enumerate(dates):
        g = df[df["date"] == d]

        if i % rebalance_every != 0:
            if prev_w is None:
                w = pd.Series(0.0, index=g["symbol"])
            else:
                w = prev_w.reindex(g["symbol"]).fillna(0.0)
            rows.append(pd.DataFrame({"date": d, "symbol": w.index, "weight": w.values}))
            continue

        if g["trade_flag"].iloc[0] == 0:
            prev_w = pd.Series(0.0, index=g["symbol"])
            rows.append(pd.DataFrame({"date": d, "symbol": prev_w.index, "weight": prev_w.values}))
            continue

        g2 = g.set_index("symbol")
        mom = zscore(g2["mom_12_1"])
        diff = zscore(-g2["diff_resid"])

        score = mom + diffusion_beta * diff

        w = build_market_neutral_weights(
            scores=score,
            gross_exposure=gross,
            max_abs_weight=max_w,
            long_short_frac=ls_frac
        )

        prev_w = w
        rows.append(pd.DataFrame({"date": d, "symbol": w.index, "weight": w.values}))

    out = pd.concat(rows, ignore_index=True)
    out_path = root / "data/processed/weights_w120_mom12_1.parquet"
    out.to_parquet(out_path)

    print("\n=== STEP 7 WEIGHTS (12–1 MOMENTUM) ===")
    print(f"Saved: {out_path}")
    print("====================================\n")


if __name__ == "__main__":
    main()