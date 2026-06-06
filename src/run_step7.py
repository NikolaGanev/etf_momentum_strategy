from pathlib import Path
import pandas as pd
import numpy as np

from src.portfolio.baseline import build_market_neutral_weights
from src.portfolio.smoothing import smooth_weights


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main():
    root = project_root()

    dataset_path = root / "data" / "processed" / "model_dataset_w120.parquet"
    df = pd.read_parquet(dataset_path).copy()
    df["date"] = pd.to_datetime(df["date"])

    # Baseline settings (we’ll tune later)
    gross_exposure = 1.0
    max_abs_weight = 0.02
    long_short_frac = 0.30
    eta = 0.8

    weights_rows = []

    for d, g in df.groupby("date"):
        trade_flag = int(g["trade_flag"].iloc[0])

        if trade_flag == 0:
            # Stay flat (no positions)
            tmp = pd.DataFrame({
                "date": d,
                "symbol": g["symbol"].values,
                "weight": np.zeros(len(g), dtype=float)
            })
            weights_rows.append(tmp)
            continue

        # IMPORTANT: flipped signal direction (momentum)
        scores = -g.set_index("symbol")["diff_resid"]

        w = build_market_neutral_weights(
            scores=scores,
            gross_exposure=gross_exposure,
            max_abs_weight=max_abs_weight,
            long_short_frac=long_short_frac
        )

        tmp = pd.DataFrame({"date": d, "symbol": w.index, "weight": w.values})
        weights_rows.append(tmp)

    wdf = pd.concat(weights_rows, ignore_index=True)

    # Smooth across time (this will naturally handle entering/exiting on gate days)
    wdf = smooth_weights(wdf, eta=eta)

    out_path = root / "data" / "processed" / "weights_w120.parquet"
    wdf.to_parquet(out_path)

    last_day = wdf["date"].max()
    last = wdf[wdf["date"] == last_day]
    print("\n=== STEP 7 WEIGHTS REPORT (REGIME-GATED) ===")
    print(f"Saved: {out_path.resolve()}")
    print(f"Dates: {wdf['date'].nunique()}, Symbols: {wdf['symbol'].nunique()}")
    print(f"Last day gross: {float(np.abs(last['weight']).sum()):.6f}")
    print("==========================================\n")


if __name__ == "__main__":
    main()