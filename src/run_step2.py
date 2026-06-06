from __future__ import annotations

from datetime import date
from pathlib import Path
import pandas as pd

from src.data.provider_yahoo import download_yahoo_data
from src.preprocessing.returns import compute_log_returns
from src.preprocessing.sanity_checks import run_sanity_checks


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main():
    from src.data.universe_etf import get_etf_universe
    symbols = get_etf_universe()

    out_dir = project_root() / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    live_path = out_dir / "market_data_live.parquet"

    # Try download
    try:
        df = download_yahoo_data(
            symbols=symbols,
            start_date="2019-01-01",
            end_date=None,
            max_retries=6,
            chunk_size=5,
            base_sleep=2.0,
            progress=True,
        )
        df["date"] = pd.to_datetime(df["date"])
        raw_last = df["date"].max()

        today = pd.Timestamp(date.today())
        days_stale = (today.normalize() - raw_last.normalize()).days
        print(f"\n[STEP2] Raw download last date: {raw_last.date()}")
        print(f"[STEP2] Days stale vs today: {days_stale}")

        # compute returns (needed for trading)
        df = compute_log_returns(df)
        live_df = df.dropna(subset=["adj_close", "ret_1d"]).reset_index(drop=True)

        print("\n=== SANITY CHECK (LIVE) ===")
        run_sanity_checks(live_df)

        live_df.to_parquet(live_path)
        print(f"\nSaved LIVE dataset : {live_path.resolve()}\n")

    except Exception as e:
        # Fallback to last saved live dataset so pipeline can continue
        print("\n!!! WARNING: Yahoo download failed !!!")
        print("Reason:", repr(e))
        if not live_path.exists():
            raise  # nothing to fall back to

        live_df = pd.read_parquet(live_path).copy()
        live_df["date"] = pd.to_datetime(live_df["date"])
        print("\nUsing cached LIVE dataset instead:")
        print("Cached last date:", live_df["date"].max().date())
        print(f"Path: {live_path.resolve()}\n")

        # still sanity check cached data
        print("=== SANITY CHECK (CACHED LIVE) ===")
        run_sanity_checks(live_df)


if __name__ == "__main__":
    main()