from pathlib import Path
import pandas as pd

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def main():
    root = project_root()

    market_path = root / "data" / "processed" / "market_data.parquet"
    weights_path = root / "data" / "processed" / "weights_ts_mom.parquet"
    last_trade_path = root / "data" / "processed" / "last_trade_date.txt"

    m = pd.read_parquet(market_path, columns=["date"])
    m["date"] = pd.to_datetime(m["date"])
    market_last = m["date"].max()

    w = pd.read_parquet(weights_path, columns=["date"])
    w["date"] = pd.to_datetime(w["date"])
    weights_last = w["date"].max()

    last_trade = last_trade_path.read_text().strip() if last_trade_path.exists() else "(missing)"

    print("\n=== DATE DEBUG ===")
    print("market_data last date :", market_last)
    print("weights last date     :", weights_last)
    print("last_trade_date.txt   :", last_trade)
    print("==================\n")

if __name__ == "__main__":
    main()