from __future__ import annotations
from pathlib import Path
import pandas as pd


def project_root() -> Path:
    # src/data/fetch_sp500.py -> parents[0]=data, [1]=src, [2]=project root
    return Path(__file__).resolve().parents[2]


def fetch_sp500_tickers(cache_rel_path: str = "data/raw/sp500_tickers.csv") -> list[str]:
    """
    Fetch S&P 500 tickers from a public CSV source and cache locally.
    Cache path is anchored at project root so it works from any working directory.
    """
    root = project_root()
    p = root / cache_rel_path
    p.parent.mkdir(parents=True, exist_ok=True)

    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    df = pd.read_csv(url)

    tickers = df["Symbol"].astype(str).tolist()
    tickers = [t.strip().upper().replace(".", "-") for t in tickers if isinstance(t, str)]
    tickers = sorted(list(dict.fromkeys(tickers)))

    pd.DataFrame({"symbol": tickers}).to_csv(p, index=False)
    return tickers


if __name__ == "__main__":
    tickers = fetch_sp500_tickers()
    print(f"Saved {len(tickers)} tickers to data/raw/sp500_tickers.csv")