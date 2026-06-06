from __future__ import annotations
from pathlib import Path
import pandas as pd


def project_root() -> Path:
    # src/data/universe.py -> parents[0]=data, [1]=src, [2]=project root
    return Path(__file__).resolve().parents[2]


def get_universe(cache_rel_path: str = "data/raw/sp500_tickers.csv") -> list[str]:
    """
    Load S&P 500 tickers from a local cache path anchored at project root.
    """
    root = project_root()
    p = root / cache_rel_path

    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. Run: python src/data/fetch_sp500.py"
        )

    tickers = pd.read_csv(p)["symbol"].astype(str).tolist()
    tickers = [t.strip().upper().replace(".", "-") for t in tickers if isinstance(t, str)]
    return sorted(list(dict.fromkeys(tickers)))