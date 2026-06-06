from __future__ import annotations

from typing import List, Optional
import random
import time
import pandas as pd
import yfinance as yf


def _download_chunk(tickers: List[str], start_date: str, end_date: Optional[str], progress: bool) -> pd.DataFrame:
    return yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,          # None => latest
        auto_adjust=True,
        group_by="ticker",
        progress=progress,
        threads=True,
    )


def download_yahoo_data(
    symbols: List[str],
    start_date: str,
    end_date: Optional[str],
    *,
    max_retries: int = 6,
    chunk_size: int = 5,
    base_sleep: float = 2.0,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Robust Yahoo download with retries + chunking to reduce rate limiting.

    Output LONG format:
    date | symbol | adj_close | volume | dollar_volume
    """
    if not symbols:
        raise ValueError("No symbols provided")

    all_rows = []
    symbols = list(dict.fromkeys(symbols))  # de-dupe, keep order

    chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]

    for chunk in chunks:
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                df = _download_chunk(chunk, start_date, end_date, progress=progress)
                if df is None or df.empty:
                    raise ValueError("Empty response")

                # Normalize to LONG
                if isinstance(df.columns, pd.MultiIndex):
                    # MultiIndex: (Ticker, Field)
                    for sym in chunk:
                        if sym not in df.columns.get_level_values(0):
                            continue
                        sub = df[sym].copy()
                        if sub.empty:
                            continue
                        sub = sub.reset_index().rename(columns={
                            "Date": "date",
                            "Close": "adj_close",
                            "Volume": "volume",
                        })
                        if "adj_close" not in sub.columns:
                            continue
                        if "volume" not in sub.columns:
                            sub["volume"] = 0.0

                        sub["symbol"] = sym
                        sub = sub[["date", "symbol", "adj_close", "volume"]]
                        sub["dollar_volume"] = sub["adj_close"] * sub["volume"]
                        all_rows.append(sub)
                else:
                    # single ticker case
                    sub = df.reset_index().rename(columns={
                        "Date": "date",
                        "Close": "adj_close",
                        "Volume": "volume",
                    })
                    sub["symbol"] = chunk[0]
                    sub = sub[["date", "symbol", "adj_close", "volume"]]
                    sub["dollar_volume"] = sub["adj_close"] * sub["volume"]
                    all_rows.append(sub)

                # Success for this chunk
                last_err = None
                break

            except Exception as e:
                last_err = e
                # exponential backoff + jitter
                sleep_s = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                print(f"[Yahoo] Chunk {chunk} attempt {attempt}/{max_retries} failed: {e}. Sleeping {sleep_s:.2f}s")
                time.sleep(sleep_s)

        if last_err is not None:
            print(f"[Yahoo] Giving up on chunk {chunk} after {max_retries} retries. Last error: {last_err}")

    if not all_rows:
        raise ValueError("No data downloaded from Yahoo (all chunks failed).")

    out = pd.concat(all_rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.dropna(subset=["adj_close"])
    return out