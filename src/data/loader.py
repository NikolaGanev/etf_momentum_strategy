import pandas as pd


def load_processed_market_data(path: str) -> pd.DataFrame:
    """
    Load the processed dataset produced in Step 2.

    Ensures:
    - date is datetime
    - date is timezone-naive (important for consistent indexing)
    """
    df = pd.read_parquet(path).copy()

    df["date"] = pd.to_datetime(df["date"])
    # If timezone-aware (Yahoo sometimes is), drop tz info for consistency
    if getattr(df["date"].dt, "tz", None) is not None:
        df["date"] = df["date"].dt.tz_convert(None)

    return df