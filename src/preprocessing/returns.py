import pandas as pd
import numpy as np


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 1-day log returns per symbol.

    Adds column:
    - ret_1d
    """

    df = df.sort_values(["symbol", "date"]).copy()

    df["ret_1d"] = (
        np.log(df["adj_close"]) -
        np.log(df.groupby("symbol")["adj_close"].shift(1))
    )

    return df