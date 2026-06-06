import pandas as pd
import numpy as np


def compute_forward_returns(
    df: pd.DataFrame,
    horizons: list[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Compute forward log returns for multiple horizons.

    Adds columns:
    - fwd_ret_5
    - fwd_ret_10
    - fwd_ret_20
    """

    df = df.sort_values(["symbol", "date"]).copy()

    for h in horizons:
        df[f"fwd_ret_{h}"] = (
            np.log(df.groupby("symbol")["adj_close"].shift(-h)) -
            np.log(df["adj_close"])
        )

    return df