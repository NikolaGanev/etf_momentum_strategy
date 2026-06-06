import numpy as np
import pandas as pd


def add_basic_controls(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Add a couple of cheap, useful control features:
    - vol_20d: rolling std of ret_1d per symbol
    - log_dollar_volume: log(1 + dollar_volume)
    """
    df = df_prices.sort_values(["symbol", "date"]).copy()

    df["vol_20d"] = (
        df.groupby("symbol")["ret_1d"]
          .rolling(20, min_periods=10)
          .std()
          .reset_index(level=0, drop=True)
    )

    df["log_dollar_volume"] = np.log1p(df["dollar_volume"])

    return df


def build_model_dataset(
    market_df: pd.DataFrame,
    diffusion_df: pd.DataFrame,
    tda_df: pd.DataFrame,
    corr_window: int = 120,
    alpha: float = 1.0,
    signal_lookback: int = 5
) -> pd.DataFrame:
    """
    Build the modeling dataset for a specific corr_window.

    market_df: Step 2 output (with labels)
    diffusion_df: Step 4 output
    tda_df: Step 5 output

    Returns a leakage-safe dataset keyed by (date, symbol).
    """
    # --- Filter diffusion to the chosen regime config ---
    resid_col = f"signal_resid_alpha{alpha}_w{corr_window}"
    x_col = f"signal_x_{signal_lookback}d"
    x_smooth_col = f"signal_x_smooth_alpha{alpha}"

    dff = diffusion_df[diffusion_df["corr_window"] == corr_window].copy()

    # Keep only relevant cols (and rename to simple names)
    dff = dff[["date", "symbol", resid_col, x_col, x_smooth_col]].rename(columns={
        resid_col: "diff_resid",
        x_col: "signal_5d",
        x_smooth_col: "signal_5d_smooth"
    })

    # --- Filter TDA features to chosen window ---
    tdf = tda_df[tda_df["corr_window"] == corr_window].copy()

    # Ensure date is datetime and unique per date/window
    tdf["date"] = pd.to_datetime(tdf["date"])
    tdf = tdf.drop(columns=["corr_window"])

    # --- Add basic control features to market_df ---
    mkt = add_basic_controls(market_df)

    # Keep relevant columns
    keep_cols = [
        "date", "symbol", "ret_1d", "adj_close", "dollar_volume",
        "vol_20d", "log_dollar_volume",
        "fwd_ret_5", "fwd_ret_10", "fwd_ret_20"
    ]
    mkt = mkt[keep_cols].copy()
    mkt["date"] = pd.to_datetime(mkt["date"])

    # --- Merge: market + diffusion on (date, symbol) ---
    out = mkt.merge(dff, on=["date", "symbol"], how="inner")

    # --- Merge: add TDA regime features on date ---
    out = out.merge(tdf, on="date", how="inner")

    # Drop rows with missing values (vol_20d starts later, etc.)
    out = out.dropna().sort_values(["date", "symbol"]).reset_index(drop=True)

    # Add metadata columns (useful later)
    out["corr_window"] = corr_window
    out["alpha"] = alpha
    out["signal_lookback"] = signal_lookback

    return out