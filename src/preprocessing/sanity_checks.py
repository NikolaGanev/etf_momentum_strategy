import pandas as pd


def run_sanity_checks(df: pd.DataFrame) -> None:
    print("\n=== SANITY CHECK REPORT ===")
    print(f"Rows: {len(df):,}")
    print(f"Symbols: {df['symbol'].nunique()}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")

    print("\nMissing values (%):")
    print(df.isna().mean()[df.isna().mean() > 0])

    print("\n1-day return stats:")
    print(df["ret_1d"].describe())

    print("\nDollar volume stats:")
    print(df["dollar_volume"].describe())

    print("\n===========================\n")