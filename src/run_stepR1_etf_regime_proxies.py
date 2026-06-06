from pathlib import Path
import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def mst_total_weight(dist: np.ndarray) -> float:
    """
    Prim's algorithm MST total weight for a dense distance matrix.
    dist: (n,n) symmetric, zeros on diagonal.
    """
    n = dist.shape[0]
    in_mst = np.zeros(n, dtype=bool)
    min_edge = np.full(n, np.inf, dtype=float)

    # start at node 0
    min_edge[0] = 0.0
    total = 0.0

    for _ in range(n):
        u = int(np.argmin(np.where(in_mst, np.inf, min_edge)))
        in_mst[u] = True
        total += float(min_edge[u])

        # relax edges
        for v in range(n):
            if not in_mst[v] and dist[u, v] < min_edge[v]:
                min_edge[v] = dist[u, v]

    return total


def compute_regime_proxies(
    prices: pd.DataFrame,
    window: int = 120,
    min_assets: int = 6
) -> pd.DataFrame:
    """
    Compute simple regime proxies from ETF correlation structure:
      - avg_abs_corr: average absolute correlation (excluding diagonal)
      - mst_total_dist: MST total distance on correlation-distance graph
      - mst_mean_dist: mst_total_dist / (n-1)
      - corr_dispersion: std of off-diagonal correlations
    """
    df = prices.sort_values(["date", "symbol"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    # pivot prices -> log returns
    px = df.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    logp = np.log(px)
    rets = logp.diff().dropna()

    dates = rets.index.to_list()
    out_rows = []

    for i in range(window, len(dates) + 1):
        end_date = dates[i - 1]
        R = rets.iloc[i - window:i].dropna(axis=1, how="any")
        if R.shape[1] < min_assets:
            continue

        C = R.corr().to_numpy(dtype=float)
        n = C.shape[0]

        # off-diagonal mask
        mask = ~np.eye(n, dtype=bool)
        off = C[mask]

        avg_abs_corr = float(np.mean(np.abs(off)))
        corr_dispersion = float(np.std(off))

        # correlation distance: d = sqrt(2(1-corr))
        dist = np.sqrt(np.clip(2.0 * (1.0 - C), 0.0, None))
        np.fill_diagonal(dist, 0.0)

        mst_total = mst_total_weight(dist)
        mst_mean = mst_total / max(1, (n - 1))

        out_rows.append({
            "date": end_date,
            "corr_window": window,
            "n_assets": int(n),
            "avg_abs_corr": avg_abs_corr,
            "corr_dispersion": corr_dispersion,
            "mst_total_dist": float(mst_total),
            "mst_mean_dist": float(mst_mean),
        })

    return pd.DataFrame(out_rows)


def main():
    root = project_root()
    market_path = root / "data" / "processed" / "market_data.parquet"
    mkt = pd.read_parquet(market_path).copy()
    mkt["date"] = pd.to_datetime(mkt["date"])

    # windows: you can add 60 too, but start with 120 (stable)
    windows = [120]

    all_feats = []
    for w in windows:
        print(f"Computing ETF regime proxies for window={w} ...")
        feats = compute_regime_proxies(mkt, window=w)
        all_feats.append(feats)

    out = pd.concat(all_feats, ignore_index=True).sort_values("date").reset_index(drop=True)

    out_path = root / "data" / "processed" / "etf_regime_proxies.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)

    print("\n=== ETF REGIME PROXIES REPORT ===")
    print(f"Saved: {out_path.resolve()}")
    print(f"Rows: {len(out):,} | Date range: {out['date'].min()} → {out['date'].max()}")
    print("Columns:", list(out.columns))
    print("================================\n")


if __name__ == "__main__":
    main()