from __future__ import annotations

import math
import os
from pathlib import Path
import pandas as pd

from src.live.config import load_config
from src.live.broker_alpaca import AlpacaBroker


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_latest_target_weights(weights_path: Path) -> tuple[pd.Timestamp, pd.DataFrame]:
    w = pd.read_parquet(weights_path).copy()
    w["date"] = pd.to_datetime(w["date"])
    last_day = pd.Timestamp(w["date"].max())
    w = w[w["date"] == last_day].copy()
    # Keep all rows as provided; we will treat missing symbols as 0 later
    return last_day, w[["symbol", "weight"]].copy()


def load_latest_prices(market_path: Path) -> tuple[pd.Timestamp, pd.Series]:
    m = pd.read_parquet(market_path).copy()
    m["date"] = pd.to_datetime(m["date"])
    last_day = pd.Timestamp(m["date"].max())
    m = m[m["date"] == last_day].copy()
    return last_day, m.set_index("symbol")["adj_close"].astype(float)


def load_market_calendar(market_path: Path) -> pd.DatetimeIndex:
    m = pd.read_parquet(market_path, columns=["date"]).copy()
    d = pd.to_datetime(m["date"].unique())
    return pd.DatetimeIndex(sorted(d))


def load_last_trade_date(path: Path) -> pd.Timestamp | None:
    if not path.exists():
        return None
    txt = path.read_text().strip()
    if not txt:
        return None
    try:
        return pd.Timestamp(txt)
    except Exception:
        return None


def save_last_trade_date(path: Path, date: pd.Timestamp) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pd.Timestamp(date).date()))


def is_rebalance_date(weights_date: pd.Timestamp, market_dates: pd.DatetimeIndex, rebalance_every: int) -> bool:
    market_dates = pd.to_datetime(market_dates).sort_values()
    if weights_date not in market_dates:
        return False
    idx = int(market_dates.get_loc(weights_date))
    return (idx % rebalance_every) == 0


def round_buy_qty(qty: float) -> float:
    # allow fractional buys
    return max(0.0, math.floor(qty * 10000) / 10000.0)


def round_sell_qty_for_long(qty: float, current_long: float) -> float:
    # selling a long can be fractional, but cannot exceed current long in this leg
    qty = min(qty, current_long)
    return max(0.0, math.floor(qty * 10000) / 10000.0)


def round_sell_qty_for_short(qty: float) -> float:
    # opening/increasing a short must be integer shares
    return float(max(0, int(math.floor(qty))))


def main():
    cfg = load_config()
    root = project_root()

    weights_path = root / "data" / "processed" / "weights_ts_mom.parquet"
    market_path = root / "data" / "processed" / "market_data_live.parquet"
    last_trade_path = root / "data" / "processed" / "last_trade_date.txt"

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")
    if not market_path.exists():
        raise FileNotFoundError(f"Missing market file: {market_path}")

    weights_date, w = load_latest_target_weights(weights_path)
    market_date, px = load_latest_prices(market_path)
    mkt_dates = load_market_calendar(market_path)

    force_trade = os.getenv("FORCE_TRADE", "0").strip() == "1"

    # Gate 1: refuse stale weights vs market (unless forced)
    if weights_date.date() != market_date.date() and not force_trade:
        print("\n=== PAPER TRADING RUN (GATED) ===")
        print(f"Latest market date: {market_date.date()} | Latest weights date: {weights_date.date()}")
        print("Refusing to trade stale weights. (Set FORCE_TRADE=1 to override.)\n")
        return

    # Gate 2: rebalance schedule
    if not is_rebalance_date(weights_date, mkt_dates, rebalance_every=5) and not force_trade:
        print("\n=== PAPER TRADING RUN (GATED) ===")
        print(f"Latest weights date {weights_date.date()} is NOT a rebalance date (every 5 trading days).")
        print("No trades placed. (Set FORCE_TRADE=1 to override.)\n")
        return

    # Gate 3: don't trade twice
    last_traded = load_last_trade_date(last_trade_path)
    if last_traded is not None and last_traded.date() == weights_date.date() and not force_trade:
        print("\n=== PAPER TRADING RUN (GATED) ===")
        print(f"Already traded for weights date {weights_date.date()}.")
        print("No trades placed. (Set FORCE_TRADE=1 to override.)\n")
        return

    broker = AlpacaBroker(cfg.alpaca_api_key, cfg.alpaca_secret_key, paper=True)
    equity = float(broker.get_equity())
    positions = broker.get_positions()  # dict symbol -> position object with .qty
    current_symbols = set(positions.keys())

    # Build full target symbol set:
    # - anything in weights
    # - anything currently held (so we can close if target=0)
    target_symbols = set(w["symbol"].astype(str).tolist()) | current_symbols

    # Build target weights map (missing => 0)
    w_map = {str(s): float(v) for s, v in zip(w["symbol"], w["weight"])}

    # Build a table for all symbols we might trade
    rows = []
    for sym in sorted(target_symbols):
        price = float(px.get(sym, float("nan")))
        if not math.isfinite(price) or price <= 0:
            continue  # can't trade without a price
        weight = float(w_map.get(sym, 0.0))
        current = float(positions[sym].qty) if sym in positions else 0.0

        target_notional = weight * equity
        target_shares = target_notional / price
        rows.append((sym, weight, price, current, target_shares))

    t = pd.DataFrame(rows, columns=["symbol", "weight", "price", "current_shares", "target_shares"])

    orders = []
    skipped_non_shortable = []

    # Build orders symbol-by-symbol with flip handling
    for _, r in t.iterrows():
        sym = str(r["symbol"])
        price = float(r["price"])
        current = float(r["current_shares"])
        target = float(r["target_shares"])

        delta = target - current
        if abs(delta) < 1e-6:
            continue

        # Notional threshold
        if abs(delta) * price < cfg.min_trade_notional_usd:
            continue

        # Case 1: need to increase (buy)
        if delta > 0:
            qty = min(delta, cfg.max_order_notional_usd / price)
            qty = round_buy_qty(qty)
            if qty * price < cfg.min_trade_notional_usd:
                continue
            orders.append((sym, "buy", qty, current, target))
            continue

        # delta < 0 => selling / reducing
        sell_needed = abs(delta)

        # If target >= 0: we only need to reduce a long (or sell if already long)
        if target >= 0:
            if current <= 0:
                # already flat/short; if target>=0 and current<0 you'd need buy-to-cover,
                # but this case is handled by delta>0 above (because target - current > 0)
                continue
            qty = min(sell_needed, cfg.max_order_notional_usd / price)
            qty = round_sell_qty_for_long(qty, current_long=current)
            if qty * price < cfg.min_trade_notional_usd:
                continue
            orders.append((sym, "sell", qty, current, target))
            continue

        # target < 0: we want to be short
        # If currently long: do a two-leg flip: sell-to-flat (fractional), then short (integer)
        if current > 0:
            # Leg A: sell current long to 0
            qty_flat = round_sell_qty_for_long(current, current_long=current)
            if qty_flat * price >= cfg.min_trade_notional_usd:
                orders.append((sym, "sell", qty_flat, current, 0.0))

            # Leg B: open short to reach target (<0)
            short_shares = abs(target)
            short_shares = min(short_shares, cfg.max_order_notional_usd / price)
            short_qty = round_sell_qty_for_short(short_shares)
            if short_qty <= 0:
                continue
            if not broker.is_shortable(sym):
                skipped_non_shortable.append(sym)
                continue
            if short_qty * price < cfg.min_trade_notional_usd:
                continue
            orders.append((sym, "sell", short_qty, 0.0, target))
            continue

        # If currently flat or already short: just sell more (integer)
        short_shares = sell_needed
        short_shares = min(short_shares, cfg.max_order_notional_usd / price)
        short_qty = round_sell_qty_for_short(short_shares)
        if short_qty <= 0:
            continue
        if not broker.is_shortable(sym):
            skipped_non_shortable.append(sym)
            continue
        if short_qty * price < cfg.min_trade_notional_usd:
            continue
        orders.append((sym, "sell", short_qty, current, target))

    print("\n=== PAPER TRADING RUN ===")
    print(f"Market date: {market_date.date()} | Weights date: {weights_date.date()}")
    print(f"Equity: {equity:.2f}")
    print(f"Planned orders: {len(orders)}")

    for sym, side, qty, current, tgt in orders:
        tag = "SHORT" if (side == "sell" and current <= 0 and tgt < 0) else "TRADE"
        print(f"{side.upper():4s} {sym:5s} qty={qty:.4f} ({tag})")

    if skipped_non_shortable:
        print("Skipped non-shortable shorts:", ", ".join(sorted(set(skipped_non_shortable))))

    if cfg.dry_run:
        print("\nDRY RUN enabled — no orders submitted.\n")
        return

    failures = 0
    for sym, side, qty, _, _ in orders:
        try:
            broker.submit_market_order(sym, qty, side)
        except Exception as e:
            failures += 1
            print(f"ORDER FAILED: {side.upper()} {sym} qty={qty} | {e}")

    if failures == 0:
        save_last_trade_date(last_trade_path, weights_date)
        print(f"\nSubmitted orders. Recorded last trade date: {weights_date.date()}\n")
    else:
        print(f"\nSubmitted with {failures} failures. Not updating last_trade_date.txt.\n")


if __name__ == "__main__":
    main()