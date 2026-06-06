from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


@dataclass
class Position:
    symbol: str
    qty: float


class AlpacaBroker:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.client = TradingClient(api_key, secret_key, paper=paper)

    def get_equity(self) -> float:
        acct = self.client.get_account()
        return float(acct.equity)

    def get_positions(self) -> Dict[str, Position]:
        out: Dict[str, Position] = {}
        for p in self.client.get_all_positions():
            out[p.symbol] = Position(symbol=p.symbol, qty=float(p.qty))
        return out

    def is_shortable(self, symbol: str) -> bool:
        """
        Ask Alpaca whether an asset is shortable.
        """
        asset = self.client.get_asset(symbol)
        # asset.shortable exists in Alpaca asset model
        return bool(getattr(asset, "shortable", False))

    def submit_market_order(self, symbol: str, qty: float, side: str) -> None:
        if qty <= 0:
            return

        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side_enum,
            time_in_force=TimeInForce.DAY
        )
        self.client.submit_order(req)