from __future__ import annotations

def get_etf_universe() -> list[str]:
    # Liquid, broad ETFs (free to fetch from Yahoo)
    return [
        "SPY", "QQQ", "IWM", "DIA",
        "TLT", "IEF",
        "GLD", "SLV",
        "USO",
        "UUP"
    ]