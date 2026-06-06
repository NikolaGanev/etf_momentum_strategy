from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv


@dataclass(frozen=True)
class LiveConfig:
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str

    # Safety settings
    max_order_notional_usd: float = 10_000.0
    min_trade_notional_usd: float = 50.0
    dry_run: bool = False  # start with True


def project_root() -> Path:
    # src/live/config.py -> src/live -> src -> project root
    return Path(__file__).resolve().parents[2]


def load_config() -> LiveConfig:
    env_path = project_root() / "configs" / "secrets.env"

    if not env_path.exists():
        raise FileNotFoundError(f"Missing secrets file at: {env_path}")

    load_dotenv(dotenv_path=env_path, override=True)

    key = os.getenv("ALPACA_API_KEY", "").strip()
    sec = os.getenv("ALPACA_SECRET_KEY", "").strip()
    url = os.getenv("ALPACA_BASE_URL", "").strip()

    if not key or not sec or not url:
        raise ValueError(
            "Missing Alpaca env vars. Check configs/secrets.env has:\n"
            "ALPACA_API_KEY=...\nALPACA_SECRET_KEY=...\nALPACA_BASE_URL=..."
        )

    return LiveConfig(
        alpaca_api_key=key,
        alpaca_secret_key=sec,
        alpaca_base_url=url,
    )