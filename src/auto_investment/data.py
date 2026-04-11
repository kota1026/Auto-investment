"""Market data — OHLCV fetcher built on ccxt.

Defaults to Binance testnet so this module is safe to run without real API
keys. The same code path works against any of the 100+ exchanges ccxt supports
— just change `EXCHANGE_ID`.

Includes a deterministic synthetic-data generator (`synthetic_ohlcv`) so the
backtest, server, and frontend can all run end-to-end with no network access.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .config import settings

logger = logging.getLogger(__name__)


def _make_exchange():
    """Lazy-import ccxt and return a configured exchange instance.

    Lazy because ccxt is a heavy import; we want unit tests that don't touch
    the network to skip it entirely.
    """
    import ccxt  # noqa: PLC0415

    exchange_class = getattr(ccxt, settings.exchange_id)
    exchange = exchange_class(
        {
            "apiKey": settings.exchange_api_key or None,
            "secret": settings.exchange_api_secret or None,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
    )
    if settings.exchange_testnet and hasattr(exchange, "set_sandbox_mode"):
        exchange.set_sandbox_mode(True)
        logger.info("ccxt sandbox mode enabled for %s", settings.exchange_id)
    return exchange


def fetch_ohlcv(
    symbol: str | None = None,
    timeframe: str | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch recent OHLCV bars and return a tz-aware indexed dataframe.

    Falls back to synthetic data on any network/import error so the rest of
    the system stays usable offline. The fallback is logged at WARNING.
    """
    symbol = symbol or settings.symbol
    timeframe = timeframe or settings.timeframe
    try:
        exchange = _make_exchange()
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as exc:  # noqa: BLE001 — broad: network/import/auth all fall back
        logger.warning("ccxt fetch failed (%s); falling back to synthetic data", exc)
        return synthetic_ohlcv(limit=limit, timeframe=timeframe)

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    return df


def synthetic_ohlcv(
    limit: int = 500,
    timeframe: str = "1h",
    seed: int = 42,
    start_price: float = 30_000.0,
) -> pd.DataFrame:
    """Generate deterministic OHLCV bars for offline development and tests.

    Uses a geometric Brownian motion + sinusoidal trend so EMA crossovers
    actually fire (otherwise tests for the strategy never trigger).
    """
    rng = np.random.default_rng(seed)
    n = int(limit)

    # Price path: trend (sin wave) + random walk + drift
    t = np.arange(n)
    trend = 0.05 * np.sin(t / 30) + 0.0002 * t  # gentle drift + cycle
    noise = rng.normal(0, 0.012, n).cumsum()
    log_returns = trend + noise
    closes = start_price * np.exp(log_returns)

    # Build OHLC around closes
    opens = np.concatenate([[start_price], closes[:-1]])
    spreads = np.abs(rng.normal(0, 0.004, n)) * closes
    highs = np.maximum(opens, closes) + spreads
    lows = np.minimum(opens, closes) - spreads
    volumes = rng.uniform(50, 500, n)

    # Index — UTC timestamps spaced by timeframe
    delta = _timeframe_to_timedelta(timeframe)
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    index = pd.date_range(end=end, periods=n, freq=delta, tz="UTC")

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=index,
    )
    df.index.name = "timestamp"
    return df


def _timeframe_to_timedelta(tf: str) -> timedelta:
    """Parse a ccxt-style timeframe string ('1h', '15m', '1d') to a timedelta."""
    unit = tf[-1].lower()
    n = int(tf[:-1])
    if unit == "m":
        return timedelta(minutes=n)
    if unit == "h":
        return timedelta(hours=n)
    if unit == "d":
        return timedelta(days=n)
    raise ValueError(f"Unsupported timeframe: {tf}")
