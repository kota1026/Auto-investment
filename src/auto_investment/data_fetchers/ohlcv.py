"""CEX OHLCV fetcher used by S3 cross-exchange stat-arb.

Pulls 1m or 5m bars from two venues (default: Binance + Bybit) for the same
symbol, persists to parquet, and exposes a helper that builds a *spread
series* for the strategy.

Cache layout:
    data/ohlcv/<exchange>_<symbol>_<timeframe>_<since>_<until>.parquet
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/ohlcv")


def _safe_filename(s: str) -> str:
    return s.replace("/", "-").replace(":", "_")


def cache_path(
    exchange: str,
    symbol: str,
    timeframe: str,
    since: datetime,
    until: datetime,
    base: Path = DEFAULT_CACHE_DIR,
) -> Path:
    name = (
        f"{exchange}_{_safe_filename(symbol)}_{timeframe}"
        f"_{since.strftime('%Y%m%d')}_{until.strftime('%Y%m%d')}.parquet"
    )
    return base / name


def fetch_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str = "1m",
    since: datetime | None = None,
    until: datetime | None = None,
    page_limit: int = 1000,
    sleep_s: float = 0.2,
) -> pd.DataFrame:
    """Paginated OHLCV pull through ccxt. Returns tz-aware indexed DataFrame."""
    import ccxt  # noqa: PLC0415

    until = until or datetime.now(timezone.utc)
    since = since or (until - timedelta(days=14))
    ex = getattr(ccxt, exchange)({"enableRateLimit": True})

    rows: list[list] = []
    cursor_ms = int(since.timestamp() * 1000)
    until_ms = int(until.timestamp() * 1000)
    while cursor_ms < until_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor_ms,
                               limit=page_limit)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cursor_ms:
            break
        cursor_ms = last_ts + 1
        time.sleep(sleep_s)

    if not rows:
        raise RuntimeError(f"No OHLCV rows for {exchange} {symbol}")

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    df = df[(df.index >= since) & (df.index < until)]
    return df


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Wrote OHLCV cache: %s (%d bars)", path, len(df))


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_or_fetch(
    exchange: str,
    symbol: str,
    timeframe: str = "1m",
    since: datetime | None = None,
    until: datetime | None = None,
    base: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    until = until or datetime.now(timezone.utc).replace(second=0, microsecond=0)
    since = since or (until - timedelta(days=14))
    path = cache_path(exchange, symbol, timeframe, since, until, base=base)
    if path.exists():
        return load(path)
    raise FileNotFoundError(path)


def build_spread_series(
    venue_a: str,
    venue_b: str,
    symbol: str,
    timeframe: str = "1m",
    base: Path = DEFAULT_CACHE_DIR,
    since: datetime | None = None,
    until: datetime | None = None,
) -> pd.DataFrame:
    """Build a paired close-price series for cross-exchange spread modelling.

    Returns DataFrame with columns:
      - close_a, close_b   : per-venue close
      - spread             : close_a - close_b
      - spread_bps         : 10000 * (close_a - close_b) / mid
      - mid                : 0.5 * (close_a + close_b)
    Aligns timestamps with an inner join (drops any bar missing from either
    venue), which is appropriate because the strategy can only fire when
    both venues have a fresh tick.
    """
    a = load_or_fetch(venue_a, symbol, timeframe, since=since, until=until, base=base)
    b = load_or_fetch(venue_b, symbol, timeframe, since=since, until=until, base=base)
    df = pd.DataFrame({"close_a": a["close"], "close_b": b["close"]}).dropna()
    df["mid"] = (df["close_a"] + df["close_b"]) / 2
    df["spread"] = df["close_a"] - df["close_b"]
    df["spread_bps"] = 10_000 * df["spread"] / df["mid"]
    return df
