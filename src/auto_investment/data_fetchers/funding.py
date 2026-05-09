"""Hyperliquid (and CEX-perp) funding-rate history fetcher.

Hyperliquid pays funding **hourly**. Their REST endpoint is exposed both
through ccxt (`fetch_funding_rate_history`) and directly at
`https://api.hyperliquid.xyz/info` (POST type=fundingHistory).

We use ccxt for portability — same code works against Binance/Bybit perps
when we add them as a redundancy in Phase 3.

Cache layout:
    data/funding/<exchange>_<symbol>_<since>_<until>.parquet
    columns: ts (UTC), funding_rate (fraction), mark_price (optional)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/funding")


def _safe_filename(s: str) -> str:
    return s.replace("/", "-").replace(":", "_")


def cache_path(
    exchange: str, symbol: str, since: datetime, until: datetime,
    base: Path = DEFAULT_CACHE_DIR,
) -> Path:
    name = (
        f"{exchange}_{_safe_filename(symbol)}"
        f"_{since.strftime('%Y%m%d')}_{until.strftime('%Y%m%d')}.parquet"
    )
    return base / name


def fetch_funding_rate_history(
    exchange: str = "hyperliquid",
    symbol: str = "BTC/USDC:USDC",
    since: datetime | None = None,
    until: datetime | None = None,
    page_limit: int = 500,
    sleep_s: float = 0.25,
) -> pd.DataFrame:
    """Pull funding history from a perp venue via ccxt.

    Hyperliquid returns at most ~500 entries per call; we paginate by
    advancing `since` until we reach `until` (or the API stops returning
    new data). Returns a tz-aware DataFrame indexed by `ts` with a
    `funding_rate` column (fractional, e.g. 0.000125 = 1.25 bps/h).
    """
    import ccxt  # noqa: PLC0415

    until = until or datetime.now(timezone.utc)
    since = since or (until - timedelta(days=90))

    ex_class = getattr(ccxt, exchange)
    ex = ex_class({"enableRateLimit": True})

    rows: list[dict] = []
    cursor_ms = int(since.timestamp() * 1000)
    until_ms = int(until.timestamp() * 1000)

    while cursor_ms < until_ms:
        batch = ex.fetch_funding_rate_history(symbol, since=cursor_ms, limit=page_limit)
        if not batch:
            break
        for entry in batch:
            ts_ms = entry.get("timestamp")
            if ts_ms is None or ts_ms >= until_ms:
                continue
            rows.append({
                "ts": ts_ms,
                "funding_rate": float(entry.get("fundingRate") or 0.0),
            })
        last_ts = batch[-1].get("timestamp")
        if last_ts is None or last_ts <= cursor_ms:
            break
        cursor_ms = last_ts + 1
        time.sleep(sleep_s)

    if not rows:
        raise RuntimeError(f"No funding rows returned for {exchange} {symbol}")

    df = pd.DataFrame(rows).drop_duplicates("ts").sort_values("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    return df


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.index.name != "ts":
        df = df.set_index("ts")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_or_fetch(
    exchange: str = "hyperliquid",
    symbol: str = "BTC/USDC:USDC",
    since: datetime | None = None,
    until: datetime | None = None,
    base: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Cache-first loader. Falls back to synthetic if no cache and no network."""
    until = until or datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    since = since or (until - timedelta(days=90))
    path = cache_path(exchange, symbol, since, until, base=base)
    if path.exists():
        logger.info("Loading funding from cache: %s", path)
        return load(path)
    logger.warning(
        "No funding cache at %s. Run scripts/fetch_real_data.py to populate.", path
    )
    raise FileNotFoundError(path)


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Wrote funding cache: %s (%d rows)", path, len(df))
