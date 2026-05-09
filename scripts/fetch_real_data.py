"""Populate data/{funding,yields,ohlcv,preipo} with real-data caches.

Run this on a machine with network access (your laptop / VPS), then commit
the parquet files (or .gitignore them and re-fetch on each Phase-2 cycle).
This script is idempotent — already-cached files are skipped unless
`--force` is passed.

Examples:
    # Default: 90 days of Hyperliquid BTC funding + DefiLlama L2 USDC yields
    python scripts/fetch_real_data.py

    # Just S3 — Binance vs Bybit BTC 1m for the last 14 days
    python scripts/fetch_real_data.py --target s3 --days 14

    # Refresh everything (overwrite cache)
    python scripts/fetch_real_data.py --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch")


def fetch_s1(days: int, force: bool) -> None:
    """Hyperliquid funding history for BTC and ETH."""
    from auto_investment.data_fetchers import funding as fdg
    until = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    since = until - timedelta(days=days)
    for sym in ("BTC/USDC:USDC", "ETH/USDC:USDC"):
        path = fdg.cache_path("hyperliquid", sym, since, until)
        if path.exists() and not force:
            log.info("[skip] funding cache present: %s", path)
            continue
        log.info("Fetching Hyperliquid funding for %s …", sym)
        df = fdg.fetch_funding_rate_history("hyperliquid", sym, since=since, until=until)
        fdg.save(df, path)


def fetch_s2(force: bool) -> None:
    """DefiLlama pool universe + 30-day APY history per pool."""
    from auto_investment.data_fetchers import yields as yld
    log.info("Fetching DefiLlama pool universe …")
    universe = yld.fetch_pool_universe()
    yld.save_universe(universe)
    log.info("Universe contains %d pools after Phase-1 filters.", len(universe))
    for pid in universe["pool_id"]:
        chart_path = yld.DEFAULT_CACHE_DIR / f"chart_{pid.replace('/', '-')}.parquet"
        if chart_path.exists() and not force:
            log.info("[skip] chart cache present: %s", chart_path)
            continue
        log.info("Fetching pool chart %s …", pid)
        try:
            chart = yld.fetch_pool_chart(pid)
            if not chart.empty:
                yld.save_chart(pid, chart)
        except Exception as exc:  # noqa: BLE001 — keep going on per-pool failure
            log.warning("Failed pool %s: %s", pid, exc)


def fetch_s3(days: int, force: bool, symbol: str = "BTC/USDT") -> None:
    """Paired 1-minute OHLCV from Binance + Bybit."""
    from auto_investment.data_fetchers import ohlcv as oh
    until = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    since = until - timedelta(days=days)
    for venue in ("binance", "bybit"):
        path = oh.cache_path(venue, symbol, "1m", since, until)
        if path.exists() and not force:
            log.info("[skip] OHLCV cache present: %s", path)
            continue
        log.info("Fetching %s %s 1m bars …", venue, symbol)
        df = oh.fetch_ohlcv(venue, symbol, "1m", since=since, until=until)
        oh.save(df, path)


def fetch_s4(force: bool) -> None:
    """Aevo Pre-Markets daily marks for our watchlist."""
    from auto_investment.data_fetchers import preipo as pi
    try:
        instruments = pi.fetch_instruments()
    except Exception as exc:  # noqa: BLE001
        log.warning("Aevo instruments fetch failed: %s", exc)
        return
    log.info("Aevo watchlist hits: %d instruments", len(instruments))
    for it in instruments:
        sym = it.get("instrument_name") or it.get("symbol") or "UNKNOWN"
        path = pi.DEFAULT_CACHE_DIR / f"{sym.lower()}.parquet"
        if path.exists() and not force:
            log.info("[skip] pre-IPO cache present: %s", path)
            continue
        try:
            df = pi.fetch_index_history(sym)
            if not df.empty:
                pi.save_marks(sym, df)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed pre-IPO %s: %s", sym, exc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["all", "s1", "s2", "s3", "s4"], default="all")
    ap.add_argument("--days", type=int, default=90,
                    help="History window; applies to S1 funding and S3 OHLCV")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing cache entries")
    args = ap.parse_args()

    if args.target in ("all", "s1"):
        fetch_s1(args.days, args.force)
    if args.target in ("all", "s2"):
        fetch_s2(args.force)
    if args.target in ("all", "s3"):
        fetch_s3(min(args.days, 14), args.force)  # 14d × 2 venues × 1m is plenty
    if args.target in ("all", "s4"):
        fetch_s4(args.force)
    log.info("Done. Cache lives under %s/data/", REPO_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
