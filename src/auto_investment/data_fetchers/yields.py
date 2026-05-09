"""DefiLlama Yields fetcher for S2.

Public REST, no key required:
  - GET https://yields.llama.fi/pools         → pool universe + current APY
  - GET https://yields.llama.fi/chart/<pool>  → 30-day APY history per pool

Cache layout:
    data/yields/pools_<YYYYMMDD>.parquet     (one row per pool, snapshot)
    data/yields/chart_<pool_id>.parquet      (per-pool APY time series)

Tip: DefiLlama returns thousands of pools globally. Filter to
USDC/USDT/DAI on Arbitrum/Base/Optimism + min TVL before persisting,
otherwise the cache balloons.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/yields")
POOLS_URL = "https://yields.llama.fi/pools"
CHART_URL = "https://yields.llama.fi/chart/{pool_id}"

# Phase-1 universe filters per spec §3 (S2)
ALLOWED_CHAINS = {"Arbitrum", "Base", "Optimism"}
ALLOWED_SYMBOLS = {"USDC", "USDT", "DAI", "USDC.E"}
ALLOWED_PROJECTS = {
    "aave-v3", "compound-v3", "morpho-blue", "curve-dex",
    "pendle", "yearn-finance", "yearn-v3", "fluid-lending",
}
MIN_TVL_USD = 20_000_000


def _http_json(url: str, timeout: float = 15.0) -> dict | list:
    """Lightweight stdlib GET → JSON. Avoids extra deps in a low-budget setup.

    DefiLlama doesn't require auth but does require a real User-Agent on
    some plans/regions (otherwise returns 403).
    """
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "auto-investment/0.2 (+github.com/kota1026/Auto-investment)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_pool_universe(
    chains: set[str] = ALLOWED_CHAINS,
    symbols: set[str] = ALLOWED_SYMBOLS,
    projects: set[str] = ALLOWED_PROJECTS,
    min_tvl_usd: float = MIN_TVL_USD,
) -> pd.DataFrame:
    """Fetch the global pool list, then filter to our Phase-1 universe."""
    raw = _http_json(POOLS_URL)
    pools = raw["data"] if isinstance(raw, dict) else raw
    rows = []
    for p in pools:
        chain = p.get("chain", "")
        symbol = (p.get("symbol") or "").upper()
        project = p.get("project", "")
        tvl = float(p.get("tvlUsd") or 0)
        if (
            chain in chains
            and symbol in symbols
            and project in projects
            and tvl >= min_tvl_usd
        ):
            rows.append({
                "pool_id": p["pool"],
                "chain": chain,
                "project": project,
                "symbol": symbol,
                "tvl_usd": tvl,
                "apy_base": float(p.get("apyBase") or 0),
                "apy_reward": float(p.get("apyReward") or 0),
                "apy_total": float(p.get("apy") or 0),
                "exposure": p.get("exposure", ""),
                "audits": int(p.get("audits") or 0),
                "stablecoin": bool(p.get("stablecoin", False)),
            })
    df = pd.DataFrame(rows)
    df["snapshot_ts"] = datetime.now(timezone.utc)
    return df


def fetch_pool_chart(pool_id: str, sleep_s: float = 0.25) -> pd.DataFrame:
    """Fetch 30-day APY history for a single pool. Returns DataFrame indexed by ts."""
    raw = _http_json(CHART_URL.format(pool_id=pool_id))
    series = raw["data"] if isinstance(raw, dict) else raw
    rows = []
    for entry in series:
        rows.append({
            "ts": entry["timestamp"],
            "apy_base": float(entry.get("apyBase") or 0),
            "apy_reward": float(entry.get("apyReward") or 0),
            "apy_total": float(entry.get("apy") or 0),
            "tvl_usd": float(entry.get("tvlUsd") or 0),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    time.sleep(sleep_s)
    return df


def save_universe(df: pd.DataFrame, base: Path = DEFAULT_CACHE_DIR) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"pools_{datetime.now(timezone.utc).strftime('%Y%m%d')}.parquet"
    df.to_parquet(path)
    logger.info("Wrote yields universe cache: %s (%d pools)", path, len(df))
    return path


def save_chart(pool_id: str, df: pd.DataFrame, base: Path = DEFAULT_CACHE_DIR) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    safe = pool_id.replace("/", "-")
    path = base / f"chart_{safe}.parquet"
    df.to_parquet(path)
    return path


def load_universe_latest(base: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """Load the most recent pool snapshot from cache."""
    candidates = sorted(base.glob("pools_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No yield cache in {base}")
    return pd.read_parquet(candidates[-1])


def load_chart(pool_id: str, base: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    safe = pool_id.replace("/", "-")
    path = base / f"chart_{safe}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def build_apy_grid(
    pool_ids: list[str], base: Path = DEFAULT_CACHE_DIR
) -> pd.DataFrame:
    """Combine per-pool charts into a single time-aligned grid for the router.

    Output shape: (n_periods, n_pools), values = apy_total fraction.
    Drops timestamps where any pool is missing data, since the backtester
    expects a fully aligned grid.
    """
    frames = {}
    for pid in pool_ids:
        df = load_chart(pid, base=base)
        frames[pid] = df["apy_total"] / 100.0  # DefiLlama gives % — convert to fraction
    grid = pd.DataFrame(frames).dropna()
    return grid
