"""S4 — Pre-IPO token data layer.

Aevo Pre-Markets exposes daily settlement marks for "pre-IPO" tokens
(SpaceX, Anthropic, Stripe, OpenAI, xAI). Their REST is public:

    GET https://api.aevo.xyz/instruments         → list pre-IPO instruments
    GET https://api.aevo.xyz/markets/<symbol>    → current mark
    GET https://api.aevo.xyz/index-history       → daily mark history

Whales Market on Solana also runs OTC pre-launch markets but their print
data is not REST-accessible without scraping; we leave that as Phase 3.

Per spec §3 (S4) this is **alert-only** in Phase 1: we don't auto-trade
these names, we just emit a Slack alert when our heuristics fire.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/preipo")
INSTRUMENTS_URL = "https://api.aevo.xyz/instruments"
INDEX_HISTORY_URL = "https://api.aevo.xyz/index-history"

# Names we surface alerts for. Keep the list short so alerts stay actionable.
WATCHLIST = ["SPACEX", "OPENAI", "ANTHROPIC", "STRIPE", "XAI"]


def _http_json(url: str, timeout: float = 15.0) -> dict | list:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "auto-investment/0.2",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_instruments() -> list[dict]:
    """List all pre-IPO instruments on Aevo. Filtered to our watchlist."""
    raw = _http_json(INSTRUMENTS_URL)
    items = raw if isinstance(raw, list) else raw.get("data", [])
    out = []
    for it in items:
        sym = (it.get("instrument_name") or it.get("symbol") or "").upper()
        if any(name in sym for name in WATCHLIST):
            out.append(it)
    return out


def fetch_index_history(
    instrument: str,
    since: datetime | None = None,
    until: datetime | None = None,
) -> pd.DataFrame:
    """Daily marks for one pre-IPO instrument."""
    qs = f"?instrument_name={instrument}"
    if since:
        qs += f"&start_time={int(since.timestamp())}"
    if until:
        qs += f"&end_time={int(until.timestamp())}"
    raw = _http_json(INDEX_HISTORY_URL + qs)
    items = raw["data"] if isinstance(raw, dict) else raw
    rows = []
    for it in items:
        rows.append({
            "ts": int(it["timestamp"]),
            "mark": float(it["index_price"]),
        })
    if not rows:
        return pd.DataFrame(columns=["mark"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.set_index("ts").sort_index()
    return df


def save_marks(symbol: str, df: pd.DataFrame, base: Path = DEFAULT_CACHE_DIR) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{symbol.lower()}.parquet"
    df.to_parquet(path)
    logger.info("Wrote pre-IPO cache: %s (%d marks)", path, len(df))
    return path


def load_marks(symbol: str, base: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    path = base / f"{symbol.lower()}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    return df
