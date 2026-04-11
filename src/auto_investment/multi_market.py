"""Multi-symbol market data — synthetic + real (yfinance fallback).

The single-symbol synthetic generator in `data.py` is fine for the EMA-cross
strategy, but Alpha Arena trades a portfolio of correlated assets. We need:

  - A correlated multi-symbol generator that produces realistic cross-asset
    relationships (BTC leads, ETH/SOL follow with noise + lag)
  - A real-data path that fetches multiple symbols from yfinance and aligns
    them on a common timestamp index
  - A `MarketSnapshot` view that the decision agent can consume at any
    time-step (current + recent history per symbol)

This is the data layer for the Alpha Arena contest loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Universe definitions
# -----------------------------------------------------------------------------

# Default Alpha Arena universe — three high-volume crypto perps
ALPHA_ARENA_UNIVERSE = ["BTC", "ETH", "SOL"]

# Yahoo Finance symbol map for the real data path
YAHOO_SYMBOLS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "BNB": "BNB-USD",
    "XRP": "XRP-USD",
}


@dataclass
class MarketSnapshot:
    """A point-in-time view of all symbols in the universe.

    Used as input to the Alpha Arena decision agent. Includes the current
    price, recent close history (for indicator computation), and basic
    derived metrics so the LLM doesn't have to rederive them.

    Optional macro / news fields are attached once per contest (they're
    slow to fetch) and shared across all steps. The decision agent uses
    them to add cross-asset context to its reasoning.
    """

    timestamp: pd.Timestamp
    prices: dict[str, float]
    recent_closes: dict[str, list[float]]   # last N closes per symbol
    returns_1h: dict[str, float]            # last-hour return
    returns_24h: dict[str, float]           # last-24-hour return
    volatility_24h: dict[str, float]        # rolling 24h volatility

    # Optional cross-asset context (fetched once per contest)
    macro: Optional[dict] = None            # from FRED (DGS10, VIX, DXY, ...)
    news_summary: Optional[str] = None      # from Tavily

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "prices": self.prices,
            "returns_1h": self.returns_1h,
            "returns_24h": self.returns_24h,
            "volatility_24h": self.volatility_24h,
            "macro": self.macro,
            "news_summary": self.news_summary,
        }

    def to_prompt_block(self) -> str:
        """Human-readable rendering for the LLM decision prompt."""
        lines = [f"Timestamp: {self.timestamp.isoformat()}", ""]
        lines.append(
            f"{'Symbol':<8}{'Price':>12}{'1h':>10}{'24h':>10}{'Vol24h':>10}"
        )
        for sym in self.prices:
            lines.append(
                f"{sym:<8}{self.prices[sym]:>12,.2f}"
                f"{self.returns_1h[sym]*100:>9.2f}%"
                f"{self.returns_24h[sym]*100:>9.2f}%"
                f"{self.volatility_24h[sym]*100:>9.2f}%"
            )

        if self.macro:
            lines.append("")
            lines.append("Macro (FRED — free, replaces $24k/yr Bloomberg Terminal):")
            for k, v in self.macro.items():
                lines.append(f"  {k}: {v}")

        if self.news_summary:
            lines.append("")
            lines.append("Recent news (Tavily):")
            # Wrap long news summary to readable line length
            summary = self.news_summary.replace("\n", " ")
            for i in range(0, len(summary), 100):
                lines.append(f"  {summary[i:i+100]}")

        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Real data fetching (yfinance)
# -----------------------------------------------------------------------------


def fetch_real_multi(
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    limit: int = 1000,
) -> dict[str, pd.DataFrame]:
    """Fetch real OHLCV from yfinance for multiple symbols, aligned on time.

    Each symbol's dataframe has the standard columns. The returned dict has
    one entry per symbol; all dataframes share the same time index after
    inner-join alignment.

    Falls back to `synthetic_multi` if yfinance is unavailable or all fetches
    fail. The caller can check the source via `df.attrs.get('source')`.
    """
    symbols = symbols or ALPHA_ARENA_UNIVERSE
    try:
        from .data_providers import _fetch_yfinance  # noqa: PLC0415
    except ImportError:
        logger.warning("data_providers unavailable; using synthetic multi-market")
        return synthetic_multi(symbols=symbols, limit=limit)

    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        yahoo = YAHOO_SYMBOLS.get(sym, f"{sym}-USD")
        try:
            df = _fetch_yfinance(yahoo, timeframe, limit)
            if not df.empty:
                df.attrs["source"] = "yfinance"
                frames[sym] = df
                logger.info("Fetched %s: %d bars", sym, len(df))
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance failed for %s: %s", sym, exc)

    if not frames:
        logger.warning("All yfinance fetches failed; using synthetic")
        return synthetic_multi(symbols=symbols, limit=limit)

    # Align on common timestamps via inner join on index
    common_idx = None
    for df in frames.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)
    aligned = {sym: df.loc[common_idx].copy() for sym, df in frames.items()}
    for sym, df in aligned.items():
        df.attrs["source"] = "yfinance"
    return aligned


# -----------------------------------------------------------------------------
# Synthetic multi-market generator (correlated)
# -----------------------------------------------------------------------------


def synthetic_multi(
    symbols: list[str] | None = None,
    limit: int = 1000,
    timeframe: str = "1h",
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate correlated synthetic OHLCV for multiple symbols.

    BTC is the leader. ETH and SOL follow with high correlation (0.85, 0.75)
    plus their own idiosyncratic noise. This produces realistic cross-asset
    behavior so portfolio strategies aren't trivially decoupled.

    Starting prices are calibrated to roughly real-world levels for the
    common symbols.
    """
    symbols = symbols or ALPHA_ARENA_UNIVERSE
    n = int(limit)
    rng = np.random.default_rng(seed)

    # BTC as the "factor" — geometric Brownian motion + cycle
    t = np.arange(n)
    btc_trend = 0.06 * np.sin(t / 40) + 0.0003 * t
    btc_noise = rng.normal(0, 0.015, n).cumsum()
    btc_log_returns = btc_trend + btc_noise

    starting_prices = {
        "BTC": 30_000.0,
        "ETH": 2_000.0,
        "SOL": 100.0,
        "BNB": 350.0,
        "XRP": 0.55,
    }
    correlations = {
        "BTC": 1.00,
        "ETH": 0.88,
        "SOL": 0.78,
        "BNB": 0.82,
        "XRP": 0.65,
    }
    idiosyncratic_vol = {
        "BTC": 0.0,
        "ETH": 0.012,
        "SOL": 0.020,
        "BNB": 0.014,
        "XRP": 0.025,
    }

    delta = _timeframe_to_timedelta(timeframe)
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    index = pd.date_range(end=end, periods=n, freq=delta, tz="UTC")

    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        if sym not in starting_prices:
            logger.warning("Unknown synthetic symbol %s; using defaults", sym)
            sp = 100.0
            corr = 0.7
            iv = 0.015
        else:
            sp = starting_prices[sym]
            corr = correlations[sym]
            iv = idiosyncratic_vol[sym]

        # Mix BTC factor with idiosyncratic noise according to correlation
        idio = rng.normal(0, iv, n).cumsum()
        log_returns = corr * btc_log_returns + math.sqrt(max(0, 1 - corr * corr)) * idio
        closes = sp * np.exp(log_returns)

        opens = np.concatenate([[sp], closes[:-1]])
        spread = np.abs(rng.normal(0, 0.005, n)) * closes
        highs = np.maximum(opens, closes) + spread
        lows = np.minimum(opens, closes) - spread
        volumes = rng.uniform(100, 1000, n)

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
        df.attrs["source"] = "synthetic"
        out[sym] = df

    return out


# -----------------------------------------------------------------------------
# Snapshot construction
# -----------------------------------------------------------------------------


def make_snapshot(
    frames: dict[str, pd.DataFrame],
    step_index: int,
    history_window: int = 24,
    *,
    macro: Optional[dict] = None,
    news_summary: Optional[str] = None,
) -> MarketSnapshot:
    """Build a MarketSnapshot from aligned multi-symbol frames at a given step.

    Used by the contest loop to feed the decision agent. The decision agent
    sees the current price, recent closes, and a few derived metrics — but
    NOT future bars (no leakage).

    Optional `macro` and `news_summary` are attached verbatim — the caller
    (the contest loop) fetches these once at the start and passes them to
    every step to keep cost low.
    """
    first_frame = next(iter(frames.values()))
    timestamp = first_frame.index[step_index]

    prices: dict[str, float] = {}
    recent_closes: dict[str, list[float]] = {}
    returns_1h: dict[str, float] = {}
    returns_24h: dict[str, float] = {}
    vol_24h: dict[str, float] = {}

    for sym, df in frames.items():
        window = df.iloc[max(0, step_index - history_window) : step_index + 1]
        prices[sym] = float(window["close"].iloc[-1])
        recent_closes[sym] = window["close"].tolist()

        if len(window) >= 2:
            returns_1h[sym] = float(
                (window["close"].iloc[-1] / window["close"].iloc[-2]) - 1.0
            )
        else:
            returns_1h[sym] = 0.0

        if len(window) >= 24:
            returns_24h[sym] = float(
                (window["close"].iloc[-1] / window["close"].iloc[-24]) - 1.0
            )
            vol_24h[sym] = float(window["close"].pct_change().std() or 0.0)
        else:
            returns_24h[sym] = 0.0
            vol_24h[sym] = 0.0

    return MarketSnapshot(
        timestamp=timestamp,
        prices=prices,
        recent_closes=recent_closes,
        returns_1h=returns_1h,
        returns_24h=returns_24h,
        volatility_24h=vol_24h,
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


import math  # noqa: E402  (placed late so the file is read top-down)


def _timeframe_to_timedelta(tf: str) -> timedelta:
    unit = tf[-1].lower()
    n = int(tf[:-1])
    if unit == "m":
        return timedelta(minutes=n)
    if unit == "h":
        return timedelta(hours=n)
    if unit == "d":
        return timedelta(days=n)
    raise ValueError(f"Unsupported timeframe: {tf}")
