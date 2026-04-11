"""Macroeconomic data fetcher — Federal Reserve via fredapi.

Replaces the $25k/year Bloomberg Terminal with a free FRED API key. Pulls a
configurable basket of macro indicators (10Y yield, VIX, DXY, Fed funds rate,
unemployment) and feeds the latest values into the Claude advisor as context.

Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html and
set FRED_API_KEY in `.env`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from .config import settings

logger = logging.getLogger(__name__)

# Default basket of macro series IDs. Tweak via the `series` argument.
# All are daily-frequency series so they update every business day.
DEFAULT_SERIES: dict[str, str] = {
    "DGS10": "10-Year Treasury Yield (%)",
    "DGS2": "2-Year Treasury Yield (%)",
    "VIXCLS": "VIX (CBOE volatility index)",
    "DTWEXBGS": "USD broad index (DXY proxy)",
    "DFF": "Effective Fed Funds Rate (%)",
}


def fetch_macro_snapshot(series: dict[str, str] | None = None) -> dict | None:
    """Pull the latest value of each FRED series and return a flat dict.

    Returns None if FRED_API_KEY is missing or the call fails. The caller
    should handle None — the AI advisor will simply skip the macro section.
    """
    if not settings.fred_api_key:
        logger.debug("FRED disabled (no FRED_API_KEY)")
        return None

    series_map = series or DEFAULT_SERIES
    try:
        from fredapi import Fred  # noqa: PLC0415
    except ImportError:
        logger.warning("fredapi not installed; pip install fredapi")
        return None

    try:
        fred = Fred(api_key=settings.fred_api_key)
        snapshot: dict = {}
        # Look back ~14 days to be safe with weekends/holidays
        start = (datetime.utcnow() - timedelta(days=14)).strftime("%Y-%m-%d")
        for series_id, label in series_map.items():
            try:
                s = fred.get_series(series_id, observation_start=start)
                s = s.dropna()
                if len(s) == 0:
                    continue
                latest = float(s.iloc[-1])
                snapshot[label] = round(latest, 4)
            except Exception as exc:  # noqa: BLE001
                logger.debug("FRED series %s failed: %s", series_id, exc)
        return snapshot or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("FRED snapshot failed: %s", exc)
        return None


def macro_to_prompt_line(snapshot: dict) -> str:
    """Render the snapshot as a single human-readable line."""
    if not snapshot:
        return "No macro data."
    return ", ".join(f"{k}={v}" for k, v in snapshot.items())
