"""Real market data providers — yfinance, CoinGecko, Alpha Vantage.

ccxt is great for live trading but its public OHLCV endpoints are rate-limited
and sometimes blocked by hosting providers. For backtesting we want
free, reliable, no-key sources. This module adds three:

  - yfinance     : free, no key, supports stocks AND crypto, 1h granularity
                   for ~730 days, daily for years
  - CoinGecko    : free, no key, crypto-only, hourly for last 90 days
  - Alpha Vantage: free with API key, stocks, FX, crypto, intraday

Usage:
    from auto_investment.data_providers import fetch_real
    df = fetch_real("BTC-USD", "1h", limit=1000, provider="yfinance")
    # or "auto" tries yfinance → coingecko → alpha vantage in order

Each provider returns the standard OHLCV dataframe used everywhere else
(columns: open/high/low/close/volume, tz-aware UTC index).

Why this exists separately from data.py:
  - data.py is the *trading* path (ccxt, exchange-bound, supports orders)
  - data_providers.py is the *backtesting* path (read-only, no auth)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import settings

logger = logging.getLogger(__name__)


def fetch_real(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 1000,
    provider: str = "auto",
) -> pd.DataFrame:
    """Fetch real OHLCV from a free data source.

    Args:
        symbol: For yfinance, use Yahoo tickers like "BTC-USD", "ETH-USD",
            "AAPL", "7203.T" (Toyota on TSE). For CoinGecko, pass
            "BTC/USDT" — it parses the base symbol. For Alpha Vantage, use
            their notation.
        timeframe: "1h", "4h", "1d" — providers map this to their own units.
        limit: Number of bars desired.
        provider: "yfinance" | "coingecko" | "alphavantage" | "auto".

    Raises ValueError if no provider works.
    """
    providers = (
        [provider]
        if provider != "auto"
        else ["yfinance", "coingecko", "alphavantage"]
    )

    last_error: Exception | None = None
    for p in providers:
        try:
            if p == "yfinance":
                return _fetch_yfinance(symbol, timeframe, limit)
            if p == "coingecko":
                return _fetch_coingecko(symbol, timeframe, limit)
            if p == "alphavantage":
                return _fetch_alphavantage(symbol, timeframe, limit)
            raise ValueError(f"Unknown provider: {p}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Provider %s failed: %s", p, exc)
            last_error = exc

    raise ValueError(
        f"All data providers failed for {symbol}. Last error: {last_error}"
    )


# -----------------------------------------------------------------------------
# yfinance — Yahoo Finance, free, no key, broad coverage
# -----------------------------------------------------------------------------


def _fetch_yfinance(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Fetch from yfinance. Maps our timeframes to Yahoo's intervals.

    Yahoo limits intraday history:
      - 1m: 7 days max
      - 5m / 15m / 30m: 60 days
      - 1h: 730 days
      - 1d / 1wk: years

    For "BTC/USDT" style inputs, convert to Yahoo's "BTC-USD" naming.
    """
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("yfinance not installed: pip install yfinance") from exc

    # Convert ccxt-style "BTC/USDT" → yahoo "BTC-USD"
    if "/" in symbol:
        base = symbol.split("/")[0]
        yahoo_symbol = f"{base}-USD"
    else:
        yahoo_symbol = symbol

    # Map our timeframe → yahoo interval and a sensible lookback period
    interval_map = {
        "1m": ("1m", "7d"),
        "5m": ("5m", "60d"),
        "15m": ("15m", "60d"),
        "30m": ("30m", "60d"),
        "1h": ("1h", "730d"),
        "4h": ("1h", "730d"),  # Yahoo has no 4h — fetch 1h and resample
        "1d": ("1d", "5y"),
        "1wk": ("1wk", "10y"),
    }
    if timeframe not in interval_map:
        raise ValueError(f"Unsupported timeframe for yfinance: {timeframe}")
    yf_interval, period = interval_map[timeframe]

    ticker = yf.Ticker(yahoo_symbol)
    df = ticker.history(period=period, interval=yf_interval, auto_adjust=False)
    if df.empty:
        raise ValueError(f"yfinance returned no data for {yahoo_symbol}")

    # Standardize columns
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )[["open", "high", "low", "close", "volume"]]

    # Make index UTC tz-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"

    # If user asked for 4h, resample 1h → 4h
    if timeframe == "4h":
        df = (
            df.resample("4h")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

    # Trim to last `limit` bars
    return df.tail(limit)


# -----------------------------------------------------------------------------
# CoinGecko — free crypto data, no API key, hourly granularity for ≤90 days
# -----------------------------------------------------------------------------


COINGECKO_SLUG = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "AVAX": "avalanche-2",
    "MATIC": "matic-network",
    "LINK": "chainlink",
    "DOT": "polkadot",
    "BNB": "binancecoin",
}


def _fetch_coingecko(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Fetch from CoinGecko's free public API.

    CoinGecko returns prices at varying granularity depending on `days`:
      - 1 day:   5-minute bars
      - 2-90 days: hourly bars
      - >90 days: daily bars
    """
    import httpx  # noqa: PLC0415

    base = symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()
    if base.endswith("-USD"):
        base = base[:-4]
    slug = COINGECKO_SLUG.get(base)
    if not slug:
        raise ValueError(
            f"Unknown CoinGecko symbol {base!r}. Add it to COINGECKO_SLUG."
        )

    # Choose `days` to get the granularity we want
    if timeframe in ("1m", "5m", "15m", "30m"):
        days = 1
    elif timeframe in ("1h", "4h"):
        days = min(90, max(2, limit // 24 + 5))
    else:
        days = min(365, max(30, limit + 5))

    url = (
        f"https://api.coingecko.com/api/v3/coins/{slug}/ohlc"
        f"?vs_currency=usd&days={days}"
    )
    with httpx.Client(timeout=15.0) as client:
        r = client.get(url)
        r.raise_for_status()
        ohlc = r.json()

    if not ohlc:
        raise ValueError(f"CoinGecko returned empty data for {slug}")

    df = pd.DataFrame(ohlc, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    # CoinGecko OHLC endpoint doesn't return volume — fetch separately
    vol_url = (
        f"https://api.coingecko.com/api/v3/coins/{slug}/market_chart"
        f"?vs_currency=usd&days={days}"
    )
    try:
        with httpx.Client(timeout=15.0) as client:
            v = client.get(vol_url).json()
        vols = pd.DataFrame(v.get("total_volumes", []), columns=["timestamp", "volume"])
        vols["timestamp"] = pd.to_datetime(vols["timestamp"], unit="ms", utc=True)
        vols = vols.set_index("timestamp")
        df = df.join(vols, how="left")
        df["volume"] = df["volume"].ffill().fillna(0)
    except Exception as exc:  # noqa: BLE001
        logger.debug("CoinGecko volume fetch failed: %s", exc)
        df["volume"] = 0.0

    if timeframe == "4h":
        df = (
            df.resample("4h")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
    return df.tail(limit)


# -----------------------------------------------------------------------------
# Alpha Vantage — free with API key, broader coverage
# -----------------------------------------------------------------------------


def _fetch_alphavantage(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Fetch from Alpha Vantage. Requires ALPHAVANTAGE_API_KEY."""
    api_key = getattr(settings, "alphavantage_api_key", "") or ""
    if not api_key:
        raise ValueError("ALPHAVANTAGE_API_KEY not set")

    import httpx  # noqa: PLC0415

    interval_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "60min",
    }
    if timeframe not in interval_map:
        raise ValueError(f"Unsupported timeframe for Alpha Vantage: {timeframe}")

    if "/" in symbol:
        base, quote = symbol.split("/")
        function = "CRYPTO_INTRADAY"
        params = {
            "function": function,
            "symbol": base,
            "market": quote,
            "interval": interval_map[timeframe],
            "outputsize": "full",
            "apikey": api_key,
        }
    else:
        function = "TIME_SERIES_INTRADAY"
        params = {
            "function": function,
            "symbol": symbol,
            "interval": interval_map[timeframe],
            "outputsize": "full",
            "apikey": api_key,
        }

    with httpx.Client(timeout=15.0) as client:
        r = client.get("https://www.alphavantage.co/query", params=params)
        r.raise_for_status()
        data = r.json()

    series_key = next((k for k in data if "Time Series" in k), None)
    if not series_key:
        raise ValueError(f"Alpha Vantage returned no data: {data}")

    rows = []
    for ts, vals in data[series_key].items():
        # Alpha Vantage uses different key prefixes for crypto vs stocks
        rows.append(
            {
                "timestamp": pd.Timestamp(ts, tz="UTC"),
                "open": float(vals.get("1. open") or vals.get("1a. open (USD)")),
                "high": float(vals.get("2. high") or vals.get("2a. high (USD)")),
                "low": float(vals.get("3. low") or vals.get("3a. low (USD)")),
                "close": float(vals.get("4. close") or vals.get("4a. close (USD)")),
                "volume": float(vals.get("5. volume", 0)),
            }
        )
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return df.tail(limit)
