"""Technical indicators implemented with pure pandas/numpy.

No TA-Lib dependency — pip-installable on any platform.
All functions take a `pd.Series` of close prices (or OHLC for ATR) and return
a `pd.Series` aligned to the input index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (Wilder smoothing not used — standard EMA)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    Returns values in [0, 100]. Standard interpretation:
      - RSI > 70: overbought
      - RSI < 30: oversold
      - RSI > 50: bullish bias, < 50: bearish bias
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing — equivalent to EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_value = 100 - (100 / (1 + rs))
    # When avg_loss is zero, RSI is technically 100
    rsi_value = rsi_value.where(avg_loss != 0, 100.0)
    return rsi_value


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing.

    Used as a volatility proxy for stop-loss/take-profit sizing.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_indicators(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    rsi_period: int = 14,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Append EMA(fast), EMA(slow), RSI, ATR columns to an OHLCV dataframe.

    Input columns: open, high, low, close, volume (case-insensitive).
    Returns a new dataframe — does not mutate input.
    """
    out = df.copy()
    # Normalize column names
    out.columns = [c.lower() for c in out.columns]
    out["ema_fast"] = ema(out["close"], fast)
    out["ema_slow"] = ema(out["close"], slow)
    out["rsi"] = rsi(out["close"], rsi_period)
    out["atr"] = atr(out["high"], out["low"], out["close"], atr_period)
    return out
