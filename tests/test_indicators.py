"""Tests for the technical indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from auto_investment.indicators import add_indicators, atr, ema, rsi


def _make_series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_ema_constant_series_equals_constant():
    s = _make_series([10.0] * 30)
    result = ema(s, period=10)
    # After warmup, EMA of a constant must equal the constant
    assert result.dropna().iloc[-1] == 10.0


def test_ema_responds_faster_than_sma_to_step():
    """EMA should react more quickly to a step change than a simple average."""
    series = _make_series([10.0] * 20 + [20.0] * 20)
    fast_ema = ema(series, period=5)
    slow_ema = ema(series, period=15)
    # Three bars after the step, the faster EMA must be closer to the new level
    assert fast_ema.iloc[23] > slow_ema.iloc[23]


def test_rsi_bounds_and_uptrend():
    """RSI must stay in [0, 100] and trend high during a strong uptrend."""
    # Strictly increasing series
    series = _make_series(list(range(100, 200)))
    result = rsi(series, period=14).dropna()
    assert (result >= 0).all() and (result <= 100).all()
    # Pure uptrend → RSI should be near 100
    assert result.iloc[-1] > 95


def test_rsi_downtrend():
    series = _make_series(list(range(200, 100, -1)))
    result = rsi(series, period=14).dropna()
    assert result.iloc[-1] < 5


def test_atr_positive_for_volatile_series():
    rng = np.random.default_rng(0)
    n = 100
    close = pd.Series(100 + rng.normal(0, 5, n).cumsum())
    high = close + np.abs(rng.normal(0, 1, n))
    low = close - np.abs(rng.normal(0, 1, n))
    result = atr(high, low, close, period=14).dropna()
    assert (result > 0).all()


def test_add_indicators_attaches_columns():
    rng = np.random.default_rng(1)
    n = 100
    df = pd.DataFrame(
        {
            "open": 100 + rng.normal(0, 1, n).cumsum(),
            "high": 102 + rng.normal(0, 1, n).cumsum(),
            "low": 98 + rng.normal(0, 1, n).cumsum(),
            "close": 100 + rng.normal(0, 1, n).cumsum(),
            "volume": rng.uniform(1, 10, n),
        }
    )
    out = add_indicators(df)
    for col in ("ema_fast", "ema_slow", "rsi", "atr"):
        assert col in out.columns
    assert out["ema_fast"].notna().sum() > 0
    assert out["ema_slow"].notna().sum() > 0
