"""Tests for the EMA-cross + RSI strategy."""

from __future__ import annotations

import pandas as pd

from auto_investment.data import synthetic_ohlcv
from auto_investment.indicators import add_indicators
from auto_investment.strategy import generate_signals, latest_signal


def test_generate_signals_only_returns_valid_labels():
    df = synthetic_ohlcv(limit=300)
    df = add_indicators(df)
    df = generate_signals(df)
    assert set(df["signal"].unique()).issubset({"long", "short", "flat"})


def test_synthetic_data_produces_at_least_one_signal():
    """The synthetic generator is tuned to ensure crossovers actually fire,
    so the strategy isn't a no-op in tests."""
    df = synthetic_ohlcv(limit=500)
    df = add_indicators(df)
    df = generate_signals(df)
    nonflat = df[df["signal"] != "flat"]
    assert len(nonflat) > 0, "Expected at least one signal in 500 synthetic bars"


def test_long_signal_requires_rsi_above_50():
    df = synthetic_ohlcv(limit=500)
    df = add_indicators(df)
    df = generate_signals(df)
    longs = df[df["signal"] == "long"]
    if len(longs):
        assert (longs["rsi"] > 50).all()


def test_short_signal_requires_rsi_below_50():
    df = synthetic_ohlcv(limit=500)
    df = add_indicators(df)
    df = generate_signals(df)
    shorts = df[df["signal"] == "short"]
    if len(shorts):
        assert (shorts["rsi"] < 50).all()


def test_latest_signal_returns_none_when_flat():
    """If we hand-craft a dataframe whose last bar is flat, latest_signal returns None."""
    df = synthetic_ohlcv(limit=300)
    df = add_indicators(df)
    df = generate_signals(df)
    df.iloc[-1, df.columns.get_loc("signal")] = "flat"
    assert latest_signal(df) is None


def test_generate_signals_raises_without_indicators():
    df = pd.DataFrame({"close": [1.0, 2.0]})
    try:
        generate_signals(df)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")
