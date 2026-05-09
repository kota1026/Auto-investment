"""Tests for the S3 cross-exchange stat-arb strategy."""

import numpy as np
import pandas as pd

from auto_investment.strategies.cross_exchange import (
    CrossExchangeConfig,
    backtest_cross_exchange,
    synth_cross_exchange_spread,
)


def test_synth_spread_shape():
    df = synth_cross_exchange_spread(n=1440, seed=1)
    assert {"close_a", "close_b", "mid", "spread", "spread_bps"} <= set(df.columns)
    assert len(df) == 1440


def test_synth_spread_is_mean_reverting():
    """OU spread should have stationary variance, not random walk."""
    df = synth_cross_exchange_spread(n=60 * 24 * 14, seed=2)
    half_a = df["spread_bps"].iloc[:len(df)//2].std()
    half_b = df["spread_bps"].iloc[len(df)//2:].std()
    # Stds across halves should be similar (within 2x); a random walk would
    # have growing variance.
    assert 0.5 < half_b / half_a < 2.0


def test_strategy_makes_trades_with_default_config():
    df = synth_cross_exchange_spread(n=60 * 24 * 7, seed=3)
    res = backtest_cross_exchange(df)
    assert res.n_trades > 0


def test_strategy_unwinds_on_revert_or_timeout():
    """Every trade must have an exit_reason in the allowed set."""
    df = synth_cross_exchange_spread(n=60 * 24 * 5, seed=4)
    res = backtest_cross_exchange(df)
    for t in res.trades:
        assert t.exit_reason in ("z_revert", "timeout")
        assert t.holding_bars >= 1


def test_high_z_threshold_reduces_trade_count():
    df = synth_cross_exchange_spread(n=60 * 24 * 7, seed=5)
    low = backtest_cross_exchange(df, config=CrossExchangeConfig(z_entry=1.5))
    high = backtest_cross_exchange(df, config=CrossExchangeConfig(z_entry=3.5))
    assert high.n_trades < low.n_trades


def test_equity_curve_length_matches_input():
    df = synth_cross_exchange_spread(n=2000, seed=6)
    res = backtest_cross_exchange(df)
    assert len(res.equity_curve) == len(df)
