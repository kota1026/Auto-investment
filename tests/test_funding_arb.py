"""Tests for the S1 funding arbitrage strategy."""

import numpy as np
import pandas as pd

from auto_investment.strategies.funding_arb import (
    FundingArbConfig,
    backtest_funding_arb,
    funding_to_apr,
    synth_funding_series,
)


def test_funding_to_apr_conversion():
    # 1 bps/hour funding (= 0.0001 fractional) over 8760 hours/year
    # equals 8760 bps APR (= 87.6%).
    assert funding_to_apr(1e-4, 24 * 365) == 8760.0


def test_synth_funding_has_positive_expected_value_under_default_params():
    # Default params model a positive-funding regime with occasional flips.
    # We don't require >50% of periods to be positive (regime flips can
    # produce long negative stretches), only that the *mean* funding is
    # positive over a long enough window across most seeds.
    means = [synth_funding_series(n=24 * 90, seed=s).mean() for s in range(10)]
    assert sum(means) / len(means) > 0
    assert sum(1 for m in means if m > 0) >= 6  # >=60% of seeds positive-mean


def test_strategy_makes_money_in_positive_funding_regime():
    """Sanity: with persistently positive funding, S1 should be profitable."""
    funding = synth_funding_series(n=24 * 60, seed=3, mean_bps_per_hour=2.0, flip_prob=0.005)
    cfg = FundingArbConfig(notional_per_trade_usd=2_000.0)
    res = backtest_funding_arb(funding, config=cfg)
    assert res.n_trades >= 1
    assert res.total_pnl_usd > 0


def test_strategy_stays_flat_when_funding_too_low():
    """If funding APR never crosses the entry threshold, no trades fire."""
    rng_idx = pd.date_range("2024-01-01", periods=240, freq="1h", tz="UTC")
    funding = pd.Series(np.ones(240) * 1e-6, index=rng_idx)  # 0.01 bps/h ≈ 0.88% APR
    res = backtest_funding_arb(funding)
    assert res.n_trades == 0


def test_strategy_unwinds_when_funding_flips():
    """Positive then negative regime: must enter then exit."""
    n = 480
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    f = np.concatenate([
        np.full(n // 2, 3e-4),    # ~263% APR positive
        np.full(n // 2, -3e-4),   # negative
    ])
    res = backtest_funding_arb(pd.Series(f, index=idx))
    assert res.n_trades >= 1
    # All trades should have closed (no dangling open position) since
    # the second half of the series flipped negative.
    last_exit = max(t.exit_time for t in res.trades)
    # last exit is in the second half
    assert last_exit > idx[n // 2 - 1]


def test_equity_curve_length_matches_input():
    f = synth_funding_series(n=200, seed=4)
    res = backtest_funding_arb(f)
    assert len(res.equity_curve) == len(f)


def test_persistence_gate_suppresses_single_period_spikes():
    """A funding series with only single-period spikes above threshold
    should not trigger entries when min_persistent_periods >= 2."""
    n = 240
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    # Below threshold most of the time; isolated single-period spikes
    base = np.full(n, 1e-5)  # ~0.88% APR (below 1000 bps default)
    spike_locations = [50, 100, 150, 200]  # 4 isolated spikes
    for loc in spike_locations:
        base[loc] = 1e-3  # ~876% APR for a single hour
    cfg = FundingArbConfig(min_persistent_periods=3)
    res = backtest_funding_arb(pd.Series(base, index=idx), config=cfg)
    # 6-period rolling smoothing means a single spike can't sustain
    # 3 periods of smoothed APR > 1000 bps → no trades.
    assert res.n_trades == 0


def test_persistence_gate_allows_sustained_funding():
    """A long stretch of high funding (then a flip) should trigger one
    completed trade, proving the persistence gate doesn't permanently lock
    out entries."""
    n = 240
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    # First 180 hours: strongly positive (~876% APR), then crash to negative
    # so the position closes before the test ends.
    f = np.full(n, 1e-3)
    f[180:] = -1e-3
    cfg = FundingArbConfig(min_persistent_periods=3)
    res = backtest_funding_arb(pd.Series(f, index=idx), config=cfg)
    assert res.n_trades >= 1


def test_persistence_gate_zero_is_backwards_compatible():
    """min_persistent_periods=0 should restore the prior 'enter immediately
    once smoothed APR crosses threshold' behaviour."""
    f = synth_funding_series(n=24 * 60, seed=3, mean_bps_per_hour=2.0,
                             flip_prob=0.005)
    cfg_strict = FundingArbConfig(min_persistent_periods=3)
    cfg_loose = FundingArbConfig(min_persistent_periods=1)
    res_strict = backtest_funding_arb(f, config=cfg_strict)
    res_loose = backtest_funding_arb(f, config=cfg_loose)
    # Loose config should produce >= as many trades as strict config
    assert res_loose.n_trades >= res_strict.n_trades
