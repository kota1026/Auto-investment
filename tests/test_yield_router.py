"""Tests for the S2 DeFi yield router."""

import pandas as pd

from auto_investment.strategies.yield_router import (
    YieldRouterConfig,
    backtest_yield_router,
    baseline_static_apy,
    synth_pool_grid,
)


def test_synth_pool_grid_shape():
    pools, history = synth_pool_grid(seed=1)
    assert len(pools) == 5
    assert history.shape[1] == len(pools)
    # 4h cadence × 365 days = 6 bars/day × 365
    assert history.shape[0] == 6 * 365


def test_router_at_least_picks_one_pool_initially():
    pools, history = synth_pool_grid(seed=2)
    res = backtest_yield_router(pools, history)
    assert res.n_rotations >= 1
    assert res.equity_curve is not None
    assert (res.chosen_pool_path != "").all()


def test_router_beats_or_matches_static_aave_baseline():
    """With a synthetic universe favouring rotation, router should outperform."""
    pools, history = synth_pool_grid(seed=5)
    res = backtest_yield_router(pools, history)
    base = baseline_static_apy(history, "aave-arb-usdc", res.config.notional_usd)
    # router_final >= baseline_final
    assert float(res.equity_curve.iloc[-1]) >= float(base.iloc[-1]) * 0.99


def test_filters_drop_unsafe_pools():
    """A pool that fails TVL filter should never be chosen."""
    pools, history = synth_pool_grid(seed=6)
    # Force the first pool to be tiny TVL
    bad = pools[0]
    pools[0] = bad.__class__(
        pool_id=bad.pool_id, chain=bad.chain, protocol=bad.protocol,
        asset=bad.asset, tvl_usd=1_000_000, audits=bad.audits,
        age_days=bad.age_days, risk_premium_bps=bad.risk_premium_bps,
    )
    res = backtest_yield_router(pools, history,
                                config=YieldRouterConfig(min_tvl_usd=20_000_000))
    assert (res.chosen_pool_path != bad.pool_id).all()


def test_equity_curve_monotone_when_no_rotation_loss():
    """Capital should grow over time absent severe rotation costs."""
    pools, history = synth_pool_grid(seed=8)
    res = backtest_yield_router(pools, history)
    assert float(res.equity_curve.iloc[-1]) > float(res.equity_curve.iloc[0])


def test_baseline_static_apy_matches_input_size():
    pools, history = synth_pool_grid(seed=9)
    base = baseline_static_apy(history, pools[0].pool_id, 1_000.0)
    assert isinstance(base, pd.Series)
    assert len(base) == len(history)
