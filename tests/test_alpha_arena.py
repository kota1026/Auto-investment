"""Tests for the Alpha Arena contest loop and supporting modules."""

from __future__ import annotations

import pandas as pd

from auto_investment.alpha_arena import (
    ContestResult,
    _compute_risk_stats,
    EquityPoint,
    run_contest,
)
from auto_investment.decision_agent import (
    DecisionBatch,
    TradeDecision,
    _heuristic_decide,
)
from auto_investment.multi_market import (
    ALPHA_ARENA_UNIVERSE,
    make_snapshot,
    synthetic_multi,
)
from auto_investment.perp_sim import PerpAccount, PerpSimulator


def test_synthetic_multi_returns_aligned_frames():
    frames = synthetic_multi(symbols=["BTC", "ETH", "SOL"], limit=200)
    assert set(frames.keys()) == {"BTC", "ETH", "SOL"}
    for sym, df in frames.items():
        assert len(df) == 200
        assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
    # All should share the same index (correlated GBM design)
    btc_idx = frames["BTC"].index
    for df in frames.values():
        assert (df.index == btc_idx).all()


def test_synthetic_multi_correlations_realistic():
    """ETH and SOL should be highly but not perfectly correlated with BTC."""
    frames = synthetic_multi(symbols=["BTC", "ETH", "SOL"], limit=500)
    btc_ret = frames["BTC"]["close"].pct_change().dropna()
    eth_ret = frames["ETH"]["close"].pct_change().dropna()
    sol_ret = frames["SOL"]["close"].pct_change().dropna()
    eth_corr = btc_ret.corr(eth_ret)
    sol_corr = btc_ret.corr(sol_ret)
    # Per the generator design: ETH ~0.88, SOL ~0.78
    assert 0.5 < eth_corr < 0.99
    assert 0.4 < sol_corr < 0.99


def test_make_snapshot_has_all_universe():
    frames = synthetic_multi(limit=200)
    snap = make_snapshot(frames, step_index=100, history_window=24)
    for sym in ALPHA_ARENA_UNIVERSE:
        assert sym in snap.prices
        assert sym in snap.returns_24h
        assert sym in snap.volatility_24h
        assert len(snap.recent_closes[sym]) > 0


def test_snapshot_to_prompt_block_includes_all_symbols():
    frames = synthetic_multi(limit=200)
    snap = make_snapshot(frames, step_index=100)
    block = snap.to_prompt_block()
    for sym in ALPHA_ARENA_UNIVERSE:
        assert sym in block


def test_heuristic_decide_returns_one_per_universe():
    frames = synthetic_multi(limit=200)
    snap = make_snapshot(frames, step_index=100)
    account = PerpAccount(cash=10_000.0)
    batch = _heuristic_decide(snap, account, ALPHA_ARENA_UNIVERSE)
    assert isinstance(batch, DecisionBatch)
    assert len(batch.decisions) == len(ALPHA_ARENA_UNIVERSE)
    for d in batch.decisions:
        assert d.symbol in ALPHA_ARENA_UNIVERSE
        assert 0.0 <= d.confidence <= 1.0


def test_heuristic_closes_loser():
    """If a position is down >2%, the heuristic should close it."""
    frames = synthetic_multi(limit=200)
    snap = make_snapshot(frames, step_index=100)
    sim = PerpSimulator(starting_capital=10_000.0, taker_fee=0.0, slippage_bps_per_10k=0.0)
    # Open at a high price, then snapshot is at a much lower price
    btc_price = snap.prices["BTC"]
    sim.open_position("BTC", "long", notional_usd=1000.0, leverage=10.0, current_price=btc_price * 1.10)
    batch = _heuristic_decide(snap, sim.account, ["BTC"])
    btc_decision = next(d for d in batch.decisions if d.symbol == "BTC")
    assert btc_decision.action == "close"


def test_run_contest_smoke_synthetic_no_ai():
    """End-to-end smoke test using heuristic on synthetic data."""
    result = run_contest(
        starting_capital=10_000.0,
        duration_days=10,
        decision_interval_hours=4,
        use_real_data=False,
        use_ai=False,
        seed=42,
    )
    assert isinstance(result, ContestResult)
    assert result.starting_capital == 10_000.0
    assert result.final_equity > 0
    assert result.n_decisions > 0
    assert len(result.equity_curve) > 0
    # Sharpe should be a finite number (could be positive or negative)
    assert result.sharpe == result.sharpe  # not NaN


def test_contest_result_to_dict_serializes():
    result = run_contest(
        starting_capital=1_000.0, duration_days=5, use_ai=False, seed=1
    )
    d = result.to_dict()
    assert "final_equity" in d
    assert "sharpe" in d
    assert "equity_curve" in d
    assert "trade_log" in d
    assert "alpha_vs_benchmark" in d


def test_compute_risk_stats_handles_empty_curve():
    sharpe, sortino, max_dd = _compute_risk_stats([])
    assert sharpe == 0.0
    assert sortino == 0.0
    assert max_dd == 0.0


def test_compute_risk_stats_positive_curve():
    """A monotonically increasing equity curve should have positive Sharpe."""
    ts = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
    curve = [
        EquityPoint(timestamp=t, equity=10_000.0 + i * 10, cash=0, n_positions=0)
        for i, t in enumerate(ts)
    ]
    sharpe, _, max_dd = _compute_risk_stats(curve)
    assert sharpe > 0
    assert max_dd == 0.0  # no drawdown


def test_run_contest_30_day_synthetic_finishes():
    """Longer run should still complete and produce coherent stats."""
    result = run_contest(
        starting_capital=30.0,
        duration_days=30,
        decision_interval_hours=4,
        use_ai=False,
        seed=7,
    )
    assert result.final_equity >= 0
    assert result.n_decisions > 100  # 30 days * 24h / 4h = 180 decisions
    assert isinstance(result.benchmark_return_pct, float)
