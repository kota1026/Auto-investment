"""Tests for the Hyperliquid-style perpetual futures simulator."""

from __future__ import annotations

import pytest

from auto_investment.perp_sim import PerpAccount, PerpPosition, PerpSimulator


def test_open_long_deducts_margin_and_fee():
    sim = PerpSimulator(starting_capital=10_000.0, taker_fee=0.001)
    success, _ = sim.open_position("BTC", "long", notional_usd=1000.0, leverage=10.0, current_price=30_000.0)
    assert success
    # Margin = notional / leverage = 100, fee = notional * 0.001 = 1
    # Cash should drop by ~101
    assert 9_898 < sim.account.cash < 9_900
    assert "BTC" in sim.account.positions
    assert sim.account.fees_paid == pytest.approx(1.0)


def test_open_position_rejects_excess_leverage():
    sim = PerpSimulator(starting_capital=10_000.0, max_leverage=20.0)
    success, reason = sim.open_position("BTC", "long", 100.0, leverage=50.0, current_price=30_000.0)
    assert not success
    assert "leverage" in reason


def test_open_position_rejects_insufficient_cash():
    sim = PerpSimulator(starting_capital=100.0)
    # 10x leverage on 100k notional = 10k margin, way more than 100 cash
    success, reason = sim.open_position("BTC", "long", 100_000.0, leverage=10.0, current_price=30_000.0)
    assert not success
    assert "insufficient" in reason.lower()


def test_open_position_rejects_double_open():
    sim = PerpSimulator(starting_capital=10_000.0)
    sim.open_position("BTC", "long", 1000.0, leverage=5.0, current_price=30_000.0)
    success, reason = sim.open_position("BTC", "long", 500.0, leverage=5.0, current_price=30_000.0)
    assert not success
    assert "already open" in reason


def test_close_long_at_higher_price_realizes_profit():
    sim = PerpSimulator(starting_capital=10_000.0, taker_fee=0.0, slippage_bps_per_10k=0.0)
    sim.open_position("BTC", "long", notional_usd=1000.0, leverage=10.0, current_price=100.0)
    cash_after_open = sim.account.cash
    success, pnl = sim.close_position("BTC", current_price=110.0)
    assert success
    # 10% price move on $1000 notional = $100 PnL
    assert pnl == pytest.approx(100.0, rel=1e-2)
    # Cash returned: margin (100) + pnl (100) = 200
    assert sim.account.cash == pytest.approx(cash_after_open + 100 + 100, rel=1e-2)


def test_close_short_at_lower_price_realizes_profit():
    sim = PerpSimulator(starting_capital=10_000.0, taker_fee=0.0, slippage_bps_per_10k=0.0)
    sim.open_position("BTC", "short", notional_usd=1000.0, leverage=10.0, current_price=100.0)
    success, pnl = sim.close_position("BTC", current_price=90.0)
    assert success
    assert pnl == pytest.approx(100.0, rel=1e-2)


def test_funding_rate_long_pays_when_positive():
    sim = PerpSimulator(
        starting_capital=10_000.0,
        taker_fee=0.0,
        funding_rate_hourly=0.001,  # 0.1% per hour
        slippage_bps_per_10k=0.0,
    )
    sim.open_position("BTC", "long", notional_usd=1000.0, leverage=10.0, current_price=100.0)
    cash_before = sim.account.cash
    sim.step({"BTC": 100.0})
    # Funding payment = 1000 * 0.001 = 1
    assert sim.account.cash == pytest.approx(cash_before - 1.0, rel=1e-2)
    assert sim.account.funding_paid == pytest.approx(1.0, rel=1e-2)


def test_funding_rate_short_receives_when_positive():
    sim = PerpSimulator(
        starting_capital=10_000.0,
        taker_fee=0.0,
        funding_rate_hourly=0.001,
        slippage_bps_per_10k=0.0,
    )
    sim.open_position("BTC", "short", notional_usd=1000.0, leverage=10.0, current_price=100.0)
    cash_before = sim.account.cash
    sim.step({"BTC": 100.0})
    assert sim.account.cash == pytest.approx(cash_before + 1.0, rel=1e-2)
    assert sim.account.funding_paid == pytest.approx(-1.0, rel=1e-2)


def test_liquidation_when_margin_breached():
    """A 10x long should liquidate when the price drops more than ~10%."""
    sim = PerpSimulator(
        starting_capital=10_000.0,
        taker_fee=0.0,
        funding_rate_hourly=0.0,
        maintenance_margin=0.015,  # 1.5%
        slippage_bps_per_10k=0.0,
    )
    sim.open_position("BTC", "long", notional_usd=1000.0, leverage=10.0, current_price=100.0)
    # 12% price drop should breach margin
    liquidated = sim.step({"BTC": 88.0})
    assert "BTC" in liquidated
    assert sim.account.n_liquidations == 1
    assert "BTC" not in sim.account.positions


def test_position_pnl_calculations():
    pos = PerpPosition(
        symbol="BTC",
        side="long",
        entry=100.0,
        qty=10.0,
        leverage=5.0,
        margin=200.0,
        opened_at_step=0,
    )
    assert pos.notional(110.0) == 1100.0
    assert pos.unrealized_pnl(110.0) == 100.0
    assert pos.unrealized_pnl(90.0) == -100.0
    assert pos.equity(110.0) == 300.0  # margin + pnl


def test_account_equity_includes_unrealized_pnl():
    sim = PerpSimulator(starting_capital=10_000.0, taker_fee=0.0, slippage_bps_per_10k=0.0)
    sim.open_position("BTC", "long", notional_usd=1000.0, leverage=10.0, current_price=100.0)
    eq = sim.equity({"BTC": 110.0})
    # Started with 10k, locked 100 margin, +100 PnL → equity = 10000 + 100 = 10100
    assert eq == pytest.approx(10_100.0, rel=1e-2)


def test_to_dict_serializable():
    sim = PerpSimulator(starting_capital=10_000.0)
    sim.open_position("BTC", "long", 500.0, leverage=5.0, current_price=100.0)
    d = sim.account.to_dict({"BTC": 100.0})
    assert "cash" in d
    assert "equity" in d
    assert len(d["positions"]) == 1
