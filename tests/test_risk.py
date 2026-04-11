"""Tests for the risk management module."""

from __future__ import annotations

import pytest

from auto_investment.risk import build_trade_plan


def test_long_plan_risk_dollars_match_budget():
    plan = build_trade_plan(
        side="long",
        entry=100.0,
        atr_value=2.0,
        equity=10_000.0,
        risk_per_trade=0.01,
        sl_atr_mult=1.5,
        tp_atr_mult=3.0,
    )
    # Risk budget is 1% of $10k = $100
    assert plan.risk_usd == pytest.approx(100.0)
    # SL is at entry - 1.5 * ATR = 100 - 3 = 97
    assert plan.stop == pytest.approx(97.0)
    # Target is at entry + 3 * ATR = 100 + 6 = 106
    assert plan.target == pytest.approx(106.0)
    # Quantity such that loss at stop equals risk budget:
    # qty * (entry - stop) == risk_usd → qty * 3 == 100 → qty ≈ 33.33
    assert plan.qty == pytest.approx(100.0 / 3.0)
    # R:R must be tp_mult / sl_mult = 2.0
    assert plan.rr == pytest.approx(2.0)


def test_short_plan_inverts_stop_and_target():
    plan = build_trade_plan(
        side="short",
        entry=100.0,
        atr_value=2.0,
        equity=10_000.0,
    )
    assert plan.stop > plan.entry  # short stops are above entry
    assert plan.target < plan.entry  # short targets are below entry


def test_invalid_atr_raises():
    with pytest.raises(ValueError):
        build_trade_plan(side="long", entry=100.0, atr_value=0.0, equity=10_000.0)


def test_invalid_equity_raises():
    with pytest.raises(ValueError):
        build_trade_plan(side="long", entry=100.0, atr_value=1.0, equity=0.0)


def test_invalid_risk_fraction_raises():
    with pytest.raises(ValueError):
        build_trade_plan(side="long", entry=100.0, atr_value=1.0, equity=10_000.0, risk_per_trade=1.5)


def test_invalid_side_raises():
    with pytest.raises(ValueError):
        build_trade_plan(side="sideways", entry=100.0, atr_value=1.0, equity=10_000.0)  # type: ignore[arg-type]
