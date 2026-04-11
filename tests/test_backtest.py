"""End-to-end test of the backtester on synthetic data."""

from __future__ import annotations

from auto_investment.backtest import run_backtest
from auto_investment.data import synthetic_ohlcv


def test_backtest_runs_end_to_end():
    df = synthetic_ohlcv(limit=600)
    result = run_backtest(df, initial_equity=10_000.0, risk_per_trade=0.01)

    # Sanity: structural fields exist and have reasonable types
    assert isinstance(result.n_trades, int)
    assert result.n_trades >= 0
    assert 0.0 <= result.win_rate <= 1.0
    assert result.max_drawdown_pct <= 0.0  # drawdown is non-positive
    assert result.final_equity > 0
    assert len(result.equity_curve) > 0


def test_backtest_to_dict_serializes():
    df = synthetic_ohlcv(limit=400)
    result = run_backtest(df)
    d = result.to_dict()
    assert "trades" in d
    assert "equity_curve" in d
    assert "final_equity" in d
    # Equity curve entries are JSON-friendly dicts
    if d["equity_curve"]:
        assert "timestamp" in d["equity_curve"][0]
        assert "equity" in d["equity_curve"][0]
