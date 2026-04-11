"""Tests for the AutoAgent-style strategy parameter optimizer."""

from __future__ import annotations

from auto_investment.data import synthetic_ohlcv
from auto_investment.optimizer import (
    DEFAULT_GRID,
    Candidate,
    OptimizationResult,
    _expand_grid,
    grid_search,
    random_search,
)


def test_expand_grid_filters_invalid_ema_combos():
    """ema_fast >= ema_slow should be filtered out."""
    grid = {
        "ema_fast": [10, 20, 30],
        "ema_slow": [10, 20],
    }
    combos = list(_expand_grid(grid))
    # Valid combos: (10,20). 10/10, 20/10, 20/20, 30/10, 30/20 are filtered.
    assert len(combos) == 1
    assert combos[0] == {"ema_fast": 10, "ema_slow": 20}


def test_expand_grid_default_size():
    """Default grid expands to the expected number of combinations."""
    combos = list(_expand_grid(DEFAULT_GRID))
    # 4 ema_fast * 4 ema_slow * 3 rsi * 3 sl * 3 tp = 432.
    # All ema_fast values (max 20) are < all ema_slow values (min 21),
    # so no combinations are filtered.
    assert len(combos) == 4 * 4 * 3 * 3 * 3


def test_random_search_returns_ranked_candidates():
    df = synthetic_ohlcv(limit=600)
    result = random_search(df, n_samples=10, seed=0)
    assert isinstance(result, OptimizationResult)
    assert result.n_evaluated == 10
    # If any candidate cleared the min_trades floor, it should be sorted by Sharpe descending
    if len(result.candidates) > 1:
        sharpes = [c.sharpe for c in result.candidates]
        assert sharpes == sorted(sharpes, reverse=True)
    if result.candidates:
        assert result.best == result.candidates[0]


def test_grid_search_returns_optimization_result():
    """Run with a tiny custom grid to keep the test fast."""
    df = synthetic_ohlcv(limit=600)
    tiny_grid = {
        "ema_fast": [10, 12],
        "ema_slow": [26, 30],
        "rsi_threshold": [50],
        "sl_atr_mult": [1.5],
        "tp_atr_mult": [3.0],
    }
    result = grid_search(df, grid=tiny_grid)
    assert isinstance(result, OptimizationResult)
    # 2*2*1*1*1 = 4 combinations, all valid (fast<slow)
    assert result.n_evaluated == 4


def test_optimization_result_to_dict():
    df = synthetic_ohlcv(limit=400)
    result = random_search(df, n_samples=5, seed=1)
    d = result.to_dict()
    assert "n_evaluated" in d
    assert "top" in d
    assert "best" in d
    if result.best is not None:
        assert d["best"]["sharpe"] == result.best.sharpe


def test_candidate_to_dict_keys():
    c = Candidate(
        params={"ema_fast": 12, "ema_slow": 26},
        sharpe=1.5,
        total_return_pct=10.0,
        win_rate=0.55,
        max_drawdown_pct=-5.0,
        n_trades=20,
    )
    d = c.to_dict()
    for key in ("params", "sharpe", "total_return_pct", "win_rate", "max_drawdown_pct", "n_trades"):
        assert key in d
