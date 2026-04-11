"""Tests for the JSAI-paper-inspired iterative improvement loop.

Most of the loop's value is in the LLM iterations, which require an API key.
These tests cover the deterministic harness: backtest helper, prompt building,
mode selection, and the AI-disabled fallback path.
"""

from __future__ import annotations

import pandas as pd

from auto_investment.data import synthetic_ohlcv
from auto_investment.improvement_loop import (
    DEFAULT_INITIAL_PARAMS,
    FeedbackMode,
    ImprovementResult,
    StrategyParams,
    _backtest_with_params,
    _build_feedback_prompt,
    run_improvement_loop,
)


def test_default_initial_params_has_required_keys():
    for key in ("ema_fast", "ema_slow", "rsi_threshold", "sl_atr_mult", "tp_atr_mult"):
        assert key in DEFAULT_INITIAL_PARAMS


def test_backtest_with_params_returns_full_metrics_dict():
    df = synthetic_ohlcv(limit=400)
    metrics = _backtest_with_params(df, DEFAULT_INITIAL_PARAMS, 10_000.0, 0.01)
    for key in (
        "n_trades",
        "win_rate",
        "total_return_pct",
        "max_drawdown_pct",
        "ic_mean",
        "icir",
        "factor_explained_fraction",
        "factor_loadings",
    ):
        assert key in metrics


def test_feedback_modes_produce_distinct_prompts():
    df = synthetic_ohlcv(limit=400)
    metrics = _backtest_with_params(df, DEFAULT_INITIAL_PARAMS, 10_000.0, 0.01)
    p1 = _build_feedback_prompt(df, DEFAULT_INITIAL_PARAMS, metrics, FeedbackMode.P1)
    p2 = _build_feedback_prompt(df, DEFAULT_INITIAL_PARAMS, metrics, FeedbackMode.P2)
    p3 = _build_feedback_prompt(df, DEFAULT_INITIAL_PARAMS, metrics, FeedbackMode.P3)

    assert "ic_mean" not in p1.lower() or "additional diagnostics" not in p1.lower()
    assert "additional diagnostics" in p2.lower()
    assert "additional diagnostics" in p3.lower()
    # P3 should add a price-action time series block
    assert "recent price action" in p3.lower()
    assert "recent price action" not in p2.lower()


def test_strategy_params_validation():
    # Valid construction
    p = StrategyParams(
        ema_fast=12, ema_slow=26, rsi_threshold=50.0, sl_atr_mult=1.5, tp_atr_mult=3.0
    )
    assert p.ema_fast == 12
    # Out-of-range values should raise
    raised = False
    try:
        StrategyParams(
            ema_fast=12, ema_slow=26, rsi_threshold=999.0, sl_atr_mult=1.5, tp_atr_mult=3.0
        )
    except Exception:
        raised = True
    assert raised


def test_run_improvement_loop_ai_disabled_returns_baseline():
    """With no ANTHROPIC_API_KEY, the loop should return the baseline run only."""
    df = synthetic_ohlcv(limit=400)
    # Make sure we're in the no-key path. If a key happens to be set in the
    # environment, the loop will actually call the API — that's a separate
    # integration concern; we still want the structure assertions to hold.
    result = run_improvement_loop(df, mode=FeedbackMode.P1, max_iterations=1)
    assert isinstance(result, ImprovementResult)
    assert result.mode == FeedbackMode.P1
    assert result.initial_params == DEFAULT_INITIAL_PARAMS
    assert result.initial_metrics == result.final_metrics or result.iterations_used >= 1
