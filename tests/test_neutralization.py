"""Tests for the factor neutralization module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from auto_investment.data import synthetic_ohlcv
from auto_investment.indicators import add_indicators
from auto_investment.neutralization import build_factor_panel, neutralize


def test_build_factor_panel_columns():
    df = synthetic_ohlcv(limit=200)
    df_ind = add_indicators(df)
    panel = build_factor_panel(df_ind)
    for col in ("vol_20", "mom_20", "volume_z", "atr_pct"):
        assert col in panel.columns
    assert len(panel) == len(df_ind)


def test_neutralize_pure_factor_signal_explains_nearly_everything():
    """If the signal IS one of the factors, neutralization should explain ~100%."""
    rng = np.random.default_rng(0)
    n = 200
    factor = pd.Series(rng.normal(0, 1, n), name="x")
    factors = pd.DataFrame({"x": factor})
    signal = factor.copy()  # signal == factor
    residual, report = neutralize(signal, factors)
    assert report.explained_fraction > 0.99
    # Residual should be ~0
    assert residual.abs().mean() < 0.01


def test_neutralize_pure_noise_explains_nothing():
    """A signal independent of the factors should have ~0 explained fraction."""
    rng = np.random.default_rng(1)
    n = 500
    factors = pd.DataFrame({"x": rng.normal(0, 1, n)})
    signal = pd.Series(rng.normal(0, 1, n))
    _, report = neutralize(signal, factors)
    # Random signal should have explained fraction near zero (within sampling)
    assert abs(report.explained_fraction) < 0.1


def test_neutralize_short_input_returns_signal_unchanged():
    signal = pd.Series([1.0, 2.0, 3.0])
    factors = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    residual, report = neutralize(signal, factors)
    # Should return signal unchanged for short inputs
    assert (residual == signal).all()
    assert report.explained_fraction == 0.0


def test_neutralize_loadings_are_recorded():
    rng = np.random.default_rng(2)
    n = 200
    factors = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    signal = factors["a"] * 2.0 + factors["b"] * 0.5 + rng.normal(0, 0.1, n)
    _, report = neutralize(signal, factors)
    # The recovered loadings should be close to (2.0, 0.5)
    assert abs(report.factor_loadings["a"] - 2.0) < 0.2
    assert abs(report.factor_loadings["b"] - 0.5) < 0.2
