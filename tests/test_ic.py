"""Tests for the Information Coefficient module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from auto_investment.data import synthetic_ohlcv
from auto_investment.ic import (
    cross_sectional_ic,
    forward_returns,
    ic_report,
    rolling_ic,
    signal_from_indicators,
)
from auto_investment.indicators import add_indicators


def test_forward_returns_shape_and_shift():
    close = pd.Series([100, 110, 121, 121], dtype=float)
    fr = forward_returns(close, horizon=1)
    # First value: (110-100)/100 = 0.10. Last is NaN (no future bar).
    assert fr.iloc[0] == pytest.approx(0.1)
    assert pd.isna(fr.iloc[-1])


def test_perfect_signal_yields_high_ic():
    """A signal that equals the next-bar return should have IC ≈ 1."""
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0, 0.01, 200))
    signal = rets.shift(-1).fillna(0.0)  # signal IS the future return
    fwd = rets.shift(-1).fillna(0.0)
    rep = ic_report(signal, fwd)
    assert rep.mean_ic > 0.9


def test_random_signal_yields_zero_ic():
    """A random signal should have IC near zero."""
    rng = np.random.default_rng(1)
    signal = pd.Series(rng.normal(0, 1, 300))
    rets = pd.Series(rng.normal(0, 0.01, 300))
    rep = ic_report(signal, rets)
    assert abs(rep.mean_ic) < 0.3


def test_rolling_ic_length_matches_input():
    rng = np.random.default_rng(2)
    n = 200
    s = pd.Series(rng.normal(0, 1, n))
    r = pd.Series(rng.normal(0, 1, n))
    rolled = rolling_ic(s, r, window=30)
    assert len(rolled) == n
    # First (window-1) values should be NaN
    assert rolled.iloc[:29].isna().all()


def test_rolling_ic_input_length_mismatch_raises():
    s = pd.Series([1, 2, 3])
    r = pd.Series([1, 2])
    try:
        rolling_ic(s, r, window=2)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_signal_from_indicators_requires_columns():
    df = pd.DataFrame({"close": [1, 2, 3]})
    try:
        signal_from_indicators(df)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_signal_from_indicators_on_real_frame():
    df = synthetic_ohlcv(limit=200)
    df_ind = add_indicators(df)
    sig = signal_from_indicators(df_ind)
    assert len(sig) == len(df_ind)
    assert sig.notna().any()


def test_ic_report_to_dict():
    rep = ic_report(pd.Series([0.1, 0.2, 0.3] * 20), pd.Series([0.05, 0.1, 0.15] * 20))
    d = rep.to_dict()
    for key in ("mean_ic", "std_ic", "icir", "n_obs"):
        assert key in d


def test_cross_sectional_ic_basic():
    """Build a small panel where signal perfectly predicts returns row-wise."""
    n_rows = 10
    n_assets = 6
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, (n_rows, n_assets))
    signals = pd.DataFrame(base, columns=[f"a{i}" for i in range(n_assets)])
    rets = signals.copy()  # perfect prediction
    cs_ic = cross_sectional_ic(signals, rets)
    assert (cs_ic > 0.99).all()
