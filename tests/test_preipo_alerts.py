"""Tests for the S4 pre-IPO alerts module."""

import pandas as pd

from auto_investment.strategies.preipo_alerts import (
    PreIPOAlertConfig,
    scan_for_alerts,
    synth_marks,
)


def test_synth_marks_basic_shape():
    df = synth_marks(symbol="SPACEX", days=180, seed=1)
    assert "mark" in df.columns
    assert len(df) == 180
    assert (df["mark"] > 0).all()


def test_alerts_fire_on_rumor_jump():
    """A 15% jump on day 120 should generate at least one alert."""
    df = synth_marks(symbol="SPACEX", days=180, seed=1, rumor_jump_day=120,
                     rumor_jump_pct=0.30)
    alerts = scan_for_alerts(df, symbol="SPACEX")
    assert len(alerts) >= 1


def test_no_alerts_in_calm_regime():
    """Without a jump and with low vol, no alert should fire."""
    df = synth_marks(symbol="SPACEX", days=180, seed=2,
                     drift_bps_per_day=5.0, vol_bps_per_day=20.0,
                     rumor_jump_day=None)
    alerts = scan_for_alerts(df, symbol="SPACEX")
    assert len(alerts) == 0


def test_cooldown_suppresses_repeat_alerts():
    """A rolling sequence of jumps shouldn't fire on every single bar."""
    df = synth_marks(symbol="SPACEX", days=180, seed=3,
                     drift_bps_per_day=300.0,
                     vol_bps_per_day=400.0)
    cfg = PreIPOAlertConfig(cooldown_days=3)
    alerts = scan_for_alerts(df, symbol="SPACEX", config=cfg)
    # consecutive alerts must be at least cooldown_days apart
    for prev, cur in zip(alerts, alerts[1:]):
        gap = (cur.timestamp - prev.timestamp).days
        assert gap >= cfg.cooldown_days


def test_empty_input_returns_no_alerts():
    df = pd.DataFrame()
    assert scan_for_alerts(df, symbol="SPACEX") == []
