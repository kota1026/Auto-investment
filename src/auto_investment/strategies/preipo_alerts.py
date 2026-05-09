"""S4 — Pre-IPO alerts (research-grade, alert-only).

Spec: docs/strategy_spec.md §3 (S4).

We deliberately do **not** auto-trade these. We surface alerts when the
two cheap heuristics below fire, and the human (CEO) decides whether to
fill manually:

  1. Implied IPO probability rises sharply
     (logistic fit to the daily mark; alert if Δp > 5pp over 7 days).
  2. Discount-to-comps widens
     (ratio of mark to a Tavily-fetched private-market valuation
     drops by >10% week over week).

The first heuristic is computable from Aevo's own data and is the only one
implemented here. The second requires Tavily web search to get
private-market valuations from news; we do that inside
`.claude/commands/news_sentiment.md` so it benefits from Claude's
reasoning, not the Python layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreIPOAlertConfig:
    # Logistic curve fit over a rolling window of daily marks. We treat the
    # current mark as p_IPO * terminal_mark + (1-p) * 0, i.e. p ≈ mark/terminal.
    # The "terminal" is taken as the rolling max of the 90-day window — a
    # conservative proxy for "what the market thinks the post-IPO price is".
    fit_window_days: int = 90
    # Minimum implied-probability jump that triggers an alert
    delta_prob_threshold: float = 0.05
    # Look back this many days for the delta
    delta_lookback_days: int = 7
    # Don't fire repeat alerts inside this cooldown
    cooldown_days: int = 3


@dataclass
class PreIPOAlert:
    timestamp: pd.Timestamp
    symbol: str
    current_mark: float
    implied_prob_now: float
    implied_prob_then: float
    delta_prob: float
    rationale: str


def _implied_prob(mark: float, terminal: float) -> float:
    """Treat mark as p × terminal; clamp to [0, 1]. Returns NaN if terminal<=0."""
    if terminal <= 0:
        return float("nan")
    return float(min(max(mark / terminal, 0.0), 1.0))


def scan_for_alerts(
    marks: pd.DataFrame,
    *,
    symbol: str,
    config: PreIPOAlertConfig | None = None,
    as_of: datetime | None = None,
) -> list[PreIPOAlert]:
    """Walk the daily-mark history and emit alerts where the heuristic fires.

    `marks` must have a `mark` column and a tz-aware datetime index at daily
    granularity. We tolerate gaps (forward-fill up to 3 days).
    """
    cfg = config or PreIPOAlertConfig()
    if marks.empty or "mark" not in marks.columns:
        return []
    s = marks["mark"].asfreq("D").ffill(limit=3).dropna()
    if len(s) < cfg.fit_window_days // 2:
        return []

    rolling_max = s.rolling(cfg.fit_window_days, min_periods=10).max()
    prob = pd.Series(
        [_implied_prob(m, t) for m, t in zip(s.values, rolling_max.values)],
        index=s.index, name="implied_prob",
    )

    alerts: list[PreIPOAlert] = []
    last_alert_ts: pd.Timestamp | None = None
    for ts, p_now in prob.items():
        then_ts = ts - pd.Timedelta(days=cfg.delta_lookback_days)
        if then_ts not in prob.index:
            continue
        p_then = prob.loc[then_ts]
        if pd.isna(p_now) or pd.isna(p_then):
            continue
        delta = p_now - p_then
        if delta < cfg.delta_prob_threshold:
            continue
        if last_alert_ts is not None and (ts - last_alert_ts).days < cfg.cooldown_days:
            continue
        alerts.append(PreIPOAlert(
            timestamp=ts, symbol=symbol,
            current_mark=float(s.loc[ts]),
            implied_prob_now=float(p_now),
            implied_prob_then=float(p_then),
            delta_prob=float(delta),
            rationale=(
                f"Implied IPO probability for {symbol} rose "
                f"{delta*100:.1f}pp in {cfg.delta_lookback_days}d "
                f"({p_then*100:.0f}% → {p_now*100:.0f}%). Consider manual review."
            ),
        ))
        last_alert_ts = ts

    if as_of is not None:
        alerts = [a for a in alerts if a.timestamp <= pd.Timestamp(as_of)]
    return alerts


def synth_marks(
    symbol: str = "SPACEX",
    days: int = 180,
    seed: int = 23,
    drift_bps_per_day: float = 30.0,
    vol_bps_per_day: float = 250.0,
    rumor_jump_day: int | None = 120,
    rumor_jump_pct: float = 0.15,
) -> pd.DataFrame:
    """Synthetic daily marks for a pre-IPO name. Includes optional rumor jump.

    Default profile: gentle uptrend + occasional volatility, with a single
    "IPO rumor" jump on `rumor_jump_day`. The jump is what S4 should detect.
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift_bps_per_day / 1e4, vol_bps_per_day / 1e4, days)
    if rumor_jump_day is not None and 0 <= rumor_jump_day < days:
        log_returns[rumor_jump_day] += math.log(1 + rumor_jump_pct)
    price_path = 100.0 * np.exp(np.cumsum(log_returns))
    idx = pd.date_range(
        end=datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0),
        periods=days, freq="D",
    )
    df = pd.DataFrame({"mark": price_path}, index=idx)
    df.index.name = "ts"
    return df
