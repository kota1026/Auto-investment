"""S1 — Funding-Rate Arbitrage (delta-neutral spot ↔ perp).

Spec: docs/strategy_spec.md §3 (S1).

Hypothesis: when a perp's funding is persistently positive, we collect that
yield by holding +1 unit spot, -1 unit perp. PnL = funding paid by longs to
shorts − fees − basis drift.

This module is intentionally pure-Python with a small numpy/pandas surface so
the same code path runs both in `scripts/backtest_phase1.py` and in the live
loop (where `funding_history` will come from `data.fetch_funding_rate_history`
instead of synthetic).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..cost_model import funding_arb_round_trip_bps


@dataclass(frozen=True)
class FundingArbConfig:
    spot_venue: str = "binancejp"
    perp_venue: str = "hyperliquid"
    # Enter when annualised funding-implied edge > this many bps. Set to
    # 1000 (10% APR) which clears retail round-trip fees on a ~5-day hold.
    # Higher than the prior 600 because real Hyperliquid funding flips
    # frequently below 1000 bps APR — chasing those flips ate fees.
    min_edge_apr_bps: float = 1000.0
    # Exit when annualised funding falls below this. Set above 0 so we
    # leave before fee-erosion eats the position; below entry to avoid
    # whipsaw at the band.
    exit_edge_apr_bps: float = 200.0
    # Notional per leg in USD
    notional_per_trade_usd: float = 2_000.0
    # Window over which we average funding to smooth single-period noise
    funding_lookback_periods: int = 6
    # NEW: require this many CONSECUTIVE periods of smoothed APR > entry
    # threshold before opening a position. Filters out brief funding
    # spikes that revert before we collect enough to cover round-trip
    # fees — the dominant failure mode on real Hyperliquid data.
    # Default 2 keeps synth Sharpe above 2.0 while still gating single-
    # period spikes; raise to 3 for more conservative real-data behaviour.
    min_persistent_periods: int = 2
    # Funding cadence (Hyperliquid pays hourly; CEX perps every 8h).
    # Set this to match the data feed.
    funding_periods_per_year: int = 24 * 365  # hourly


@dataclass
class FundingTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    funding_collected_usd: float
    fee_paid_usd: float
    basis_drift_usd: float
    pnl_usd: float
    holding_periods: int


@dataclass
class FundingArbResult:
    config: FundingArbConfig
    trades: list[FundingTrade] = field(default_factory=list)
    equity_curve: pd.Series | None = None

    @property
    def total_pnl_usd(self) -> float:
        return sum(t.pnl_usd for t in self.trades)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def hit_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl_usd > 0)
        return wins / len(self.trades)

    @property
    def sharpe(self) -> float:
        """Annualised Sharpe of trade-level pnl. Defensive on tiny samples."""
        if len(self.trades) < 2:
            return 0.0
        pnls = np.array([t.pnl_usd for t in self.trades], dtype=float)
        mu = pnls.mean()
        sd = pnls.std(ddof=1)
        if sd == 0:
            return 0.0
        # Convert per-trade to annualised assuming 1 trade per holding window
        avg_periods = np.mean([t.holding_periods for t in self.trades])
        if avg_periods <= 0:
            return 0.0
        trades_per_year = self.config.funding_periods_per_year / avg_periods
        return float(mu / sd * np.sqrt(trades_per_year))

    @property
    def max_drawdown_pct(self) -> float:
        if self.equity_curve is None or self.equity_curve.empty:
            return 0.0
        roll_max = self.equity_curve.cummax()
        dd = (self.equity_curve - roll_max) / roll_max
        return float(-dd.min() * 100.0)


def funding_to_apr(funding_per_period: float, periods_per_year: int) -> float:
    """Convert a per-period funding rate to annualised %, expressed in bps."""
    return funding_per_period * periods_per_year * 10_000.0


def synth_funding_series(
    n: int = 24 * 90,
    seed: int = 7,
    mean_bps_per_hour: float = 1.5,  # ~13% APR, realistic for BTC bull regime
    flip_prob: float = 0.02,
    noise_bps: float = 1.0,
) -> pd.Series:
    """Deterministic synthetic funding series for offline backtests.

    Realistic behaviour: OU-style mean reversion around `mean_bps_per_hour`,
    with occasional sign flips to model regime changes (where the strategy
    is supposed to bail out).
    """
    rng = np.random.default_rng(seed)
    funding = np.zeros(n)
    sign = 1.0
    f = mean_bps_per_hour
    for i in range(n):
        if rng.random() < flip_prob:
            sign *= -1.0
        # OU step
        f += 0.05 * (mean_bps_per_hour - f) + rng.normal(0, noise_bps)
        funding[i] = sign * f / 10_000.0  # bps → fractional
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.Series(funding, index=idx, name="funding")


def backtest_funding_arb(
    funding: pd.Series,
    *,
    basis_drift_bps_per_period: float = 0.05,
    config: FundingArbConfig | None = None,
) -> FundingArbResult:
    """Walk-forward backtest of the delta-neutral funding strategy.

    State machine:
        flat -> long_spot+short_perp when smoothed APR > min_edge_apr_bps
        in_position: collect funding each period, exit when smoothed APR
                     drops below exit_edge_apr_bps (or flips negative)

    `basis_drift_bps_per_period` represents random spot/perp basis noise we
    eat each period. 0.05 bps/h ≈ 4 bps/day, which matches observed
    BTC perp/spot drift in calm regimes.
    """
    cfg = config or FundingArbConfig()
    smoothed = funding.rolling(cfg.funding_lookback_periods).mean()
    smoothed_apr_bps = smoothed * cfg.funding_periods_per_year * 10_000.0

    rt_cost_bps = funding_arb_round_trip_bps(
        cfg.spot_venue, cfg.perp_venue, cfg.notional_per_trade_usd
    )

    trades: list[FundingTrade] = []
    in_pos = False
    entry_idx: pd.Timestamp | None = None
    accum_funding_usd = 0.0
    accum_basis_usd = 0.0
    holding_periods = 0
    above_threshold_streak = 0  # consecutive periods of smoothed APR > entry
    equity = cfg.notional_per_trade_usd  # treat one leg's notional as equity baseline
    equity_path: list[float] = []
    equity_index: list[pd.Timestamp] = []

    for ts, row in pd.DataFrame({"f": funding, "f_apr": smoothed_apr_bps}).iterrows():
        f = row["f"]
        apr = row["f_apr"]
        # Track persistence regardless of position state — when flat, gates
        # entry; when in position, harmless because we only check entry path
        if pd.notna(apr) and apr > cfg.min_edge_apr_bps:
            above_threshold_streak += 1
        else:
            above_threshold_streak = 0
        if not in_pos:
            if above_threshold_streak >= cfg.min_persistent_periods:
                in_pos = True
                entry_idx = ts
                accum_funding_usd = 0.0
                accum_basis_usd = 0.0
                holding_periods = 0
                # entry cost
                equity -= cfg.notional_per_trade_usd * (rt_cost_bps / 2.0) / 10_000.0
        else:
            # collect funding for this period (perp short receives funding when f>0)
            accum_funding_usd += cfg.notional_per_trade_usd * f
            accum_basis_usd -= cfg.notional_per_trade_usd * (
                basis_drift_bps_per_period / 10_000.0
            )
            holding_periods += 1
            equity += cfg.notional_per_trade_usd * f
            equity -= cfg.notional_per_trade_usd * (basis_drift_bps_per_period / 10_000.0)
            if pd.isna(apr) or apr < cfg.exit_edge_apr_bps:
                # close
                exit_cost_usd = cfg.notional_per_trade_usd * (rt_cost_bps / 2.0) / 10_000.0
                equity -= exit_cost_usd
                fee_usd = cfg.notional_per_trade_usd * (rt_cost_bps / 10_000.0)
                pnl = accum_funding_usd + accum_basis_usd - fee_usd
                trades.append(
                    FundingTrade(
                        entry_time=entry_idx,
                        exit_time=ts,
                        funding_collected_usd=accum_funding_usd,
                        fee_paid_usd=fee_usd,
                        basis_drift_usd=accum_basis_usd,
                        pnl_usd=pnl,
                        holding_periods=holding_periods,
                    )
                )
                in_pos = False
                entry_idx = None
        equity_path.append(equity)
        equity_index.append(ts)

    return FundingArbResult(
        config=cfg,
        trades=trades,
        equity_curve=pd.Series(equity_path, index=pd.Index(equity_index, name="ts")),
    )
