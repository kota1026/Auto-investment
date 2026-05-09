"""S2 — L2 DeFi yield router (YEARN V3 style).

Spec: docs/strategy_spec.md §3 (S2).

Hypothesis: stable-coin pool APYs mean-revert on a days-to-weeks horizon.
A "rotate to top forecast APY when expected gain > 2× rotation cost" rule
beats holding any single pool.

This module is data-source agnostic. In production, `apy_history` comes from
DefiLlama (`https://yields.llama.fi/chartLendingPool/<pool_id>`). For tests
and offline backtests we build a synthetic multi-pool grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..cost_model import yield_rotation_cost_bps


@dataclass(frozen=True)
class PoolMeta:
    """Static info about a DeFi pool. Mirrors DefiLlama fields we care about."""

    pool_id: str
    chain: str
    protocol: str
    asset: str
    tvl_usd: float
    audits: int
    age_days: int
    risk_premium_bps: float = 0.0  # extra hurdle for risky pools

    def passes_filters(
        self, *, min_tvl_usd: float = 20_000_000, min_age_days: int = 90
    ) -> bool:
        return (
            self.tvl_usd >= min_tvl_usd
            and self.audits >= 1
            and self.age_days >= min_age_days
        )


@dataclass(frozen=True)
class YieldRouterConfig:
    notional_usd: float = 5_000.0
    holding_window_days: int = 7
    forecast_halflife_periods: int = 12  # at 4h cadence = 48h half-life
    rotation_safety_factor: float = 2.0  # require 2× rotation cost to switch
    min_tvl_usd: float = 20_000_000
    min_age_days: int = 90


@dataclass
class YieldRotation:
    timestamp: pd.Timestamp
    from_pool: str | None
    to_pool: str
    cost_usd: float
    forecast_uplift_bps: float


@dataclass
class YieldRouterResult:
    config: YieldRouterConfig
    rotations: list[YieldRotation] = field(default_factory=list)
    equity_curve: pd.Series | None = None
    chosen_pool_path: pd.Series | None = None  # which pool per timestamp

    @property
    def n_rotations(self) -> int:
        return len(self.rotations)

    @property
    def total_return_pct(self) -> float:
        if self.equity_curve is None or self.equity_curve.empty:
            return 0.0
        first = float(self.equity_curve.iloc[0])
        last = float(self.equity_curve.iloc[-1])
        if first == 0:
            return 0.0
        return (last / first - 1.0) * 100.0

    def annualised_return_pct(self) -> float:
        if self.equity_curve is None or self.equity_curve.empty:
            return 0.0
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        if days <= 0:
            return 0.0
        ratio = float(self.equity_curve.iloc[-1] / self.equity_curve.iloc[0])
        if ratio <= 0:
            return 0.0
        return (ratio ** (365.0 / days) - 1.0) * 100.0

    @property
    def max_drawdown_pct(self) -> float:
        if self.equity_curve is None or self.equity_curve.empty:
            return 0.0
        roll = self.equity_curve.cummax()
        return float(-((self.equity_curve - roll) / roll).min() * 100.0)


def synth_pool_grid(seed: int = 11) -> tuple[list[PoolMeta], pd.DataFrame]:
    """Build a synthetic 5-pool universe with 90 days of 4h APY history.

    We model APY as a noisy mean-reverting process around different long-run
    means per pool, with occasional incentive boosts for one pool to reward
    the rotation strategy. This is intentionally favourable to the strategy
    so we can sanity-check whether the implementation extracts the obvious
    edge; an OOS test on real DefiLlama data is what actually qualifies it.
    """
    rng = np.random.default_rng(seed)
    pools = [
        PoolMeta("aave-arb-usdc", "arbitrum", "aave-v3", "USDC",
                 250_000_000, audits=3, age_days=900),
        PoolMeta("compound-base-usdc", "base", "compound-v3", "USDC",
                 80_000_000, audits=2, age_days=400),
        PoolMeta("morpho-base-usdc", "base", "morpho-blue", "USDC",
                 120_000_000, audits=2, age_days=300, risk_premium_bps=20),
        PoolMeta("yearn-arb-usdc", "arbitrum", "yearn-v3", "USDC",
                 30_000_000, audits=2, age_days=180, risk_premium_bps=30),
        PoolMeta("aave-op-usdc", "optimism", "aave-v3", "USDC",
                 50_000_000, audits=3, age_days=600),
    ]
    n = 6 * 365  # 4h cadence × 1 year (6 bars/day)
    means = np.array([0.045, 0.052, 0.061, 0.068, 0.043])  # base APY
    apy = np.zeros((n, len(pools)))
    apy[0] = means
    for t in range(1, n):
        drift = 0.05 * (means - apy[t - 1])
        noise = rng.normal(0, 0.0015, len(pools))
        apy[t] = np.clip(apy[t - 1] + drift + noise, 0.005, 0.30)
        # Occasional incentive spike on a random pool (rotation opportunity)
        if rng.random() < 0.01:
            i = rng.integers(0, len(pools))
            apy[t, i] += rng.uniform(0.02, 0.05)
    idx = pd.date_range("2025-12-01", periods=n, freq="4h", tz="UTC")
    df = pd.DataFrame(apy, index=idx, columns=[p.pool_id for p in pools])
    return pools, df


def backtest_yield_router(
    pools: list[PoolMeta],
    apy_history: pd.DataFrame,
    *,
    config: YieldRouterConfig | None = None,
) -> YieldRouterResult:
    """Walk-forward backtest of the yield router strategy.

    At each timestamp:
      1. Apply EWMA forecast over each pool's APY
      2. Filter pools through risk gates (TVL, audits, age, risk premium)
      3. Compute uplift from current pool to best candidate, net of rotation cost
      4. If uplift > safety_factor × cost, switch
      5. Accrue interest at the *realised* APY of the held pool over [t-1, t]
    """
    cfg = config or YieldRouterConfig()
    eligible = [p for p in pools if p.passes_filters(
        min_tvl_usd=cfg.min_tvl_usd, min_age_days=cfg.min_age_days)]
    eligible_ids = [p.pool_id for p in eligible]
    pool_meta = {p.pool_id: p for p in eligible}

    # Defend against empty/single-row apy_history (real-data fetch may
    # produce a tiny grid if pools have non-overlapping timestamps).
    if apy_history is None or len(apy_history) < 2 or not eligible_ids:
        return YieldRouterResult(config=cfg, rotations=[], equity_curve=None,
                                 chosen_pool_path=None)

    # Drop any eligible_ids missing from the apy_history columns
    eligible_ids = [pid for pid in eligible_ids if pid in apy_history.columns]
    if not eligible_ids:
        return YieldRouterResult(config=cfg, rotations=[], equity_curve=None,
                                 chosen_pool_path=None)
    pool_meta = {pid: pool_meta[pid] for pid in eligible_ids}

    apy = apy_history[eligible_ids]
    forecast = apy.ewm(halflife=cfg.forecast_halflife_periods, adjust=False).mean()

    # Cadence in years per period
    period_seconds = (apy.index[1] - apy.index[0]).total_seconds()
    period_years = period_seconds / (365 * 24 * 3600)

    equity = cfg.notional_usd
    equity_path: list[float] = []
    equity_index: list[pd.Timestamp] = []
    chosen: list[str] = []
    rotations: list[YieldRotation] = []

    current_pool: str | None = None
    for i, ts in enumerate(apy.index):
        if i > 0 and current_pool is not None:
            realised = float(apy.iloc[i - 1][current_pool])  # earned on prior period
            equity *= 1.0 + realised * period_years

        # Decide post-accrual whether to rotate going forward
        f_row = forecast.iloc[i]
        # apply per-pool risk premium hurdle (deduct from forecast)
        adj_forecast = f_row.copy()
        for pid, p in pool_meta.items():
            adj_forecast[pid] = max(0.0, adj_forecast[pid] - p.risk_premium_bps / 10_000.0)
        target = adj_forecast.idxmax()

        if current_pool is None:
            # initial allocation — pay only deposit cost
            cost_bps = yield_rotation_cost_bps(
                pool_meta[target].chain, equity
            ) / 2.0  # one-leg gas only
            equity -= equity * cost_bps / 10_000.0
            rotations.append(
                YieldRotation(
                    timestamp=ts,
                    from_pool=None,
                    to_pool=target,
                    cost_usd=equity * cost_bps / 10_000.0,
                    forecast_uplift_bps=float(adj_forecast[target] * 10_000),
                )
            )
            current_pool = target
        elif target != current_pool:
            uplift_apr_bps = float(
                (adj_forecast[target] - adj_forecast[current_pool]) * 10_000.0
            )
            uplift_over_window_bps = uplift_apr_bps * cfg.holding_window_days / 365.0
            cost_bps = yield_rotation_cost_bps(pool_meta[target].chain, equity)
            if uplift_over_window_bps > cfg.rotation_safety_factor * cost_bps:
                cost_usd = equity * cost_bps / 10_000.0
                equity -= cost_usd
                rotations.append(
                    YieldRotation(
                        timestamp=ts,
                        from_pool=current_pool,
                        to_pool=target,
                        cost_usd=cost_usd,
                        forecast_uplift_bps=uplift_apr_bps,
                    )
                )
                current_pool = target
        equity_path.append(equity)
        equity_index.append(ts)
        chosen.append(current_pool or "")

    eq = pd.Series(equity_path, index=pd.Index(equity_index, name="ts"))
    return YieldRouterResult(
        config=cfg,
        rotations=rotations,
        equity_curve=eq,
        chosen_pool_path=pd.Series(chosen, index=eq.index, name="pool"),
    )


def baseline_static_apy(
    apy_history: pd.DataFrame, pool_id: str, notional_usd: float
) -> pd.Series:
    """Buy-and-hold baseline for a single pool, same cadence as the input."""
    period_seconds = (apy_history.index[1] - apy_history.index[0]).total_seconds()
    period_years = period_seconds / (365 * 24 * 3600)
    s = apy_history[pool_id]
    growth = (1.0 + s.shift(1).fillna(s.iloc[0]) * period_years).cumprod()
    return growth * notional_usd
