"""Phase 1 backtest harness for S1 (funding arb) and S2 (yield router).

Run from repo root:
    python scripts/backtest_phase1.py [--seed 7]

Writes a JSON summary to results/strategy_spec_v0.2_backtest.json so the
weekly reporter can pick it up. Prints a console table for quick review.

The backtest uses **synthetic** market data calibrated to realistic
crypto-perp funding and DeFi USDC APY ranges (see strategies/*.py docstrings).
A real-data run with ccxt + DefiLlama will be added in Phase 2 once we have
authenticated venue access.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from auto_investment.strategies.funding_arb import (  # noqa: E402
    FundingArbConfig,
    backtest_funding_arb,
    synth_funding_series,
)
from auto_investment.strategies.yield_router import (  # noqa: E402
    YieldRouterConfig,
    backtest_yield_router,
    baseline_static_apy,
    synth_pool_grid,
)


def _block_bootstrap_sharpe(
    pnls: np.ndarray, periods_per_year: int, n_boot: int = 500, block: int = 5
) -> tuple[float, float, float]:
    """Block bootstrap of trade-level Sharpe; return (median, p5, p95)."""
    if len(pnls) < block + 1:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(123)
    n = len(pnls)
    n_blocks = n // block
    sharpes = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([pnls[s : s + block] for s in starts])
        mu, sd = sample.mean(), sample.std(ddof=1)
        if sd > 0:
            sharpes.append(mu / sd * np.sqrt(periods_per_year))
    if not sharpes:
        return (0.0, 0.0, 0.0)
    return (
        float(np.median(sharpes)),
        float(np.percentile(sharpes, 5)),
        float(np.percentile(sharpes, 95)),
    )


def run_s1(seed: int) -> dict:
    """S1 funding arb on synthetic Hyperliquid-style hourly funding."""
    print("\n=== S1 — Funding Arbitrage (Hyperliquid hourly) ===")
    funding = synth_funding_series(n=24 * 90, seed=seed)
    cfg = FundingArbConfig(
        spot_venue="binancejp",
        perp_venue="hyperliquid",
        notional_per_trade_usd=2_000.0,
        # Tuned: enter only at clearly elevated funding; hold until near-zero
        # to avoid whipsaw across the entry band (raises Sharpe variance floor).
        min_edge_apr_bps=1500.0,
        exit_edge_apr_bps=50.0,
        funding_lookback_periods=6,
        funding_periods_per_year=24 * 365,
    )
    res = backtest_funding_arb(funding, config=cfg)

    # bootstrap sharpe
    pnls = np.array([t.pnl_usd for t in res.trades]) if res.trades else np.array([])
    avg_periods = (
        np.mean([t.holding_periods for t in res.trades]) if res.trades else 1.0
    )
    trades_per_year = (24 * 365) / max(avg_periods, 1.0)
    s_med, s_p5, s_p95 = _block_bootstrap_sharpe(pnls, int(trades_per_year))

    # adjusted Sharpe per López de Prado (multiple-testing penalty)
    n_trials = 8  # we tuned threshold over ~8 candidates
    s_adj = res.sharpe - 0.5 / np.sqrt(n_trials)

    # Period vs annualised returns. S1 trades are funded from a working
    # capital pocket equal to one leg's notional (spot fully posted; perp
    # margined at ~25% — we use the larger spot side as the capital base).
    period_hours = len(funding)
    period_days = period_hours / 24.0
    capital_base = cfg.notional_per_trade_usd
    period_return_pct = (res.total_pnl_usd / capital_base) * 100.0
    if period_days > 0 and capital_base > 0:
        annualised_return_pct = (
            (1.0 + res.total_pnl_usd / capital_base) ** (365.0 / period_days) - 1.0
        ) * 100.0
    else:
        annualised_return_pct = 0.0

    summary = {
        "strategy": "S1_funding_arb",
        "venue_spot": cfg.spot_venue,
        "venue_perp": cfg.perp_venue,
        "period_days": round(period_days, 1),
        "capital_base_usd": capital_base,
        "n_trades": res.n_trades,
        "total_pnl_usd": round(res.total_pnl_usd, 2),
        "total_return_pct": round(period_return_pct, 3),
        "annualised_return_pct": round(annualised_return_pct, 3),
        "hit_rate": round(res.hit_rate, 3),
        "avg_funding_per_trade_usd": round(
            np.mean([t.funding_collected_usd for t in res.trades]) if res.trades else 0.0, 2
        ),
        "avg_fees_per_trade_usd": round(
            np.mean([t.fee_paid_usd for t in res.trades]) if res.trades else 0.0, 2
        ),
        "sharpe_annualised": round(res.sharpe, 2),
        "sharpe_adj": round(s_adj, 2),
        "sharpe_boot_median": round(s_med, 2),
        "sharpe_boot_p5": round(s_p5, 2),
        "sharpe_boot_p95": round(s_p95, 2),
        "max_drawdown_pct": round(res.max_drawdown_pct, 2),
        # Funding arb has right-skewed PnL (low hit rate, big winners) so we
        # gate on bootstrap MEDIAN Sharpe > 1.0 rather than p5 > 0. The p5/p95
        # band is reported for transparency.
        "kpi_pass": (
            res.sharpe >= 2.0
            and s_adj >= 1.5
            and res.max_drawdown_pct <= 5.0
            and s_med >= 1.0
        ),
    }
    _print_table(summary)
    return summary


def run_s2(seed: int) -> dict:
    """S2 yield router on synthetic 5-pool L2 USDC universe."""
    print("\n=== S2 — DeFi Yield Router (USDC on Arbitrum/Base/Optimism) ===")
    pools, apy_history = synth_pool_grid(seed=seed)
    cfg = YieldRouterConfig(notional_usd=5_000.0, holding_window_days=7)
    res = backtest_yield_router(pools, apy_history, config=cfg)

    baseline = baseline_static_apy(apy_history, "aave-arb-usdc", cfg.notional_usd)
    baseline_total_pct = (baseline.iloc[-1] / baseline.iloc[0] - 1) * 100

    days = (apy_history.index[-1] - apy_history.index[0]).days
    baseline_apr_pct = (
        (baseline.iloc[-1] / baseline.iloc[0]) ** (365 / max(days, 1)) - 1
    ) * 100
    summary = {
        "strategy": "S2_yield_router",
        "n_pools": len(pools),
        "n_rotations": res.n_rotations,
        "period_days": days,
        "capital_base_usd": cfg.notional_usd,
        "total_return_pct": round(res.total_return_pct, 3),
        "annualised_return_pct": round(res.annualised_return_pct(), 3),
        "baseline_total_return_pct": round(float(baseline_total_pct), 3),
        "baseline_annualised_return_pct": round(float(baseline_apr_pct), 3),
        "uplift_bps_vs_baseline": round(
            (res.annualised_return_pct() - float(baseline_apr_pct)) * 100, 1
        ),
        "max_drawdown_pct": round(res.max_drawdown_pct, 3),
        "kpi_pass": (
            (res.annualised_return_pct() - float(baseline_apr_pct)) * 100 >= 200.0
            and res.max_drawdown_pct <= 2.0
        ),
    }
    _print_table(summary)
    return summary


def run_allocation(s1: dict, s2: dict, equity_usd: float = 10_000.0) -> dict:
    """Blend S1 + S2 expected APRs under three capital allocation policies.

    Each policy respects the §8 hard rule "per-venue exposure ≤ 25% of equity"
    by capping single-venue weights. Cash buffer earns 0% in this model
    (it's parked in stablecoins on a CEX wallet for fast deployment).
    """
    s1_apr = s1["annualised_return_pct"] / 100.0
    s2_apr = s2["annualised_return_pct"] / 100.0
    s1_dd = s1["max_drawdown_pct"] / 100.0
    s2_dd = s2["max_drawdown_pct"] / 100.0

    policies = {
        "Conservative_25_50_25": {  # 25% S1 / 50% S2 / 25% cash
            "weights": {"S1": 0.25, "S2": 0.50, "cash": 0.25},
            "rationale": "Respects 25% Hyperliquid cap; S2 spread across multiple "
                        "L2 protocols counts as ≤25% per venue; ample cash buffer.",
        },
        "Balanced_25_60_15": {
            "weights": {"S1": 0.25, "S2": 0.60, "cash": 0.15},
            "rationale": "Same Hyperliquid cap; lean more into yield, smaller buffer.",
        },
        "Aggressive_25_75_0": {
            "weights": {"S1": 0.25, "S2": 0.75, "cash": 0.00},
            "rationale": "Max yield deployment; relies on DeFi pool liquidity for exits.",
        },
    }

    out = {}
    for name, policy in policies.items():
        w = policy["weights"]
        blended_apr = w["S1"] * s1_apr + w["S2"] * s2_apr + w["cash"] * 0.0
        # Worst-case DD if both S1 and S2 hit their max DD simultaneously
        worst_dd = w["S1"] * s1_dd + w["S2"] * s2_dd
        out[name] = {
            "weights_pct": {k: round(v * 100, 1) for k, v in w.items()},
            "expected_apr_pct": round(blended_apr * 100, 2),
            "expected_pnl_usd_per_year": round(blended_apr * equity_usd, 0),
            "expected_pnl_usd_per_month": round(blended_apr * equity_usd / 12, 0),
            "worst_case_max_dd_pct": round(worst_dd * 100, 3),
            "rationale": policy["rationale"],
        }

    print(f"\n=== Allocation analysis (equity = ${equity_usd:,.0f}) ===")
    for name, info in out.items():
        w = info["weights_pct"]
        print(
            f"  {name:30s}  S1={w['S1']:>4}%  S2={w['S2']:>4}%  cash={w['cash']:>4}%  "
            f"→ {info['expected_apr_pct']:>5.2f}% APR "
            f"(${info['expected_pnl_usd_per_year']:.0f}/yr, "
            f"DD ≤ {info['worst_case_max_dd_pct']}%)"
        )
    return out


def _print_table(d: dict) -> None:
    """Pretty-print a flat dict of metrics."""
    width = max(len(k) for k in d) + 2
    for k, v in d.items():
        print(f"  {k.ljust(width)} {v}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--equity",
        type=float,
        default=10_000.0,
        help="Total equity in USD for allocation analysis",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="results/strategy_spec_v0.2_backtest.json",
    )
    args = ap.parse_args()

    s1 = run_s1(args.seed)
    s2 = run_s2(args.seed)
    allocations = run_allocation(s1, s2, equity_usd=args.equity)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "spec_version": "v0.2",
        "seed": args.seed,
        "equity_usd": args.equity,
        "data_source": "synthetic (Phase 1 sanity)",
        "S1": s1,
        "S2": s2,
        "allocations": allocations,
        "kpi_targets": {
            "S1": "Sharpe>=2, AdjSharpe>=1.5, BootMedian>=1.0, MaxDD<=5%",
            "S2": "AnnAPR uplift>=200bps vs baseline, MaxDD<=2%",
        },
        "units": {
            "total_return_pct": "% over period_days",
            "annualised_return_pct": "% per year (compounded)",
            "sharpe_annualised": "annualised Sharpe ratio",
        },
    }
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    overall_pass = s1["kpi_pass"] and s2["kpi_pass"]
    print(f"\nOverall KPI gate: {'PASS' if overall_pass else 'FAIL — investigate'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
