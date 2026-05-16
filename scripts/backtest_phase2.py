"""Phase 2 backtest harness — adds S3, S4, and a real-data path for S1/S2.

Default mode (--mode synth) runs everything on synthetic fixtures so it's
hermetic and reproducible. Pass --mode real to load the parquet caches
populated by `scripts/fetch_real_data.py`.

Output: results/strategy_spec_v0.3_backtest.json with metrics and a blended
allocation analysis (now includes S3 in the mix).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from auto_investment.strategies.cross_exchange import (  # noqa: E402
    CrossExchangeConfig,
    backtest_cross_exchange,
    synth_cross_exchange_spread,
)
from auto_investment.strategies.funding_arb import (  # noqa: E402
    FundingArbConfig,
    backtest_funding_arb,
    synth_funding_series,
)
from auto_investment.strategies.preipo_alerts import (  # noqa: E402
    scan_for_alerts,
    synth_marks,
)
from auto_investment.strategies.yield_router import (  # noqa: E402
    YieldRouterConfig,
    backtest_yield_router,
    baseline_static_apy,
    synth_pool_grid,
)

log = logging.getLogger("phase2")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------


def run_s1(seed: int, mode: str) -> dict:
    print("\n=== S1 — Funding Arbitrage ===")
    if mode == "real":
        from auto_investment.data_fetchers import funding as fdg
        try:
            cache = next((REPO_ROOT / "data/funding").glob("hyperliquid_BTC*.parquet"))
            df = fdg.load(cache)
            funding = df["funding_rate"]
            print(f"  data: real ({len(funding)} hourly bars from {cache.name})")
        except (StopIteration, FileNotFoundError):
            print("  data: real cache missing — falling back to synth")
            funding = synth_funding_series(n=24 * 90, seed=seed)
    else:
        funding = synth_funding_series(n=24 * 90, seed=seed)
        print(f"  data: synthetic ({len(funding)} hourly bars, seed={seed})")

    # Use FundingArbConfig defaults (1000/200 + 3-period persistence) which
    # are documented in funding_arb.py and tuned against the first real-data
    # GHA run that showed -32% APR with 600/100 (whipsaw on Hyperliquid
    # funding flips). Override only the venue and size here.
    cfg = FundingArbConfig(
        spot_venue="binancejp", perp_venue="hyperliquid",
        notional_per_trade_usd=2_000.0,
    )
    res = backtest_funding_arb(funding, config=cfg)
    period_hours = len(funding)
    period_days = period_hours / 24.0
    capital = cfg.notional_per_trade_usd
    period_pct = (res.total_pnl_usd / capital) * 100.0
    apr_pct = ((1 + res.total_pnl_usd / capital) ** (365.0 / max(period_days, 1)) - 1) * 100
    summary = {
        "strategy": "S1_funding_arb", "data": mode,
        "period_days": round(period_days, 1), "capital_base_usd": capital,
        "n_trades": res.n_trades, "total_pnl_usd": round(res.total_pnl_usd, 2),
        "total_return_pct": round(period_pct, 3),
        "annualised_return_pct": round(apr_pct, 3),
        "sharpe_annualised": round(res.sharpe, 2),
        "max_drawdown_pct": round(res.max_drawdown_pct, 2),
    }
    _print_table(summary); return summary


def run_s2(seed: int, mode: str) -> dict:
    print("\n=== S2 — DeFi Yield Router ===")
    real_loaded = False
    if mode == "real":
        from auto_investment.data_fetchers import yields as yld
        try:
            universe = yld.load_universe_latest()
            pool_ids = universe["pool_id"].tolist()[:5]
            grid = yld.build_apy_grid(pool_ids)
            min_bars_required = 30  # ~5 days at 4h cadence
            if grid.empty or len(grid) < min_bars_required or grid.shape[1] < 2:
                raise ValueError(
                    f"real grid too small ({len(grid)} bars × "
                    f"{grid.shape[1] if not grid.empty else 0} pools); "
                    f"need >= {min_bars_required} bars and >= 2 pools"
                )
            from auto_investment.strategies.yield_router import PoolMeta
            # Only build PoolMeta for pools that survived build_apy_grid's
            # coverage filter — otherwise we feed the backtester a column
            # name that doesn't exist.
            kept_ids = grid.columns.tolist()
            kept_universe = universe[universe["pool_id"].isin(kept_ids)]
            pools = [
                PoolMeta(
                    pool_id=row["pool_id"], chain=row["chain"],
                    protocol=row["project"], asset=row["symbol"],
                    tvl_usd=float(row["tvl_usd"]),
                    audits=int(row["audits"]),
                    age_days=180, risk_premium_bps=0.0,
                )
                for _, row in kept_universe.iterrows()
            ]
            apy_history = grid
            real_loaded = True
            print(f"  data: real ({len(grid)} bars × {grid.shape[1]} pools)")
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"  data: real cache unusable ({exc}) — falling back to synth")
    if not real_loaded:
        pools, apy_history = synth_pool_grid(seed=seed)
        if mode == "synth":
            print(f"  data: synthetic ({len(apy_history)} 4h bars × {len(pools)} pools)")

    cfg = YieldRouterConfig(notional_usd=5_000.0, holding_window_days=7)
    res = backtest_yield_router(pools, apy_history, config=cfg)
    base_pool = pools[0].pool_id
    base = baseline_static_apy(apy_history, base_pool, cfg.notional_usd)
    days = max((apy_history.index[-1] - apy_history.index[0]).days, 1)
    base_apr = ((float(base.iloc[-1]) / float(base.iloc[0])) ** (365 / days) - 1) * 100
    summary = {
        "strategy": "S2_yield_router", "data": mode,
        "n_pools": len(pools), "n_rotations": res.n_rotations,
        "period_days": days, "capital_base_usd": cfg.notional_usd,
        "total_return_pct": round(res.total_return_pct, 3),
        "annualised_return_pct": round(res.annualised_return_pct(), 3),
        "baseline_annualised_return_pct": round(float(base_apr), 3),
        "uplift_bps_vs_baseline": round((res.annualised_return_pct() - base_apr) * 100, 1),
        "max_drawdown_pct": round(res.max_drawdown_pct, 3),
    }
    _print_table(summary); return summary


def run_s3(seed: int, mode: str) -> dict:
    print("\n=== S3 — Cross-exchange Stat-Arb ===")
    real_loaded = False
    if mode == "real":
        from auto_investment.data_fetchers import ohlcv as oh
        try:
            spread_df = oh.build_spread_series("binance", "bybit", "BTC/USDT", "1m")
            # Need at least 1.5× the rolling-window-span of bars to produce
            # a usable z-score series.
            if len(spread_df) < 360:
                raise ValueError(f"spread series too short ({len(spread_df)} bars)")
            real_loaded = True
            print(f"  data: real ({len(spread_df)} 1m bars)")
        except (FileNotFoundError, ValueError) as exc:
            print(f"  data: real cache unusable ({exc}) — falling back to synth")
    if not real_loaded:
        spread_df = synth_cross_exchange_spread(seed=seed)
        if mode == "synth":
            print(f"  data: synthetic ({len(spread_df)} 1m bars, seed={seed})")

    # Use the dataclass defaults (z_entry=3.0, timeout=120) which are
    # calibrated above retail-taker fees per cross_exchange.py docstring.
    cfg = CrossExchangeConfig(
        venue_a="binance", venue_b="bybit",
        notional_per_leg_usd=1_000.0,
    )
    res = backtest_cross_exchange(spread_df, config=cfg)
    period_min = len(spread_df)
    period_days = period_min / (60 * 24)
    capital = cfg.notional_per_leg_usd  # per-leg capital base
    period_pct = (res.total_pnl_usd / capital) * 100.0
    apr_pct = ((1 + res.total_pnl_usd / capital) ** (365.0 / max(period_days, 0.001)) - 1) * 100
    summary = {
        "strategy": "S3_cross_exchange", "data": mode,
        "period_days": round(period_days, 1), "capital_base_usd": capital,
        "n_trades": res.n_trades,
        "hit_rate": round(res.hit_rate, 3),
        "total_pnl_usd": round(res.total_pnl_usd, 2),
        "total_return_pct": round(period_pct, 3),
        "annualised_return_pct": round(apr_pct, 3),
        "sharpe_annualised": round(res.sharpe_annualised, 2),
        "max_drawdown_pct": round(res.max_drawdown_pct, 3),
    }
    _print_table(summary); return summary


def run_s4(seed: int, mode: str) -> dict:
    print("\n=== S4 — Pre-IPO Alerts (alert-only) ===")
    if mode == "real":
        from auto_investment.data_fetchers import preipo as pi
        results = []
        for sym in pi.WATCHLIST:
            try:
                df = pi.load_marks(sym)
                alerts = scan_for_alerts(df, symbol=sym)
                results.extend(alerts)
            except FileNotFoundError:
                continue
        if not results:
            print("  data: real cache missing — falling back to synth")
            df = synth_marks(symbol="SPACEX", seed=seed)
            results = scan_for_alerts(df, symbol="SPACEX")
    else:
        df = synth_marks(symbol="SPACEX", seed=seed)
        results = scan_for_alerts(df, symbol="SPACEX")
        print(f"  data: synthetic (180 daily marks, seed={seed})")
    summary = {
        "strategy": "S4_preipo_alerts", "data": mode,
        "n_alerts": len(results),
        "alerts": [
            {"ts": a.timestamp.isoformat(), "symbol": a.symbol,
             "delta_prob": round(a.delta_prob, 3),
             "current_mark": round(a.current_mark, 2)}
            for a in results[:5]  # cap output noise
        ],
    }
    _print_table({"n_alerts": summary["n_alerts"]})
    return summary


# ---------------------------------------------------------------------------
# Allocation across S1+S2+S3 (S4 is alert-only, no capital allocated)
# ---------------------------------------------------------------------------


def run_allocation(s1: dict, s2: dict, s3: dict, equity_usd: float) -> dict:
    s1_apr = s1.get("annualised_return_pct", 0) / 100.0
    s2_apr = s2.get("annualised_return_pct", 0) / 100.0
    s3_apr = s3.get("annualised_return_pct", 0) / 100.0
    policies = {
        # Each respects §8 hard rule: max 25% per venue.
        "Conservative_S1_S2_S3_cash": {
            "weights": {"S1": 0.20, "S2": 0.50, "S3": 0.10, "cash": 0.20},
            "rationale": "Heavier yield base; small S3 sleeve until real-data Sharpe lands.",
        },
        "Balanced_25_50_15_10": {
            "weights": {"S1": 0.25, "S2": 0.50, "S3": 0.15, "cash": 0.10},
            "rationale": "Hits the 25% Hyperliquid cap; reasonable S3 exposure.",
        },
        "Aggressive_25_50_25_0": {
            "weights": {"S1": 0.25, "S2": 0.50, "S3": 0.25, "cash": 0.00},
            "rationale": "Max all sleeves; no cash buffer (depends on DeFi liquidity).",
        },
    }
    out = {}
    for name, p in policies.items():
        w = p["weights"]
        apr = w["S1"] * s1_apr + w["S2"] * s2_apr + w["S3"] * s3_apr
        out[name] = {
            "weights_pct": {k: round(v * 100, 1) for k, v in w.items()},
            "expected_apr_pct": round(apr * 100, 2),
            "expected_pnl_usd_per_year": round(apr * equity_usd, 0),
            "rationale": p["rationale"],
        }
    print(f"\n=== Allocation across S1+S2+S3 (equity = ${equity_usd:,.0f}) ===")
    for name, info in out.items():
        w = info["weights_pct"]
        print(f"  {name:32s}  S1={w['S1']:>4}%  S2={w['S2']:>4}%  "
              f"S3={w['S3']:>4}%  cash={w['cash']:>4}%  → "
              f"{info['expected_apr_pct']:>5.2f}% APR "
              f"(${info['expected_pnl_usd_per_year']:.0f}/yr)")
    return out


def _print_table(d: dict) -> None:
    width = max(len(k) for k in d) + 2
    for k, v in d.items():
        print(f"  {k.ljust(width)} {v}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--equity", type=float, default=10_000.0)
    ap.add_argument("--mode", choices=["synth", "real"], default="synth")
    ap.add_argument("--out", default="results/strategy_spec_v0.3_backtest.json")
    args = ap.parse_args()

    s1 = run_s1(args.seed, args.mode)
    s2 = run_s2(args.seed, args.mode)
    s3 = run_s3(args.seed, args.mode)
    s4 = run_s4(args.seed, args.mode)
    alloc = run_allocation(s1, s2, s3, args.equity)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "spec_version": "v0.3",
        "mode": args.mode, "seed": args.seed, "equity_usd": args.equity,
        "S1": s1, "S2": s2, "S3": s3, "S4": s4,
        "allocations": alloc,
    }
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
