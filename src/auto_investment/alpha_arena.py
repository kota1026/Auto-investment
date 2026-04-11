"""Alpha Arena contest loop — replicate the Nof1 format end-to-end.

The contest:
  - Multi-symbol crypto perp portfolio (default BTC/ETH/SOL)
  - Cross-margin USDC account, configurable starting capital
  - Hourly market data, decision every 4 hours (configurable)
  - LLM agent (or heuristic) makes all trading decisions autonomously
  - Realistic perp simulator: fees, funding, slippage, liquidations
  - Run on real (yfinance) or synthetic data
  - Compare against BTC buy-and-hold benchmark
  - Compute Sharpe ratio, max drawdown, full trade log

This is the highest-level user-facing module — it ties together perp_sim,
multi_market, and decision_agent into a single `run_contest()` call that
the FastAPI server and CLI both use.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .decision_agent import DecisionBatch, TradeDecision, decide
from .multi_market import (
    ALPHA_ARENA_UNIVERSE,
    MarketSnapshot,
    fetch_real_multi,
    make_snapshot,
    synthetic_multi,
)
from .perp_sim import PerpAccount, PerpSimulator

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Result types
# -----------------------------------------------------------------------------


@dataclass
class EquityPoint:
    timestamp: pd.Timestamp
    equity: float
    cash: float
    n_positions: int


@dataclass
class TradeLogEntry:
    timestamp: pd.Timestamp
    symbol: str
    action: str
    side: Optional[str]
    price: float
    qty: float
    leverage: float
    pnl: float
    rationale: str


@dataclass
class ContestResult:
    """Final output of a contest run."""

    starting_capital: float
    final_equity: float
    total_return_pct: float
    benchmark_return_pct: float           # BTC buy-and-hold over the same period
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    n_decisions: int
    n_trades: int
    n_liquidations: int
    fees_paid: float
    funding_paid: float
    duration_hours: float
    universe: list[str]
    used_ai: bool

    equity_curve: list[EquityPoint] = field(default_factory=list)
    trade_log: list[TradeLogEntry] = field(default_factory=list)
    decision_log: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "starting_capital": self.starting_capital,
            "final_equity": round(self.final_equity, 4),
            "total_return_pct": round(self.total_return_pct, 4),
            "benchmark_return_pct": round(self.benchmark_return_pct, 4),
            "alpha_vs_benchmark": round(
                self.total_return_pct - self.benchmark_return_pct, 4
            ),
            "sharpe": round(self.sharpe, 3),
            "sortino": round(self.sortino, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "n_decisions": self.n_decisions,
            "n_trades": self.n_trades,
            "n_liquidations": self.n_liquidations,
            "fees_paid": round(self.fees_paid, 4),
            "funding_paid": round(self.funding_paid, 4),
            "duration_hours": self.duration_hours,
            "universe": self.universe,
            "used_ai": self.used_ai,
            "equity_curve": [
                {
                    "timestamp": p.timestamp.isoformat(),
                    "equity": round(p.equity, 4),
                    "cash": round(p.cash, 4),
                    "n_positions": p.n_positions,
                }
                for p in self.equity_curve
            ],
            "trade_log": [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "symbol": t.symbol,
                    "action": t.action,
                    "side": t.side,
                    "price": round(t.price, 4),
                    "qty": round(t.qty, 8),
                    "leverage": t.leverage,
                    "pnl": round(t.pnl, 4),
                    "rationale": t.rationale,
                }
                for t in self.trade_log
            ],
            "decision_log": self.decision_log,
        }


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def run_contest(
    *,
    starting_capital: float = 10_000.0,
    universe: list[str] | None = None,
    duration_days: int = 30,
    decision_interval_hours: int = 4,
    use_real_data: bool = False,
    use_ai: bool = False,
    seed: int = 42,
) -> ContestResult:
    """Run a full Alpha Arena contest from start to finish.

    Args:
        starting_capital: USDC amount to start with. Alpha Arena uses $10k.
        universe: list of symbols (default ["BTC", "ETH", "SOL"])
        duration_days: total contest length in days
        decision_interval_hours: how often the agent makes decisions
        use_real_data: True → yfinance hourly data; False → synthetic
        use_ai: True → use Claude agent; False → use heuristic baseline
        seed: synthetic data seed for reproducibility

    Returns a ContestResult with full trade log and equity curve.
    """
    universe = universe or ALPHA_ARENA_UNIVERSE
    n_bars = duration_days * 24  # 1h bars

    # 1. Get market data
    if use_real_data:
        try:
            frames = fetch_real_multi(symbols=universe, timeframe="1h", limit=n_bars + 50)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Real data fetch failed (%s); falling back to synthetic", exc)
            frames = synthetic_multi(symbols=universe, limit=n_bars + 50, seed=seed)
    else:
        frames = synthetic_multi(symbols=universe, limit=n_bars + 50, seed=seed)

    # Trim to common length
    common_len = min(len(df) for df in frames.values())
    if common_len < 50:
        raise ValueError(
            f"Insufficient data: only {common_len} bars across symbols (need ≥50)"
        )
    n_steps = min(common_len, n_bars)
    for sym in frames:
        frames[sym] = frames[sym].iloc[-n_steps:]

    # 2. Initialize simulator
    sim = PerpSimulator(starting_capital=starting_capital)

    # 3. Run the loop
    equity_curve: list[EquityPoint] = []
    trade_log: list[TradeLogEntry] = []
    decision_log: list[dict] = []
    n_decisions = 0

    history_window = 24

    for step in range(history_window, n_steps):
        snapshot = make_snapshot(frames, step, history_window=history_window)

        # Step the simulator (apply funding, check liquidations)
        liquidated = sim.step(snapshot.prices)
        for sym in liquidated:
            trade_log.append(
                TradeLogEntry(
                    timestamp=snapshot.timestamp,
                    symbol=sym,
                    action="liquidated",
                    side=None,
                    price=snapshot.prices.get(sym, 0.0),
                    qty=0.0,
                    leverage=0.0,
                    pnl=0.0,
                    rationale="margin ratio fell below maintenance — forced close",
                )
            )

        # Decision time?
        if step % decision_interval_hours == 0:
            n_decisions += 1
            try:
                batch = decide(snapshot, sim.account, universe, use_ai=use_ai)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Decision step failed: %s", exc)
                batch = DecisionBatch(decisions=[], overall_thesis=f"error: {exc}")

            decision_log.append(
                {
                    "step": step,
                    "timestamp": snapshot.timestamp.isoformat(),
                    "thesis": batch.overall_thesis,
                    "decisions": [d.model_dump() for d in batch.decisions],
                    "equity_at_decision": sim.equity(snapshot.prices),
                }
            )

            for d in batch.decisions:
                _execute_decision(d, sim, snapshot, trade_log)

        # Record equity at every step
        equity = sim.equity(snapshot.prices)
        equity_curve.append(
            EquityPoint(
                timestamp=snapshot.timestamp,
                equity=equity,
                cash=sim.account.cash,
                n_positions=len(sim.account.positions),
            )
        )

    # 4. Finalize
    final_prices = {sym: float(frames[sym]["close"].iloc[-1]) for sym in universe}
    for sym in list(sim.account.positions.keys()):
        sim.close_position(sym, final_prices[sym])
    final_equity = sim.account.cash

    # 5. Compute stats
    total_return_pct = (final_equity / starting_capital - 1.0) * 100.0

    # BTC buy-and-hold benchmark
    btc_frame = frames.get("BTC")
    if btc_frame is not None and len(btc_frame) >= 2:
        benchmark_return_pct = (
            float(btc_frame["close"].iloc[-1]) / float(btc_frame["close"].iloc[0]) - 1.0
        ) * 100.0
    else:
        benchmark_return_pct = 0.0

    sharpe, sortino, max_dd = _compute_risk_stats(equity_curve)

    return ContestResult(
        starting_capital=starting_capital,
        final_equity=final_equity,
        total_return_pct=total_return_pct,
        benchmark_return_pct=benchmark_return_pct,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_pct=max_dd,
        n_decisions=n_decisions,
        n_trades=sim.account.n_trades,
        n_liquidations=sim.account.n_liquidations,
        fees_paid=sim.account.fees_paid,
        funding_paid=sim.account.funding_paid,
        duration_hours=float(n_steps - history_window),
        universe=universe,
        used_ai=use_ai,
        equity_curve=equity_curve,
        trade_log=trade_log,
        decision_log=decision_log,
    )


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------


def _execute_decision(
    d: TradeDecision,
    sim: PerpSimulator,
    snapshot: MarketSnapshot,
    trade_log: list[TradeLogEntry],
) -> None:
    """Translate one TradeDecision into PerpSimulator calls."""
    price = snapshot.prices.get(d.symbol)
    if price is None:
        return

    if d.action == "hold":
        return

    if d.action == "close":
        success, pnl = sim.close_position(d.symbol, price)
        if success:
            trade_log.append(
                TradeLogEntry(
                    timestamp=snapshot.timestamp,
                    symbol=d.symbol,
                    action="close",
                    side=None,
                    price=price,
                    qty=0.0,
                    leverage=0.0,
                    pnl=pnl,
                    rationale=d.rationale,
                )
            )
        return

    # open_long / open_short
    if d.confidence < 0.6:
        return  # discipline rule from system prompt
    side = "long" if d.action == "open_long" else "short"
    equity = sim.equity(snapshot.prices)
    notional = equity * d.size_pct_of_equity * d.leverage
    if notional <= 0:
        return
    success, reason = sim.open_position(
        symbol=d.symbol,
        side=side,
        notional_usd=notional,
        leverage=d.leverage,
        current_price=price,
    )
    if success:
        pos = sim.account.positions[d.symbol]
        trade_log.append(
            TradeLogEntry(
                timestamp=snapshot.timestamp,
                symbol=d.symbol,
                action=d.action,
                side=side,
                price=pos.entry,
                qty=pos.qty,
                leverage=d.leverage,
                pnl=0.0,
                rationale=d.rationale,
            )
        )
    else:
        logger.debug("Open rejected for %s: %s", d.symbol, reason)


def _compute_risk_stats(curve: list[EquityPoint]) -> tuple[float, float, float]:
    """Sharpe, Sortino, and max drawdown from a per-bar equity curve.

    Sharpe is annualized assuming 1h bars (24*365 periods/year). For 4h
    decisions, the underlying equity is still updated every hour (positions
    are marked-to-market every step), so the 1h annualization is correct.
    """
    if len(curve) < 2:
        return 0.0, 0.0, 0.0

    equity = np.array([p.equity for p in curve])
    returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2 or returns.std() == 0:
        sharpe = 0.0
    else:
        sharpe = float(returns.mean() / returns.std() * math.sqrt(24 * 365))
        if not np.isfinite(sharpe):
            sharpe = 0.0

    downside = returns[returns < 0]
    if len(downside) < 2 or downside.std() == 0:
        sortino = 0.0
    else:
        sortino = float(returns.mean() / downside.std() * math.sqrt(24 * 365))
        if not np.isfinite(sortino):
            sortino = 0.0

    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_dd = float(drawdowns.min() * 100.0)

    return round(sharpe, 4), round(sortino, 4), round(max_dd, 4)
