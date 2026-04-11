"""AutoAgent-style strategy parameter optimizer.

Inspired by the AutoAgent (kevinrgu/autoagent) hill-climbing meta-loop:
iteratively probe a parameter space, score each candidate by a numeric
metric (Sharpe ratio here), and return a ranking. Designed to run overnight
and surface a handful of promising configurations for manual review.

Two search strategies:
  - `grid_search`: exhaustive sweep over a parameter grid
  - `random_search`: uniform-random sampling (better when the grid is huge)

Both score each candidate by Sharpe ratio of the equity curve, with
secondary tiebreakers on win rate and max drawdown.

The output `OptimizationResult` is JSON-serializable for the FastAPI server
and contains a top-N ranking the user can pick from. The rationale for
choosing one config over another is left to the human (or to a follow-up
Claude prompt — see `explain_top_with_claude` below).
"""

from __future__ import annotations

import itertools
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .indicators import add_indicators
from .strategy import generate_signals

logger = logging.getLogger(__name__)


# Default search space — small enough to run in <30s on a laptop CPU.
DEFAULT_GRID: dict[str, list] = {
    "ema_fast": [8, 12, 16, 20],
    "ema_slow": [21, 26, 34, 50],
    "rsi_threshold": [45, 50, 55],
    "sl_atr_mult": [1.0, 1.5, 2.0],
    "tp_atr_mult": [2.0, 3.0, 4.0],
}


@dataclass
class Candidate:
    """One parameter combination + its backtest metrics."""

    params: dict
    sharpe: float
    total_return_pct: float
    win_rate: float
    max_drawdown_pct: float
    n_trades: int

    def to_dict(self) -> dict:
        return {
            "params": self.params,
            "sharpe": float(self.sharpe),
            "total_return_pct": float(self.total_return_pct),
            "win_rate": float(self.win_rate),
            "max_drawdown_pct": float(self.max_drawdown_pct),
            "n_trades": int(self.n_trades),
        }


@dataclass
class OptimizationResult:
    """Full result of an optimization run."""

    candidates: list[Candidate] = field(default_factory=list)
    n_evaluated: int = 0
    best: Candidate | None = None

    def top(self, n: int = 5) -> list[Candidate]:
        return self.candidates[:n]

    def to_dict(self) -> dict:
        return {
            "n_evaluated": self.n_evaluated,
            "best": self.best.to_dict() if self.best else None,
            "top": [c.to_dict() for c in self.top(10)],
        }


def grid_search(
    df: pd.DataFrame,
    grid: dict[str, list] | None = None,
    *,
    initial_equity: float = 10_000.0,
    risk_per_trade: float = 0.01,
    min_trades: int = 5,
) -> OptimizationResult:
    """Exhaustively evaluate every combination in the grid.

    Skips combinations where ema_fast >= ema_slow (those are nonsensical) and
    candidates with fewer than `min_trades` (statistically meaningless).
    """
    grid = grid or DEFAULT_GRID
    combos = _expand_grid(grid)
    return _evaluate_all(df, combos, initial_equity, risk_per_trade, min_trades)


def random_search(
    df: pd.DataFrame,
    grid: dict[str, list] | None = None,
    *,
    n_samples: int = 30,
    initial_equity: float = 10_000.0,
    risk_per_trade: float = 0.01,
    min_trades: int = 5,
    seed: int = 0,
) -> OptimizationResult:
    """Randomly sample `n_samples` combinations from the grid."""
    grid = grid or DEFAULT_GRID
    rng = random.Random(seed)
    all_combos = list(_expand_grid(grid))
    combos = rng.sample(all_combos, min(n_samples, len(all_combos)))
    return _evaluate_all(df, combos, initial_equity, risk_per_trade, min_trades)


def _expand_grid(grid: dict[str, list]) -> Iterator[dict]:
    """Cartesian product of the grid as dicts, with sanity filter."""
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, values))
        if "ema_fast" in params and "ema_slow" in params:
            if params["ema_fast"] >= params["ema_slow"]:
                continue
        yield params


def _evaluate_all(
    df: pd.DataFrame,
    combos: Iterator[dict] | list[dict],
    initial_equity: float,
    risk_per_trade: float,
    min_trades: int,
) -> OptimizationResult:
    candidates: list[Candidate] = []
    n_evaluated = 0
    for params in combos:
        n_evaluated += 1
        try:
            metrics = _backtest_with_params(
                df,
                params,
                initial_equity=initial_equity,
                risk_per_trade=risk_per_trade,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Optimizer skipped %s (%s)", params, exc)
            continue
        if metrics["n_trades"] < min_trades:
            continue
        candidates.append(Candidate(params=params, **metrics))

    candidates.sort(
        key=lambda c: (c.sharpe, c.total_return_pct, -c.max_drawdown_pct),
        reverse=True,
    )
    best = candidates[0] if candidates else None
    return OptimizationResult(candidates=candidates, n_evaluated=n_evaluated, best=best)


def _backtest_with_params(
    df: pd.DataFrame,
    params: dict,
    initial_equity: float,
    risk_per_trade: float,
) -> dict:
    """Run a single backtest with custom strategy parameters."""
    # Recompute indicators with the candidate's EMA periods
    df_ind = add_indicators(
        df,
        fast=params.get("ema_fast", 12),
        slow=params.get("ema_slow", 26),
        rsi_period=14,
        atr_period=14,
    )
    # Re-generate signals with the candidate's RSI threshold
    df_sig = generate_signals(df_ind, rsi_threshold=params.get("rsi_threshold", 50.0))
    # Run the backtester with the candidate's risk multipliers.
    # We bypass run_backtest's internal indicator/signal call by passing the
    # already-augmented frame in via a tiny shim.
    result = _run_backtest_pre_augmented(
        df_sig,
        initial_equity=initial_equity,
        risk_per_trade=risk_per_trade,
        sl_atr_mult=params.get("sl_atr_mult", 1.5),
        tp_atr_mult=params.get("tp_atr_mult", 3.0),
    )

    sharpe = _sharpe_from_equity(result.equity_curve)
    return {
        "sharpe": sharpe,
        "total_return_pct": float(result.total_return_pct),
        "win_rate": float(result.win_rate),
        "max_drawdown_pct": float(result.max_drawdown_pct),
        "n_trades": int(result.n_trades),
    }


def _run_backtest_pre_augmented(
    df_sig: pd.DataFrame,
    initial_equity: float,
    risk_per_trade: float,
    sl_atr_mult: float,
    tp_atr_mult: float,
):
    """Run the backtester on a frame that already has indicators+signals.

    Recreates the entry/exit loop from `backtest.run_backtest` but skips the
    re-augmentation step so the optimizer's parameter choices stick.
    """
    from .backtest import _close, _summarize  # noqa: PLC0415
    from .risk import build_trade_plan  # noqa: PLC0415

    df_sig = df_sig.dropna(subset=["ema_fast", "ema_slow", "rsi", "atr"]).copy()

    equity = initial_equity
    equity_history: list[tuple] = []
    trades: list = []
    open_trade: dict | None = None

    for i, (ts, bar) in enumerate(df_sig.iterrows()):
        if open_trade is not None:
            high = float(bar["high"])
            low = float(bar["low"])
            if open_trade["side"] == "long":
                if low <= open_trade["stop"]:
                    pnl = (open_trade["stop"] - open_trade["entry"]) * open_trade["qty"]
                    trades.append(_close(open_trade, ts, open_trade["stop"], pnl, "sl"))
                    equity += pnl
                    open_trade = None
                elif high >= open_trade["target"]:
                    pnl = (open_trade["target"] - open_trade["entry"]) * open_trade["qty"]
                    trades.append(_close(open_trade, ts, open_trade["target"], pnl, "tp"))
                    equity += pnl
                    open_trade = None
            else:
                if high >= open_trade["stop"]:
                    pnl = (open_trade["entry"] - open_trade["stop"]) * open_trade["qty"]
                    trades.append(_close(open_trade, ts, open_trade["stop"], pnl, "sl"))
                    equity += pnl
                    open_trade = None
                elif low <= open_trade["target"]:
                    pnl = (open_trade["entry"] - open_trade["target"]) * open_trade["qty"]
                    trades.append(_close(open_trade, ts, open_trade["target"], pnl, "tp"))
                    equity += pnl
                    open_trade = None

        if open_trade is None and bar["signal"] in ("long", "short") and i + 1 < len(df_sig):
            next_bar = df_sig.iloc[i + 1]
            entry = float(next_bar["open"])
            atr_val = float(bar["atr"])
            try:
                plan = build_trade_plan(
                    side=bar["signal"],
                    entry=entry,
                    atr_value=atr_val,
                    equity=equity,
                    risk_per_trade=risk_per_trade,
                    sl_atr_mult=sl_atr_mult,
                    tp_atr_mult=tp_atr_mult,
                )
                open_trade = {
                    "entry_time": next_bar.name,
                    "side": plan.side,
                    "entry": plan.entry,
                    "stop": plan.stop,
                    "target": plan.target,
                    "qty": plan.qty,
                }
            except ValueError:
                pass

        equity_history.append((ts, equity))

    if open_trade is not None:
        last_close = float(df_sig["close"].iloc[-1])
        if open_trade["side"] == "long":
            pnl = (last_close - open_trade["entry"]) * open_trade["qty"]
        else:
            pnl = (open_trade["entry"] - last_close) * open_trade["qty"]
        trades.append(_close(open_trade, df_sig.index[-1], last_close, pnl, "eod"))
        equity += pnl
        if equity_history:
            equity_history[-1] = (df_sig.index[-1], equity)

    equity_curve = pd.Series(
        [eq for _, eq in equity_history],
        index=pd.DatetimeIndex([ts for ts, _ in equity_history]),
        name="equity",
    )
    return _summarize(trades, equity_curve, initial_equity)


def _sharpe_from_equity(equity_curve: pd.Series, periods_per_year: int = 365 * 24) -> float:
    """Compute an annualized Sharpe ratio from a per-bar equity series.

    Defaults assume 1-hour bars (24/day, 365/year). For daily bars, pass
    `periods_per_year=252`.
    """
    if len(equity_curve) < 2:
        return 0.0
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    sharpe = float(returns.mean() / returns.std() * math.sqrt(periods_per_year))
    if not np.isfinite(sharpe):
        return 0.0
    return round(sharpe, 4)


def explain_top_with_claude(result: OptimizationResult, top_n: int = 5) -> str | None:
    """Optionally have Claude pick the best config and explain why.

    This is the AutoAgent twist: rather than blindly trusting the top-Sharpe
    candidate (which can be lucky), ask Claude Opus 4.6 to weigh the top N
    candidates and recommend one with a written rationale.

    Returns None if AI is disabled or the API call fails.
    """
    from .ai_advisor import settings as ai_settings  # noqa: PLC0415

    if not (ai_settings.ai_enabled and ai_settings.anthropic_api_key):
        return None
    if not result.candidates:
        return None

    try:
        import anthropic  # noqa: PLC0415

        client = anthropic.Anthropic(api_key=ai_settings.anthropic_api_key)
        top = result.top(top_n)
        table = "\n".join(
            f"  {i+1}. {c.params}  →  Sharpe {c.sharpe:.2f}, "
            f"return {c.total_return_pct:+.1f}%, win {c.win_rate*100:.0f}%, "
            f"DD {c.max_drawdown_pct:.1f}%, trades {c.n_trades}"
            for i, c in enumerate(top)
        )
        prompt = (
            f"Here are the top {len(top)} parameter sets from a strategy optimization run:\n\n"
            f"{table}\n\n"
            "Recommend ONE configuration for live trading. Consider Sharpe, drawdown, "
            "trade count (higher is more statistically reliable), and parameter robustness "
            "(don't pick a configuration that looks like an outlier vs its neighbors). "
            "Respond with: (1) the index of your pick, (2) a 2-3 sentence rationale, "
            "(3) one concrete risk to watch."
        )
        response = client.messages.create(
            model=ai_settings.ai_model,
            max_tokens=1024,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            messages=[{"role": "user", "content": prompt}],
        )
        return next(
            (b.text for b in response.content if getattr(b, "type", None) == "text"),
            None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Claude explanation failed: %s", exc)
        return None
