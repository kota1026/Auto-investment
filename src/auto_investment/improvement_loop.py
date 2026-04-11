"""Iterative LLM-driven strategy improvement loop.

This module is a faithful adaptation of the JSAI paper:

  Kawamura, Kubo, Nakagawa (2026)
  「大規模言語モデルを用いた株式投資戦略の自動生成におけるフィードバック設計」
  人工知能学会第二種研究会資料 金融情報学研究会 SIG-FIN-036-31

The paper builds an iterative loop where an LLM is shown a strategy's
backtest results and asked to propose improvements. They test 3 prompt
designs (P1, P2, P3) varying along two axes:

  - Information scope: basic only vs basic + additional (IC, factor exposure)
  - Presentation format: text only vs text + plots

Their key findings:
  1. Performance improvement depends MUCH more on model choice than feedback
     design — Claude family was best (Sonnet 4.5 = 14.12%, Opus 4.5 = 12.69%),
     GPT family worst (GPT-5 = -0.29%)
  2. Adding more info to the feedback (P1 → P2) HURT performance on average
     (-1.30%). Plots (P1 → P3) were neutral (+0.00%)
  3. BUT — feedback design DID change the *quality* of code: P2 induced more
     factor neutralization implementations, P3 induced more dynamic gating
     (IC + VIX based regime adaptation)

Adaptations for our crypto-EMA-RSI setting:

  - The paper rewrites the strategy *code* on each iteration. We instead
    propose **parameter changes** (ema_fast, ema_slow, rsi_threshold,
    sl_atr_mult, tp_atr_mult), which are safer (no exec of untrusted code)
    and align with our existing optimizer module.
  - The "APPROVED" termination signal is preserved.
  - The P1/P2/P3 prompts are translated to crypto-relevant metrics.
  - We use claude-opus-4-6 by default, validated as the best family by the
    paper's experiments.

To use:
    from auto_investment.improvement_loop import run_improvement_loop, FeedbackMode
    result = run_improvement_loop(df, mode=FeedbackMode.P2, max_iterations=5)
    print(result.history[-1].rationale)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

from .backtest import run_backtest
from .config import settings
from .ic import forward_returns, ic_report, signal_from_indicators
from .indicators import add_indicators
from .neutralization import build_factor_panel, neutralize
from .strategy import generate_signals

logger = logging.getLogger(__name__)


class FeedbackMode(str, Enum):
    """The three feedback designs from the JSAI paper."""

    P1 = "P1"  # basic info only, text
    P2 = "P2"  # basic + additional (IC, factor exposure), text
    P3 = "P3"  # basic + additional + plots (we substitute richer text)


# -----------------------------------------------------------------------------
# Pydantic schema for the LLM's response
# -----------------------------------------------------------------------------


class StrategyParams(BaseModel):
    """Strategy parameters the LLM is allowed to tune."""

    ema_fast: int = Field(ge=2, le=200)
    ema_slow: int = Field(ge=3, le=400)
    rsi_threshold: float = Field(ge=20.0, le=80.0)
    sl_atr_mult: float = Field(ge=0.5, le=5.0)
    tp_atr_mult: float = Field(ge=0.5, le=10.0)


class ImprovementProposal(BaseModel):
    """One iteration's output: either an improved param set or APPROVED."""

    approved: bool = Field(
        description="True if the current params are good enough to deploy. "
        "When True, no parameter changes should be made."
    )
    new_params: Optional[StrategyParams] = Field(
        default=None,
        description="The recommended new parameters. Required when approved=False, "
        "ignored when approved=True.",
    )
    rationale: str = Field(
        description="One paragraph explaining the change (or the approval). Must "
        "reference at least two concrete metrics from the input."
    )
    expected_change: str = Field(
        default="",
        description="What you expect to improve (e.g., 'reduce drawdown', "
        "'increase trade frequency', 'capture more trends').",
    )


# -----------------------------------------------------------------------------
# Loop result containers
# -----------------------------------------------------------------------------


@dataclass
class IterationRecord:
    """One step of the improvement loop."""

    iteration: int
    params: dict
    metrics: dict
    proposal: dict
    approved: bool


@dataclass
class ImprovementResult:
    """Full result of an improvement run."""

    mode: FeedbackMode
    initial_params: dict
    initial_metrics: dict
    final_params: dict
    final_metrics: dict
    history: list[IterationRecord] = field(default_factory=list)
    converged: bool = False
    iterations_used: int = 0

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "initial_params": self.initial_params,
            "initial_metrics": self.initial_metrics,
            "final_params": self.final_params,
            "final_metrics": self.final_metrics,
            "history": [
                {
                    "iteration": r.iteration,
                    "params": r.params,
                    "metrics": r.metrics,
                    "proposal": r.proposal,
                    "approved": r.approved,
                }
                for r in self.history
            ],
            "converged": self.converged,
            "iterations_used": self.iterations_used,
        }


# -----------------------------------------------------------------------------
# System prompt — frozen, cacheable
# -----------------------------------------------------------------------------


SYSTEM_PROMPT_LOOP = """You are a senior quantitative researcher iteratively
improving an automated trading strategy. The strategy is a momentum/trend
follower:

  ENTRY: EMA(fast)/EMA(slow) crossover, filtered by RSI > rsi_threshold for
         longs and RSI < rsi_threshold for shorts.
  EXIT:  ATR-based stop-loss (sl_atr_mult * ATR) and take-profit
         (tp_atr_mult * ATR).
  RISK:  Position sized to risk 1% of equity per trade.

You can tune five parameters per iteration:

  - ema_fast       (int, 2..200)
  - ema_slow       (int, 3..400)   — must be > ema_fast
  - rsi_threshold  (float, 20..80)
  - sl_atr_mult    (float, 0.5..5.0)
  - tp_atr_mult    (float, 0.5..10.0)

You receive backtest metrics for the current parameter set. Your job is to
propose ONE changed parameter set per iteration that you believe will improve
the strategy, OR to APPROVE the current set if it's already production-ready.

# What "production-ready" means

  - Sharpe ratio >= 1.2
  - Win rate >= 45%
  - Max drawdown >= -15% (less negative is better)
  - At least 20 trades over the backtest period
  - Trades aren't all clustered in one regime

# Improvement principles (JSAI paper findings)

  - Make ONE focused change per iteration. Don't tune everything at once —
    you won't be able to attribute the result.
  - Prefer parameter directions supported by the diagnostic data, not random
    exploration. If the IC is positive but trades are losing, the issue is
    likely SL/TP placement, not entry logic.
  - Watch for over-trading: too many small losses with a high trade count
    suggests SL too tight or rsi_threshold too lax.
  - Watch for under-trading: very few trades with mediocre PnL suggests
    rsi_threshold too strict or EMAs too slow.
  - The Kawamura et al. (2026) JSAI paper found that adding more metrics to
    the feedback prompt didn't help performance — what matters is making
    *focused, supported* changes. So don't try to address every metric at
    once. Pick the most binding constraint and address that.

# Approval

If the current parameters meet the production criteria above AND the IC
is positive (signal is genuinely predictive), APPROVE — set approved=true and
leave new_params null. Do NOT keep tweaking a working strategy.

# Output

Return a structured ImprovementProposal:
  - approved: bool
  - new_params: StrategyParams (required when approved=false)
  - rationale: paragraph citing at least 2 concrete metrics
  - expected_change: short phrase like "reduce drawdown by tightening exits"
"""


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


DEFAULT_INITIAL_PARAMS = {
    "ema_fast": 12,
    "ema_slow": 26,
    "rsi_threshold": 50.0,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 3.0,
}


def run_improvement_loop(
    df: pd.DataFrame,
    *,
    mode: FeedbackMode = FeedbackMode.P2,
    initial_params: dict | None = None,
    max_iterations: int = 5,
    initial_equity: float = 10_000.0,
    risk_per_trade: float = 0.01,
) -> ImprovementResult:
    """Iteratively improve strategy parameters using Claude's feedback.

    Implements the JSAI paper's loop with our parameter-tuning adaptation.
    Each iteration:
      1. Run backtest with current params
      2. Compute the diagnostic bundle for the chosen feedback mode
      3. Send to Claude → ImprovementProposal
      4. If proposal.approved, stop; otherwise apply new_params and loop
    """
    initial = dict(initial_params or DEFAULT_INITIAL_PARAMS)
    current = dict(initial)

    initial_metrics = _backtest_with_params(df, current, initial_equity, risk_per_trade)

    result = ImprovementResult(
        mode=mode,
        initial_params=initial,
        initial_metrics=initial_metrics,
        final_params=initial,
        final_metrics=initial_metrics,
    )

    if not (settings.ai_enabled and settings.anthropic_api_key):
        logger.info("Improvement loop: AI disabled, returning baseline-only result")
        return result

    try:
        import anthropic  # noqa: PLC0415

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cannot import anthropic SDK: %s", exc)
        return result

    for iteration in range(1, max_iterations + 1):
        metrics = _backtest_with_params(df, current, initial_equity, risk_per_trade)
        feedback = _build_feedback_prompt(df, current, metrics, mode)

        try:
            response = client.messages.parse(
                model=settings.ai_model,
                max_tokens=2048,
                thinking={"type": "adaptive"},
                output_config={"effort": "medium"},
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT_LOOP,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": feedback}],
                output_format=ImprovementProposal,
            )
            proposal = response.parsed_output
        except Exception as exc:  # noqa: BLE001
            logger.exception("Improvement iteration %d failed: %s", iteration, exc)
            break

        if proposal is None:
            logger.warning("Iteration %d returned no parsed proposal", iteration)
            break

        record = IterationRecord(
            iteration=iteration,
            params=dict(current),
            metrics=metrics,
            proposal=proposal.model_dump(),
            approved=proposal.approved,
        )
        result.history.append(record)
        result.iterations_used = iteration

        if proposal.approved:
            result.converged = True
            result.final_params = dict(current)
            result.final_metrics = metrics
            logger.info("Improvement loop converged at iteration %d (APPROVED)", iteration)
            break

        if proposal.new_params is None:
            logger.warning("Iteration %d not approved but no new_params; stopping", iteration)
            break

        new_params = proposal.new_params.model_dump()
        if new_params["ema_slow"] <= new_params["ema_fast"]:
            logger.warning("Invalid params (ema_slow<=ema_fast); stopping")
            break

        current = new_params
        result.final_params = dict(current)
        result.final_metrics = _backtest_with_params(
            df, current, initial_equity, risk_per_trade
        )

    return result


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------


def _backtest_with_params(
    df: pd.DataFrame, params: dict, initial_equity: float, risk_per_trade: float
) -> dict:
    """Run a backtest with the given params and return a flat metrics dict.

    Includes the JSAI paper's "additional info" metrics (IC, ICIR, factor
    explained_fraction) so the P2/P3 prompts have something to consume.
    """
    df_ind = add_indicators(
        df,
        fast=params["ema_fast"],
        slow=params["ema_slow"],
        rsi_period=14,
        atr_period=14,
    )
    df_sig = generate_signals(df_ind, rsi_threshold=params["rsi_threshold"])

    # Run the backtest via the optimizer's pre-augmented helper to keep the
    # parameter choices intact.
    from .optimizer import _run_backtest_pre_augmented  # noqa: PLC0415

    bt = _run_backtest_pre_augmented(
        df_sig,
        initial_equity=initial_equity,
        risk_per_trade=risk_per_trade,
        sl_atr_mult=params["sl_atr_mult"],
        tp_atr_mult=params["tp_atr_mult"],
    )

    # IC / ICIR on the indicator-derived signal vs forward returns
    sig = signal_from_indicators(df_ind)
    fwd = forward_returns(df_ind["close"], horizon=1)
    ic = ic_report(sig, fwd)

    # Factor neutralization "explained fraction" — how much of the signal is
    # explained by simple style factors. High = signal is mostly noise from
    # common factors.
    factors = build_factor_panel(df_ind)
    _, neut_report = neutralize(sig, factors)

    return {
        "n_trades": int(bt.n_trades),
        "win_rate": round(float(bt.win_rate), 4),
        "avg_rr": round(float(bt.avg_rr), 4),
        "total_return_pct": round(float(bt.total_return_pct), 4),
        "max_drawdown_pct": round(float(bt.max_drawdown_pct), 4),
        "final_equity": round(float(bt.final_equity), 2),
        "ic_mean": ic.mean_ic,
        "ic_std": ic.std_ic,
        "icir": ic.icir,
        "factor_explained_fraction": neut_report.explained_fraction,
        "factor_loadings": neut_report.factor_loadings,
    }


def _build_feedback_prompt(
    df: pd.DataFrame, params: dict, metrics: dict, mode: FeedbackMode
) -> str:
    """Build the per-iteration user message in the chosen JSAI feedback mode.

    Adapts the paper's P1/P2/P3 prompts to our crypto-EMA-RSI strategy.
    """
    base_block = f"""# Current Strategy Parameters
  ema_fast: {params['ema_fast']}
  ema_slow: {params['ema_slow']}
  rsi_threshold: {params['rsi_threshold']}
  sl_atr_mult: {params['sl_atr_mult']}
  tp_atr_mult: {params['tp_atr_mult']}

# Backtest Metrics (Basic)
  n_trades: {metrics['n_trades']}
  win_rate: {metrics['win_rate'] * 100:.1f}%
  avg_rr: {metrics['avg_rr']:.2f}
  total_return: {metrics['total_return_pct']:+.2f}%
  max_drawdown: {metrics['max_drawdown_pct']:.2f}%
  final_equity: ${metrics['final_equity']:,.0f}
"""

    if mode == FeedbackMode.P1:
        return base_block + """
# Your task

Propose ONE focused parameter change to improve the strategy, or APPROVE if
the current parameters meet production criteria. Reference at least two
concrete metrics in your rationale.
"""

    additional_block = f"""
# Additional Diagnostics (P2/P3)

## Signal Quality (IC / ICIR)
  ic_mean: {metrics['ic_mean']:+.4f}     (correlation of signal with next-bar returns)
  ic_std: {metrics['ic_std']:.4f}
  icir: {metrics['icir']:+.3f}            (mean / std — higher = more consistent)

## Factor Exposure (proxy for signal cleanliness)
  factor_explained_fraction: {metrics['factor_explained_fraction']:.2%}
    (Fraction of signal variance explained by simple style factors —
     vol_20, mom_20, volume_z, atr_pct. Lower is better; high values mean
     the signal is mostly common-factor noise rather than alpha.)
  factor_loadings: {metrics['factor_loadings']}
"""

    if mode == FeedbackMode.P2:
        return base_block + additional_block + """
# Your task

Propose ONE focused parameter change to improve the strategy, or APPROVE if
the current parameters meet production criteria. Reference at least two
concrete metrics in your rationale, and weave in at least one of the
additional diagnostics (IC/ICIR or factor exposure) — that's the whole point
of this feedback mode.
"""

    # P3 — paper uses plots; we substitute denser textual time-series
    n = len(df)
    sample_close = df["close"].iloc[max(0, n - 60) :: 5]
    series_block = "\n  ".join(
        f"{ts.date()}: {price:.2f}" for ts, price in sample_close.items()
    )

    return base_block + additional_block + f"""
## Recent Price Action (every 5th bar over last ~60 bars)
  {series_block}

# Your task

Propose ONE focused parameter change to improve the strategy, or APPROVE if
the current parameters meet production criteria. Reference at least two
concrete metrics, weave in IC/ICIR or factor exposure, and consider whether
the recent price action shows a regime where dynamic gating (e.g., trade only
when IC is positive over the last 30 bars) would help. The JSAI paper found
that giving plots to the model induced more dynamic-gating implementations —
think along those lines.
"""
