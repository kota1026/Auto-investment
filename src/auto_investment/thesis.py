"""Investment thesis builder — Dexter-inspired multi-agent loop.

Inspired by virattt/dexter, the open-source "Claude Code for finance":
  Planning → Action → Validation → Answer

The core insight from Dexter is the **Validation Agent**: after the main
analysis runs, a second LLM pass is fed the analysis + the underlying data
and asked to spot inaccuracies, hallucinations, or unsupported claims. The
final output is either the validated thesis or a downgraded one with the
validator's caveats appended.

Adapted to our setting:
  1. Planning  : decide what data to fetch for this symbol
                 (handled implicitly — the function pulls our standard set)
  2. Action    : fetch price history + indicators + news + macro + (Novaquity)
  3. Analysis  : Claude Opus 4.6 writes a structured InvestmentThesis
  4. Validation: a second Claude pass checks the thesis against the data
                 and emits a ValidationReport (verdict + flagged issues)

Use case: "give me a one-paragraph investment thesis for BTC right now,
backed by current price action, macro, and news, and double-checked by an
independent validator."
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .config import settings
from .context import MarketContext
from .data import fetch_ohlcv
from .forecaster import forecast_close
from .indicators import add_indicators
from .macro import fetch_macro_snapshot
from .news import fetch_news
from .novaquity import fetch_fundamentals

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Output schemas
# -----------------------------------------------------------------------------


class InvestmentThesis(BaseModel):
    """Structured investment thesis emitted by the Analysis Agent."""

    symbol: str = Field(description="The market symbol this thesis is about.")
    direction: Literal["bullish", "bearish", "neutral"] = Field(
        description="Overall directional view over the relevant horizon."
    )
    horizon: str = Field(
        description="Investment horizon — e.g. '1-3 days', '2 weeks', '1-3 months'."
    )
    summary: str = Field(
        description="One-paragraph thesis. Must reference at least three concrete "
        "data points from the input (price level, indicator value, news event, "
        "macro reading, etc.)."
    )
    catalysts: list[str] = Field(
        default_factory=list,
        description="Specific events or data points that support the thesis.",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Specific risks that could invalidate the thesis.",
    )
    invalidation_level: Optional[str] = Field(
        default=None,
        description="A concrete price level or event that would falsify the thesis.",
    )


class ValidationReport(BaseModel):
    """Output of the Validation Agent — independent check on the thesis."""

    verdict: Literal["validated", "amended", "rejected"] = Field(
        description="validated = thesis stands; amended = stands with caveats; "
        "rejected = thesis is unsupported by the data."
    )
    issues_found: list[str] = Field(
        default_factory=list,
        description="Specific problems with the thesis: factual errors, "
        "unsupported claims, missing context, internal contradictions.",
    )
    confidence_adjustment: float = Field(
        ge=-1.0,
        le=1.0,
        description="Adjustment to apply to thesis confidence in [-1, 1]. "
        "Negative downgrades, positive upgrades.",
    )
    final_recommendation: str = Field(
        description="One sentence: should the trader act on this thesis?"
    )


class ThesisResult(BaseModel):
    """Combined output: the thesis plus its validation report."""

    thesis: InvestmentThesis
    validation: ValidationReport
    context_used: dict


# -----------------------------------------------------------------------------
# Agent prompts
# -----------------------------------------------------------------------------


ANALYST_SYSTEM_PROMPT = """You are an expert buy-side investment analyst writing
short-form theses for an automated trading system. Each thesis must be:

  - Grounded in the data the system gives you (price, indicators, news, macro)
  - Specific: cite numbers, headlines, and indicator readings, not vague phrases
  - Actionable: include catalysts, risks, and a concrete invalidation level
  - Honest about uncertainty — if the data doesn't support a conviction call,
    say "neutral" rather than forcing a directional view

You are NOT permitted to invent facts. Every claim in the thesis must be
supportable from the input data the user provides. If a claim depends on
information you don't have, say so and downgrade conviction.

The thesis horizon should match the timeframe of the input data — for hourly
bars, think days to weeks, not months.
"""

VALIDATOR_SYSTEM_PROMPT = """You are a senior risk officer reviewing an
investment thesis written by a junior analyst. Your job is to spot:

  - Factual errors (claims that contradict the underlying data)
  - Unsupported claims (statements with no data backing)
  - Internal contradictions (the summary says bullish but the catalysts are
    all bearish, or the invalidation level is on the wrong side of price)
  - Missing context (a major red flag in the data the analyst ignored)
  - Selection bias (cherry-picked data points that don't represent the whole)

You receive:
  1. The original input data (the same data the analyst was given)
  2. The thesis the analyst produced

Return a structured ValidationReport with verdict, issues_found,
confidence_adjustment, and a one-sentence final_recommendation.

VERDICT GUIDE:
  - "validated" → the thesis is well-supported and you'd act on it
  - "amended"  → the thesis is mostly right but has caveats worth noting
  - "rejected" → the thesis is unsupported or contradicts the data

CONFIDENCE ADJUSTMENT:
  - 0.0 means no change
  - -0.3 means downgrade (caveats reduce conviction)
  - -1.0 means flip to neutral / no trade
  - +0.2 means the thesis is actually understating the case

Be specific in `issues_found` — say WHAT and WHERE, not just "has issues".
"""


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def build_thesis(symbol: str | None = None, limit: int = 300) -> ThesisResult | None:
    """Run the full Dexter-style loop and return a validated thesis.

    Returns None if AI is disabled. The function ALWAYS gathers context (so
    even bypass mode users can see what was collected) — only the LLM passes
    are skipped.
    """
    symbol = symbol or settings.symbol

    # ---- Action: gather all context ------------------------------------------
    df = fetch_ohlcv(symbol=symbol, limit=limit)
    df = add_indicators(df)

    last = df.iloc[-1]
    indicators_snapshot = {
        "last_close": float(last["close"]),
        "ema_fast": float(last["ema_fast"]) if not _isnan(last["ema_fast"]) else None,
        "ema_slow": float(last["ema_slow"]) if not _isnan(last["ema_slow"]) else None,
        "rsi": float(last["rsi"]) if not _isnan(last["rsi"]) else None,
        "atr": float(last["atr"]) if not _isnan(last["atr"]) else None,
    }

    try:
        forecast = forecast_close(df["close"], horizon=24)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Forecast failed in thesis builder: %s", exc)
        forecast = None

    news = fetch_news(symbol)
    macro = fetch_macro_snapshot()
    fundamentals = None
    if symbol.isdigit() and len(symbol) == 4:
        snap = fetch_fundamentals(symbol)
        if snap is not None:
            fundamentals = snap.to_dict()

    context = MarketContext(
        recent_bars=[
            {
                "timestamp": ts.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
            for ts, row in df.tail(30).iterrows()
        ],
        forecast=forecast,
        news_summary=news.summary if news else None,
        news_items=news.items if news else [],
        macro_snapshot=macro,
        fundamentals=fundamentals,
    )

    context_used = {
        "symbol": symbol,
        "indicators": indicators_snapshot,
        "context": {
            "forecast": forecast.to_dict() if forecast else None,
            "news_summary": context.news_summary,
            "macro": context.macro_snapshot,
            "fundamentals": context.fundamentals,
        },
    }

    if not (settings.ai_enabled and settings.anthropic_api_key):
        logger.info("Thesis builder: AI disabled, returning context only")
        return None

    try:
        import anthropic  # noqa: PLC0415

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        # ---- Analysis Agent --------------------------------------------------
        analyst_user_message = _build_analyst_prompt(symbol, indicators_snapshot, context)
        analyst_response = client.messages.parse(
            model=settings.ai_model,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            system=[
                {
                    "type": "text",
                    "text": ANALYST_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": analyst_user_message}],
            output_format=InvestmentThesis,
        )
        thesis = analyst_response.parsed_output
        if thesis is None:
            logger.warning("Analyst returned no parsed thesis")
            return None

        # ---- Validation Agent (Dexter's key idea) ----------------------------
        validator_user_message = _build_validator_prompt(thesis, indicators_snapshot, context)
        validator_response = client.messages.parse(
            model=settings.ai_model,
            max_tokens=1024,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            system=[
                {
                    "type": "text",
                    "text": VALIDATOR_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": validator_user_message}],
            output_format=ValidationReport,
        )
        validation = validator_response.parsed_output
        if validation is None:
            # Validator failed — be conservative
            validation = ValidationReport(
                verdict="amended",
                issues_found=["Validator pass returned no parsed output"],
                confidence_adjustment=-0.3,
                final_recommendation="Manual review recommended.",
            )

        return ThesisResult(thesis=thesis, validation=validation, context_used=context_used)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Thesis builder failed: %s", exc)
        return None


# -----------------------------------------------------------------------------
# Prompt builders
# -----------------------------------------------------------------------------


def _build_analyst_prompt(symbol: str, indicators: dict, context: MarketContext) -> str:
    bars = "\n".join(
        f"  {b['timestamp']}: O={b['open']:.2f} H={b['high']:.2f} "
        f"L={b['low']:.2f} C={b['close']:.2f}"
        for b in context.recent_bars[-20:]
    )
    extra = context.to_prompt_section()
    extra_section = f"\n{extra}\n" if extra else ""
    return f"""# Symbol
{symbol}

# Latest Indicators
  last_close: {indicators['last_close']}
  ema_fast: {indicators['ema_fast']}
  ema_slow: {indicators['ema_slow']}
  rsi: {indicators['rsi']}
  atr: {indicators['atr']}

# Recent Price Action (last 20 bars)
{bars}
{extra_section}
# Your task

Write a structured investment thesis for {symbol}. Cite at least three concrete
data points from the input. Include specific catalysts, risks, and a concrete
invalidation level (a price the market would have to reach for you to abandon
this view). If the data is genuinely mixed, return direction="neutral".
"""


def _build_validator_prompt(
    thesis: InvestmentThesis, indicators: dict, context: MarketContext
) -> str:
    bars = "\n".join(
        f"  {b['timestamp']}: O={b['open']:.2f} H={b['high']:.2f} "
        f"L={b['low']:.2f} C={b['close']:.2f}"
        for b in context.recent_bars[-20:]
    )
    extra = context.to_prompt_section()
    extra_section = f"\n{extra}\n" if extra else ""
    thesis_block = (
        f"  symbol: {thesis.symbol}\n"
        f"  direction: {thesis.direction}\n"
        f"  horizon: {thesis.horizon}\n"
        f"  summary: {thesis.summary}\n"
        f"  catalysts: {'; '.join(thesis.catalysts)}\n"
        f"  risks: {'; '.join(thesis.risks)}\n"
        f"  invalidation_level: {thesis.invalidation_level}\n"
    )
    return f"""# Original Input Data

## Indicators
  last_close: {indicators['last_close']}
  ema_fast: {indicators['ema_fast']}
  ema_slow: {indicators['ema_slow']}
  rsi: {indicators['rsi']}
  atr: {indicators['atr']}

## Recent Price Action
{bars}
{extra_section}
# Analyst's Thesis (to be reviewed)

{thesis_block}

# Your task

Independently validate the thesis above against the input data. Look for
factual errors, unsupported claims, internal contradictions, and missing
context. Return a ValidationReport.
"""


def _isnan(x) -> bool:
    return x != x
