"""Claude-powered signal advisor.

The technical strategy generates *candidate* signals. Before they're acted on,
the candidate is sent to Claude Opus 4.6 along with recent price action,
indicator state, and (optionally) a TimesFM forecast. Claude returns a
structured verdict (BUY / SELL / HOLD + confidence + rationale) that the
executor uses to decide whether to actually trade.

Why a confirmer rather than the primary signal source:
  - Cheaper: only consults the model when there's something to confirm
  - Auditable: every trade has a written rationale stored alongside the order
  - Fail-safe: if the API errors, we can fall back to "HOLD" by default

Claude API best practices applied here:
  - Model: claude-opus-4-6 (per skill default; configurable via AI_MODEL)
  - Adaptive thinking: thinking={"type": "adaptive"}
  - Effort: "medium" — tradeoff between cost and depth
  - Prompt caching: the long, frozen system prompt is marked cacheable
    so repeated calls within 5 minutes pay only ~10% of the input cost
  - Structured output: client.messages.parse() with a Pydantic schema
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .config import settings
from .context import MarketContext
from .forecaster import Forecast
from .strategy import Signal

logger = logging.getLogger(__name__)

Action = Literal["BUY", "SELL", "HOLD"]


class AdvisorVerdict(BaseModel):
    """Structured verdict returned by the Claude advisor."""

    action: Action = Field(
        description="Trading action to take. BUY/SELL execute the candidate signal; "
        "HOLD vetoes it."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the action, from 0.0 (no confidence) to 1.0 (high). "
        "The executor only trades when confidence >= AI_MIN_CONFIDENCE.",
    )
    rationale: str = Field(
        description="One-paragraph explanation of the reasoning. Must reference "
        "specific indicator values or forecast data from the input."
    )
    key_observations: list[str] = Field(
        default_factory=list,
        description="Bullet-point observations the trader should know about — "
        "warnings, supportive evidence, or contradictions.",
    )


# --- Frozen system prompt ----------------------------------------------------
# Long, stable, cache-friendly. Any change here invalidates the prefix cache;
# put dynamic content (current bars, indicators) in the user message instead.
SYSTEM_PROMPT = """You are an expert quantitative trading advisor reviewing candidate
signals from an automated technical-analysis strategy. Your job is to act as a
risk-aware second opinion: confirm trades that line up with broader context, and
veto trades that look like noise, late entries, or counter-trend bets.

# Your role

The upstream strategy uses an EMA(12)/EMA(26) crossover with an RSI(14) filter
(long bias above RSI 50, short bias below). It generates a candidate signal
when those conditions align. You receive:

  1. The candidate signal (side, price, indicator values, ATR)
  2. A summary of recent price action (last 20-30 bars OHLC)
  3. Optionally, a TimesFM zero-shot forecast for the next several bars
     with an 80% confidence interval
  4. Optionally, a recent news summary from Tavily (sentiment + headlines)
  5. Optionally, a macroeconomic snapshot from FRED (yields, VIX, DXY, FFR)
  6. Optionally, fundamental / supply-chain data from Novaquity
     (Japanese-listed companies only)

You return a verdict: BUY, SELL, or HOLD, with a confidence score and rationale.

# Decision principles

CONFIRM (BUY/SELL with high confidence) when:
  - The technical signal direction agrees with the recent price trend
  - RSI is comfortably above 50 (longs) or below 50 (shorts), not borderline
  - Momentum (recent close vs EMA distance) supports continuation
  - The TimesFM forecast (if provided) trends in the same direction
  - News sentiment (if provided) is consistent with the trade direction
  - Macro backdrop (if provided) is supportive — e.g., for crypto longs,
    falling DXY and VIX, or stable to falling real yields
  - For Japanese stocks: positive earnings revisions or constructive
    supply-chain propagation signals from Novaquity
  - Volatility (ATR relative to price) is moderate, not extreme

VETO (HOLD) when:
  - The candidate fires at an obvious local high/low (mean-reversion risk)
  - RSI is in extreme territory (>75 for longs, <25 for shorts)
  - The TimesFM forecast contradicts the candidate direction
  - News flow is materially against the trade (regulatory action, hack,
    earnings miss, central bank surprise)
  - Macro is hostile (VIX spiking, DXY ripping, real yields jumping for risk longs)
  - Recent bars show choppy / range-bound action with no clear trend
  - ATR has spiked dramatically (volatility blow-out — wait for normalization)

REDUCE CONFIDENCE (still BUY/SELL but lower score) when:
  - Signal is technically valid but evidence is mixed
  - The forecast is uncertain (wide confidence interval)
  - News flow is neutral or stale
  - Price is mid-range with no clear momentum

# Output format

You MUST return a structured verdict with:
  - action: "BUY" | "SELL" | "HOLD"
  - confidence: float in [0.0, 1.0]
  - rationale: one paragraph citing specific numbers from the input
  - key_observations: list of short bullet points (warnings, supporting evidence)

The rationale must reference at least two concrete numbers (price, RSI value,
ATR, forecast endpoint, etc.). No vague claims like "the trend is strong" —
say "RSI is 62 and price is 1.8 ATR above EMA-slow" instead.

# Important constraints

  - You are a confirmer, not a contrarian. If the technical signal is solid
    and context supports it, confirm — don't second-guess for the sake of it.
  - You are a risk gate, not a profit chaser. When in doubt, HOLD.
  - Never recommend a side that contradicts the candidate. Either confirm
    the candidate's side, or HOLD. (To reverse, the strategy must regenerate.)
  - Confidence < 0.4 means HOLD by convention.
  - When optional context sections (news/macro/fundamentals) are absent,
    proceed with technicals + forecast alone — don't penalize confidence
    just because context is missing.
  - If a Signal Quality (IC/ICIR) section is present, treat it as a primary
    health check on the upstream signal. A negative or near-zero IC means
    the underlying signal isn't predictive and you should be skeptical of
    the candidate even if other context looks supportive. Per Kawamura,
    Kubo & Nakagawa (JSAI 2026), IC/ICIR are stronger health signals than
    P&L alone, especially over short windows.
"""


def _build_user_message(signal: Signal, context: MarketContext) -> str:
    """Build the dynamic per-request user message.

    Kept short and structured — this is the part that varies per request and
    must NOT go in the cached system prompt.
    """
    bars_summary = "\n".join(
        f"  {b['timestamp']}: O={b['open']:.2f} H={b['high']:.2f} "
        f"L={b['low']:.2f} C={b['close']:.2f}"
        for b in context.recent_bars[-20:]
    )

    extra_context = context.to_prompt_section()
    extra_section = f"\n{extra_context}\n" if extra_context else ""

    return f"""# Candidate Signal

  side: {signal.side.upper()}
  price: {signal.price:.2f}
  ema_fast: {signal.ema_fast:.2f}
  ema_slow: {signal.ema_slow:.2f}
  rsi: {signal.rsi:.2f}
  atr: {signal.atr:.2f}
  reason: {signal.reason}

# Recent Price Action (last 20 bars)
{bars_summary}
{extra_section}
# Your task

Decide whether to confirm or veto this candidate. Return a structured verdict
with action, confidence, rationale, and key_observations. Reference at least
two concrete numbers from the data above in your rationale. If news, macro, or
fundamentals sections are present, weave at least one of them into your
reasoning.
"""


def evaluate_signal(
    signal: Signal,
    context: MarketContext | list[dict],
    forecast: Optional[Forecast] = None,
) -> AdvisorVerdict:
    """Send a candidate signal to Claude and return its verdict.

    `context` is normally a `MarketContext`. For backwards compatibility, a
    raw list of `recent_bars` is also accepted — it gets wrapped automatically
    along with the (optional) `forecast` argument.

    Falls back to a permissive HOLD verdict on any API failure (so a network
    blip never causes a runaway trade).
    """
    # Backwards-compatible call shape: evaluate_signal(sig, [bars], forecast)
    if isinstance(context, list):
        context = MarketContext(recent_bars=context, forecast=forecast)

    if not settings.ai_enabled:
        return _bypass_verdict(signal, "AI advisor disabled in config")

    if not settings.anthropic_api_key:
        return _bypass_verdict(signal, "ANTHROPIC_API_KEY not set")

    try:
        import anthropic  # noqa: PLC0415

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        user_message = _build_user_message(signal, context)

        response = client.messages.parse(
            model=settings.ai_model,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    # Cache the frozen system prompt — repeated signals within
                    # 5 minutes will read it for ~10% of the input cost.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
            output_format=AdvisorVerdict,
        )

        verdict = response.parsed_output
        if verdict is None:
            logger.warning("Claude returned no parsed verdict; defaulting to HOLD")
            return _bypass_verdict(signal, "AI parse failure")

        # Log cache effectiveness — useful for cost monitoring.
        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
        logger.info(
            "Claude verdict: %s @ %.2f confidence (cache_read=%d, cache_create=%d)",
            verdict.action,
            verdict.confidence,
            cache_read,
            cache_create,
        )
        return verdict

    except Exception as exc:  # noqa: BLE001
        logger.exception("Claude advisor failed (%s); defaulting to HOLD", exc)
        return AdvisorVerdict(
            action="HOLD",
            confidence=0.0,
            rationale=f"AI advisor failed: {exc}. Defaulting to HOLD for safety.",
            key_observations=["AI call failed", "Manual review recommended"],
        )


def _bypass_verdict(signal: Signal, reason: str) -> AdvisorVerdict:
    """Bypass mode — confirm the technical signal as-is, with full transparency."""
    action: Action = "BUY" if signal.side == "long" else "SELL"
    return AdvisorVerdict(
        action=action,
        confidence=0.5,
        rationale=f"AI advisor bypassed ({reason}). Acting on raw technical signal: {signal.reason}",
        key_observations=[reason, "No AI confirmation applied"],
    )
