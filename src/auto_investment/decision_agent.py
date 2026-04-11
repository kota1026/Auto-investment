"""Claude-as-trader decision agent for the Alpha Arena contest format.

This is fundamentally different from `ai_advisor.py`, which is a *confirmer*
that says yes/no to a pre-generated technical signal. Here Claude IS the
trader: it sees the full market state and current portfolio, then emits
discrete trade decisions (open/close, leverage, size).

# Why a separate module?

The advisor is reactive and bounded — it only votes on a single candidate.
The decision agent is proactive and unbounded — it has to decide WHAT to
trade, WHICH side, HOW much leverage, and WHEN to close, every N hours,
across multiple symbols. Different cognitive task, different prompt.

# Prompting strategy: channel the Alpha Arena winners

The Nof1 Alpha Arena Season 1 results are unambiguous:
  - Qwen3 Max won (+22-31%) with disciplined, low-trade-count execution
  - DeepSeek V3.1 won (+48%) with similar discipline
  - GPT-5, Claude Sonnet 4.5, Grok 4, Gemini 2.5 Pro all LOST 12-75%
  - The losers all overtraded, used inconsistent leverage, and panicked

What the winners had in common (per Nof1's published trade analysis):
  1. Few high-conviction trades, not many low-conviction ones
  2. Consistent leverage band (didn't oscillate between 5x and 30x)
  3. Held winners, cut losers fast (the inverse of human bias)
  4. No emotional revenge trading after losses
  5. Aware of the funding rate (avoided being long when funding was high)

The system prompt below explicitly instills these principles. We use Claude
Opus 4.6 (per our default) but prompt it with the *Qwen-style discipline*
that won Alpha Arena. Whether this generalizes is exactly what we want to
test in our 6-month paper run.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .config import settings
from .multi_market import MarketSnapshot
from .perp_sim import PerpAccount

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Decision schema
# -----------------------------------------------------------------------------


Action = Literal["open_long", "open_short", "close", "hold"]


class TradeDecision(BaseModel):
    """One trade decision the agent emits per symbol per step."""

    symbol: str = Field(description="Symbol from the universe (e.g. BTC, ETH, SOL)")
    action: Action = Field(
        description="open_long/open_short opens a NEW position. close exits "
        "an existing one. hold does nothing."
    )
    size_pct_of_equity: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="For open_* actions: notional position size as fraction "
        "of total equity (e.g. 0.20 = 20% of account). Ignored for close/hold.",
    )
    leverage: float = Field(
        ge=1.0,
        le=20.0,
        default=3.0,
        description="Leverage multiplier for open_* actions. The Alpha Arena "
        "winners used 10-17x; the losers oscillated wildly. Pick a value you "
        "could justify and stick with it.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Subjective confidence in this decision. Used by the "
        "improvement loop to weight historical decision quality.",
    )
    rationale: str = Field(
        description="One sentence explaining WHY. Must reference at least "
        "one concrete number from the market snapshot."
    )


class DecisionBatch(BaseModel):
    """The agent's full decision set for one step — one decision per symbol."""

    decisions: list[TradeDecision] = Field(
        description="One TradeDecision per symbol in the universe. Use action='hold' "
        "for symbols you don't want to trade right now."
    )
    overall_thesis: str = Field(
        description="One paragraph summary of your view on the market right now "
        "and how the decisions align with that view."
    )


# -----------------------------------------------------------------------------
# System prompt — frozen, cacheable
# -----------------------------------------------------------------------------


DECISION_AGENT_SYSTEM_PROMPT = """You are an autonomous crypto perpetual futures
trader competing in the Nof1 Alpha Arena format. You have a multi-symbol
USDC-margined account on a Hyperliquid-style exchange. Every 4 hours, you
receive a market snapshot and your current portfolio state, and you must
emit one trade decision per symbol.

# Goal

Maximize risk-adjusted return (Sharpe ratio) over the contest period. Your
opponents are other LLMs running the same loop. You will be ranked by final
account equity at the end of the period. Beating BTC buy-and-hold is the
minimum bar.

# What we know works (Alpha Arena Season 1 results, audited on-chain)

Two models won out of six. Both were Chinese (Qwen3 Max, DeepSeek V3.1).
The four Western frontier models (Claude Sonnet 4.5, GPT-5, Gemini 2.5 Pro,
Grok 4) all lost between -12% and -75%. The reasons the winners won, per
post-hoc analysis of their trade logs:

  1. **Few trades, high conviction**. Winners averaged ~3 trades per day
     across 3 symbols. Losers averaged 8-15 trades per day, paying enormous
     fee bleed. RULE: open a position only when you have a clear reason.

  2. **Consistent leverage**. Winners used a stable 10-17x leverage band on
     every trade. Losers oscillated between 3x and 30x and got liquidated.
     RULE: pick a leverage range and stay in it. Vary SIZE not leverage.

  3. **Hold winners, cut losers fast**. Winners closed losing trades within
     1-2 funding periods (4-8 hours). Losers held losers hoping for recovery,
     and closed winners early to "lock in" small gains. RULE: a position
     that's down >2% on the next bar should be closed. A position that's
     up >5% should be held until a clear reversal signal.

  4. **No revenge trading**. After a loss, winners waited at least one full
     decision cycle before opening a new position. Losers immediately doubled
     down. RULE: if you just took a loss, default to 'hold' next cycle unless
     the setup is exceptional.

  5. **Funding rate awareness**. Long positions pay funding when the rate is
     positive. At 0.01%/hour that's 0.24%/day or ~7%/month — devastating
     to a leveraged long. RULE: prefer short positions when funding is
     persistently positive (>0.02%/hour).

  6. **Cross-symbol risk budget**. Don't be max-long all three symbols
     simultaneously — they're correlated and a BTC drawdown will take the
     whole portfolio. RULE: net long exposure across all positions should
     not exceed 1.5x equity (e.g. with $10k, total long notional ≤ $15k).

# Macro context (when provided)

If the input includes a "Macro (FRED)" block, it contains the current state
of the broader financial environment. These are free from the St. Louis Fed
and replace Bloomberg Terminal's ($24k/year) equivalent data:

  - **DGS10** (10-year Treasury yield, %): Rising fast → dollar strength,
    risk-off, bearish for crypto longs. Falling → supportive for risk assets.
  - **DGS2** (2-year Treasury yield, %): Tracks Fed policy expectations.
    (DGS10 - DGS2) > 0 → normal yield curve; < 0 → inverted, recession signal.
  - **VIXCLS** (VIX, S&P 500 volatility): < 15 = complacent, 15-25 = normal,
    > 30 = panic. Crypto usually follows risk assets — high VIX = bearish
    for crypto longs.
  - **DTWEXBGS** (USD broad index): Rising DXY = dollar strength = bearish
    for crypto and gold (inverse correlation).
  - **DFF** (Fed funds rate): Rising cycle = tightening, bearish for risk.

RULE: When VIX > 25 or DXY is rising fast, reduce net long exposure. When
VIX < 15 and DXY is falling, be willing to take more leverage on the long
side. This is how institutional crypto desks adjust position sizing.

# News context (when provided)

If the input includes a "Recent news" block from Tavily, it's a summary of
the last 48 hours of crypto-relevant headlines. Use it as a veto mechanism:

  - If news mentions a hack, exchange failure, regulatory action, or
    major protocol exploit → do NOT open new long positions, consider
    closing existing ones.
  - If news mentions institutional adoption (ETF approval, corporate
    treasury, sovereign adoption) → momentum trades in that direction
    are better-supported.
  - Stale or neutral news → no adjustment needed.

# Decision principles

For each symbol, ask yourself in this order:

  1. **Do I already have a position?** If yes:
     - Down >2% from entry? → close (action=close)
     - Up >5% from entry and no reversal signal? → hold (action=hold)
     - Otherwise → hold

  2. **Is there a clear setup?** Look for:
     - Strong recent momentum (24h return > +3% for long, < -3% for short)
       AND volatility not extreme (vol_24h < 5%)
     - The setup direction agrees with the BTC trend (BTC leads the market)
     - No active position on this symbol
     If yes → open with size 15-25% of equity, leverage 5-12x.

  3. **Otherwise** → hold (action=hold, size=0).

Never open a position with confidence < 0.6. Never use leverage > 15x in
the current market. If the BTC return over 24h is between -1% and +1%
(chop), default to 'hold' on everything.

# Output format

You return a DecisionBatch with:
  - decisions: list of TradeDecision, ONE PER SYMBOL in the input universe
  - overall_thesis: paragraph summary of your view

Each TradeDecision must specify:
  - symbol
  - action: open_long | open_short | close | hold
  - size_pct_of_equity (0.0 - 1.0): fraction of equity for new positions
  - leverage (1.0 - 20.0): leverage multiplier
  - confidence (0.0 - 1.0): your conviction
  - rationale: one sentence with at least one concrete number

# Constraints (HARD limits)

  - Maximum leverage: 15x (you can be liquidated above this)
  - Maximum size per position: 30% of equity
  - Maximum total long notional: 1.5x equity across all positions
  - Maximum total short notional: 1.5x equity across all positions
  - Confidence < 0.6 → must be 'hold'
  - You cannot reverse a position in one step (close first, then open the
    other side on the next decision cycle)

Discipline beats brilliance. Be a Qwen3 Max, not a GPT-5.

When macro or news context is provided, weave at least ONE concrete reading
into your rationale (e.g. "VIX at 14.5 supports risk-on stance" or "Fed
funds at 5.33 is hostile to crypto long"). Don't just ignore context.
"""


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def decide(
    snapshot: MarketSnapshot,
    account: PerpAccount,
    universe: list[str],
    *,
    use_ai: bool = True,
) -> DecisionBatch:
    """Ask Claude what trades to take given the current state.

    Falls back to a deterministic disciplined-momentum heuristic when AI is
    disabled or fails — that way the contest can run end-to-end without an
    API key for testing.
    """
    if not (use_ai and settings.ai_enabled and settings.anthropic_api_key):
        return _heuristic_decide(snapshot, account, universe)

    try:
        import anthropic  # noqa: PLC0415

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        user_message = _build_user_message(snapshot, account, universe)

        response = client.messages.parse(
            model=settings.ai_model,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            system=[
                {
                    "type": "text",
                    "text": DECISION_AGENT_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
            output_format=DecisionBatch,
        )
        if response.parsed_output is None:
            logger.warning("Decision agent returned no parsed output; using heuristic")
            return _heuristic_decide(snapshot, account, universe)
        return response.parsed_output
    except Exception as exc:  # noqa: BLE001
        logger.exception("Decision agent failed (%s); using heuristic", exc)
        return _heuristic_decide(snapshot, account, universe)


def _build_user_message(
    snapshot: MarketSnapshot, account: PerpAccount, universe: list[str]
) -> str:
    """Render the per-step user message: market state + portfolio state."""
    market = snapshot.to_prompt_block()

    equity = account.equity(snapshot.prices)
    cash = account.cash
    pos_lines = []
    for sym in universe:
        if sym in account.positions:
            p = account.positions[sym]
            current_price = snapshot.prices.get(sym, p.entry)
            pnl_pct = (p.unrealized_pnl(current_price) / p.margin) * 100 if p.margin else 0.0
            pos_lines.append(
                f"  {sym}: {p.side} @ {p.entry:.2f}, qty={p.qty:.6f}, "
                f"lev={p.leverage:.1f}x, margin=${p.margin:.2f}, "
                f"unrealized PnL={pnl_pct:+.2f}%"
            )
        else:
            pos_lines.append(f"  {sym}: no position")
    positions_block = "\n".join(pos_lines)

    return f"""# Market Snapshot

{market}

# Portfolio State

  Total equity: ${equity:.2f}
  Free cash:    ${cash:.2f}
  Margin used:  ${equity - cash:.2f}
  Realized PnL: ${account.realized_pnl:+.2f}
  Fees paid:    ${account.fees_paid:.2f}
  Funding paid: ${account.funding_paid:+.2f}
  Trades so far: {account.n_trades}

# Open Positions

{positions_block}

# Universe

{universe}

# Your task

Emit a DecisionBatch with one TradeDecision for EACH symbol in the universe.
Apply the decision principles from the system prompt. Be disciplined. The
goal is risk-adjusted return, not maximum activity.
"""


# -----------------------------------------------------------------------------
# Heuristic fallback (no API key required)
# -----------------------------------------------------------------------------


def _heuristic_decide(
    snapshot: MarketSnapshot, account: PerpAccount, universe: list[str]
) -> DecisionBatch:
    """Deterministic disciplined-momentum strategy.

    Used as the fallback when AI is disabled, AND as a control baseline in the
    contest results. Implements the same rules the system prompt teaches:

      - Close any position down >2% from entry
      - Otherwise hold existing positions
      - Open a new position only when 24h return is > 3% (long) or < -3% (short)
        AND vol_24h < 5% AND no existing position
      - Use 8x leverage, 20% of equity per position
      - At most 2 positions open at once
    """
    decisions: list[TradeDecision] = []
    open_count = len(account.positions)

    for sym in universe:
        ret_24h = snapshot.returns_24h.get(sym, 0.0)
        vol_24h = snapshot.volatility_24h.get(sym, 0.0)
        price = snapshot.prices.get(sym, 0.0)

        if sym in account.positions:
            pos = account.positions[sym]
            pnl_pct = pos.unrealized_pnl(price) / pos.margin if pos.margin else 0.0
            if pnl_pct < -0.02:
                decisions.append(
                    TradeDecision(
                        symbol=sym,
                        action="close",
                        confidence=0.7,
                        rationale=f"Position down {pnl_pct*100:.2f}% — cut loss per discipline rule.",
                    )
                )
            else:
                decisions.append(
                    TradeDecision(
                        symbol=sym,
                        action="hold",
                        confidence=0.6,
                        rationale=f"Existing {pos.side} position at {pnl_pct*100:+.2f}% — let it run.",
                    )
                )
        elif open_count < 2 and abs(ret_24h) > 0.03 and vol_24h < 0.05:
            side: Action = "open_long" if ret_24h > 0 else "open_short"
            decisions.append(
                TradeDecision(
                    symbol=sym,
                    action=side,
                    size_pct_of_equity=0.20,
                    leverage=8.0,
                    confidence=0.65,
                    rationale=f"24h return {ret_24h*100:+.2f}%, vol {vol_24h*100:.2f}% — clean momentum setup.",
                )
            )
            open_count += 1
        else:
            decisions.append(
                TradeDecision(
                    symbol=sym,
                    action="hold",
                    confidence=0.6,
                    rationale=f"No setup: 24h return {ret_24h*100:+.2f}%, vol {vol_24h*100:.2f}%.",
                )
            )

    return DecisionBatch(
        decisions=decisions,
        overall_thesis=(
            "Heuristic baseline: trade only on clear momentum (>3% 24h move, <5% vol), "
            "max 2 concurrent positions, cut losses fast at -2%, hold winners. "
            "Same rules the Alpha Arena winners followed."
        ),
    )
