"""Live trading skeleton — wires the strategy + AI advisor + risk into ccxt.

Runs by default against Binance testnet (set EXCHANGE_TESTNET=false to go
real, but think twice before doing that). The loop:

  1. Fetch latest OHLCV from the exchange
  2. Compute indicators + technical signal
  3. If a candidate signal fires, optionally consult TimesFM + Claude
  4. If the AI verdict's confidence clears AI_MIN_CONFIDENCE, build a trade
     plan and submit a market order via ccxt
  5. Sleep until the next bar close, repeat

This module is intentionally a *skeleton*. Productionizing it requires:
  - Position state persistence (DB or file) so a restart doesn't lose track
  - Reconciliation against actual exchange positions on startup
  - Retry & circuit-breaker logic on API failures
  - Bracket order management (most exchanges have a one-cancels-other primitive)
  - PnL accounting and slippage tracking
"""

from __future__ import annotations

import logging
import time

from .ai_advisor import evaluate_signal
from .config import settings
from .data import fetch_ohlcv
from .forecaster import forecast_close
from .indicators import add_indicators
from .risk import build_trade_plan
from .strategy import generate_signals, latest_signal

logger = logging.getLogger(__name__)


def run_once() -> dict:
    """Single tick of the live loop. Returns a dict describing what happened."""
    df = fetch_ohlcv(limit=300)
    df = add_indicators(df)
    df = generate_signals(df)

    sig = latest_signal(df)
    if sig is None:
        return {"status": "no_signal", "last_close": float(df["close"].iloc[-1])}

    logger.info("Candidate signal: %s @ %.2f", sig.side, sig.price)

    # Optional forecast layer
    try:
        fc = forecast_close(df["close"], horizon=24)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Forecast failed: %s", exc)
        fc = None

    # AI confirmation
    recent_bars = [
        {
            "timestamp": ts.isoformat(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        for ts, row in df.tail(30).iterrows()
    ]
    verdict = evaluate_signal(sig, recent_bars, fc)
    logger.info(
        "AI verdict: %s @ %.2f — %s",
        verdict.action,
        verdict.confidence,
        verdict.rationale[:120],
    )

    # Confidence gate
    if verdict.action == "HOLD" or verdict.confidence < settings.ai_min_confidence:
        return {
            "status": "vetoed",
            "signal": sig.to_dict(),
            "verdict": verdict.model_dump(),
        }

    # Build the trade plan
    plan = build_trade_plan(
        side=sig.side,
        entry=sig.price,
        atr_value=sig.atr,
        equity=settings.equity_usd,
        risk_per_trade=settings.risk_per_trade,
    )

    # Submit the order via ccxt — guarded so a missing key doesn't crash the loop
    order_result = _submit_market_order(plan, settings.symbol)

    return {
        "status": "executed",
        "signal": sig.to_dict(),
        "verdict": verdict.model_dump(),
        "plan": plan.to_dict(),
        "order": order_result,
    }


def _submit_market_order(plan, symbol: str) -> dict:
    """Try to submit a market order; degrade to dry-run on missing credentials."""
    if not (settings.exchange_api_key and settings.exchange_api_secret):
        logger.warning("No exchange credentials — running in DRY_RUN mode")
        return {"dry_run": True, **plan.to_dict()}

    try:
        from .data import _make_exchange  # noqa: PLC0415

        exchange = _make_exchange()
        side = "buy" if plan.side == "long" else "sell"
        order = exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=plan.qty,
        )
        # Real implementations should also place the SL/TP as separate
        # stop-market and take-profit orders, ideally as a bracket OCO if the
        # exchange supports it.
        return {"dry_run": False, "order": order, **plan.to_dict()}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Order submission failed: %s", exc)
        return {"dry_run": True, "error": str(exc), **plan.to_dict()}


def run_loop(poll_interval_seconds: int = 60) -> None:
    """Run the live loop forever. Ctrl-C to stop."""
    logger.info(
        "Starting live loop on %s (%s) every %ds",
        settings.symbol,
        settings.timeframe,
        poll_interval_seconds,
    )
    while True:
        try:
            result = run_once()
            logger.info("Tick result: %s", result["status"])
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tick failed: %s", exc)
        time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_loop()
