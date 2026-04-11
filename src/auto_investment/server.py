"""FastAPI server exposing OHLCV, indicators, signals, AI verdicts, and
backtest results to the lightweight-charts frontend.

Run with:
    uvicorn auto_investment.server:app --reload --port 8000

Endpoints:
    GET  /                  → serves web/index.html
    GET  /api/health        → liveness check
    GET  /api/candles       → OHLCV + indicators (JSON ready for lightweight-charts)
    GET  /api/backtest      → run a backtest and return trades + equity curve
    POST /api/evaluate      → run one full tick: signal → forecast → AI → plan
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .ai_advisor import evaluate_signal
from .backtest import run_backtest
from .config import settings
from .context import MarketContext
from .data import fetch_ohlcv
from .forecaster import forecast_close
from .indicators import add_indicators
from .macro import fetch_macro_snapshot
from .news import fetch_news
from .novaquity import fetch_fundamentals
from .optimizer import explain_top_with_claude, grid_search, random_search
from .risk import build_trade_plan
from .strategy import generate_signals, latest_signal
from .thesis import build_thesis

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Auto-Investment",
    description="Hybrid technical + AI-confirmed trading system",
    version="0.1.0",
)

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def root() -> FileResponse:
    """Serve the lightweight-charts dashboard."""
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return FileResponse(index)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "symbol": settings.symbol,
        "timeframe": settings.timeframe,
        "ai_enabled": settings.ai_enabled,
        "exchange": settings.exchange_id,
        "testnet": settings.exchange_testnet,
    }


@app.get("/api/candles")
def candles(limit: int = 300) -> dict:
    """OHLCV + indicators in lightweight-charts JSON shape.

    Returns three arrays:
      - candles: [{time, open, high, low, close}, ...]
      - ema_fast / ema_slow: [{time, value}, ...]  (line series)
      - rsi: [{time, value}, ...]
      - markers: [{time, position, color, shape, text}, ...] for signals
    """
    df = fetch_ohlcv(limit=limit)
    df = add_indicators(df)
    df = generate_signals(df)

    candles = []
    ema_fast = []
    ema_slow = []
    rsi_series = []
    markers = []

    for ts, row in df.iterrows():
        time_unix = int(ts.timestamp())
        candles.append(
            {
                "time": time_unix,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        )
        if not _isnan(row["ema_fast"]):
            ema_fast.append({"time": time_unix, "value": float(row["ema_fast"])})
        if not _isnan(row["ema_slow"]):
            ema_slow.append({"time": time_unix, "value": float(row["ema_slow"])})
        if not _isnan(row["rsi"]):
            rsi_series.append({"time": time_unix, "value": float(row["rsi"])})

        if row["signal"] == "long":
            markers.append(
                {
                    "time": time_unix,
                    "position": "belowBar",
                    "color": "#22c55e",
                    "shape": "arrowUp",
                    "text": "LONG",
                }
            )
        elif row["signal"] == "short":
            markers.append(
                {
                    "time": time_unix,
                    "position": "aboveBar",
                    "color": "#ef4444",
                    "shape": "arrowDown",
                    "text": "SHORT",
                }
            )

    return {
        "candles": candles,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "rsi": rsi_series,
        "markers": markers,
        "symbol": settings.symbol,
        "timeframe": settings.timeframe,
    }


@app.get("/api/backtest")
def backtest(limit: int = 1000) -> dict:
    """Run a backtest and return trades + equity curve."""
    df = fetch_ohlcv(limit=limit)
    result = run_backtest(
        df,
        initial_equity=settings.equity_usd,
        risk_per_trade=settings.risk_per_trade,
    )
    return result.to_dict()


@app.post("/api/evaluate")
def evaluate(limit: int = 300) -> dict:
    """Run one full evaluation tick: signal → forecast → context → AI verdict → plan.

    Same logic as `live.run_once()` but never submits an order — purely
    informational. Use this from the frontend to preview what the live loop
    would do at this moment.
    """
    df = fetch_ohlcv(limit=limit)
    df = add_indicators(df)
    df = generate_signals(df)

    sig = latest_signal(df)
    if sig is None:
        return {"status": "no_signal", "last_close": float(df["close"].iloc[-1])}

    context = _build_context_for_request(df)
    verdict = evaluate_signal(sig, context)

    plan = None
    if verdict.action != "HOLD" and verdict.confidence >= settings.ai_min_confidence:
        plan = build_trade_plan(
            side=sig.side,
            entry=sig.price,
            atr_value=sig.atr,
            equity=settings.equity_usd,
            risk_per_trade=settings.risk_per_trade,
        ).to_dict()

    return {
        "status": "evaluated",
        "signal": sig.to_dict(),
        "forecast": context.forecast.to_dict() if context.forecast else None,
        "news": {
            "summary": context.news_summary,
            "items": context.news_items,
        } if context.news_summary else None,
        "macro": context.macro_snapshot,
        "fundamentals": context.fundamentals,
        "verdict": verdict.model_dump(),
        "plan": plan,
    }


@app.get("/api/context")
def context_only(limit: int = 300) -> dict:
    """Return all context layers without consulting Claude.

    Useful for the frontend to display news, macro, and fundamentals
    independently of running an AI evaluation.
    """
    df = fetch_ohlcv(limit=limit)
    ctx = _build_context_for_request(df)
    return {
        "forecast": ctx.forecast.to_dict() if ctx.forecast else None,
        "news": {"summary": ctx.news_summary, "items": ctx.news_items} if ctx.news_summary else None,
        "macro": ctx.macro_snapshot,
        "fundamentals": ctx.fundamentals,
    }


@app.post("/api/optimize")
def optimize(
    limit: int = 1000,
    method: str = "random",
    n_samples: int = 30,
    explain: bool = True,
) -> dict:
    """Run an AutoAgent-style strategy parameter optimization.

    Query params:
      - limit: bars to backtest against (default 1000)
      - method: "random" or "grid" (default random — much faster)
      - n_samples: only used for random search (default 30)
      - explain: if true and ANTHROPIC_API_KEY is set, also have Claude
        recommend one of the top configurations with a written rationale
    """
    df = fetch_ohlcv(limit=limit)
    if method == "grid":
        result = grid_search(
            df, initial_equity=settings.equity_usd, risk_per_trade=settings.risk_per_trade
        )
    else:
        result = random_search(
            df,
            n_samples=n_samples,
            initial_equity=settings.equity_usd,
            risk_per_trade=settings.risk_per_trade,
        )

    payload = result.to_dict()
    payload["method"] = method
    if explain:
        payload["claude_recommendation"] = explain_top_with_claude(result, top_n=5)
    return payload


@app.post("/api/thesis")
def thesis(symbol: str | None = None, limit: int = 300) -> dict:
    """Run the Dexter-inspired multi-agent thesis builder.

    Returns an investment thesis from an Analyst pass and a ValidationReport
    from an independent Validator pass. The validator is the key add — it
    catches hallucinated claims and unsupported assertions before the trader
    sees the thesis.

    If AI is disabled or unconfigured, returns a payload with `thesis: null`
    and just the gathered context (so the user still sees what data is
    available).
    """
    result = build_thesis(symbol=symbol, limit=limit)
    if result is None:
        return {
            "status": "ai_disabled",
            "message": "AI disabled or no ANTHROPIC_API_KEY — context-only mode unavailable from /api/thesis. "
            "Use /api/context for raw data.",
        }
    return {
        "status": "ok",
        "thesis": result.thesis.model_dump(),
        "validation": result.validation.model_dump(),
        "context_used": result.context_used,
    }


def _build_context_for_request(df) -> MarketContext:
    """Build a MarketContext from a fetched OHLCV frame.

    Mirrors `live._build_context` but kept here so the server doesn't import
    the live loop module (which is meant for long-running processes).
    """
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
    try:
        forecast = forecast_close(df["close"], horizon=24)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Forecast failed: %s", exc)
        forecast = None

    news_bundle = fetch_news(settings.symbol)
    macro = fetch_macro_snapshot()

    fundamentals = None
    if settings.symbol.isdigit() and len(settings.symbol) == 4:
        snap = fetch_fundamentals(settings.symbol)
        if snap is not None:
            fundamentals = snap.to_dict()

    return MarketContext(
        recent_bars=recent_bars,
        forecast=forecast,
        news_summary=news_bundle.summary if news_bundle else None,
        news_items=news_bundle.items if news_bundle else [],
        macro_snapshot=macro,
        fundamentals=fundamentals,
    )


def _isnan(x) -> bool:
    return x != x  # NaN is the only value where x != x
