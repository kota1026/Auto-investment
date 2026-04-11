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
from .data import fetch_ohlcv
from .forecaster import forecast_close
from .indicators import add_indicators
from .risk import build_trade_plan
from .strategy import generate_signals, latest_signal

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
    """Run one full evaluation tick: signal → forecast → AI verdict → plan.

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

    try:
        fc = forecast_close(df["close"], horizon=24)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Forecast failed: %s", exc)
        fc = None

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
        "forecast": fc.to_dict() if fc else None,
        "verdict": verdict.model_dump(),
        "plan": plan,
    }


def _isnan(x) -> bool:
    return x != x  # NaN is the only value where x != x
