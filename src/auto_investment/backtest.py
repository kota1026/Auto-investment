"""Vectorized backtester for the EMA-cross + RSI strategy.

Walks bar-by-bar over a dataframe with `signal` column applied. On each
signal it opens a position sized via risk.py, simulates the trade until SL or
TP is hit (using the next bar's high/low for fills), and records the trade.

Intentionally simple: no slippage model, no funding fees, no shorts on spot
(short trades are simulated as if on a margin/perpetual venue). Replace with
ccxt.create_order() in live.py for real execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .indicators import add_indicators
from .risk import build_trade_plan
from .strategy import generate_signals


@dataclass
class Trade:
    """Closed trade record for the equity curve and stats."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry: float
    exit: float
    qty: float
    pnl_usd: float
    pnl_pct: float
    exit_reason: str  # "tp", "sl", or "eod"


@dataclass
class BacktestResult:
    """Summary of a backtest run."""

    trades: list[Trade]
    equity_curve: pd.Series
    final_equity: float
    total_return_pct: float
    win_rate: float
    avg_rr: float
    max_drawdown_pct: float
    n_trades: int

    def to_dict(self) -> dict:
        return {
            "trades": [
                {**asdict(t), "entry_time": t.entry_time.isoformat(), "exit_time": t.exit_time.isoformat()}
                for t in self.trades
            ],
            "equity_curve": [
                {"timestamp": ts.isoformat(), "equity": float(v)}
                for ts, v in self.equity_curve.items()
            ],
            "final_equity": self.final_equity,
            "total_return_pct": self.total_return_pct,
            "win_rate": self.win_rate,
            "avg_rr": self.avg_rr,
            "max_drawdown_pct": self.max_drawdown_pct,
            "n_trades": self.n_trades,
        }


def run_backtest(
    df: pd.DataFrame,
    initial_equity: float = 10_000.0,
    risk_per_trade: float = 0.01,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 3.0,
) -> BacktestResult:
    """Run an in-sample backtest over the given OHLCV dataframe.

    The dataframe is expected to be raw OHLCV (open/high/low/close/volume);
    indicators and signals are added internally.
    """
    df = add_indicators(df)
    df = generate_signals(df)
    df = df.dropna(subset=["ema_fast", "ema_slow", "rsi", "atr"]).copy()

    equity = initial_equity
    equity_history: list[tuple[pd.Timestamp, float]] = []
    trades: list[Trade] = []

    open_trade: dict | None = None  # holds the active position, if any

    for i, (ts, bar) in enumerate(df.iterrows()):
        # 1. If a position is open, check whether SL or TP was hit *this* bar
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
            else:  # short
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

        # 2. If flat, look for a new signal — execute on the *next* bar's open
        if open_trade is None and bar["signal"] in ("long", "short") and i + 1 < len(df):
            next_bar = df.iloc[i + 1]
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
                pass  # skip degenerate signals (zero ATR etc.)

        equity_history.append((ts, equity))

    # Close any still-open trade at the last close
    if open_trade is not None:
        last_close = float(df["close"].iloc[-1])
        if open_trade["side"] == "long":
            pnl = (last_close - open_trade["entry"]) * open_trade["qty"]
        else:
            pnl = (open_trade["entry"] - last_close) * open_trade["qty"]
        trades.append(_close(open_trade, df.index[-1], last_close, pnl, "eod"))
        equity += pnl
        equity_history[-1] = (df.index[-1], equity)

    equity_curve = pd.Series(
        [eq for _, eq in equity_history],
        index=pd.DatetimeIndex([ts for ts, _ in equity_history]),
        name="equity",
    )

    return _summarize(trades, equity_curve, initial_equity)


def _close(open_trade: dict, exit_time, exit_price: float, pnl: float, reason: str) -> Trade:
    return Trade(
        entry_time=open_trade["entry_time"],
        exit_time=exit_time,
        side=open_trade["side"],
        entry=open_trade["entry"],
        exit=float(exit_price),
        qty=open_trade["qty"],
        pnl_usd=float(pnl),
        pnl_pct=float(pnl / (open_trade["entry"] * open_trade["qty"]) * 100.0),
        exit_reason=reason,
    )


def _summarize(trades: list[Trade], equity_curve: pd.Series, initial: float) -> BacktestResult:
    final = float(equity_curve.iloc[-1]) if len(equity_curve) else initial
    total_return_pct = (final / initial - 1.0) * 100.0
    n = len(trades)
    wins = [t for t in trades if t.pnl_usd > 0]
    win_rate = (len(wins) / n) if n else 0.0

    avg_rr = 0.0
    if wins and (losses := [t for t in trades if t.pnl_usd <= 0]):
        avg_win = sum(t.pnl_usd for t in wins) / len(wins)
        avg_loss = abs(sum(t.pnl_usd for t in losses) / len(losses))
        avg_rr = avg_win / avg_loss if avg_loss > 0 else 0.0

    # Max drawdown
    if len(equity_curve):
        running_max = equity_curve.cummax()
        drawdowns = (equity_curve - running_max) / running_max
        max_dd = float(drawdowns.min() * 100.0)
    else:
        max_dd = 0.0

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        final_equity=final,
        total_return_pct=total_return_pct,
        win_rate=win_rate,
        avg_rr=avg_rr,
        max_drawdown_pct=max_dd,
        n_trades=n,
    )
