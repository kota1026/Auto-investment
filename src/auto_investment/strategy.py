"""Trading strategy — EMA cross with RSI trend filter.

Pure function design: takes a dataframe with indicators applied, returns a
signal dataframe. Easy to backtest, easy to extend, easy to swap.

Entry rules
-----------
LONG  when EMA_fast crosses above EMA_slow AND RSI > 50
SHORT when EMA_fast crosses below EMA_slow AND RSI < 50

Exit handled by ATR-based stop-loss / take-profit (see risk.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

SignalType = Literal["long", "short", "flat"]


@dataclass(frozen=True)
class Signal:
    """A discrete trading decision tied to a single bar."""

    timestamp: pd.Timestamp
    side: SignalType
    price: float
    ema_fast: float
    ema_slow: float
    rsi: float
    atr: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "side": self.side,
            "price": float(self.price),
            "ema_fast": float(self.ema_fast),
            "ema_slow": float(self.ema_slow),
            "rsi": float(self.rsi),
            "atr": float(self.atr),
            "reason": self.reason,
        }


def generate_signals(df: pd.DataFrame, rsi_threshold: float = 50.0) -> pd.DataFrame:
    """Append a `signal` column to an indicator-augmented dataframe.

    Values: "long" / "short" / "flat".

    Crossovers are detected on the *current* bar (i.e. when the prior bar had
    fast<=slow and the current bar has fast>slow). This is the standard
    backtest convention — execution would happen on the next bar's open.
    """
    if not {"ema_fast", "ema_slow", "rsi"}.issubset(df.columns):
        raise ValueError("DataFrame must have ema_fast, ema_slow, rsi columns. "
                         "Run indicators.add_indicators() first.")

    out = df.copy()
    fast = out["ema_fast"]
    slow = out["ema_slow"]
    rsi_val = out["rsi"]

    crossed_up = (fast.shift(1) <= slow.shift(1)) & (fast > slow)
    crossed_down = (fast.shift(1) >= slow.shift(1)) & (fast < slow)

    long_signal = crossed_up & (rsi_val > rsi_threshold)
    short_signal = crossed_down & (rsi_val < rsi_threshold)

    out["signal"] = "flat"
    out.loc[long_signal, "signal"] = "long"
    out.loc[short_signal, "signal"] = "short"
    return out


def latest_signal(df_with_signals: pd.DataFrame) -> Signal | None:
    """Return the most recent non-flat signal, or None.

    Looks at the last row only — for the streaming/live use case.
    """
    if df_with_signals.empty:
        return None
    last = df_with_signals.iloc[-1]
    side = last["signal"]
    if side == "flat":
        return None

    if side == "long":
        reason = (
            f"EMA fast({last['ema_fast']:.2f}) crossed above slow({last['ema_slow']:.2f}) "
            f"with RSI {last['rsi']:.1f} > 50 — bullish trend confirmed"
        )
    else:
        reason = (
            f"EMA fast({last['ema_fast']:.2f}) crossed below slow({last['ema_slow']:.2f}) "
            f"with RSI {last['rsi']:.1f} < 50 — bearish trend confirmed"
        )

    return Signal(
        timestamp=last.name if isinstance(last.name, pd.Timestamp) else pd.Timestamp(last.name),
        side=side,
        price=float(last["close"]),
        ema_fast=float(last["ema_fast"]),
        ema_slow=float(last["ema_slow"]),
        rsi=float(last["rsi"]),
        atr=float(last["atr"]),
        reason=reason,
    )
