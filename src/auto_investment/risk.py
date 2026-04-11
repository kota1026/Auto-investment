"""Risk management — position sizing and ATR-based stop/target placement.

The core principle: never risk more than `risk_per_trade` of equity on a single
position. Position size is derived from the distance to the stop loss, not from
a fixed dollar amount.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["long", "short"]


@dataclass(frozen=True)
class TradePlan:
    """Concrete order parameters ready for execution."""

    side: Side
    entry: float
    stop: float
    target: float
    qty: float
    risk_usd: float
    rr: float  # reward-to-risk ratio

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "entry": self.entry,
            "stop": self.stop,
            "target": self.target,
            "qty": self.qty,
            "risk_usd": self.risk_usd,
            "rr": self.rr,
        }


def build_trade_plan(
    side: Side,
    entry: float,
    atr_value: float,
    equity: float,
    risk_per_trade: float = 0.01,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 3.0,
) -> TradePlan:
    """Build a trade plan sized to risk exactly `equity * risk_per_trade`.

    Stop is `sl_atr_mult * ATR` away from entry; target is `tp_atr_mult * ATR`.
    Quantity is computed so the dollar loss at stop equals the risk budget.

    Raises ValueError on degenerate inputs (zero ATR, zero equity).
    """
    if atr_value <= 0:
        raise ValueError(f"ATR must be positive (got {atr_value})")
    if equity <= 0:
        raise ValueError(f"Equity must be positive (got {equity})")
    if not 0 < risk_per_trade < 1:
        raise ValueError(f"risk_per_trade must be in (0, 1), got {risk_per_trade}")

    risk_usd = equity * risk_per_trade
    sl_distance = sl_atr_mult * atr_value
    tp_distance = tp_atr_mult * atr_value

    if side == "long":
        stop = entry - sl_distance
        target = entry + tp_distance
    elif side == "short":
        stop = entry + sl_distance
        target = entry - tp_distance
    else:
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")

    qty = risk_usd / sl_distance
    rr = tp_distance / sl_distance

    return TradePlan(
        side=side,
        entry=float(entry),
        stop=float(stop),
        target=float(target),
        qty=float(qty),
        risk_usd=float(risk_usd),
        rr=float(rr),
    )
