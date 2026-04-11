"""Hyperliquid-style perpetual futures simulator.

Implements the minimum viable simulation needed to faithfully replicate the
Nof1 Alpha Arena contest format:
  - Cross-margin USDC-quoted perp futures
  - Up to ~50x leverage
  - Hourly funding payments (longs pay shorts when funding > 0)
  - Maintenance-margin liquidations
  - Taker fees with simple slippage model
  - Multi-symbol portfolio tracking

This is deliberately deterministic and side-effect-free so backtests are
reproducible. The decision agent (alpha_arena.py) is the only source of
"intelligence" — the simulator is dumb plumbing.

Why we need this on top of backtest.py:
  backtest.py is a single-symbol, spot, no-leverage, no-funding backtester
  designed for the EMA-cross strategy. Alpha Arena needs perp futures with
  multiple concurrent positions across multiple symbols, leverage, funding
  payments, and liquidations. Different beast — separate module.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

Side = Literal["long", "short"]


@dataclass
class PerpPosition:
    """An open perpetual futures position with cross-margin accounting."""

    symbol: str
    side: Side
    entry: float
    qty: float           # base units (e.g. BTC)
    leverage: float
    margin: float        # USDC locked as collateral
    opened_at_step: int  # for trade duration accounting

    def notional(self, current_price: float) -> float:
        """Current notional value of the position in USDC."""
        return abs(self.qty * current_price)

    def unrealized_pnl(self, current_price: float) -> float:
        """Mark-to-market unrealized PnL in USDC."""
        sign = 1.0 if self.side == "long" else -1.0
        return sign * self.qty * (current_price - self.entry)

    def equity(self, current_price: float) -> float:
        """Margin + unrealized PnL — what you'd get if you closed now (pre-fee)."""
        return self.margin + self.unrealized_pnl(current_price)

    def margin_ratio(self, current_price: float) -> float:
        """equity / notional. Below maintenance_margin → liquidation."""
        n = self.notional(current_price)
        if n <= 0:
            return float("inf")
        return self.equity(current_price) / n

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry": self.entry,
            "qty": self.qty,
            "leverage": self.leverage,
            "margin": self.margin,
        }


@dataclass
class PerpAccount:
    """Cross-margin perp account state."""

    cash: float                                     # free USDC
    positions: dict[str, PerpPosition] = field(default_factory=dict)
    fees_paid: float = 0.0
    funding_paid: float = 0.0  # net (positive = paid funding, negative = received)
    realized_pnl: float = 0.0
    n_trades: int = 0
    n_liquidations: int = 0

    def equity(self, prices: dict[str, float]) -> float:
        """Total account value = cash + (margin + unrealized PnL) for each position."""
        total = self.cash
        for pos in self.positions.values():
            price = prices.get(pos.symbol, pos.entry)
            total += pos.equity(price)
        return total

    def to_dict(self, prices: dict[str, float]) -> dict:
        return {
            "cash": self.cash,
            "equity": self.equity(prices),
            "positions": [p.to_dict() for p in self.positions.values()],
            "fees_paid": self.fees_paid,
            "funding_paid": self.funding_paid,
            "realized_pnl": self.realized_pnl,
            "n_trades": self.n_trades,
            "n_liquidations": self.n_liquidations,
        }


class PerpSimulator:
    """Deterministic perp simulator with funding, fees, slippage, and liquidations.

    Default parameters approximate Hyperliquid mainnet:
      - Taker fee: 2.5 bps
      - Maintenance margin: 1.5% (≈ 67x max effective leverage)
      - Funding rate: 0.01%/hour (the long-run average; real funding varies)

    Slippage model: linear in size relative to a reference depth. Crude but
    matches the dominant cost on small accounts.
    """

    def __init__(
        self,
        starting_capital: float,
        *,
        taker_fee: float = 0.00025,
        maintenance_margin: float = 0.015,
        funding_rate_hourly: float = 0.0001,
        slippage_bps_per_10k: float = 1.0,
        max_leverage: float = 50.0,
    ):
        if starting_capital <= 0:
            raise ValueError("starting_capital must be positive")
        self.account = PerpAccount(cash=starting_capital)
        self.starting_capital = starting_capital
        self.taker_fee = taker_fee
        self.maintenance_margin = maintenance_margin
        self.funding_rate_hourly = funding_rate_hourly
        self.slippage_bps_per_10k = slippage_bps_per_10k
        self.max_leverage = max_leverage
        self.step_count = 0

    # --- public API -----------------------------------------------------------

    def open_position(
        self,
        symbol: str,
        side: Side,
        notional_usd: float,
        leverage: float,
        current_price: float,
    ) -> tuple[bool, str]:
        """Open a new position. Returns (success, reason).

        Rejects when:
          - Position already exists for this symbol (close first)
          - Notional <= 0 or leverage out of range
          - Insufficient free cash for margin + fees
        """
        if leverage <= 0 or leverage > self.max_leverage:
            return False, f"leverage {leverage} out of range (0, {self.max_leverage}]"
        if notional_usd <= 0:
            return False, "notional must be positive"
        if symbol in self.account.positions:
            return False, f"position already open on {symbol}"

        margin_required = notional_usd / leverage
        slippage = self._slippage(notional_usd, current_price)
        fill_price = current_price + slippage if side == "long" else current_price - slippage
        fee = notional_usd * self.taker_fee
        total_cost = margin_required + fee

        if total_cost > self.account.cash:
            return False, f"insufficient cash: need {total_cost:.2f}, have {self.account.cash:.2f}"

        qty = notional_usd / fill_price
        self.account.cash -= total_cost
        self.account.fees_paid += fee
        self.account.n_trades += 1
        self.account.positions[symbol] = PerpPosition(
            symbol=symbol,
            side=side,
            entry=fill_price,
            qty=qty,
            leverage=leverage,
            margin=margin_required,
            opened_at_step=self.step_count,
        )
        return True, "ok"

    def close_position(self, symbol: str, current_price: float) -> tuple[bool, float]:
        """Close a position at the current price. Returns (success, realized_pnl_net)."""
        if symbol not in self.account.positions:
            return False, 0.0
        pos = self.account.positions[symbol]
        notional = pos.notional(current_price)
        slippage = self._slippage(notional, current_price)
        fill_price = current_price - slippage if pos.side == "long" else current_price + slippage

        # Recompute PnL at fill price (slippage acts against us)
        sign = 1.0 if pos.side == "long" else -1.0
        gross_pnl = sign * pos.qty * (fill_price - pos.entry)
        fee = notional * self.taker_fee
        net_pnl = gross_pnl - fee

        self.account.cash += pos.margin + net_pnl
        self.account.fees_paid += fee
        self.account.realized_pnl += net_pnl
        del self.account.positions[symbol]
        return True, net_pnl

    def step(self, prices: dict[str, float]) -> list[str]:
        """Advance one bar.

        Applies hourly funding and checks for liquidations. Returns a list of
        symbols that were liquidated (empty list if none).

        `prices` must contain a price for every symbol with an open position.
        """
        self.step_count += 1
        liquidated: list[str] = []

        for symbol, pos in list(self.account.positions.items()):
            if symbol not in prices:
                continue
            price = prices[symbol]

            # Apply funding (longs pay shorts when funding rate > 0)
            funding_payment = pos.notional(price) * self.funding_rate_hourly
            if pos.side == "long":
                self.account.cash -= funding_payment
                self.account.funding_paid += funding_payment
            else:
                self.account.cash += funding_payment
                self.account.funding_paid -= funding_payment

            # Liquidation check
            if pos.margin_ratio(price) < self.maintenance_margin:
                logger.info(
                    "LIQUIDATION %s @ %.2f (margin_ratio=%.4f < %.4f)",
                    symbol,
                    price,
                    pos.margin_ratio(price),
                    self.maintenance_margin,
                )
                # Forced close — no margin returned, full position lost
                pos_equity_at_liq = max(0.0, pos.equity(price))
                self.account.cash += pos_equity_at_liq * 0.5  # 50% slashing penalty
                self.account.realized_pnl -= pos.margin
                self.account.n_liquidations += 1
                del self.account.positions[symbol]
                liquidated.append(symbol)

        return liquidated

    def equity(self, prices: dict[str, float]) -> float:
        """Total account equity at the given prices."""
        return self.account.equity(prices)

    def margin_used(self) -> float:
        return sum(p.margin for p in self.account.positions.values())

    def free_cash(self) -> float:
        return self.account.cash

    # --- helpers --------------------------------------------------------------

    def _slippage(self, notional_usd: float, price: float) -> float:
        """Linear slippage model: bps per $10k of notional, in price units."""
        bps = self.slippage_bps_per_10k * (notional_usd / 10_000.0)
        return price * (bps / 10_000.0)
