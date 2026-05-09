"""Per-venue trading cost model used by every strategy backtest.

Numbers are calibrated to public fee schedules as of 2026-05. They live in
one place so backtests, the live executor, and the reporter all charge the
same fees. If the user uses a different fee tier (VIP, BNB discount, etc.),
override via `cost_model.override(...)` at the start of the run.

Spec reference: docs/strategy_spec.md §4.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class VenueCosts:
    """Maker/taker fees in bps (1 bp = 0.01%)."""

    taker_bps: float
    maker_bps: float


# CEX spot venues. binancejp = Binance Japan retail tier (no VIP).
SPOT_VENUE_COSTS: dict[str, VenueCosts] = {
    "binance": VenueCosts(taker_bps=4.5, maker_bps=1.0),
    "binancejp": VenueCosts(taker_bps=7.5, maker_bps=1.5),
    "bybit": VenueCosts(taker_bps=5.5, maker_bps=1.0),
    "okx": VenueCosts(taker_bps=5.0, maker_bps=0.8),
}

# Perp venues. hyperliquid is the Phase-1 default per spec §9.
PERP_VENUE_COSTS: dict[str, VenueCosts] = {
    "hyperliquid": VenueCosts(taker_bps=2.5, maker_bps=0.2),
    "dydx_v4": VenueCosts(taker_bps=5.0, maker_bps=2.0),
    "aevo": VenueCosts(taker_bps=5.0, maker_bps=2.0),
    "binance_perp": VenueCosts(taker_bps=4.0, maker_bps=2.0),
    "bybit_perp": VenueCosts(taker_bps=5.5, maker_bps=2.0),
}

# L2 gas (USD per typical tx; rounded up for conservatism).
L2_GAS_USD: dict[str, float] = {
    "arbitrum": 0.20,
    "base": 0.10,
    "optimism": 0.25,
}

# Misc DeFi.
SWAP_BPS = 5.0           # uniswap v3 5bp tier
LP_DEPOSIT_BPS = 0.0
LP_WITHDRAW_BPS = 0.0
PREIPO_SPREAD_BPS = 200.0  # honest acknowledgement of wide pre-IPO spreads


def slippage_bps(notional_usd: float) -> float:
    """Linear-in-size slippage estimate.

    Floor of 2 bps, growing by 1.5 bps per $100k notional. Crude but matches
    observed BTC/ETH market-impact curves at the $1k–$50k size we trade.
    """
    return max(2.0, 1.5 * (notional_usd / 100_000.0))


def round_trip_bps(
    venue: str,
    notional_usd: float,
    *,
    perp: bool = False,
    taker_legs: int = 2,
) -> float:
    """Total round-trip cost (entry + exit) in bps for a single asset.

    `taker_legs` lets us model maker-on-entry-taker-on-exit (=1) for the
    cases where we sit on the book with a passive limit. Default is the
    pessimistic both-taker case.
    """
    table = PERP_VENUE_COSTS if perp else SPOT_VENUE_COSTS
    if venue not in table:
        raise KeyError(f"Unknown venue: {venue!r}; known={list(table)}")
    fees = table[venue]
    taker = fees.taker_bps * taker_legs
    maker = fees.maker_bps * (2 - taker_legs)
    return taker + maker + slippage_bps(notional_usd) * 2  # 2 fills


def funding_arb_round_trip_bps(
    spot_venue: str,
    perp_venue: str,
    notional_usd: float,
) -> float:
    """Round-trip cost of opening + closing a delta-neutral funding arb.

    Spot leg + perp leg are independent fills; both pay taker on entry and
    exit by default (we don't assume we make).
    """
    spot = round_trip_bps(spot_venue, notional_usd, perp=False)
    perp = round_trip_bps(perp_venue, notional_usd, perp=True)
    return spot + perp


def yield_rotation_cost_bps(
    chain: str,
    notional_usd: float,
    swaps: int = 0,
) -> float:
    """Cost of moving capital from one DeFi pool to another on the same chain.

    Cost = 2 × gas (withdraw + deposit) / notional + `swaps` × Uniswap fee.
    Cross-chain rotation is out of Phase 1 scope, so we don't model bridges.
    """
    gas_usd = L2_GAS_USD.get(chain, 0.5)
    gas_bps = (2 * gas_usd) / max(notional_usd, 1.0) * 10_000
    return gas_bps + swaps * SWAP_BPS


@dataclass
class CostOverrides:
    """Lightweight override container for one-off backtest scenarios."""

    spot: dict[str, VenueCosts] = field(default_factory=dict)
    perp: dict[str, VenueCosts] = field(default_factory=dict)


def apply_overrides(overrides: CostOverrides) -> None:
    """Merge user overrides into the module-level fee tables. Mutates in place."""
    SPOT_VENUE_COSTS.update(overrides.spot)
    PERP_VENUE_COSTS.update(overrides.perp)


# Keep `replace` imported so external code can construct modified VenueCosts
# without importing dataclasses themselves.
__all__ = [
    "VenueCosts",
    "SPOT_VENUE_COSTS",
    "PERP_VENUE_COSTS",
    "L2_GAS_USD",
    "SWAP_BPS",
    "LP_DEPOSIT_BPS",
    "LP_WITHDRAW_BPS",
    "PREIPO_SPREAD_BPS",
    "slippage_bps",
    "round_trip_bps",
    "funding_arb_round_trip_bps",
    "yield_rotation_cost_bps",
    "CostOverrides",
    "apply_overrides",
    "replace",
]
