"""Sanity tests for the venue cost model."""

from auto_investment import cost_model as cm


def test_hyperliquid_is_cheapest_perp():
    assert (
        cm.PERP_VENUE_COSTS["hyperliquid"].taker_bps
        < cm.PERP_VENUE_COSTS["dydx_v4"].taker_bps
    )
    assert (
        cm.PERP_VENUE_COSTS["hyperliquid"].taker_bps
        < cm.PERP_VENUE_COSTS["aevo"].taker_bps
    )


def test_round_trip_bps_includes_two_slippage_charges():
    rt = cm.round_trip_bps("binancejp", 1_000.0, perp=False)
    fees = cm.SPOT_VENUE_COSTS["binancejp"].taker_bps * 2
    slip = cm.slippage_bps(1_000.0) * 2
    assert abs(rt - (fees + slip)) < 1e-6


def test_funding_arb_round_trip_under_50bps_at_2k():
    rt = cm.funding_arb_round_trip_bps("binancejp", "hyperliquid", 2_000.0)
    # Spot taker (2×) + perp taker (2×) + slippage (4×) at $2k notional
    # should land well under 50 bps for a delta-neutral entry at small size.
    assert rt < 50.0, f"funding arb round-trip too expensive: {rt:.2f} bps"


def test_yield_rotation_cost_dominates_at_small_notional():
    """At $100 notional, two L2 gas tx (~$0.40) is ~40 bps — material vs profit."""
    cost = cm.yield_rotation_cost_bps("arbitrum", 100.0)
    # Round-trip $0.20 × 2 = $0.40 on $100 = 40 bps. Material enough to gate.
    assert cost >= 30.0


def test_yield_rotation_cost_negligible_at_large_notional():
    """At $50k, two L2 gas tx should round to <1 bp."""
    cost = cm.yield_rotation_cost_bps("base", 50_000.0)
    assert cost < 1.0


def test_unknown_venue_raises():
    try:
        cm.round_trip_bps("nonexistent", 1_000.0)
    except KeyError:
        return
    raise AssertionError("expected KeyError")
