"""Tests for the data fetcher modules — round-trip cache only.

These don't hit the network; they test the parquet save/load and shape
helpers. The user runs `scripts/fetch_real_data.py` on their machine to
populate caches with real data; that's covered by manual smoke testing.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from auto_investment.data_fetchers import funding as fdg
from auto_investment.data_fetchers import ohlcv as oh
from auto_investment.data_fetchers import yields as yld


def test_funding_save_load_roundtrip(tmp_path: Path):
    idx = pd.date_range("2026-01-01", periods=24, freq="1h", tz="UTC", name="ts")
    df = pd.DataFrame({"funding_rate": np.linspace(1e-4, 2e-4, 24)}, index=idx)
    path = tmp_path / "hl_BTC.parquet"
    fdg.save(df, path)
    loaded = fdg.load(path)
    # Parquet drops index freq metadata; compare values + index timestamps only.
    np.testing.assert_array_equal(loaded["funding_rate"].values, df["funding_rate"].values)
    assert (loaded.index == df.index).all()


def test_funding_cache_path_is_safe(tmp_path: Path):
    since = datetime(2026, 1, 1, tzinfo=timezone.utc)
    until = datetime(2026, 4, 1, tzinfo=timezone.utc)
    p = fdg.cache_path("hyperliquid", "BTC/USDC:USDC", since, until, base=tmp_path)
    # No raw slashes or colons in the path
    assert "/" not in p.name
    assert ":" not in p.name


def test_ohlcv_build_spread_series(tmp_path: Path):
    """Synthetic 2-venue cache → spread frame round trip."""
    idx = pd.date_range("2026-04-01", periods=120, freq="1min", tz="UTC", name="ts")
    a = pd.DataFrame(
        {"open": 1, "high": 1, "low": 1, "close": np.linspace(60_000, 60_120, 120),
         "volume": 1.0}, index=idx)
    b = pd.DataFrame(
        {"open": 1, "high": 1, "low": 1, "close": np.linspace(60_005, 60_115, 120),
         "volume": 1.0}, index=idx)
    since = idx[0].to_pydatetime()
    until = (idx[-1] + pd.Timedelta(minutes=1)).to_pydatetime()
    oh.save(a, oh.cache_path("binance", "BTC/USDT", "1m", since, until, base=tmp_path))
    oh.save(b, oh.cache_path("bybit", "BTC/USDT", "1m", since, until, base=tmp_path))
    spread = oh.build_spread_series("binance", "bybit", "BTC/USDT", "1m",
                                    since=since, until=until, base=tmp_path)
    assert "spread_bps" in spread.columns
    assert len(spread) == 120
    # Spread is positive in this synthetic
    assert (spread["spread"] > 0).any()


def test_yields_build_apy_grid(tmp_path: Path):
    idx = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC", name="ts")
    for pid, base in [("pool-a", 5.0), ("pool-b", 6.5)]:
        df = pd.DataFrame({"apy_total": np.full(30, base), "apy_base": np.full(30, base),
                           "apy_reward": np.zeros(30), "tvl_usd": np.full(30, 1e8)},
                          index=idx)
        yld.save_chart(pid, df, base=tmp_path)
    grid = yld.build_apy_grid(["pool-a", "pool-b"], base=tmp_path)
    assert grid.shape == (30, 2)
    # DefiLlama % → fraction conversion
    assert abs(grid["pool-a"].iloc[0] - 0.05) < 1e-9
    assert abs(grid["pool-b"].iloc[0] - 0.065) < 1e-9


def test_yields_build_apy_grid_handles_mismatched_indices(tmp_path: Path):
    """Real DefiLlama charts have non-aligned timestamps; outer-join + ffill
    must keep them usable instead of producing an empty grid."""
    base_start = pd.Timestamp("2026-01-01", tz="UTC")
    # Pool A: 30 daily bars
    idx_a = pd.date_range(base_start, periods=30, freq="D", tz="UTC", name="ts")
    # Pool B: same range but offset by 6h (different cadence in real life)
    idx_b = idx_a + pd.Timedelta(hours=6)
    for pid, idx, apy in [("pool-a", idx_a, 5.0), ("pool-b", idx_b, 6.5)]:
        df = pd.DataFrame({"apy_total": np.full(len(idx), apy),
                           "apy_base": np.full(len(idx), apy),
                           "apy_reward": np.zeros(len(idx)),
                           "tvl_usd": np.full(len(idx), 1e8)}, index=idx)
        yld.save_chart(pid, df, base=tmp_path)
    grid = yld.build_apy_grid(["pool-a", "pool-b"], base=tmp_path)
    assert not grid.empty
    assert grid.shape[1] == 2


def test_yields_build_apy_grid_skips_missing_pools(tmp_path: Path):
    """Missing chart files are skipped silently; we still get a usable grid."""
    idx = pd.date_range("2026-01-01", periods=20, freq="D", tz="UTC", name="ts")
    df = pd.DataFrame({"apy_total": np.full(20, 5.0), "apy_base": np.full(20, 5.0),
                       "apy_reward": np.zeros(20), "tvl_usd": np.full(20, 1e8)},
                      index=idx)
    yld.save_chart("pool-only", df, base=tmp_path)
    grid = yld.build_apy_grid(["pool-only", "pool-missing"], base=tmp_path)
    assert grid.shape == (20, 1)
    assert "pool-only" in grid.columns
    assert "pool-missing" not in grid.columns


def test_yields_build_apy_grid_returns_empty_on_no_pools(tmp_path: Path):
    grid = yld.build_apy_grid(["a", "b"], base=tmp_path)
    assert grid.empty
