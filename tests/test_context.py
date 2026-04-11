"""Tests for the MarketContext aggregation dataclass."""

from __future__ import annotations

from auto_investment.context import MarketContext
from auto_investment.forecaster import Forecast


def test_empty_context_renders_empty_string():
    ctx = MarketContext()
    assert ctx.to_prompt_section() == ""


def test_context_with_only_forecast():
    fc = Forecast(point=[100.0, 101.0], lower=[99.0, 99.5], upper=[101.0, 102.5], horizon=2, backend="naive")
    ctx = MarketContext(forecast=fc)
    rendered = ctx.to_prompt_section()
    assert "TimesFM Forecast" in rendered
    assert "naive" in rendered


def test_context_with_news_summary_and_items():
    ctx = MarketContext(
        news_summary="BTC rallying on ETF inflows.",
        news_items=[
            {"source": "coindesk.com", "title": "Bitcoin ETFs see record inflows"},
            {"source": "reuters.com", "title": "Crypto market rebounds"},
        ],
    )
    rendered = ctx.to_prompt_section()
    assert "Recent News" in rendered
    assert "ETF inflows" in rendered
    assert "coindesk.com" in rendered


def test_context_with_macro():
    ctx = MarketContext(macro_snapshot={"VIX": 14.5, "DGS10": 4.32})
    rendered = ctx.to_prompt_section()
    assert "Macro Snapshot" in rendered
    assert "VIX" in rendered
    assert "14.5" in rendered


def test_context_with_fundamentals():
    ctx = MarketContext(
        fundamentals={
            "ticker": "7203",
            "pe_ratio": 9.5,
            "earnings_revision": 0.04,
            "supply_chain_score": None,  # None values should be skipped
        }
    )
    rendered = ctx.to_prompt_section()
    assert "Fundamentals" in rendered
    assert "pe_ratio" in rendered
    assert "supply_chain_score" not in rendered  # filtered


def test_context_combines_all_sections():
    fc = Forecast(point=[100.0], lower=[99.0], upper=[101.0], horizon=1, backend="naive")
    ctx = MarketContext(
        forecast=fc,
        news_summary="Test summary",
        macro_snapshot={"VIX": 14.0},
        fundamentals={"pe": 10.0},
    )
    rendered = ctx.to_prompt_section()
    assert "TimesFM Forecast" in rendered
    assert "Recent News" in rendered
    assert "Macro Snapshot" in rendered
    assert "Fundamentals" in rendered
