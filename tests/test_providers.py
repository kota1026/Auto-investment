"""Tests for optional context providers (news, macro, novaquity).

These exercise the no-key fallback paths so the test suite stays hermetic.
The actual API calls are not exercised here — they would require live network
access and valid credentials.
"""

from __future__ import annotations

from unittest.mock import patch

from auto_investment.macro import fetch_macro_snapshot, macro_to_prompt_line
from auto_investment.news import _build_query, _domain_of, fetch_news
from auto_investment.novaquity import fetch_fundamentals


def test_news_disabled_without_key():
    with patch("auto_investment.news.settings") as mock_settings:
        mock_settings.tavily_api_key = ""
        result = fetch_news("BTC/USDT")
        assert result is None


def test_news_query_builder_uses_friendly_name():
    q = _build_query("BTC/USDT")
    assert "Bitcoin" in q
    q2 = _build_query("ETH/USDT")
    assert "Ethereum" in q2
    q3 = _build_query("XYZ/USDT")
    # Falls back to the symbol itself for unknown bases
    assert "XYZ" in q3


def test_news_domain_extraction():
    assert _domain_of("https://www.coindesk.com/markets/article") == "coindesk.com"
    assert _domain_of("https://reuters.com/foo") == "reuters.com"
    assert _domain_of("") == "?"
    # Strings without a scheme are returned as-is (no parsing happens) — fine
    assert _domain_of("not-a-url") == "not-a-url"


def test_macro_disabled_without_key():
    with patch("auto_investment.macro.settings") as mock_settings:
        mock_settings.fred_api_key = ""
        result = fetch_macro_snapshot()
        assert result is None


def test_macro_to_prompt_line_empty():
    assert macro_to_prompt_line({}) == "No macro data."


def test_macro_to_prompt_line_formats_pairs():
    line = macro_to_prompt_line({"VIX": 14.5, "DGS10": 4.32})
    assert "VIX=14.5" in line
    assert "DGS10=4.32" in line


def test_novaquity_disabled_without_key():
    with patch("auto_investment.novaquity.settings") as mock_settings:
        mock_settings.novaquity_api_key = ""
        mock_settings.novaquity_base_url = ""
        result = fetch_fundamentals("7203")
        assert result is None
