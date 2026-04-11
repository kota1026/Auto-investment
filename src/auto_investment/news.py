"""News sentiment fetcher built on Tavily.

Tavily (https://tavily.com) is a search API tuned for AI agents — results
come pre-filtered and structured. We use it to grab the most recent news
items for the symbol being traded and feed a short summary into the Claude
advisor's prompt.

Set TAVILY_API_KEY in `.env` to enable. Without it, this module is a no-op
that returns `None`, and the AI advisor simply ignores the news section.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from .config import settings

logger = logging.getLogger(__name__)

TAVILY_ENDPOINT = "https://api.tavily.com/search"


@dataclass
class NewsBundle:
    """A short news summary plus the underlying items for citation."""

    summary: str
    items: list[dict]

    def to_dict(self) -> dict:
        return {"summary": self.summary, "items": self.items}


def fetch_news(symbol: str, max_results: int = 5, days: int = 2) -> NewsBundle | None:
    """Fetch recent news for a trading symbol via Tavily.

    Returns None if `TAVILY_API_KEY` is not configured or the API call fails.
    The caller (typically `live.run_once()`) should handle None gracefully.
    """
    if not settings.tavily_api_key:
        logger.debug("Tavily disabled (no TAVILY_API_KEY)")
        return None

    query = _build_query(symbol)
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                TAVILY_ENDPOINT,
                json={
                    "api_key": settings.tavily_api_key,
                    "query": query,
                    "search_depth": "basic",
                    "topic": "news",
                    "days": days,
                    "max_results": max_results,
                    "include_answer": True,
                },
            )
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tavily fetch failed: %s", exc)
        return None

    answer = data.get("answer") or "No summary available."
    items = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "source": _domain_of(r.get("url", "")),
            "snippet": (r.get("content") or "")[:300],
            "score": r.get("score", 0.0),
        }
        for r in data.get("results", [])
    ]
    return NewsBundle(summary=answer, items=items)


def _build_query(symbol: str) -> str:
    """Convert a market symbol like 'BTC/USDT' into a natural news query."""
    base = symbol.split("/")[0].upper()
    nice_name = {
        "BTC": "Bitcoin",
        "ETH": "Ethereum",
        "SOL": "Solana",
        "XRP": "XRP",
        "DOGE": "Dogecoin",
    }.get(base, base)
    return (
        f"{nice_name} price news, market sentiment, regulatory developments, "
        f"and macro events affecting crypto in the past 48 hours"
    )


def _domain_of(url: str) -> str:
    """Extract a short domain label for citation display."""
    if not url:
        return "?"
    try:
        host = url.split("//", 1)[-1].split("/", 1)[0]
        return host.replace("www.", "")
    except Exception:  # noqa: BLE001
        return "?"
