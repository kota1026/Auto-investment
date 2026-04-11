"""Novaquity client — Japanese alternative-data platform from Tokyo University.

Novaquity (https://novaquity.net) provides AI-friendly alternative data for
Japanese-listed companies: earnings text features, supply-chain network
relationships, and event-propagation tracking. Both REST and MCP interfaces
are advertised.

This module wraps the REST surface. The MCP integration is left as a hook
because it requires the Anthropic Managed Agents flow (or a local MCP host),
which depends on the deployment environment.

Set NOVAQUITY_API_KEY and NOVAQUITY_BASE_URL in `.env` to enable. Without
those, the client returns None and the AI advisor skips the fundamentals
section gracefully.

NOTE: The exact REST schema is not yet public; the field/path names below
match the patterns described on novaquity.net and should be confirmed against
the actual API docs once you have access. The client falls back gracefully
on schema mismatches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class FundamentalSnapshot:
    """A flat dict of the most recent fundamental signals for one Japanese ticker."""

    ticker: str
    features: dict
    related_tickers: list[str]

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "features": self.features,
            "related_tickers": self.related_tickers,
        }


def fetch_fundamentals(ticker: str) -> FundamentalSnapshot | None:
    """Fetch the latest fundamental snapshot for a Japanese ticker.

    `ticker` is a 4-digit Tokyo Stock Exchange code (e.g., "7203" for Toyota).
    Returns None if Novaquity isn't configured or the call fails — the AI
    advisor will skip this section.
    """
    if not (settings.novaquity_api_key and settings.novaquity_base_url):
        logger.debug("Novaquity disabled (no NOVAQUITY_API_KEY/BASE_URL)")
        return None

    base = settings.novaquity_base_url.rstrip("/")
    headers = {
        "Authorization": f"Bearer {settings.novaquity_api_key}",
        "Accept": "application/json",
    }

    try:
        with httpx.Client(timeout=15.0, headers=headers) as client:
            features = _safe_get(client, f"{base}/companies/{ticker}/features")
            related = _safe_get(client, f"{base}/companies/{ticker}/network/related")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Novaquity fetch failed: %s", exc)
        return None

    if features is None and related is None:
        return None

    return FundamentalSnapshot(
        ticker=ticker,
        features=(features or {}).get("features", features or {}),
        related_tickers=(related or {}).get("related_tickers", []),
    )


def _safe_get(client: httpx.Client, url: str) -> dict | None:
    """GET a URL and return parsed JSON, swallowing errors."""
    try:
        r = client.get(url)
        if r.status_code == 200:
            return r.json()
        logger.debug("Novaquity GET %s -> %s", url, r.status_code)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Novaquity GET %s failed: %s", url, exc)
    return None


# -----------------------------------------------------------------------------
# MCP integration hook (placeholder)
# -----------------------------------------------------------------------------
#
# To use Novaquity as an MCP server with the Claude advisor, the deployment
# would attach the Novaquity MCP endpoint to a Managed Agents Vault and have
# the Claude advisor query it directly. Pseudo-code:
#
#     vault = client.beta.vaults.create(name="novaquity")
#     client.beta.vaults.credentials.create(
#         vault_id=vault.id,
#         display_name="Novaquity API",
#         auth={
#             "type": "mcp_oauth",
#             "mcp_server_url": "https://mcp.novaquity.net",
#             "access_token": settings.novaquity_api_key,
#         },
#     )
#     agent = client.beta.agents.create(
#         name="JP Equity Trader",
#         model="claude-opus-4-6",
#         mcp_servers=[{"type": "url", "name": "novaquity", "url": "https://mcp.novaquity.net"}],
#         tools=[{"type": "mcp_toolset", "mcp_server_name": "novaquity"}],
#     )
#
# This would let Claude query Novaquity tools (e.g., supply-chain
# propagation queries) live during signal evaluation. Implementation deferred
# until the official MCP server URL and auth schema are published.
