"""MarketContext — bundles all optional advisory inputs for the AI advisor.

Each field is optional: the advisor builds the prompt from whatever is
available. This keeps the system gracefully degradable when optional services
(Tavily, FRED, Novaquity, TimesFM) are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .forecaster import Forecast


@dataclass
class MarketContext:
    """Aggregated context handed to the Claude advisor."""

    recent_bars: list[dict] = field(default_factory=list)
    forecast: Optional[Forecast] = None

    # Tavily news search results — short summaries with sentiment
    news_summary: Optional[str] = None
    news_items: list[dict] = field(default_factory=list)

    # FRED macroeconomic snapshot — e.g., {"DGS10": 4.32, "VIXCLS": 14.5, ...}
    macro_snapshot: Optional[dict[str, Any]] = None

    # Novaquity fundamental / supply-chain data (Japanese stocks only)
    fundamentals: Optional[dict[str, Any]] = None

    # IC / ICIR signal-quality summary (from the JSAI paper). Single string
    # like "IC mean=+0.0123, std=0.0567, ICIR=+0.217 (240 obs)".
    ic_summary: Optional[str] = None

    def to_prompt_section(self) -> str:
        """Render the optional context as markdown the LLM can parse.

        Sections that have no data are omitted entirely so the model isn't
        confused by empty placeholders.
        """
        sections = []

        if self.forecast is not None:
            sections.append(f"## TimesFM Forecast\n{self.forecast.summary()}")

        if self.news_summary:
            news_text = self.news_summary
            if self.news_items:
                top = "\n".join(
                    f"  - [{i.get('source', '?')}] {i.get('title', '')[:120]}"
                    for i in self.news_items[:5]
                )
                news_text = f"{news_text}\n\nTop headlines:\n{top}"
            sections.append(f"## Recent News\n{news_text}")

        if self.macro_snapshot:
            macro_lines = "\n".join(
                f"  - {k}: {v}" for k, v in self.macro_snapshot.items()
            )
            sections.append(f"## Macro Snapshot (FRED)\n{macro_lines}")

        if self.fundamentals:
            fund_lines = "\n".join(
                f"  - {k}: {v}" for k, v in self.fundamentals.items() if v is not None
            )
            sections.append(f"## Fundamentals (Novaquity)\n{fund_lines}")

        if self.ic_summary:
            sections.append(
                f"## Signal Quality (IC / ICIR)\n  {self.ic_summary}\n"
                "  (Higher absolute IC and ICIR means the upstream signal is "
                "more predictive. Per Kawamura et al. JSAI 2026, this is a "
                "stronger health check than P&L alone.)"
            )

        return "\n\n".join(sections) if sections else ""
