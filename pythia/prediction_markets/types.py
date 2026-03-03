# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class PredictionMarketQuestion:
    """A single prediction market question with its current probability."""

    platform: str  # "metaculus" | "polymarket" | "manifold"
    question_title: str
    url: str
    probability: float | None = None  # 0–1 for binary; None for numeric
    num_forecasters: int | None = None
    volume_usd: float | None = None  # Polymarket USD; Manifold in Mana
    close_date: str | None = None  # ISO date
    resolve_date: str | None = None
    question_type: str = "binary"  # "binary" | "numeric" | "multiple_choice"
    relevance_score: float = 0.0  # 0–10 from LLM filter
    relevance_note: str = ""  # Brief explanation

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "question_title": self.question_title,
            "url": self.url,
            "probability": self.probability,
            "num_forecasters": self.num_forecasters,
            "volume_usd": self.volume_usd,
            "close_date": self.close_date,
            "resolve_date": self.resolve_date,
            "question_type": self.question_type,
            "relevance_score": self.relevance_score,
            "relevance_note": self.relevance_note,
        }


@dataclass
class MarketBundle:
    """Collection of prediction market questions for a single Fred question."""

    questions: list[PredictionMarketQuestion] = field(default_factory=list)
    query_terms_used: list[str] = field(default_factory=list)
    retrieval_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    errors: list[str] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Format as text block for injection into prompts."""
        if not self.questions:
            return ""

        lines: list[str] = []
        for q in sorted(self.questions, key=lambda x: -x.relevance_score):
            prob_str = f"{q.probability:.0%}" if q.probability is not None else "N/A"
            parts = [f"[{q.platform}] \"{q.question_title}\" → {prob_str}"]
            if q.num_forecasters:
                parts.append(f"{q.num_forecasters} forecasters")
            if q.volume_usd is not None and q.volume_usd > 0:
                if q.platform == "manifold":
                    parts.append(f"M${q.volume_usd:,.0f} volume")
                else:
                    parts.append(f"${q.volume_usd:,.0f} volume")
            if q.relevance_note:
                parts.append(f"relevance: {q.relevance_note}")
            lines.append("- " + " | ".join(parts))
            lines.append(f"  URL: {q.url}")
        return "\n".join(lines)

    def to_research_dict(self) -> dict[str, Any]:
        """Serialize for inclusion in research_json."""
        return {
            "questions": [q.to_dict() for q in self.questions],
            "query_terms_used": list(self.query_terms_used),
            "retrieval_timestamp": self.retrieval_timestamp,
            "errors": list(self.errors),
        }
