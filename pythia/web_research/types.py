# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceSource:
    title: str
    url: str
    publisher: str = ""
    date: Optional[str] = None
    summary: str = ""


@dataclass
class EvidencePack:
    query: str
    retrieved_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    recency_days: int = 120
    structural_context: str = ""
    recent_signals: List[str] = field(default_factory=list)
    sources: List[EvidenceSource] = field(default_factory=list)
    backend: str = "gemini"
    grounded: bool = False
    debug: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "retrieved_at": self.retrieved_at,
            "recency_days": self.recency_days,
            "structural_context": self.structural_context,
            "recent_signals": list(self.recent_signals),
            "sources": [s.__dict__ for s in self.sources],
            "backend": self.backend,
            "grounded": bool(self.grounded),
            "debug": dict(self.debug or {}),
            "error": self.error,
        }
