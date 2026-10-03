# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The evidence ledger a trial builds as it reads (Oct 2026).

Each step the model returns ``ledger_add``: the NEW items it found, each with
the URL, the date, the source tier, the kind of claim, the figure or quote
word for word, and the direction it pushes. The code assigns the ids
(``E1``, ``E2``, ...), drops repeats and keeps the ledger across steps; the
final ledger is stored with the trial.

Source tiers: 1 resolving source or official statistics; 2 UN, cluster or
NGO report; 3 wire service or major outlet; 4 national or local media;
5 aggregator, blog or social.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

LEDGER_TIERS = (1, 2, 3, 4, 5)
LEDGER_KINDS = ("measurement", "forecast", "statement", "speculation")
LEDGER_DIRECTIONS = ("higher", "lower", "neutral")


def _tier(raw: Any) -> Optional[int]:
    try:
        t = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return t if t in LEDGER_TIERS else None


def normalise_item(raw: Any) -> Optional[Dict[str, Any]]:
    """One ``ledger_add`` item, cleaned; None when it carries no quote."""
    if not isinstance(raw, dict):
        return None
    quote = str(raw.get("quote") or raw.get("figure") or "").strip()
    if not quote:
        return None
    kind = str(raw.get("kind") or "").strip().lower()
    direction = str(raw.get("direction") or "").strip().lower()
    return {
        "url": str(raw.get("url") or "").strip(),
        "date": str(raw.get("date") or "").strip() or None,
        "tier": _tier(raw.get("tier")),
        "kind": kind if kind in LEDGER_KINDS else None,
        "quote": quote[:1000],
        "direction": direction if direction in LEDGER_DIRECTIONS else "neutral",
    }


def parse_ledger_add(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The cleaned items of a step response's ``ledger_add`` (never raises)."""
    raw = obj.get("ledger_add") if isinstance(obj, dict) else None
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        clean = normalise_item(item)
        if clean is not None:
            out.append(clean)
    return out


@dataclass
class EvidenceLedger:
    items: List[Dict[str, Any]] = field(default_factory=list)

    def _key(self, item: Dict[str, Any]) -> tuple:
        return (item.get("url") or "", " ".join(str(item.get("quote") or "").lower().split()))

    def add(self, new_items: List[Dict[str, Any]], *, step: int) -> List[Dict[str, Any]]:
        """Add the items not already held; returns those added, with ids."""
        seen = {self._key(it) for it in self.items}
        added: List[Dict[str, Any]] = []
        for item in new_items:
            key = self._key(item)
            if key in seen:
                continue
            seen.add(key)
            entry = {"id": f"E{len(self.items) + 1}", "step": int(step), **item}
            self.items.append(entry)
            added.append(entry)
        return added

    def to_list(self) -> List[Dict[str, Any]]:
        return [dict(it) for it in self.items]


def render_added(added: List[Dict[str, Any]]) -> str:
    """The line block the transcript shows for ids the code assigned."""
    if not added:
        return "(no new ledger items)"
    lines = []
    for it in added:
        tier = it.get("tier") or "?"
        lines.append(
            f"[{it['id']}] tier {tier}, {it.get('kind') or 'unclassified'}, "
            f"{it.get('date') or 'undated'}, {it['direction']}: {it['quote']}"
            + (f" ({it['url']})" if it.get("url") else "")
        )
    return "\n".join(lines)
