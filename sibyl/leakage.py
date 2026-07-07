# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl leakage controls for backtests.

Gated on ``asOf < now`` (active in backtest; inert in live mode, where there
is nothing to leak):

1. every search is date-filtered to ``asOf`` (applied ALWAYS — harmless
   live, load-bearing in backtest),
2. a leak-classifier pass over retrieved snippets drops post-``asOf``
   material,
3. optional live lookups are clamped to ``asOf`` (extension point; lookups
   are disabled by default),
4. known resolution-source URLs are blocked so the agent cannot read the
   ground truth it is being scored against.

The residual leak rate (snippets dropped by the classifier / total
retrieved) is tracked per trial and persisted for backtest audits.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from pythia.web_research.types import EvidenceSource

from sibyl.config import SEARCH_WINDOW_DAYS

logger = logging.getLogger(__name__)

# Domains of Pythia's resolution sources (the ground truth Sibyl is scored
# against). Always blocked, in live and backtest mode alike.
RESOLUTION_SOURCE_DOMAINS = (
    "go.ifrc.org",  # IFRC GO / Montandon (natural-hazard PA resolution)
    "acleddata.com",  # ACLED (ACE fatalities resolution)
    "internal-displacement.org",  # IDMC
    "gdacs.org",  # GDACS
    "fews.net",  # FEWS NET (DR Phase 3+ resolution)
    "fdw.fews.net",
    "ipcinfo.org",  # IPC API
)

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")
_TEXT_DATE_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\.?\s+(?:(\d{1,2})(?:st|nd|rd|th)?,?\s+)?(\d{4})\b",
    re.IGNORECASE,
)


@dataclass
class LeakageStats:
    """Per-trial leak accounting, persisted for backtest audits."""

    total_retrieved: int = 0
    dropped_post_asof: int = 0
    dropped_blocked_domain: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def residual_leak_rate(self) -> float:
        if self.total_retrieved <= 0:
            return 0.0
        return self.dropped_post_asof / self.total_retrieved

    def merge(self, other: "LeakageStats") -> None:
        self.total_retrieved += other.total_retrieved
        self.dropped_post_asof += other.dropped_post_asof
        self.dropped_blocked_domain += other.dropped_blocked_domain
        self.notes.extend(other.notes)

    def to_dict(self) -> dict:
        return {
            "total_retrieved": self.total_retrieved,
            "dropped_post_asof": self.dropped_post_asof,
            "dropped_blocked_domain": self.dropped_blocked_domain,
            "residual_leak_rate": round(self.residual_leak_rate, 4),
            "notes": list(self.notes),
        }


def is_backtest(as_of: date, *, today: Optional[date] = None) -> bool:
    """True when *as_of* is in the past — i.e. there is something to leak."""
    return as_of < (today or date.today())


def date_range_freshness(as_of: date, window_days: int = SEARCH_WINDOW_DAYS) -> str:
    """Brave freshness date range ending at *as_of*.

    Applied to EVERY search (live and backtest): live, the range simply ends
    today; backtest, it caps results at the as-of date.
    """
    start = as_of - timedelta(days=max(1, int(window_days)))
    return f"{start.isoformat()}to{as_of.isoformat()}"


def is_blocked_url(url: str) -> bool:
    """True when *url* belongs to a known resolution source."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return True  # malformed URL: refuse rather than fetch
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in RESOLUTION_SOURCE_DOMAINS)


def extract_dates(text: str, *, limit: int = 8) -> List[date]:
    """Best-effort extraction of explicit dates from a snippet.

    Heuristic classifier core: recognizes ISO ``YYYY-MM[-DD]`` and textual
    ``Month [DD,] YYYY`` forms. A trained/LLM classifier could replace this;
    the interface (text -> dates) is deliberately narrow to allow that swap.
    """
    found: List[date] = []
    for m in _ISO_DATE_RE.finditer(text or ""):
        year, month = int(m.group(1)), int(m.group(2))
        day = int(m.group(3)) if m.group(3) else 1
        try:
            if 1990 <= year <= 2100 and 1 <= month <= 12:
                found.append(date(year, month, day))
        except ValueError:
            continue
        if len(found) >= limit:
            return found
    for m in _TEXT_DATE_RE.finditer(text or ""):
        month = _MONTHS.get(m.group(1).lower().rstrip("."))
        day = int(m.group(2)) if m.group(2) else 1
        year = int(m.group(3))
        if not month:
            continue
        try:
            if 1990 <= year <= 2100:
                found.append(date(year, month, day))
        except ValueError:
            continue
        if len(found) >= limit:
            break
    return found


def _parse_source_date(raw: Optional[str]) -> Optional[date]:
    """Parse a Brave ``page_age``/date field (ISO-ish) into a date."""
    if not raw:
        return None
    txt = str(raw).strip()
    try:
        return datetime.fromisoformat(txt.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    dates = extract_dates(txt, limit=1)
    return dates[0] if dates else None


def snippet_leaks(text: str, as_of: date) -> bool:
    """Leak-classifier pass over one snippet: any explicit post-asOf date."""
    return any(d > as_of for d in extract_dates(text))


def filter_sources(
    sources: Iterable[EvidenceSource],
    as_of: date,
    *,
    today: Optional[date] = None,
) -> Tuple[List[EvidenceSource], LeakageStats]:
    """Apply blocked-domain and (in backtest) post-asOf filters to sources.

    Blocked resolution-source domains are dropped unconditionally. The
    date-based classifier only runs when ``as_of`` is in the past — in live
    mode there is nothing to leak and news snippets legitimately carry
    today's date.
    """
    stats = LeakageStats()
    backtest = is_backtest(as_of, today=today)
    kept: List[EvidenceSource] = []

    for src in sources:
        stats.total_retrieved += 1
        if is_blocked_url(src.url):
            stats.dropped_blocked_domain += 1
            continue
        if backtest:
            src_date = _parse_source_date(src.date)
            if src_date is not None and src_date > as_of:
                stats.dropped_post_asof += 1
                continue
            if snippet_leaks(f"{src.title} {src.summary}", as_of):
                stats.dropped_post_asof += 1
                continue
        kept.append(src)

    if backtest and stats.dropped_post_asof:
        stats.notes.append(
            f"dropped {stats.dropped_post_asof}/{stats.total_retrieved} "
            f"post-{as_of.isoformat()} sources"
        )
        logger.info(
            "sibyl.leakage: dropped %d/%d post-asOf sources (asOf=%s)",
            stats.dropped_post_asof, stats.total_retrieved, as_of.isoformat(),
        )
    return kept, stats


def clamp_live_lookup_date(requested: date, as_of: date) -> date:
    """Clamp an optional-live-lookup date to asOf (extension point, §3.5.3).

    Live lookups are disabled by default (``LIVE_LOOKUPS_ENABLED=False``);
    any future implementation must route its effective date through here.
    """
    return min(requested, as_of)
