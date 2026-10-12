# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Which countries are asked the DR/PHASE3PLUS_IN_NEED (IPC Phase 3+) question.

Why (12 Oct 2026). Question creation asked a drought Phase 3+ question for any
country with a ``phase3plus_in_need`` row in the last 36 months. On the release
of 12 Oct 2026, 56 DR/PHASE3PLUS_IN_NEED horizons were due up to 2026-09 and 43
resolved; all 13 misses were countries whose source had stopped publishing:
ETH (FEWS NET, newest 2026-01), MRT and NIC (FEWS NET, newest 2024-11) and NAM
(IPC, newest 2025-09). A question its source no longer publishes for cannot
resolve, so asking it costs model calls and adds nothing to the record.

The rule. A DR Phase 3+ question is asked only when the country's newest
Current Situation row (metric ``phase3plus_in_need``, FEWS NET or IPC) is no
more than ``PHASE3_MAX_STALENESS_MONTHS`` (9; env
``PYTHIA_PHASE3_MAX_STALENESS_MONTHS``) months before the window's first month.
Nine months covers a twice-yearly analysis cycle plus publication lag; a
country whose newest analysis is older than that has missed a cycle.
DR/EVENT_OCCURRENCE (GDACS) is not affected.

The same bound decides which FEWS NET countries the IPC API connector still
excludes (``resolver/connectors/ipc_api.py``): a FEWS NET country whose FEWS NET
rows are stale is let through, so IPC can fill it where IPC publishes.

The guard. When ``facts_resolved`` holds no Current Situation row at all (a
fresh checkout, or the table cannot be read) the rule is not trusted to drop
every question: every country passes and the curated lists in
``create_questions_from_triage`` decide, as before. The decision is stored in
``phase3_eligibility`` (one row per HS run) and the debug bundle reports it.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger(__name__)

PHASE3_MAX_STALENESS_MONTHS = 9
ENV_MAX_STALENESS = "PYTHIA_PHASE3_MAX_STALENESS_MONTHS"

SOURCE_RULE = "rule"
SOURCE_NO_ROWS = "no_rows"
SOURCE_UNREADABLE = "unreadable"

TABLE = "phase3_eligibility"


def max_staleness_months() -> int:
    """The staleness bound in months (env override, else 9)."""
    raw = os.environ.get(ENV_MAX_STALENESS, "").strip()
    if not raw:
        return PHASE3_MAX_STALENESS_MONTHS
    try:
        value = int(raw)
    except ValueError:
        LOG.warning("%s=%r is not an integer; using %d", ENV_MAX_STALENESS, raw,
                    PHASE3_MAX_STALENESS_MONTHS)
        return PHASE3_MAX_STALENESS_MONTHS
    if value < 0:
        LOG.warning("%s=%d is negative; using %d", ENV_MAX_STALENESS, value,
                    PHASE3_MAX_STALENESS_MONTHS)
        return PHASE3_MAX_STALENESS_MONTHS
    return value


def months_between(later: str, earlier: str) -> int:
    """Whole months from ``earlier`` to ``later`` (both ``YYYY-MM``)."""
    ly, lm = (int(x) for x in str(later)[:7].split("-"))
    ey, em = (int(x) for x in str(earlier)[:7].split("-"))
    return (ly - ey) * 12 + (lm - em)


def window_start_for(today: date) -> str:
    """The first window month a forecast run on ``today`` would ask about
    (the month after today; the rule ``create_questions_from_triage`` uses)."""
    month = today.month + 1
    year = today.year + (month - 1) // 12
    month = ((month - 1) % 12) + 1
    return f"{year:04d}-{month:02d}"


def newest_current_situation(
    con: Any, *, publisher: Optional[str] = None
) -> Dict[str, Tuple[str, str]]:
    """``{iso3: (newest ym, publisher(s))}`` over ``phase3plus_in_need`` rows.

    ``publisher`` restricts to one source (compared case-insensitively). Raises
    when the table cannot be read; callers decide what that means.
    """
    columns = {str(r[1]).lower() for r in con.execute("PRAGMA table_info('facts_resolved')").fetchall()}
    if not columns:
        raise RuntimeError("facts_resolved does not exist")
    has_publisher = "publisher" in columns
    params: List[Any] = []
    where = ""
    if publisher:
        if not has_publisher:
            return {}
        where = "AND upper(COALESCE(publisher, '')) = ?"
        params.append(publisher.upper())
    pub_expr = "COALESCE(publisher, '?')" if has_publisher else "'?'"
    rows = con.execute(
        f"""
        SELECT upper(iso3) AS iso3, {pub_expr} AS publisher, MAX(ym) AS ym
        FROM facts_resolved
        WHERE lower(metric) = 'phase3plus_in_need'
          AND iso3 IS NOT NULL AND iso3 <> '' AND ym IS NOT NULL AND ym <> ''
          {where}
        GROUP BY 1, 2
        """,
        params,
    ).fetchall()
    newest: Dict[str, Tuple[str, List[str]]] = {}
    for iso3, pub, ym in rows:
        ym = str(ym)[:7]
        cur = newest.get(iso3)
        if cur is None or ym > cur[0]:
            newest[iso3] = (ym, [str(pub)])
        elif ym == cur[0]:
            cur[1].append(str(pub))
    return {iso: (ym, "+".join(sorted(set(pubs)))) for iso, (ym, pubs) in newest.items()}


@dataclass
class Decision:
    """Which countries may be asked a DR Phase 3+ question in one run."""

    window_start: str
    max_staleness: int
    source: str
    fresh: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    stale: Dict[str, Tuple[str, str, int]] = field(default_factory=dict)
    reason: str = ""

    @property
    def used_rule(self) -> bool:
        return self.source == SOURCE_RULE

    def admits(self, iso3: str) -> bool:
        if not self.used_rule:
            return True
        return str(iso3 or "").upper() in self.fresh

    def why_not(self, iso3: str) -> str:
        iso = str(iso3 or "").upper()
        if iso in self.stale:
            ym, pub, months = self.stale[iso]
            return (f"newest Phase 3+ Current Situation row {ym} ({pub}) is {months} "
                    f"months before {self.window_start}, more than {self.max_staleness}")
        return "no Phase 3+ Current Situation row from FEWS NET or IPC"


def decide(con: Any, *, window_start: str, max_staleness: Optional[int] = None) -> Decision:
    """The Phase 3+ decision for a window opening in ``window_start``. Never raises."""
    bound = max_staleness_months() if max_staleness is None else int(max_staleness)
    try:
        newest = newest_current_situation(con)
    except Exception as exc:  # noqa: BLE001 - an unreadable table is a guard condition
        return Decision(window_start, bound, SOURCE_UNREADABLE,
                        reason=f"facts_resolved unreadable: {exc}")
    if not newest:
        return Decision(window_start, bound, SOURCE_NO_ROWS,
                        reason="facts_resolved holds no phase3plus_in_need row; the curated lists decide")
    fresh: Dict[str, Tuple[str, str]] = {}
    stale: Dict[str, Tuple[str, str, int]] = {}
    for iso3, (ym, pub) in newest.items():
        months = months_between(window_start, ym)
        if months > bound:
            stale[iso3] = (ym, pub, months)
        else:
            fresh[iso3] = (ym, pub)
    return Decision(window_start, bound, SOURCE_RULE, fresh, stale)


def fewsnet_exclusions(
    con: Any,
    fewsnet_list: set,
    *,
    reference_ym: str,
    max_staleness: Optional[int] = None,
) -> Optional[Tuple[set, Dict[str, str]]]:
    """FEWS NET countries the IPC connector should still exclude.

    Returns ``(excluded, let_through)`` where ``let_through`` maps each FEWS
    NET-listed country NOT excluded to the reason. Returns None when the DB
    holds no FEWS NET Phase 3+ row at all or cannot be read: then the caller
    keeps the whole list (the old behaviour), because "FEWS NET published
    nothing" and "FEWS NET was not ingested" cannot be told apart.
    """
    bound = max_staleness_months() if max_staleness is None else int(max_staleness)
    try:
        newest = newest_current_situation(con, publisher="FEWS NET")
    except Exception as exc:  # noqa: BLE001
        LOG.warning("[phase3] FEWS NET recency unreadable (%s); keeping the full list", exc)
        return None
    if not newest:
        return None
    excluded: set = set()
    let_through: Dict[str, str] = {}
    for iso3 in sorted(c.upper() for c in fewsnet_list):
        row = newest.get(iso3)
        if row is None:
            let_through[iso3] = "no FEWS NET phase3plus_in_need row"
            continue
        months = months_between(reference_ym, row[0])
        if months > bound:
            let_through[iso3] = (f"newest FEWS NET row {row[0]} is {months} months before "
                                 f"{reference_ym}, more than {bound}")
        else:
            excluded.add(iso3)
    return excluded, let_through


def ensure_table(con: Any) -> None:
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            hs_run_id TEXT PRIMARY KEY,
            window_start TEXT,
            max_staleness_months INTEGER,
            source TEXT,
            reason TEXT,
            fresh_json TEXT,
            stale_json TEXT,
            asked_json TEXT,
            dropped_json TEXT,
            is_test BOOLEAN DEFAULT FALSE,
            decided_at TIMESTAMP DEFAULT now()
        )
        """
    )


def record(con: Any, decision: Decision, *, hs_run_id: str, asked: List[str],
           dropped: List[str], is_test: bool) -> None:
    """Store the decision (one row per HS run) and say it where a person looks."""
    ensure_table(con)
    stale_payload = {iso: {"newest_month": ym, "source": pub, "months_before_window": m}
                     for iso, (ym, pub, m) in sorted(decision.stale.items())}
    fresh_payload = {iso: {"newest_month": ym, "source": pub}
                     for iso, (ym, pub) in sorted(decision.fresh.items())}
    dropped_payload = {iso: decision.why_not(iso) for iso in sorted(set(dropped))}
    con.execute(f"DELETE FROM {TABLE} WHERE hs_run_id = ?", [hs_run_id])
    con.execute(
        f"""
        INSERT INTO {TABLE} (hs_run_id, window_start, max_staleness_months, source, reason,
            fresh_json, stale_json, asked_json, dropped_json, is_test)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [hs_run_id, decision.window_start, decision.max_staleness, decision.source,
         decision.reason, json.dumps(fresh_payload), json.dumps(stale_payload),
         json.dumps(sorted(set(asked))), json.dumps(dropped_payload), bool(is_test)],
    )
    line = (
        f"DR Phase 3+ questions for the window opening {decision.window_start}: asked for "
        f"{len(set(asked))} countr{'y' if len(set(asked)) == 1 else 'ies'}, dropped "
        f"{len(dropped_payload)} whose source stopped publishing "
        f"({', '.join(dropped_payload) or 'none'}); bound {decision.max_staleness} months; "
        f"source: {decision.source}"
    )
    print(line)
    for iso, why in dropped_payload.items():
        print(f"  {iso}: {why}")
    if not decision.used_rule:
        print(f"::warning title=Phase 3+ eligibility guard::{decision.reason}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(f"\n#### DR Phase 3+ eligibility\n\n{line}\n")
                for iso, why in dropped_payload.items():
                    fh.write(f"\n- {iso}: {why}")
                fh.write("\n")
        except OSError:
            pass


def latest_decision(con: Any, hs_run_id: Optional[str] = None) -> Optional[dict]:
    """The stored decision for ``hs_run_id`` (or the newest), else None."""
    try:
        where = "WHERE hs_run_id = ?" if hs_run_id else ""
        params = [hs_run_id] if hs_run_id else []
        cur = con.execute(f"SELECT * FROM {TABLE} {where} ORDER BY decided_at DESC LIMIT 1", params)
        row = cur.fetchone()
        if not row:
            return None
        out = dict(zip([d[0] for d in cur.description], row))
        for key in ("fresh_json", "stale_json", "asked_json", "dropped_json"):
            try:
                out[key.replace("_json", "")] = json.loads(out.get(key) or "null")
            except (TypeError, ValueError):
                out[key.replace("_json", "")] = None
        return out
    except Exception:  # noqa: BLE001 - an older DB has no table
        return None


__all__ = [
    "Decision",
    "ENV_MAX_STALENESS",
    "PHASE3_MAX_STALENESS_MONTHS",
    "decide",
    "fewsnet_exclusions",
    "latest_decision",
    "max_staleness_months",
    "months_between",
    "newest_current_situation",
    "record",
    "window_start_for",
]
