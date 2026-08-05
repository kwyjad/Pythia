# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Resolution writers for the machine — deterministic, rules only.

Hard rules enforced here:

1. Every resolution row carries provenance: source, source record ids /
   document URLs, retrieval timestamp, and which rule fired.
2. Reconciliation is deterministic — no LLM calls anywhere in this
   module (or its callers in the detection layer).
4. Resolutions freeze at month-end + ``freeze.days_after_month_end``
   and are never reopened: an existing row past its freeze date is
   IMMUTABLE — a re-run that would have changed it writes an
   append-only record to ``pa_resolution_revisions`` instead.
6. GDACS never appears here as a resolution value (Phase 1 has no GDACS
   input at all; later phases use it for detection/ceiling only).
"""

from __future__ import annotations

import calendar
import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from resolver.resolution_machine.rulebook import Rulebook
from resolver.resolution_machine.schema import ensure_schema, utcnow_iso

LOG = logging.getLogger(__name__)

STATUS_RESOLVED_ZERO = "RESOLVED_ZERO"
STATUS_RESOLVED_VALUE = "RESOLVED_VALUE"
STATUS_NO_DATA = "NO_DATA"

WRITE_WRITTEN = "written"
WRITE_FROZEN_SKIP = "frozen_skip"


def freeze_date_for_ym(ym: str, rulebook: Rulebook) -> date:
    """Freeze date for a month: month-end + freeze.days_after_month_end."""
    days = int(rulebook["freeze.days_after_month_end"])
    y, m = (int(p) for p in ym.split("-"))
    month_end = date(y, m, calendar.monthrange(y, m)[1])
    return month_end + timedelta(days=days)


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _log_revision(
    con,
    *,
    iso3: str,
    hazard_code: str,
    metric: str,
    ym: str,
    observed_status: str,
    observed_value: float | None,
    source: str,
    note: str,
    evidence: dict[str, Any] | None,
) -> None:
    con.execute(
        """
        INSERT INTO pa_resolution_revisions
            (iso3, hazard_code, metric, ym, observed_status, observed_value,
             source, note, evidence_json, logged_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            iso3,
            hazard_code,
            metric,
            ym,
            observed_status,
            observed_value,
            source,
            note,
            json.dumps(evidence) if evidence is not None else None,
            utcnow_iso(),
        ],
    )


def write_zero_resolution(
    con,
    *,
    iso3: str,
    hazard_code: str,
    ym: str,
    evidence_of_absence: dict[str, Any],
    rulebook: Rulebook,
    today: date | None = None,
) -> str:
    """Write a ``RESOLVED_ZERO`` row with full evidence of absence.

    ``evidence_of_absence`` must contain the IBTrACS query summary and
    the ReliefWeb sweep record (queries, hit counts, timestamps) — the
    caller assembles it from the detection + sweep layers.

    Returns :data:`WRITE_WRITTEN`, or :data:`WRITE_FROZEN_SKIP` when an
    existing resolution for the key is past its freeze date (rule 4:
    the existing row is untouched and the attempt is logged to
    ``pa_resolution_revisions``).
    """
    ensure_schema(con)
    metric = str(rulebook["resolution.metric"])
    unit = str(rulebook["resolution.unit"])
    today = today or _today()

    existing = con.execute(
        """
        SELECT status, value, freeze_at FROM pa_resolutions
        WHERE iso3 = ? AND hazard_code = ? AND metric = ? AND ym = ?
        """,
        [iso3, hazard_code, metric, ym],
    ).fetchone()

    source = "resolution_machine:absence"
    rule_fired = (
        f"cyclone.resolved_zero.v{rulebook.version}: "
        "no IBTrACS trigger and ReliefWeb sweep silent"
    )

    if existing is not None:
        frozen_from = date.fromisoformat(str(existing[2]))
        if today >= frozen_from:
            LOG.warning(
                "[resolutions] %s/%s/%s/%s frozen since %s — skip, logging revision",
                iso3,
                hazard_code,
                metric,
                ym,
                frozen_from.isoformat(),
            )
            _log_revision(
                con,
                iso3=iso3,
                hazard_code=hazard_code,
                metric=metric,
                ym=ym,
                observed_status=STATUS_RESOLVED_ZERO,
                observed_value=0.0,
                source=source,
                note="re-run after freeze; resolved value not altered",
                evidence=evidence_of_absence,
            )
            return WRITE_FROZEN_SKIP

    now = utcnow_iso()
    source_urls = _collect_urls(evidence_of_absence)
    con.execute(
        """
        INSERT OR REPLACE INTO pa_resolutions
            (iso3, hazard_code, metric, ym, status, value, unit,
             source, source_record_ids, source_urls, rule_fired,
             evidence_json, retrieved_at, resolved_at, freeze_at,
             rulebook_version, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            iso3,
            hazard_code,
            metric,
            ym,
            STATUS_RESOLVED_ZERO,
            0.0,
            unit,
            source,
            json.dumps([]),  # a zero rests on absence — no source records
            json.dumps(source_urls),
            rule_fired,
            json.dumps(evidence_of_absence),
            str(evidence_of_absence.get("retrieved_at") or now),
            now,
            freeze_date_for_ym(ym, rulebook).isoformat(),
            rulebook.version,
            now,
        ],
    )
    return WRITE_WRITTEN


def _collect_urls(evidence: dict[str, Any]) -> list[str]:
    """Pull every URL the evidence cites (queries + samples + sources)."""
    urls: list[str] = []
    ibtracs = evidence.get("ibtracs") or {}
    for url in ibtracs.get("source_urls") or []:
        if url:
            urls.append(str(url))
    sweep = evidence.get("reliefweb") or {}
    for q in sweep.get("queries") or []:
        if q.get("url"):
            urls.append(str(q["url"]))
        for s in q.get("sample") or []:
            if s.get("url"):
                urls.append(str(s["url"]))
    # De-duplicate, order-preserving.
    seen: set[str] = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out
