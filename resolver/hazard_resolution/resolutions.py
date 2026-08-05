# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Resolution writers for the machine — deterministic, rules only.

Hard rules enforced here:

1. Every resolution row carries provenance: source, source record ids /
   document URLs, retrieval timestamp, and which rule fired — all in
   ``haz_resolutions.provenance_json`` plus the ``rule_fired`` column.
2. Reconciliation is deterministic — no LLM calls anywhere in this
   module (or its callers in the detection layer).
4. Resolutions freeze at month-end + ``freeze_days`` and are never
   reopened: an existing row past its stored ``frozen_at`` deadline is
   IMMUTABLE — a re-run that would have changed it writes an
   append-only record to ``haz_revisions`` instead.
6. GDACS never appears here as a resolution value (Phase 1 has no GDACS
   input at all; later phases use it for detection/ceiling only).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import TYPE_CHECKING, Any

from resolver.hazard_resolution.rulebook import Rulebook
from resolver.hazard_resolution.rules import freeze_deadline
from resolver.hazard_resolution.schema import ensure_haz_schema

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

STATUS_RESOLVED_ZERO = "RESOLVED_ZERO"
STATUS_RESOLVED_VALUE = "RESOLVED_VALUE"
STATUS_NO_DATA = "NO_DATA"

WRITE_WRITTEN = "written"
WRITE_FROZEN_SKIP = "frozen_skip"

ZERO_SOURCE = "detection:absence"
ZERO_RULE_FIRED = "cyclone_zero:no_ibtracs_trigger+reliefweb_silent"


def _today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def _log_revision(
    con,
    *,
    iso3: str,
    year: int,
    month: int,
    hazard: str,
    source: str,
    source_ref: str,
    old_value: float | None,
    new_value: float | None,
    detail: dict[str, Any] | None,
) -> None:
    con.execute(
        """
        INSERT INTO haz_revisions
            (iso3, year, month, hazard, source, source_ref,
             old_value, new_value, detail_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            iso3,
            year,
            month,
            hazard,
            source,
            source_ref,
            old_value,
            new_value,
            json.dumps(detail) if detail is not None else None,
        ],
    )


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
    seen: set[str] = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def write_zero_resolution(
    con: "duckdb.DuckDBPyConnection",
    *,
    iso3: str,
    year: int,
    month: int,
    hazard: str,
    evidence_of_absence: dict[str, Any],
    rulebook: Rulebook,
    today: dt.date | None = None,
) -> str:
    """Write a ``RESOLVED_ZERO`` row with full evidence of absence.

    ``evidence_of_absence`` must contain the IBTrACS query summary and
    the ReliefWeb sweep record (queries, hit counts, timestamps) — the
    caller assembles it from the detection + sweep layers.

    Returns :data:`WRITE_WRITTEN`, or :data:`WRITE_FROZEN_SKIP` when an
    existing resolution for the cell is past its stored ``frozen_at``
    deadline (hard rule 4: the existing row is untouched and the
    attempt is logged to ``haz_revisions``).  Writing the FIRST
    resolution for an old cell is always allowed — that is the backcast
    path; freezing protects existing answers, it does not forbid late
    ones.
    """
    ensure_haz_schema(con)
    today = today or _today()

    existing = con.execute(
        """
        SELECT status, value, frozen_at FROM haz_resolutions
        WHERE iso3 = ? AND year = ? AND month = ? AND hazard = ?
        """,
        [iso3, year, month, hazard],
    ).fetchone()

    if existing is not None:
        frozen_at = existing[2]
        deadline = (
            frozen_at.date()
            if isinstance(frozen_at, dt.datetime)
            else freeze_deadline(year, month, rulebook)
        )
        if today > deadline:
            LOG.warning(
                "[resolutions] %s/%s/%d-%02d frozen since %s — skip, logging revision",
                iso3,
                hazard,
                year,
                month,
                deadline.isoformat(),
            )
            _log_revision(
                con,
                iso3=iso3,
                year=year,
                month=month,
                hazard=hazard,
                source=ZERO_SOURCE,
                source_ref="post-freeze re-run",
                old_value=existing[1],
                new_value=0.0,
                detail={
                    "note": "re-run after freeze; resolved value not altered",
                    "observed_status": STATUS_RESOLVED_ZERO,
                    "evidence_of_absence": evidence_of_absence,
                },
            )
            return WRITE_FROZEN_SKIP

    retrieved_at = str(evidence_of_absence.get("retrieved_at") or "")
    provenance = {
        "source": ZERO_SOURCE,
        "source_record_ids": [],  # a zero rests on absence — no source records
        "source_urls": _collect_urls(evidence_of_absence),
        "retrieved_at": retrieved_at,
        "rule_fired": ZERO_RULE_FIRED,
        "evidence_of_absence": evidence_of_absence,
    }
    frozen_at = dt.datetime.combine(
        freeze_deadline(year, month, rulebook), dt.time.min
    )
    try:
        con.execute("BEGIN TRANSACTION")
        con.execute(
            """
            DELETE FROM haz_resolutions
            WHERE iso3 = ? AND year = ? AND month = ? AND hazard = ?
            """,
            [iso3, year, month, hazard],
        )
        con.execute(
            """
            INSERT INTO haz_resolutions
                (iso3, year, month, hazard, status, value,
                 provenance_json, rule_fired, flagged, frozen_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, FALSE, ?)
            """,
            [
                iso3,
                year,
                month,
                hazard,
                STATUS_RESOLVED_ZERO,
                0.0,
                json.dumps(provenance),
                ZERO_RULE_FIRED,
                frozen_at,
            ],
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return WRITE_WRITTEN
