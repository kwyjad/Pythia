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
from resolver.hazard_resolution.rules import freeze_deadline, is_provisional
from resolver.hazard_resolution.schema import ensure_haz_schema

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

STATUS_RESOLVED_ZERO = "RESOLVED_ZERO"
STATUS_RESOLVED_VALUE = "RESOLVED_VALUE"
STATUS_NO_DATA = "NO_DATA"

WRITE_WRITTEN = "written"
WRITE_FROZEN_SKIP = "frozen_skip"
WRITE_PENDING = "pending_skip"

#: reconcile.STATUS_PENDING, duplicated as a literal to keep this module
#: free of an import cycle (reconcile imports candidates, which imports the
#: source connectors). Pinned by a test so the two cannot drift.
WRITE_PENDING_STATUS = "PENDING"

ZERO_SOURCE = "detection:absence"
ZERO_RULE_FIRED = "cyclone_zero:no_ibtracs_trigger+reliefweb_silent"

#: Per-hazard zero rules. A zero means different evidence for each hazard
#: (no qualifying storm track vs. no qualifying GDACS alert), and the
#: resolution must say which, so the two are never conflated in analysis.
ZERO_RULE_BY_HAZARD = {
    "TC": ZERO_RULE_FIRED,
    "FL": "flood_zero:no_gdacs_trigger+reliefweb_silent",
    # Drought has no ReliefWeb sweep: it is not an event anyone files a
    # report about on the day. Its absence evidence is the pair of
    # statements the rule needs — the indicators saw no drought, and IPC
    # recorded no qualifying Phase 3+ increase.
    "DR": "drought_zero:no_indicator_signal+no_ipc_deterioration",
}


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
    # Detection evidence: IBTrACS for cyclones, GDACS for floods, the IPC
    # analyses for droughts.
    for detector in ("ibtracs", "gdacs", "ipc"):
        for url in (evidence.get(detector) or {}).get("source_urls") or []:
            if url:
                urls.append(str(url))
    # Drought: the analysis windows the zero rests on, and every indicator
    # feed consulted. A zero has to cite what it checked.
    for key in ("covering", "previous"):
        analysis = (evidence.get("ipc") or {}).get(key) or {}
        if analysis.get("source_url"):
            urls.append(str(analysis["source_url"]))
    for reading in (evidence.get("indicators") or {}).get("readings") or []:
        if reading.get("source_url"):
            urls.append(str(reading["source_url"]))
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


def _frozen_row(
    con,
    *,
    iso3: str,
    year: int,
    month: int,
    hazard: str,
    rulebook: Rulebook,
    today: dt.date,
) -> tuple[bool, tuple | None]:
    """Is there an EXISTING resolution for this cell past its freeze date?

    Hard rule 4, in one place: freezing protects answers that already
    exist. Writing the FIRST resolution for an old cell is always allowed
    — that is the backcast path.
    """

    existing = con.execute(
        """
        SELECT status, value, frozen_at FROM haz_resolutions
        WHERE iso3 = ? AND year = ? AND month = ? AND hazard = ?
        """,
        [iso3, year, month, hazard],
    ).fetchone()
    if existing is None:
        return False, None

    frozen_at = existing[2]
    deadline = (
        frozen_at.date()
        if isinstance(frozen_at, dt.datetime)
        else freeze_deadline(year, month, rulebook)
    )
    return today > deadline, existing


def _write_row(
    con,
    *,
    iso3: str,
    year: int,
    month: int,
    hazard: str,
    status: str,
    value: float | None,
    provenance: dict[str, Any],
    rule_fired: str,
    flagged: bool,
    provisional: bool,
    rulebook: Rulebook,
) -> None:
    """Replace this cell's resolution row inside one transaction."""

    frozen_at = dt.datetime.combine(freeze_deadline(year, month, rulebook), dt.time.min)
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
                 provenance_json, rule_fired, flagged, provisional, frozen_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                iso3,
                year,
                month,
                hazard,
                status,
                value,
                json.dumps(provenance),
                rule_fired,
                flagged,
                provisional,
                frozen_at,
            ],
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def write_reconciliation(
    con: "duckdb.DuckDBPyConnection",
    reconciliation,
    rulebook: Rulebook,
    *,
    today: dt.date | None = None,
) -> str:
    """Persist a :class:`~resolver.hazard_resolution.reconcile.Reconciliation`.

    The ladder's verdict is written verbatim — this function makes no
    decision of its own beyond the freeze guard. A cell already frozen is
    left untouched and the attempt is logged to ``haz_revisions``, so a
    post-freeze upstream revision is visible without ever altering the
    resolved value (hard rule 4).

    A ``PENDING`` verdict writes nothing: the cell is triggered but no rung
    has reported yet and it has not frozen, so there is no answer to record.
    """

    ensure_haz_schema(con)
    today = today or _today()

    if reconciliation.status == WRITE_PENDING_STATUS:
        LOG.debug(
            "[resolutions] %s/%s/%s pending — no rung has reported and the cell "
            "has not frozen; nothing written",
            reconciliation.iso3, reconciliation.hazard, reconciliation.ym,
        )
        return WRITE_PENDING

    iso3 = reconciliation.iso3
    year, month = (int(p) for p in reconciliation.ym.split("-"))
    hazard = reconciliation.hazard

    frozen, existing = _frozen_row(
        con, iso3=iso3, year=year, month=month, hazard=hazard,
        rulebook=rulebook, today=today,
    )
    if frozen:
        LOG.warning(
            "[resolutions] %s/%s/%d-%02d frozen — skip, logging revision",
            iso3, hazard, year, month,
        )
        _log_revision(
            con,
            iso3=iso3,
            year=year,
            month=month,
            hazard=hazard,
            source=(reconciliation.winner.source if reconciliation.winner else "ladder"),
            source_ref=(
                reconciliation.winner.source_ref if reconciliation.winner else "post-freeze re-run"
            ),
            old_value=existing[1] if existing else None,
            new_value=reconciliation.value,
            detail={
                "note": "re-run after freeze; resolved value not altered",
                "observed_status": reconciliation.status,
                "observed_rule_fired": reconciliation.rule_fired,
                "provenance": reconciliation.provenance,
            },
        )
        return WRITE_FROZEN_SKIP

    _write_row(
        con,
        iso3=iso3,
        year=year,
        month=month,
        hazard=hazard,
        status=reconciliation.status,
        value=reconciliation.value,
        provenance=reconciliation.provenance,
        rule_fired=reconciliation.rule_fired,
        flagged=reconciliation.flagged,
        provisional=reconciliation.provisional,
        rulebook=rulebook,
    )
    return WRITE_WRITTEN


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
    rule_fired = ZERO_RULE_BY_HAZARD.get(hazard, ZERO_RULE_FIRED)

    frozen, existing = _frozen_row(
        con, iso3=iso3, year=year, month=month, hazard=hazard,
        rulebook=rulebook, today=today,
    )
    if frozen:
        LOG.warning(
            "[resolutions] %s/%s/%d-%02d frozen — skip, logging revision",
            iso3,
            hazard,
            year,
            month,
        )
        _log_revision(
            con,
            iso3=iso3,
            year=year,
            month=month,
            hazard=hazard,
            source=ZERO_SOURCE,
            source_ref="post-freeze re-run",
            old_value=existing[1] if existing else None,
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
        "rule_fired": rule_fired,
        "evidence_of_absence": evidence_of_absence,
    }
    _write_row(
        con,
        iso3=iso3,
        year=year,
        month=month,
        hazard=hazard,
        status=STATUS_RESOLVED_ZERO,
        value=0.0,
        provenance=provenance,
        rule_fired=rule_fired,
        flagged=False,
        provisional=is_provisional(year, month, rulebook, today=today),
        rulebook=rulebook,
    )
    return WRITE_WRITTEN
