# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Build ``haz_impact_candidates`` — every figure the ladder may choose from.

One row per (cell, source, source record, value type). This module reads
the raw caches and transcribes what they state; it makes no precedence
decision at all — that is :mod:`resolver.hazard_resolution.reconcile`'s
job, and keeping the two apart is what makes reconciliation auditable
("here is everything we knew; here is the rule that picked one").

Three value types, and the distinction between them is load-bearing:

``affected``
    A stated people-affected figure (EM-DAT, IFRC GO, and from Phase 3
    ReliefWeb extraction). The quantity the machine resolves.
``displaced_lower_bound``
    A displacement figure (IDMC IDU). A floor under people-affected, not
    an estimate of it.
``exposed_ceiling``
    GDACS modelled exposure. **Never a resolution value** — it exists so
    the reconciler can bound the others. It is written as a candidate so
    the ceiling that applied to a decision is visible in the record, and
    :func:`ladder_candidates` filters it out of the ladder's view.

Attribution follows ``event_attribution.figure = start_month``: a figure
lands wholly in the month its event started, never split across months.
The event's full span rides along in ``span_start``/``span_end`` so a
reader can see the figure covers more than the month it sits in.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from resolver.hazard_resolution import emdat as emdat_mod
from resolver.hazard_resolution import gdacs as gdacs_mod
from resolver.hazard_resolution import idmc_idu as idu_mod
from resolver.hazard_resolution import ifrc_go as go_mod
from resolver.hazard_resolution.rulebook import Rulebook
from resolver.hazard_resolution.schema import ensure_haz_schema

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

VALUE_AFFECTED = "affected"
VALUE_LOWER_BOUND = "displaced_lower_bound"
VALUE_CEILING = "exposed_ceiling"

SOURCE_GDACS = "gdacs"


@dataclass
class Candidate:
    """One figure a source states for a country-month-hazard."""

    iso3: str
    ym: str
    hazard: str
    value: float
    value_type: str
    source: str
    source_ref: str
    stated_by: str | None = None
    doc_url: str | None = None
    extraction_model: str | None = None
    span_start: str | None = None
    span_end: str | None = None
    retrieved_at: str | None = None
    #: Order WITHIN a rung when that rung has its own precedence rule
    #: (0 = preferred). Only ``reliefweb_extracted`` sets it: the rulebook
    #: orders its figures by attributed authority then recency, which is not
    #: the "largest stated figure" rule the record-based rungs use. None
    #: leaves that default alone.
    preference_rank: int | None = None
    #: Free-form per-source extras carried into provenance and stored as
    #: ``detail_json``. For an extracted figure this is where the verbatim
    #: quote, the stated unit and any household conversion live — the
    #: evidence that the figure was transcribed rather than invented.
    detail: dict[str, Any] | None = None

    def provenance(self) -> dict[str, Any]:
        """This candidate's audit trail, as embedded in a resolution."""
        record = {
            "source": self.source,
            "source_ref": self.source_ref,
            "value": self.value,
            "value_type": self.value_type,
            "stated_by": self.stated_by,
            "doc_url": self.doc_url,
            "extraction_model": self.extraction_model,
            "event_span": {"start": self.span_start, "end": self.span_end},
            "retrieved_at": self.retrieved_at,
        }
        if self.preference_rank is not None:
            record["preference_rank"] = self.preference_rank
        if self.detail:
            record["detail"] = self.detail
        return record


def _year_month(ym: str) -> tuple[int, int]:
    year, month = ym.split("-")
    return int(year), int(month)


def _from_emdat(records: list[dict], iso3: str, ym: str, hazard: str) -> list[Candidate]:
    out = []
    for record in records:
        out.append(
            Candidate(
                iso3=iso3,
                ym=ym,
                hazard=hazard,
                value=float(record["total_affected"]),
                value_type=VALUE_AFFECTED,
                source="emdat",
                source_ref=str(record.get("disno") or record.get("_record_id")),
                stated_by="EM-DAT",
                doc_url=record.get("_source_url"),
                span_start=record.get("start_date"),
                span_end=record.get("end_date"),
                retrieved_at=record.get("_retrieved_at"),
            )
        )
    return out


def _from_ifrc_go(records: list[dict], iso3: str, ym: str, hazard: str) -> list[Candidate]:
    out = []
    for record in records:
        out.append(
            Candidate(
                iso3=iso3,
                ym=ym,
                hazard=hazard,
                value=float(record["num_affected"]),
                value_type=VALUE_AFFECTED,
                source="ifrc_go",
                source_ref=str(record.get("report_id") or record.get("_record_id")),
                # Which GO field stated the figure: a real num_affected and
                # a governmental estimate are not interchangeable, and the
                # record must say which one this was.
                stated_by=f"IFRC GO:{record.get('affected_field')}",
                doc_url=record.get("_source_url"),
                span_start=record.get("start_date"),
                span_end=record.get("end_date"),
                retrieved_at=record.get("_retrieved_at"),
            )
        )
    return out


def _from_idu(records: list[dict], iso3: str, ym: str, hazard: str) -> list[Candidate]:
    out = []
    for record in records:
        out.append(
            Candidate(
                iso3=iso3,
                ym=ym,
                hazard=hazard,
                value=float(record["displaced"]),
                # The label travels with the value from the moment it is
                # created — never applied later, never inferred downstream.
                value_type=VALUE_LOWER_BOUND,
                source="idmc_idu",
                source_ref=str(record.get("idu_id") or record.get("_record_id")),
                stated_by="IDMC IDU (displacement)",
                doc_url=record.get("_source_url"),
                span_start=record.get("start_date"),
                span_end=record.get("end_date"),
                retrieved_at=record.get("_retrieved_at"),
            )
        )
    return out


def _from_gdacs(events: list[dict], iso3: str, ym: str, hazard: str) -> list[Candidate]:
    """GDACS exposure as a CEILING candidate — never a resolution value."""

    out = []
    for event in events:
        exposed = event.get("exposed_population")
        if exposed is None:
            continue
        out.append(
            Candidate(
                iso3=iso3,
                ym=ym,
                hazard=hazard,
                value=float(exposed),
                value_type=VALUE_CEILING,
                source=SOURCE_GDACS,
                source_ref=str(event.get("event_id") or event.get("_record_id")),
                stated_by="GDACS modelled exposure (NOT reported impact)",
                doc_url=event.get("_source_url"),
                span_start=event.get("start_date"),
                span_end=event.get("end_date"),
                retrieved_at=event.get("_retrieved_at"),
            )
        )
    return out


def build_candidates(
    con: "duckdb.DuckDBPyConnection",
    iso3: str,
    ym: str,
    hazard: str,
    rulebook: Rulebook,
    extracted: list[Candidate] | None = None,
) -> list[Candidate]:
    """Collect every candidate figure for one country-month-hazard.

    ``extracted`` carries ladder rung 2 — the figures
    :mod:`resolver.hazard_resolution.figures` built from LLM-transcribed
    ReliefWeb documents. They are passed IN rather than read here because
    producing them costs money and must therefore be an explicit decision
    of the orchestrator, never a side effect of assembling candidates.
    Omit it and the rung is simply unpopulated, which the ladder already
    handles as "no candidate on this rung".
    """

    iso3 = iso3.upper()
    candidates: list[Candidate] = []
    candidates += _from_emdat(
        emdat_mod.records_for_country_month(con, iso3, ym, hazard), iso3, ym, hazard
    )
    candidates += list(extracted or [])
    candidates += _from_ifrc_go(
        go_mod.records_for_country_month(con, iso3, ym, hazard), iso3, ym, hazard
    )
    candidates += _from_idu(
        idu_mod.records_for_country_month(con, iso3, ym, hazard), iso3, ym, hazard
    )
    # Ceiling only: GDACS events overlapping the month, not just starting
    # in it — a flood that began last month still bounds this month.
    candidates += _from_gdacs(
        gdacs_mod.events_for_country_month(con, iso3, ym, hazard), iso3, ym, hazard
    )
    return candidates


def exposure_ceiling(
    con: "duckdb.DuckDBPyConnection", iso3: str, ym: str, hazard: str
) -> float | None:
    """The GDACS exposure ceiling for a cell, before candidates are built.

    The extracted-figure pipeline needs the ceiling to reject implausible
    transcriptions at candidate stage, which is upstream of the full
    candidate set — so it is computed from the GDACS cache directly.
    Mirrors :func:`reconcile._ceiling`: several overlapping events each
    bound the month, and the largest is the binding one.
    """

    exposures = [
        c.value
        for c in _from_gdacs(
            gdacs_mod.events_for_country_month(con, iso3.upper(), ym, hazard),
            iso3.upper(),
            ym,
            hazard,
        )
    ]
    return max(exposures) if exposures else None


def write_candidates(
    con: "duckdb.DuckDBPyConnection", candidates: list[Candidate], iso3: str, ym: str, hazard: str
) -> int:
    """Replace this cell's candidate rows (idempotent per re-run)."""

    ensure_haz_schema(con)
    year, month = _year_month(ym)
    try:
        con.execute("BEGIN TRANSACTION")
        con.execute(
            """
            DELETE FROM haz_impact_candidates
            WHERE iso3 = ? AND year = ? AND month = ? AND hazard = ?
            """,
            [iso3.upper(), year, month, hazard],
        )
        for c in candidates:
            con.execute(
                """
                INSERT OR IGNORE INTO haz_impact_candidates
                    (iso3, year, month, hazard, value, value_type, source,
                     source_ref, stated_by, doc_url, extraction_model,
                     preference_rank, detail_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    c.iso3,
                    year,
                    month,
                    c.hazard,
                    c.value,
                    c.value_type,
                    c.source,
                    c.source_ref,
                    c.stated_by,
                    c.doc_url,
                    c.extraction_model,
                    c.preference_rank,
                    json.dumps(c.detail, separators=(",", ":"), sort_keys=True)
                    if c.detail
                    else None,
                ],
            )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(candidates)


def ladder_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Candidates the ladder may CHOOSE from — everything but the ceiling.

    This is the guard that keeps GDACS exposure out of resolution values.
    """

    return [c for c in candidates if c.value_type != VALUE_CEILING]


def ceiling_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Candidates that bound the answer rather than being it."""

    return [c for c in candidates if c.value_type == VALUE_CEILING]
