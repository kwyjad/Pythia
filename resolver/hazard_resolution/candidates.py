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
from resolver.hazard_resolution.rules import usable_exposure
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
    con: "duckdb.DuckDBPyConnection", iso3: str, ym: str, hazard: str,
    rulebook: Rulebook | None = None,
) -> float | None:
    """The GDACS exposure ceiling for a cell, before candidates are built.

    The extracted-figure pipeline needs the ceiling to reject implausible
    transcriptions at candidate stage, which is upstream of the full
    candidate set — so it is computed from the GDACS cache directly.
    Mirrors :func:`reconcile._ceiling`: several overlapping events each
    bound the month, and the largest is the binding one — and, as there, a
    non-positive exposure is GDACS declining to say rather than GDACS saying
    nobody was exposed, so it contributes no ceiling instead of a ceiling of
    zero. Extraction is where that mattered most: a ceiling of zero rejects
    every transcribed figure for the cell, and in the August 2026 run 147 of
    199 rejected figures were rejected against one.
    """

    return exposure_ceiling_basis(con, iso3, ym, hazard, rulebook)["value"]


#: The GDACS response field the ceiling is computed from, carried on every
#: rejection so "the ceiling is broken" and "the figure is wrong" can be told
#: apart. GDACS calls it ``population``; the raw cache stores it under the
#: name that says what it is.
CEILING_FIELD = "gdacs.population -> payload.exposed_population"


def exposure_ceiling_basis(
    con: "duckdb.DuckDBPyConnection", iso3: str, ym: str, hazard: str,
    rulebook: Rulebook | None = None,
) -> dict[str, Any]:
    """The ceiling AND where it came from: value, event, field, alternatives.

    :func:`exposure_ceiling` returns only the number, which is all the
    rejection rule needs. A reader auditing a rejection needs the rest: a
    ceiling of 2 against a reported 40,000 is a GDACS enrichment failure,
    not a mis-transcription, and only the event id and the field name say
    which of the two it is.
    """

    iso3 = iso3.upper()
    raw_events = gdacs_mod.events_for_country_month(con, iso3, ym, hazard)
    events = _from_gdacs(raw_events, iso3, ym, hazard)
    # Below the rulebook's plausibility floor an exposure is a parse
    # failure, not a bound (rules.usable_exposure); without a rulebook the
    # historical "> 0" test applies.
    if rulebook is not None:
        positive = [c for c in events if usable_exposure(c.value, rulebook) is not None]
    else:
        positive = [c for c in events if c.value > 0]
    implausible = [c for c in events if c.value > 0 and c not in positive]
    binding = max(positive, key=lambda c: c.value) if positive else None
    units = sorted({
        str(e.get("exposed_population_unit") or "") for e in raw_events
        if e.get("exposed_population_unit") is not None
    })
    return {
        "value": float(binding.value) if binding is not None else None,
        "source": SOURCE_GDACS if binding is not None else None,
        "source_ref": binding.source_ref if binding is not None else None,
        "field": CEILING_FIELD,
        "n_events": len(events),
        "n_events_with_exposure": len(positive),
        "n_events_below_plausible_floor": len(implausible),
        "exposure_units_seen": units,
        "all_exposures": sorted((float(c.value) for c in events), reverse=True)[:10],
    }


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
