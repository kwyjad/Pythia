# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""One row per assessed country-month-hazard cell, and per rejected figure.

The machine's tables answer "what did it decide" for every cell that got a
row. They cannot answer "why did this cell get no row at all", and that was
the largest gap in a resolver diagnostic: a run assessing 2,223 cells and
writing 1,008 of them left the other 1,215 to be derived by subtracting
summary JSON fields from each other, which is arithmetic rather than
diagnosis.

So the code paths that KNOW the reason state it here as it happens:

* ``cli.sweep_and_resolve_zeros`` — a non-triggered cell was swept and then
  zeroed, flipped to triggered, left inconclusive, or had its zero
  suppressed by the coverage gate;
* ``impact.resolve_triggered_cells`` — a triggered cell walked the ladder
  and got a value, a NO_DATA, a PENDING, a frozen skip, or an exception;
* ``drought.run_month`` — the same for the hazard with no ladder.

Recording is off unless ``PYTHIA_RUN_LOG_DIR`` is set (see
:mod:`resolver.diagnostics.run_log`), and it never raises: a machine that
fails because its ledger could not be written is a worse machine.
"""

from __future__ import annotations

from typing import Any

from resolver.diagnostics import run_log

# --- Reason codes -----------------------------------------------------------
# Why a cell that was ASSESSED carries no row in haz_resolutions. Each names
# a different repair: a pending cell needs the calendar, an inconclusive one
# needs the source fixed, a coverage-gated one needs the detector's history.

#: The freeze deadline has not passed and no candidate was found. Writing
#: NO_DATA here would record our impatience as a fact about the world.
REASON_PENDING = "pending_before_freeze"
#: The ReliefWeb silence sweep could not be read, so silence is unproven.
REASON_SWEEP_INCONCLUSIVE = "sweep_inconclusive"
#: The detector cannot demonstrate it covered this month; a zero would be an
#: ingestion gap dressed up as a quiet month.
REASON_COVERAGE_GATE = "coverage_gate_suppressed_zero"
#: A required drought indicator could not be read (fail-closed).
REASON_INCONCLUSIVE = "indicator_inconclusive"
#: The indicators were read, but none of them carried a reading for THIS
#: country — every answer was inferred from absence. An alerting feed that
#: never monitored a country has not said it is quiet, so no zero.
REASON_NO_COVERAGE = "indicator_no_coverage"
#: The indicators were read and answered, but too few feeds answered for a
#: zero to rest on their silence (``drought.indicators.min_answered_for_zero``).
#: One surviving feed is not evidence of quiet.
REASON_TOO_FEW_FEEDS = "indicator_too_few_feeds_for_zero"
#: The month has not ended. Nothing can have observed it in full, so a
#: drought verdict for it is PENDING whatever the feeds say today.
REASON_MONTH_IN_PROGRESS = "month_in_progress"
#: The cell already carries a frozen answer; this run's verdict was audited
#: to haz_revisions and the stored row stands.
REASON_FROZEN = "frozen_row_unchanged"
#: The walk raised. The cell is unresolved THIS run and a re-run retries it.
REASON_EXCEPTION = "cell_raised"
#: The sweep found reports, so the cell became triggered and the ladder owns
#: it — this row is the sweep's half of the story, not a missing answer.
REASON_FLIPPED = "flipped_to_triggered_by_sweep"

#: Every reason a cell may carry no row, so a reader (the debug bundle, a
#: contradiction check) can tell a known silence from an unexplained one.
NO_ROW_REASONS = frozenset({
    REASON_PENDING, REASON_SWEEP_INCONCLUSIVE, REASON_COVERAGE_GATE,
    REASON_INCONCLUSIVE, REASON_NO_COVERAGE, REASON_TOO_FEW_FEEDS,
    REASON_MONTH_IN_PROGRESS, REASON_FROZEN, REASON_EXCEPTION, REASON_FLIPPED,
})

#: Stage names, so a reader can tell which half of the machine spoke.
STAGE_SWEEP = "sweep"
STAGE_LADDER = "ladder"
STAGE_DROUGHT = "drought"


def enabled() -> bool:
    return run_log.enabled()


def record_cell(
    *,
    stage: str,
    iso3: str,
    hazard: str,
    ym: str,
    triggered: bool | None = None,
    trigger_source: str | None = None,
    status: str | None = None,
    value: float | None = None,
    rule_fired: str | None = None,
    flagged: bool | None = None,
    provisional: bool | None = None,
    write_outcome: str | None = None,
    reason_code: str | None = None,
    rungs_readable: list[str] | None = None,
    rungs_unavailable: list[str] | None = None,
    answering_rung: str | None = None,
    extraction: dict[str, Any] | None = None,
    detail: dict[str, Any] | None = None,
    run_type: str | None = None,
) -> None:
    """Append one assessed cell. Never raises."""

    run_log.record(
        run_log.STREAM_CELLS,
        {
            "stage": stage,
            "iso3": str(iso3).upper(),
            "hazard": hazard,
            "ym": ym,
            "triggered": triggered,
            "trigger_source": trigger_source,
            "status": status,
            "value": value,
            "rule_fired": rule_fired,
            "flagged": flagged,
            "provisional": provisional,
            "write_outcome": write_outcome,
            "reason_code": reason_code,
            "rungs_readable": list(rungs_readable or []),
            "rungs_unavailable": list(rungs_unavailable or []),
            "answering_rung": answering_rung,
            "extraction": extraction or {},
            "detail": detail or {},
            "run_type": run_type,
        },
    )


def record_figure(
    *,
    iso3: str,
    hazard: str,
    ym: str,
    outcome: str,
    doc_id: str | None = None,
    doc_url: str | None = None,
    value: float | None = None,
    unit: str | None = None,
    stated_by: str | None = None,
    quote: str | None = None,
    reason: str | None = None,
    ceiling: float | None = None,
    ceiling_multiplier: float | None = None,
    ceiling_source: str | None = None,
    ceiling_source_ref: str | None = None,
    ceiling_field: str | None = None,
    preference_rank: int | None = None,
    detail: dict[str, Any] | None = None,
    stated_value: float | None = None,
    stated_unit: str | None = None,
    value_persons: int | None = None,
    conversion_factor: float | None = None,
    figure_date: str | None = None,
    doc_date: str | None = None,
    doc_date_original: str | None = None,
    doc_primary_country: str | None = None,
) -> None:
    """Append one extracted figure and what became of it. Never raises.

    ``value`` is the figure as the ladder uses it (people). Since Sept 2026
    the row also carries what the SOURCE said — ``stated_value`` in
    ``stated_unit`` — beside ``value_persons`` (a whole number) and the
    ``conversion_factor`` that joins them, because "187 families affected"
    was being written as value 935 under unit "households", which is two
    facts wearing one label. The document's dates and primary country ride
    along so the attribution check can be run over the ledger alone.

    ``ceiling_source``/``ceiling_field`` matter as much as the ceiling: the
    August 2026 run rejected 199 figures against ceilings of 0, 2, 5 and 20
    and nothing in the artifact said which GDACS field produced them — so
    "the ceiling is broken" and "the figure is wrong" could not be told
    apart without reading the source.
    """

    run_log.record(
        run_log.STREAM_FIGURES,
        {
            "iso3": str(iso3).upper(),
            "hazard": hazard,
            "ym": ym,
            "outcome": outcome,
            "doc_id": doc_id,
            "doc_url": doc_url,
            "value": value,
            "unit": unit,
            "stated_by": stated_by,
            "quote": (quote or "")[:400],
            "reason": reason,
            "ceiling": ceiling,
            "ceiling_multiplier": ceiling_multiplier,
            "ceiling_source": ceiling_source,
            "ceiling_source_ref": ceiling_source_ref,
            "ceiling_field": ceiling_field,
            "preference_rank": preference_rank,
            "stated_value": stated_value,
            "stated_unit": stated_unit,
            "value_persons": value_persons,
            "conversion_factor": conversion_factor,
            "figure_date": figure_date,
            "doc_date": doc_date,
            "doc_date_original": doc_date_original,
            "doc_primary_country": doc_primary_country,
            "detail": detail or {},
        },
    )
