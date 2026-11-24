# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Which Resolver Update steps write which tables, and what each leaves behind.

The 11 Nov 2026 Resolver Update (run 42999258374) ran both Phase 2.6 steps,
both died on their first line (``ValueError: Unknown level: ''``), both were
``continue-on-error`` and so read green, and the four conflict displacement
tables came out of the run with the row counts they went in with. Nothing
said so. The bundle copied the steps' summary JSONs only when they existed,
so a missing one was silence, and no check compared a writer's tables
before and after.

This module is the declaration that lets both be asked:

* a step that RAN (it left a log) and was meant to leave a summary JSON
  and did not, is a finding (``missing_summaries``);
* a step that ran and left every one of its tables unchanged, without
  saying it had nothing to do, is a finding (``evaluate``).

Pure: no database and no IO beyond reading the step's own files. The
bundle supplies a ``touched`` callable that counts rows written since the
run started, because a step that rewrites a table in place (the DTM bounds
are deleted and re-inserted per country) moves no row count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class TableSpec:
    """One table a step writes, optionally narrowed to the rows it owns."""

    table: str
    #: SQL predicate naming the step's rows in a shared table. A shared
    #: table moves because OTHER steps wrote it, so its whole-table count
    #: says nothing about this step; only the stamped rows inside the
    #: predicate do.
    where: str | None = None
    #: Write-stamp column, when the table's is not one the bundle guesses.
    stamp: str | None = None

    @property
    def shared(self) -> bool:
        return self.where is not None


@dataclass(frozen=True)
class WriterStep:
    name: str
    #: Log file the workflow tees the step to (relative to ``diagnostics/``).
    log: str
    tables: tuple[TableSpec, ...]
    #: Summary JSON the step writes (relative to ``diagnostics/``), if any.
    summary: str | None = None
    #: Reads the summary and returns why the step legitimately wrote
    #: nothing, or None. A step whose summary is missing never had nothing
    #: to do: it could not say.
    nothing_to_do: Callable[[Mapping[str, Any]], str | None] | None = field(
        default=None, compare=False
    )


def _extract_nothing_to_do(summary: Mapping[str, Any]) -> str | None:
    if summary.get("error"):
        return None
    plan = summary.get("plan") if isinstance(summary.get("plan"), Mapping) else {}
    if not summary.get("countries"):
        return "no countries to assess"
    cells = plan.get("cells")
    if cells == 0:
        return "the plan held zero cells"
    if "cells_by_reason" not in summary and plan:
        return "plan only (no extraction requested)"
    if plan.get("calls_allowed_this_run") == 0 and summary.get("calls_made", 0) == 0:
        return "the monthly extraction budget was already spent"
    return None


def _dtm_nothing_to_do(summary: Mapping[str, Any]) -> str | None:
    if summary.get("error"):
        return None
    if summary.get("status") == "no_credentials":
        return "no DTM key configured"
    if not summary.get("countries"):
        return "no countries to assess"
    return None


def _haz_nothing_to_do(summary: Mapping[str, Any]) -> str | None:
    if summary.get("dry_run"):
        return "dry run"
    if not summary.get("months_run"):
        return "zero months to assess"
    return None


def _haz_step(hazard: str, code: str) -> WriterStep:
    where = f"hazard = '{code}'"
    return WriterStep(
        name=f"phase25_haz_{hazard}",
        log=f"phase25_haz_{hazard}.log",
        summary=f"haz_run_{hazard}.json",
        tables=(
            TableSpec("haz_triggers", where=where),
            TableSpec("haz_resolutions", where=where),
        ),
        nothing_to_do=_haz_nothing_to_do,
    )


#: Every Resolver Update step that exists to write tables. Add a step here
#: when it starts writing; a step absent from this map is one nothing checks.
WRITER_STEPS: tuple[WriterStep, ...] = (
    WriterStep(
        name="phase26_conflict_extract",
        log="phase26_conflict_extract.log",
        summary="conflict_displacement_extract.json",
        tables=(
            TableSpec("conflict_displacement_cell_ledger"),
            TableSpec("conflict_displacement_candidates"),
            TableSpec("conflict_doc_extractions"),
        ),
        nothing_to_do=_extract_nothing_to_do,
    ),
    WriterStep(
        name="phase26_dtm_conflict",
        log="phase26_dtm_conflict.log",
        summary="dtm_conflict.json",
        tables=(
            TableSpec("dtm_conflict_rounds", stamp="retrieved_at"),
            TableSpec("dtm_conflict_lower_bounds", stamp="computed_at"),
        ),
        nothing_to_do=_dtm_nothing_to_do,
    ),
    WriterStep(
        name="phase1_idmc_conflict",
        log="phase1_idmc_conflict.log",
        tables=(
            TableSpec(
                "facts_resolved",
                where="UPPER(publisher) = 'IDMC' AND UPPER(hazard_code) = 'ACE'",
            ),
        ),
    ),
    _haz_step("flood", "FL"),
    _haz_step("cyclone", "TC"),
    _haz_step("drought", "DR"),
    # The monthly NMME ingest (Phase 2.4, logged as phase3_nmme.log) stamps
    # fetched_at on the newest vintage every run. The vintage BACKFILL is
    # left out: it skips every vintage already held, so writing nothing is
    # its ordinary outcome.
    WriterStep(
        name="phase24_nmme",
        log="phase3_nmme.log",
        tables=(TableSpec("seasonal_forecasts", stamp="fetched_at"),),
    ),
    WriterStep(
        name="phase4_hdx_signals",
        log="phase4_hdx_signals.log",
        tables=(TableSpec("hdx_signals", stamp="fetched_at"),),
    ),
    WriterStep(
        name="phase24_population",
        log="phase24_population.log",
        tables=(TableSpec("haz_raw_population", stamp="retrieved_at"),),
    ),
)

#: Steps that must leave a summary JSON when they run. Each script writes
#: its summary on the way out, error or not, so a missing one means the
#: process died before it could: the Phase 2.6 crash, a SIGKILL at the step
#: timeout, an import error.
SUMMARY_STEPS: tuple[WriterStep, ...] = tuple(step for step in WRITER_STEPS if step.summary)


def read_summary(diagnostics_dir: Path, step: WriterStep) -> tuple[bool, Mapping[str, Any] | None]:
    """(exists, parsed). An unreadable summary exists and parses to None."""

    if not step.summary:
        return False, None
    path = Path(diagnostics_dir) / step.summary
    if not path.is_file():
        return False, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return True, None
    return True, payload if isinstance(payload, Mapping) else None


def step_ran(diagnostics_dir: Path, step: WriterStep) -> bool:
    """A step ran when it left its log. A scoped run leaves none."""

    return (Path(diagnostics_dir) / step.log).is_file()


def missing_summaries(
    diagnostics_dir: Path,
    last_error: Callable[[Path], str],
    steps: Sequence[WriterStep] = SUMMARY_STEPS,
) -> list[dict[str, str]]:
    """Steps that ran and left no summary, with the log's last ERROR line."""

    out: list[dict[str, str]] = []
    for step in steps:
        if not step.summary or not step_ran(diagnostics_dir, step):
            continue
        exists, _ = read_summary(diagnostics_dir, step)
        if exists:
            continue
        out.append({
            "step": step.name,
            "summary": step.summary,
            "log": step.log,
            "last_error": last_error(Path(diagnostics_dir) / step.log) or "",
        })
    return out


def _counts(signature: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in ("required_counts", "optional_counts", "all_counts"):
        part = (signature or {}).get(key)
        if isinstance(part, Mapping):
            merged.update(part)
    return merged


def evaluate(
    diagnostics_dir: Path,
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
    touched: Callable[[TableSpec], int | None],
    steps: Sequence[WriterStep] = WRITER_STEPS,
) -> list[dict[str, Any]]:
    """One row per writer step that ran: did any of its tables change?

    A table changed when its row count moved (whole tables only; a shared
    table's count moves for other writers too) or when ``touched`` counts
    rows stamped since the run started. ``verdict`` is ``changed``,
    ``nothing_to_do`` (unchanged, and the step's summary says why),
    ``unchanged`` (the finding) or ``unknown`` (no signature and no stamp
    to ask).
    """

    counts_before = _counts(before)
    counts_after = _counts(after)
    rows: list[dict[str, Any]] = []
    for step in steps:
        if not step_ran(diagnostics_dir, step):
            continue
        tables: list[dict[str, Any]] = []
        any_changed = False
        any_known = False
        for spec in step.tables:
            was = counts_before.get(spec.table)
            now = counts_after.get(spec.table)
            delta = (now - was) if isinstance(was, int) and isinstance(now, int) else None
            stamped = touched(spec)
            changed = False
            if not spec.shared and delta not in (None, 0):
                changed = True
            if stamped:
                changed = True
            if (delta is not None and not spec.shared) or stamped is not None:
                any_known = True
            any_changed = any_changed or changed
            tables.append({
                "table": spec.table, "where": spec.where, "before": was, "after": now,
                "delta": delta, "touched_since_start": stamped, "changed": changed,
            })
        reason = None
        if not any_changed and step.summary and step.nothing_to_do is not None:
            exists, summary = read_summary(diagnostics_dir, step)
            if exists and summary is not None:
                reason = step.nothing_to_do(summary)
        if any_changed:
            verdict = "changed"
        elif reason:
            verdict = "nothing_to_do"
        elif any_known:
            verdict = "unchanged"
        else:
            verdict = "unknown"
        rows.append({"step": step.name, "verdict": verdict, "reason": reason or "",
                     "tables": tables})
    return rows


__all__ = [
    "TableSpec", "WriterStep", "WRITER_STEPS", "SUMMARY_STEPS",
    "read_summary", "step_ran", "missing_summaries", "evaluate",
]
