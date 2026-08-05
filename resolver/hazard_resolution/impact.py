# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Layer 2 orchestration: run the impact ladder over a month's triggers.

This is the wiring, not the logic. It fetches the ladder's sources once
per month, then for every TRIGGERED country-month builds candidates
(:mod:`candidates`), asks the deterministic reconciler for a verdict
(:mod:`reconcile`), and writes it (:mod:`resolutions`). No decision is
made here; every rule the answer depends on lives in the modules above,
which is what keeps the ladder auditable.

Both hazards share this path — a triggered cyclone month and a triggered
flood month walk the same ladder, differing only in which detector
triggered them.

Source failures are carried, not swallowed. If EM-DAT could not be
reached, the run's ``NO_DATA`` rows record that the top rung was
UNAVAILABLE rather than empty. Those are very different statements, and
only one of them means "nobody reported a figure".
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from resolver.hazard_resolution import candidates as cand_mod
from resolver.hazard_resolution import emdat as emdat_mod
from resolver.hazard_resolution import extract as extract_mod
from resolver.hazard_resolution import figures as figures_mod
from resolver.hazard_resolution import gdacs as gdacs_mod
from resolver.hazard_resolution import idmc_idu as idu_mod
from resolver.hazard_resolution import ifrc_go as go_mod
from resolver.hazard_resolution import reconcile as reconcile_mod
from resolver.hazard_resolution import reliefweb_docs as docs_mod
from resolver.hazard_resolution import resolutions as res_mod
from resolver.hazard_resolution.rulebook import Rulebook
from resolver.hazard_resolution.schema import RUN_TYPE_LIVE, ensure_haz_schema

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)


@dataclass
class LadderRun:
    """What one month's ladder pass did."""

    hazard: str
    ym: str
    cells: int = 0
    resolved_value: int = 0
    no_data: int = 0
    pending: int = 0
    flagged: int = 0
    lower_bound: int = 0
    provisional: int = 0
    frozen_skipped: int = 0
    fetches: dict[str, Any] = field(default_factory=dict)
    # Countries whose ladder walk raised — their cells are unresolved this
    # run, every other country's answer stands. "iso3: error" strings.
    failed_cells: list[str] = field(default_factory=list)
    # Rung-2 extraction, aggregated over the month's cells.
    cells_extracted: int = 0
    cells_extraction_skipped: int = 0
    docs_read: int = 0
    extraction_calls: int = 0
    extraction_cost_usd: float = 0.0
    extraction_budget_capped: bool = False

    @property
    def unavailable_sources(self) -> list[str]:
        return sorted(
            name for name, meta in self.fetches.items() if not meta.get("ok", False)
        )


def _load_country_names() -> dict[str, str]:
    """iso3 -> human-readable name from the resolver country registry.

    The extraction prompt reads humanitarian prose that says "the
    Philippines", not "PHL" — so the prompt should too. stdlib csv, not
    pandas: this runs inside the ladder walk and must stay import-light.
    """

    import csv
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "countries.csv"
    names: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                iso3 = str(row.get("iso3") or "").strip().upper()
                name = str(row.get("country_name") or "").strip()
                if iso3 and name:
                    names[iso3] = name
    except Exception as exc:  # pragma: no cover - registry always ships
        LOG.warning("[impact] could not load country names (%s) — prompts fall back to ISO3", exc)
    return names


def national_population(
    con: "duckdb.DuckDBPyConnection", iso3: str, year: int
) -> float | None:
    """The country's population denominator for the sanity cap.

    Takes the most recent year at or before the target; falls back to the
    earliest available when the cell predates the table (a denominator
    from the wrong decade still bounds a figure usefully, and the
    alternative is no cap at all).
    """

    row = con.execute(
        """
        SELECT population FROM haz_raw_population
        WHERE iso3 = ? AND year <= ?
        ORDER BY year DESC LIMIT 1
        """,
        [iso3.upper(), int(year)],
    ).fetchone()
    if row is None:
        row = con.execute(
            """
            SELECT population FROM haz_raw_population
            WHERE iso3 = ? ORDER BY year ASC LIMIT 1
            """,
            [iso3.upper()],
        ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def fetch_ladder_sources(
    con: "duckdb.DuckDBPyConnection",
    ym: str,
    hazard: str,
    rulebook: Rulebook,
    *,
    skip_fetch: bool = False,
) -> dict[str, Any]:
    """Refresh every ladder source for one month. Never raises.

    Returns one entry per source describing whether it answered, so the
    caller can distinguish an empty rung from an unreachable one.
    """

    if skip_fetch:
        LOG.info("[impact] --skip-fetch: using the existing raw caches")
        return {
            name: {"ok": True, "skipped": True}
            for name in ("emdat", "ifrc_go", "idmc_idu", "gdacs")
        }

    outcomes = {
        "emdat": emdat_mod.fetch_emdat(con, ym, hazard, rulebook),
        "ifrc_go": go_mod.fetch_ifrc_go(con, ym, hazard, rulebook),
        "idmc_idu": idu_mod.fetch_idmc_idu(con, ym, hazard, rulebook),
        # GDACS is fetched here too because it supplies the sanity ceiling
        # for BOTH hazards — for floods it has already run as the detector,
        # and the raw cache makes the second call a no-op upsert.
        "gdacs": gdacs_mod.fetch_gdacs_events(con, ym, hazard, rulebook),
    }
    for name, outcome in outcomes.items():
        if not outcome.ok:
            LOG.warning(
                "[impact] ladder source %s unavailable for %s %s: %s",
                name, hazard, ym, outcome.error,
            )
    return {name: outcome.as_provenance() for name, outcome in outcomes.items()}


#: Which reader answers "does this rung have a figure for the cell?", per
#: ladder rung. Extraction consults it to avoid paying to read documents
#: for a cell a higher rung has already answered.
_RUNG_READERS = {
    "emdat": emdat_mod.records_for_country_month,
    "ifrc_go": go_mod.records_for_country_month,
    "idmc_idu": idu_mod.records_for_country_month,
}


def higher_rungs_with_figures(
    con: "duckdb.DuckDBPyConnection", iso3: str, ym: str, hazard: str, rulebook: Rulebook
) -> list[str]:
    """Ladder rungs ABOVE the extracted rung that already have a figure.

    Derived from ``rulebook.ladder`` rather than hard-coded, so reordering
    the ladder in YAML reorders this too.
    """

    ladder = [str(rung) for rung in rulebook.get("ladder")]
    if extract_mod.SOURCE not in ladder:
        return []
    above = ladder[: ladder.index(extract_mod.SOURCE)]
    return [
        rung
        for rung in above
        if rung in _RUNG_READERS and _RUNG_READERS[rung](con, iso3.upper(), ym, hazard)
    ]


#: Wall-clock of the last ReliefWeb document request, so consecutive cells
#: are spaced by ``reliefweb.documents.request_delay_sec`` without the
#: caller having to thread a "was this the first cell?" flag through.
_LAST_DOC_FETCH: list[float] = []


def _pace_reliefweb(rulebook: Rulebook) -> None:
    """Sleep just long enough to honour the configured request delay."""

    import time

    delay = float(rulebook.get("reliefweb.documents.request_delay_sec"))
    if delay <= 0:
        return
    if _LAST_DOC_FETCH:
        elapsed = time.monotonic() - _LAST_DOC_FETCH[-1]
        if elapsed < delay:
            time.sleep(delay - elapsed)
    _LAST_DOC_FETCH.clear()
    _LAST_DOC_FETCH.append(time.monotonic())


def extract_rung_for_cell(
    con: "duckdb.DuckDBPyConnection",
    *,
    iso3: str,
    ym: str,
    hazard: str,
    rulebook: Rulebook,
    budget: "extract_mod.ExtractionBudget",
    fetch_documents: bool = True,
    call: "extract_mod.CallFn | None" = None,
    post: Any = None,
    country_name: str | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Produce ladder rung 2 for ONE triggered cell. Returns (candidates, provenance).

    Fetch documents, transcribe their stated figures, then hand them to the
    deterministic post-processing. Reached only for triggered cells, which
    is what keeps the machine's one paid step off the cells that resolved
    without it.
    """

    iso3 = iso3.upper()
    cell = f"{iso3}/{hazard}/{ym}"

    if not bool(rulebook.get("extraction.enabled")):
        return [], {"skipped_reason": "extraction_disabled"}

    if bool(rulebook.get("extraction.skip_when_higher_rung_populated")):
        higher = higher_rungs_with_figures(con, iso3, ym, hazard, rulebook)
        if higher:
            LOG.info(
                "[impact] %s: skipping extraction — higher rung(s) %s already have "
                "a figure and cannot be outranked by rung 2",
                cell, ",".join(higher),
            )
            return [], {
                "skipped_reason": "higher_rung_populated",
                "higher_rungs": higher,
            }

    provenance: dict[str, Any] = {}
    if fetch_documents:
        # One ReliefWeb request per triggered cell, paced by the rulebook —
        # the silence sweep already paces itself the same way, and the two
        # steps hit the same free API.
        _pace_reliefweb(rulebook)
        outcome = docs_mod.fetch_documents(con, iso3, ym, hazard, rulebook, post=post)
        provenance["documents"] = outcome.as_provenance()
        if not outcome.ok:
            LOG.warning(
                "[impact] %s: ReliefWeb document fetch failed (%s) — rung 2 is "
                "UNREAD for this cell, not empty",
                cell, outcome.error,
            )

    extraction = extract_mod.extract_for_cell(
        con,
        iso3=iso3,
        ym=ym,
        hazard=hazard,
        rulebook=rulebook,
        budget=budget,
        call=call,
        country_name=country_name,
    )
    ceiling = cand_mod.exposure_ceiling(con, iso3, ym, hazard)
    candidates, rejected = figures_mod.build_candidates(
        extraction,
        iso3=iso3,
        ym=ym,
        hazard=hazard,
        rulebook=rulebook,
        exposure_ceiling=ceiling,
    )
    provenance["extraction"] = {**extraction.as_provenance(), "rejected": rejected}
    # Lift the extractor's own skip reason (no documents, unresolvable model)
    # to the top level, so "was this cell read?" is one question with one
    # answer rather than two places a caller has to remember to check.
    if extraction.skipped_reason and not candidates:
        provenance["skipped_reason"] = extraction.skipped_reason
    return candidates, provenance


def resolve_triggered_cells(
    con: "duckdb.DuckDBPyConnection",
    *,
    ym: str,
    hazard: str,
    iso3s: list[str],
    rulebook: Rulebook,
    fetches: dict[str, Any] | None = None,
    dry_run: bool = False,
    today: dt.date | None = None,
    extract: bool = True,
    fetch_documents: bool = True,
    call: "extract_mod.CallFn | None" = None,
    post: Any = None,
    run_type: str = RUN_TYPE_LIVE,
) -> LadderRun:
    """Walk the ladder for every triggered cell in ``iso3s``."""

    ensure_haz_schema(con)
    run = LadderRun(hazard=hazard, ym=ym, fetches=fetches or {})
    year = int(ym.split("-")[0])
    unavailable = run.unavailable_sources
    budget = extract_mod.load_budget(con, rulebook, today=today) if extract else None
    country_names = _load_country_names()

    # One country's exception must not cost the rest of the month their
    # answers: the walk is per-cell, each cell's writes are transactional,
    # and a failed cell is recorded on the run (and in the exit code) so a
    # re-run can retry exactly what is missing.
    for iso3 in sorted(iso3s):
        try:
            extracted: list[Any] = []
            extraction_provenance: dict[str, Any] = {"ran": False}
            if extract and not dry_run:
                extracted, extraction_provenance = extract_rung_for_cell(
                    con,
                    iso3=iso3,
                    ym=ym,
                    hazard=hazard,
                    rulebook=rulebook,
                    budget=budget,
                    fetch_documents=fetch_documents,
                    call=call,
                    post=post,
                    country_name=country_names.get(iso3.upper()),
                )
                extraction_provenance["ran"] = "skipped_reason" not in extraction_provenance
                stats = extraction_provenance.get("extraction") or {}
                run.cells_extracted += int(
                    bool(stats.get("docs_read") or stats.get("docs_cached"))
                )
                run.cells_extraction_skipped += int("skipped_reason" in extraction_provenance)
                run.docs_read += int(stats.get("docs_read") or 0)
                run.extraction_calls += int(stats.get("calls_made") or 0)
                run.extraction_cost_usd += float(stats.get("cost_usd") or 0.0)
                run.extraction_budget_capped |= bool(stats.get("budget_capped"))

            found = cand_mod.build_candidates(
                con, iso3, ym, hazard, rulebook, extracted=extracted
            )
            if not dry_run:
                cand_mod.write_candidates(con, found, iso3, ym, hazard)

            verdict = reconcile_mod.reconcile(
                iso3=iso3,
                ym=ym,
                hazard=hazard,
                candidates=found,
                rulebook=rulebook,
                national_population=national_population(con, iso3, year),
                today=today,
            )
            # A rung that could not be READ is not a rung that was empty. Record
            # the difference on the row itself so a NO_DATA can be re-litigated.
            if unavailable:
                verdict.provenance.setdefault("decision", {})[
                    "sources_unavailable"
                ] = unavailable
            verdict.provenance["source_fetches"] = run.fetches
            # Rung 2's own story travels with the row: how many documents were
            # read, what the extractor discarded and why, and whether the cost
            # guard cut the reading short. A budget-capped cell has UNREAD
            # documents, which must never be mistaken for ReliefWeb silence.
            verdict.provenance["reliefweb_extraction"] = extraction_provenance

            run.cells += 1
            if verdict.status == reconcile_mod.STATUS_RESOLVED_VALUE:
                run.resolved_value += 1
                run.provisional += int(verdict.provisional)
            elif verdict.status == reconcile_mod.STATUS_PENDING:
                run.pending += 1
            else:
                run.no_data += 1
            run.flagged += int(verdict.flagged)
            run.lower_bound += int(verdict.lower_bound)

            if dry_run:
                continue
            outcome = res_mod.write_reconciliation(
                con, verdict, rulebook, today=today, run_type=run_type
            )
            if outcome == res_mod.WRITE_FROZEN_SKIP:
                run.frozen_skipped += 1
        except Exception as exc:  # noqa: BLE001 - one bad cell must not end the month
            run.failed_cells.append(f"{iso3}: {type(exc).__name__}: {exc}")
            LOG.exception("[impact] %s %s %s: ladder walk raised", iso3, hazard, ym)

    LOG.info(
        "[impact] %s %s ladder: %d cells -> %d values (%d lower-bound), "
        "%d no-data, %d pending, %d flagged, %d provisional, %d frozen-skipped",
        hazard, ym, run.cells, run.resolved_value, run.lower_bound,
        run.no_data, run.pending, run.flagged, run.provisional, run.frozen_skipped,
    )
    if unavailable:
        LOG.warning(
            "[impact] %s %s: ladder ran with unavailable sources %s — NO_DATA "
            "rows from this run record that those rungs were unreadable, not empty",
            hazard, ym, ",".join(unavailable),
        )
    if run.failed_cells:
        LOG.error(
            "[impact] %s %s: %d cell(s) raised and were NOT resolved this run "
            "(re-run to retry them): %s",
            hazard, ym, len(run.failed_cells), "; ".join(run.failed_cells[:10]),
        )
    if budget is not None:
        LOG.info(
            "[impact] %s %s extraction: %d cells read, %d skipped, %d documents, "
            "%d LLM calls, $%.4f (%d of %d monthly calls now used)",
            hazard, ym, run.cells_extracted, run.cells_extraction_skipped,
            run.docs_read, run.extraction_calls, run.extraction_cost_usd,
            budget.used_this_month + budget.calls_this_run, budget.max_calls_per_month,
        )
        if budget.exhausted:
            LOG.warning(
                "[impact] extraction stopped at the monthly cap of %d calls; "
                "cells with unread documents: %s",
                budget.max_calls_per_month, ",".join(budget.capped_cells) or "(none)",
            )
    return run


def triggered_iso3s(
    con: "duckdb.DuckDBPyConnection", ym: str, hazard: str
) -> list[str]:
    """Countries whose trigger row for this month says a hazard occurred."""

    year, month = (int(p) for p in ym.split("-"))
    return [
        row[0]
        for row in con.execute(
            """
            SELECT iso3 FROM haz_triggers
            WHERE hazard = ? AND year = ? AND month = ? AND triggered
            ORDER BY iso3
            """,
            [hazard, year, month],
        ).fetchall()
    ]
