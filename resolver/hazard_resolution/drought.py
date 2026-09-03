# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The drought rule: IPC Phase 3+ delta, gated by drought indicators.

Drought is the one hazard that skips the impact ladder. There is no
discrete event for EM-DAT to write up or for a situation report to count
the affected of; what there is, is a food-security analysis cycle. So the
question "how many people were affected by drought in this country-month?"
is answered as:

    max(0, Phase3+(analysis covering the month)
           - Phase3+(previous current-period analysis))

admitted as DROUGHT impact only where the indicators
(:mod:`drought_indicators`) say the country was in drought.

**Hard rule 2 applies here exactly as it does to the ladder**:
:func:`decide_drought` is a pure function of (analyses, indicator verdict,
rulebook, population, today). No I/O, no model, no clock beyond the one
passed in. Everything above it is wiring.

The five outcomes, and why each is what it is:

``RESOLVED_VALUE``
    Indicators show drought, two analyses are in hand, and the increase
    clears ``drought.min_delta_people`` / ``min_delta_pct``.
``RESOLVED_ZERO``
    Either *nothing happened* — no indicator signal AND no IPC
    deterioration, which is the evidence-of-absence zero — or the analyses
    were read and state no qualifying increase. Two different zeros, two
    different ``rule_fired`` strings, because conflating them would make
    "we know it was quiet" indistinguishable from "the number came out at
    zero".
``NO_DATA``
    Indicators show drought but IPC has not covered the period by the
    freeze deadline (the gap the spec asks to be made explicit), or there
    is no previous analysis to difference against, or a deterioration
    occurred that the indicators do not attribute to drought. All flagged.
``PENDING``
    The same situations BEFORE the freeze deadline. IPC publishes months
    after the fact, so declaring "no data" early would record our
    impatience as a fact about the world. Nothing is written.
``INCONCLUSIVE``
    A required indicator could not be read. Not a zero, not a value —
    fail-closed, exactly as a failed ReliefWeb sweep is for the other
    hazards. Nothing is written; the trigger row keeps the evidence.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from resolver.hazard_resolution import candidates as cand_mod
from resolver.hazard_resolution import cell_ledger
from resolver.hazard_resolution import detect as detect_mod
from resolver.hazard_resolution import drought_indicators as ind_mod
from resolver.hazard_resolution import ipc as ipc_mod
from resolver.hazard_resolution import reconcile as reconcile_mod
from resolver.hazard_resolution import resolutions as res_mod
from resolver.hazard_resolution.candidates import VALUE_AFFECTED, Candidate
from resolver.hazard_resolution.ipc import IpcAnalysis
from resolver.hazard_resolution.rulebook import Rulebook
from resolver.hazard_resolution.rules import (
    drought_delta_qualifies,
    is_provisional,
    within_population_cap,
)
from resolver.hazard_resolution.schema import RUN_TYPE_LIVE, ensure_haz_schema

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

HAZARD = "DR"
SOURCE_IPC = "ipc"

STATUS_RESOLVED_VALUE = reconcile_mod.STATUS_RESOLVED_VALUE
STATUS_RESOLVED_ZERO = res_mod.STATUS_RESOLVED_ZERO
STATUS_NO_DATA = reconcile_mod.STATUS_NO_DATA
STATUS_PENDING = reconcile_mod.STATUS_PENDING

#: Not a stored status either: a required indicator was unreadable, so the
#: machine has no basis for any answer and writes no row.
STATUS_INCONCLUSIVE = "INCONCLUSIVE"

TRIGGER_SOURCE_INDICATORS = "drought_indicators"
TRIGGER_SOURCE_IPC = "ipc"
TRIGGER_SOURCE_NONE = detect_mod.TRIGGER_SOURCE_NONE

# rule_fired vocabulary — stable strings, queried by downstream analysis.
RULE_IPC_DELTA = "drought:ipc_phase3plus_delta"
RULE_ZERO_ABSENCE = "drought_zero:no_indicator_signal+no_ipc_deterioration"
RULE_ZERO_NO_INCREASE = "drought_zero:ipc_phase3plus_no_qualifying_increase"
RULE_ZERO_MID_WINDOW = "drought_zero:no_new_deterioration_in_analysis_window"
RULE_NO_ANALYSIS = "no_data:indicators_drought_no_ipc_analysis"
RULE_NO_BASELINE = "no_data:no_previous_ipc_analysis"
RULE_NOT_ATTRIBUTED = "no_data:ipc_deterioration_not_drought_attributed"
RULE_PENDING = "pending:awaiting_ipc_analysis_before_freeze"
RULE_INCONCLUSIVE = "inconclusive:drought_indicator_unavailable"

FLAG_NO_ANALYSIS = "no_ipc_analysis_past_freeze"
FLAG_NO_BASELINE = "no_baseline_analysis_past_freeze"
FLAG_NOT_ATTRIBUTED = "ipc_deterioration_without_drought_indicator"
FLAG_POPULATION_EXCEEDED = reconcile_mod.FLAG_POPULATION_EXCEEDED


@dataclass
class DroughtDecision:
    """The drought rule's verdict for one country-month."""

    iso3: str
    ym: str
    triggered: bool
    trigger_source: str
    status: str
    value: float | None
    rule_fired: str
    provisional: bool
    flags: list[str] = field(default_factory=list)
    candidate: Candidate | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    trigger_detail: dict[str, Any] = field(default_factory=dict)
    evidence_of_absence: dict[str, Any] | None = None

    @property
    def flagged(self) -> bool:
        return bool(self.flags)

    @property
    def writes_row(self) -> bool:
        return self.status in (
            STATUS_RESOLVED_VALUE, STATUS_RESOLVED_ZERO, STATUS_NO_DATA
        )

    def as_reconciliation(self) -> reconcile_mod.Reconciliation:
        """Adapt to the shape ``resolutions.write_reconciliation`` persists.

        Drought reuses the ladder's writer — and therefore the ladder's
        freeze guard, revision logging and provenance column — without
        reusing the ladder itself. The freeze rule must not have two
        implementations.
        """

        status = self.status
        if status in (STATUS_PENDING, STATUS_INCONCLUSIVE):
            status = res_mod.WRITE_PENDING_STATUS
        return reconcile_mod.Reconciliation(
            iso3=self.iso3,
            ym=self.ym,
            hazard=HAZARD,
            status=status,
            value=self.value,
            rule_fired=self.rule_fired,
            flagged=self.flagged,
            provisional=self.provisional,
            flags=list(self.flags),
            winner=self.candidate,
            lower_bound=False,
            provenance=dict(self.provenance),
        )


def _analysis_provenance(
    current: IpcAnalysis | None, previous: IpcAnalysis | None
) -> dict[str, Any]:
    """Which analysis window(s) an answer rests on (a spec requirement)."""

    return {
        "current": current.provenance() if current else None,
        "previous": previous.provenance() if previous else None,
    }


def _base_provenance(
    *,
    rule_fired: str,
    rulebook: Rulebook,
    indicators: ind_mod.IndicatorVerdict,
    current: IpcAnalysis | None,
    previous: IpcAnalysis | None,
    delta: float | None,
    national_population: float | None,
) -> dict[str, Any]:
    return {
        "source": SOURCE_IPC,
        "source_record_ids": [a.analysis_id for a in (current, previous) if a],
        "source_urls": [a.source_url for a in (current, previous) if a and a.source_url],
        "retrieved_at": current.retrieved_at if current else None,
        "rule_fired": rule_fired,
        "decision": {
            "rule": rulebook.get("drought.rule"),
            "month_attribution": rulebook.get("drought.month_attribution"),
            "min_delta_people": rulebook.get("drought.min_delta_people"),
            "min_delta_pct": rulebook.get("drought.min_delta_pct"),
            "ipc_phase_threshold": rulebook.get("drought.ipc_phase_threshold"),
            "delta": delta,
            "ipc_analysis": _analysis_provenance(current, previous),
            "indicators": indicators.as_evidence(),
            "national_population": national_population,
            "population_cap_enabled": bool(rulebook.get("sanity.population_cap")),
        },
    }


def _candidate(
    *,
    iso3: str,
    ym: str,
    delta: float,
    current: IpcAnalysis,
    previous: IpcAnalysis,
) -> Candidate:
    """The drought rule's one candidate figure, for ``haz_impact_candidates``.

    Written even though drought never walks the ladder, so that a drought
    answer is auditable through the same table as every other hazard's.
    """

    return Candidate(
        iso3=iso3,
        ym=ym,
        hazard=HAZARD,
        value=float(delta),
        value_type=VALUE_AFFECTED,
        source=SOURCE_IPC,
        source_ref=current.analysis_id,
        stated_by=(
            "IPC/CH Phase 3+ increase vs the previous current-period analysis"
        ),
        doc_url=current.source_url,
        span_start=current.window_start,
        span_end=current.window_end,
        retrieved_at=current.retrieved_at,
        detail={
            "current": current.provenance(),
            "previous": previous.provenance(),
            "delta": float(delta),
            "derivation": "max(0, current_phase3plus - previous_phase3plus)",
        },
    )


def decide_drought(
    *,
    iso3: str,
    ym: str,
    analyses: list[IpcAnalysis],
    indicators: ind_mod.IndicatorVerdict,
    rulebook: Rulebook,
    national_population: float | None = None,
    today: dt.date | None = None,
) -> DroughtDecision:
    """Resolve one drought country-month. Pure function.

    Reads: the country's cached IPC analyses, the combined indicator
    verdict, and the rulebook. Writes: nothing. Given the same inputs it
    always returns the same answer, and the answer always says which rule
    produced it.
    """

    iso3 = iso3.upper()
    year, month = (int(p) for p in ym.split("-"))
    provisional = is_provisional(year, month, rulebook, today=today)

    current = ipc_mod.covering_analysis(analyses, ym)
    previous = ipc_mod.previous_analysis(analyses, current) if current else None
    delta: float | None = None
    if current is not None and previous is not None:
        delta = max(0.0, float(current.value) - float(previous.value))
    qualifies = delta is not None and drought_delta_qualifies(
        delta, previous.value if previous else None, rulebook
    )

    trigger_detail = {
        "params": {
            "rule": rulebook.get("drought.rule"),
            "combine": indicators.combine,
            "max_observation_age_months": rulebook.get(
                "drought.indicators.max_observation_age_months"
            ),
        },
        "indicators": indicators.as_evidence(),
        "ipc": {
            "analyses_known": len(analyses),
            "covering": current.provenance() if current else None,
            "previous": previous.provenance() if previous else None,
            "delta": delta,
            "delta_qualifies": qualifies,
        },
    }

    def _decide(
        *,
        triggered: bool,
        trigger_source: str,
        status: str,
        value: float | None,
        rule_fired: str,
        flags: list[str] | None = None,
        candidate: Candidate | None = None,
        note: str | None = None,
        evidence_of_absence: dict[str, Any] | None = None,
    ) -> DroughtDecision:
        provenance = _base_provenance(
            rule_fired=rule_fired,
            rulebook=rulebook,
            indicators=indicators,
            current=current,
            previous=previous,
            delta=delta,
            national_population=national_population,
        )
        if note:
            provenance["note"] = note
        if flags:
            provenance["decision"]["flags"] = list(flags)
        return DroughtDecision(
            iso3=iso3,
            ym=ym,
            triggered=triggered,
            trigger_source=trigger_source,
            status=status,
            value=value,
            rule_fired=rule_fired,
            provisional=provisional,
            flags=list(flags or []),
            candidate=candidate,
            provenance=provenance,
            trigger_detail=trigger_detail,
            evidence_of_absence=evidence_of_absence,
        )

    # --- A required indicator could not be read: no verdict is honest. ---
    if not indicators.available:
        return _decide(
            triggered=False,
            trigger_source=TRIGGER_SOURCE_NONE,
            status=STATUS_INCONCLUSIVE,
            value=None,
            rule_fired=RULE_INCONCLUSIVE,
            note=(
                "drought indicators could not be read, so the machine can "
                f"neither attribute a deterioration nor write a zero: {indicators.note}"
            ),
        )

    # --- Nothing happened: no drought signal AND no IPC deterioration. ---
    if not indicators.shows_drought and not qualifies:
        return _decide(
            triggered=False,
            trigger_source=TRIGGER_SOURCE_NONE,
            status=STATUS_RESOLVED_ZERO,
            value=0.0,
            rule_fired=RULE_ZERO_ABSENCE,
            note=(
                "no drought indicator signal and no qualifying IPC Phase 3+ "
                "increase; resolved zero on evidence of absence"
            ),
            evidence_of_absence={
                "indicators": indicators.as_evidence(),
                "ipc": {
                    "analyses_known": len(analyses),
                    "covering": current.provenance() if current else None,
                    "previous": previous.provenance() if previous else None,
                    "delta": delta,
                },
                "retrieved_at": (current.retrieved_at if current else None),
            },
        )

    # --- A deterioration the indicators do not attribute to drought. ---
    if not indicators.shows_drought and qualifies:
        note = (
            "IPC Phase 3+ rose but no drought indicator supports attributing the "
            "increase to drought; the figure is NOT published as drought impact"
        )
        if provisional:
            return _decide(
                triggered=True,
                trigger_source=TRIGGER_SOURCE_IPC,
                status=STATUS_PENDING,
                value=None,
                rule_fired=RULE_PENDING,
                note=note + " — indicators may still be published for this month",
            )
        return _decide(
            triggered=True,
            trigger_source=TRIGGER_SOURCE_IPC,
            status=STATUS_NO_DATA,
            value=None,
            rule_fired=RULE_NOT_ATTRIBUTED,
            flags=[FLAG_NOT_ATTRIBUTED],
            note=note + "; flagged for human review",
        )

    # --- Indicators show drought. What does IPC say? ---
    if current is None:
        # The spec's explicit NO_DATA path.
        if provisional:
            return _decide(
                triggered=True,
                trigger_source=TRIGGER_SOURCE_INDICATORS,
                status=STATUS_PENDING,
                value=None,
                rule_fired=RULE_PENDING,
                note=(
                    "drought conditions indicated and no IPC analysis covers the "
                    "month yet; IPC publishes months late, so the machine waits "
                    "rather than recording NO_DATA"
                ),
            )
        return _decide(
            triggered=True,
            trigger_source=TRIGGER_SOURCE_INDICATORS,
            status=STATUS_NO_DATA,
            value=None,
            rule_fired=RULE_NO_ANALYSIS,
            flags=[FLAG_NO_ANALYSIS],
            note=(
                "drought conditions indicated but no IPC analysis covers the "
                "month by the freeze deadline; flagged for human review"
            ),
        )

    if previous is None:
        if provisional:
            return _decide(
                triggered=True,
                trigger_source=TRIGGER_SOURCE_INDICATORS,
                status=STATUS_PENDING,
                value=None,
                rule_fired=RULE_PENDING,
                note=(
                    "an IPC analysis covers the month but no earlier analysis is "
                    "cached to difference it against; widening "
                    "drought.ipc.lookback_months may recover one"
                ),
            )
        return _decide(
            triggered=True,
            trigger_source=TRIGGER_SOURCE_INDICATORS,
            status=STATUS_NO_DATA,
            value=None,
            rule_fired=RULE_NO_BASELINE,
            flags=[FLAG_NO_BASELINE],
            note=(
                "the rule is a difference between consecutive analyses and no "
                "previous analysis exists; the covering analysis's Phase 3+ "
                "population is a STOCK and must not be published as people "
                "affected in this month"
            ),
        )

    # --- Both analyses in hand: the delta is the answer. ---
    candidate = _candidate(
        iso3=iso3, ym=ym, delta=float(delta or 0.0), current=current, previous=previous
    )

    if not qualifies:
        return _decide(
            triggered=True,
            trigger_source=TRIGGER_SOURCE_INDICATORS,
            status=STATUS_RESOLVED_ZERO,
            value=0.0,
            rule_fired=RULE_ZERO_NO_INCREASE,
            candidate=candidate,
            note=(
                "both analyses were read and state no qualifying Phase 3+ "
                "increase; this is a measured zero, not an absence of data"
            ),
        )

    # ``start_month``: only the window's first month carries the delta.
    attribution = str(rulebook.get("drought.month_attribution"))
    if attribution == "start_month" and ym != current.window_start[:7]:
        return _decide(
            triggered=True,
            trigger_source=TRIGGER_SOURCE_INDICATORS,
            status=STATUS_RESOLVED_ZERO,
            value=0.0,
            rule_fired=RULE_ZERO_MID_WINDOW,
            candidate=candidate,
            note=(
                "the analysis covering this month reported no NEW deterioration: "
                "its increase is attributed wholly to the window's first month "
                f"({current.window_start[:7]})"
            ),
        )

    flags: list[str] = []
    if not within_population_cap(float(delta or 0.0), national_population, rulebook):
        flags.append(FLAG_POPULATION_EXCEEDED)
        LOG.warning(
            "[drought] %s/%s: Phase 3+ increase %.0f exceeds national population "
            "%.0f — keeping the rule's answer, flagging",
            iso3, ym, delta or 0.0, national_population or 0.0,
        )

    note = (
        "people affected = the increase in IPC Phase 3+ population between "
        "consecutive analyses, attributed to drought because the indicators "
        "show drought conditions"
    )
    if attribution == "analysis_window":
        note += (
            ". The figure repeats in every month the analysis window covers "
            "(drought.month_attribution: analysis_window) and must NOT be "
            "summed across months"
        )
    return _decide(
        triggered=True,
        trigger_source=TRIGGER_SOURCE_INDICATORS,
        status=STATUS_RESOLVED_VALUE,
        value=float(delta or 0.0),
        rule_fired=RULE_IPC_DELTA,
        flags=flags,
        candidate=candidate,
        note=note,
    )


@dataclass
class DroughtRun:
    """What one month's drought pass did."""

    ym: str
    cells: int = 0
    triggered: int = 0
    resolved_value: int = 0
    resolved_zero: int = 0
    no_data: int = 0
    pending: int = 0
    inconclusive: int = 0
    flagged: int = 0
    provisional: int = 0
    frozen_skipped: int = 0
    fetches: dict[str, Any] = field(default_factory=dict)
    # Countries whose decision raised — undecided this run, everyone else's
    # answer stands. "iso3: error" strings, mirrored from LadderRun.
    failed_cells: list[str] = field(default_factory=list)

    @property
    def unavailable_sources(self) -> list[str]:
        return sorted(
            name for name, meta in self.fetches.items() if not meta.get("ok", False)
        )


def resolve_country_month(
    con: "duckdb.DuckDBPyConnection",
    *,
    iso3: str,
    ym: str,
    rulebook: Rulebook,
    today: dt.date | None = None,
) -> DroughtDecision:
    """Load a cell's inputs and decide it. All I/O, no rules."""

    from resolver.hazard_resolution.impact import national_population

    iso3 = iso3.upper()
    year = int(ym.split("-")[0])
    return decide_drought(
        iso3=iso3,
        ym=ym,
        analyses=ipc_mod.analyses_for_country(con, iso3, rulebook),
        indicators=ind_mod.evaluate_indicators(con, iso3, ym, rulebook),
        rulebook=rulebook,
        national_population=national_population(con, iso3, year),
        today=today,
    )


def run_month(
    con: "duckdb.DuckDBPyConnection",
    *,
    ym: str,
    iso3s: list[str],
    rulebook: Rulebook,
    fetches: dict[str, Any] | None = None,
    dry_run: bool = False,
    today: dt.date | None = None,
    run_type: str = RUN_TYPE_LIVE,
) -> DroughtRun:
    """Decide, and persist, every drought country-month in ``iso3s``.

    Triggers, candidates and resolutions all go through the same tables and
    the same freeze logic as the other hazards — the drought rule replaces
    the ladder, not the bookkeeping.
    """

    ensure_haz_schema(con)
    run = DroughtRun(ym=ym, fetches=fetches or {})
    unavailable = run.unavailable_sources
    year, month = detect_mod.ym_to_year_month(ym)

    result = detect_mod.DetectionResult(
        hazard=HAZARD,
        ym=ym,
        coverage_ok=True,
        coverage_note="drought coverage is per-country (see each row's indicators)",
        n_points_month=0,
        n_points_qualifying=0,
        max_point_time=None,
    )
    decisions: list[DroughtDecision] = []

    for iso3 in sorted({c.upper() for c in iso3s}):
        try:
            decision = resolve_country_month(
                con, iso3=iso3, ym=ym, rulebook=rulebook, today=today
            )
        except Exception as exc:  # noqa: BLE001 - one bad cell must not end the month
            run.failed_cells.append(f"{iso3}: {type(exc).__name__}: {exc}")
            LOG.exception("[drought] %s %s: decision raised", iso3, ym)
            cell_ledger.record_cell(
                stage=cell_ledger.STAGE_DROUGHT,
                iso3=iso3, hazard=HAZARD, ym=ym,
                write_outcome="exception",
                reason_code=cell_ledger.REASON_EXCEPTION,
                detail={"error": f"{type(exc).__name__}: {exc}"},
                run_type=run_type,
            )
            continue
        decisions.append(decision)
        result.rows.append(
            detect_mod.CountryTrigger(
                iso3=iso3,
                triggered=decision.triggered,
                trigger_source=decision.trigger_source,
                detail=decision.trigger_detail,
                sweep_evidence=decision.evidence_of_absence,
            )
        )
        run.cells += 1
        run.triggered += int(decision.triggered)
        if decision.status == STATUS_RESOLVED_VALUE:
            run.resolved_value += 1
            run.provisional += int(decision.provisional)
        elif decision.status == STATUS_RESOLVED_ZERO:
            run.resolved_zero += 1
            run.provisional += int(decision.provisional)
        elif decision.status == STATUS_NO_DATA:
            run.no_data += 1
        elif decision.status == STATUS_INCONCLUSIVE:
            run.inconclusive += 1
        else:
            run.pending += 1
        run.flagged += int(decision.flagged)

    if dry_run:
        for decision in decisions:
            _ledger_decision(decision, outcome="dry_run", unavailable=unavailable,
                             run_type=run_type)
        _log_run(run, unavailable)
        return run

    detect_mod.write_triggers(con, result, rulebook, run_type=run_type)

    for decision in decisions:
        if decision.candidate is not None:
            cand_mod.write_candidates(
                con, [decision.candidate], decision.iso3, ym, HAZARD
            )

        if not decision.writes_row:
            _ledger_decision(
                decision, outcome="no_row", unavailable=unavailable, run_type=run_type
            )
            continue

        # A source that could not be READ is not a source that was empty.
        decision.provenance.setdefault("decision", {})["sources_unavailable"] = (
            unavailable
        )
        decision.provenance["source_fetches"] = run.fetches

        if decision.evidence_of_absence is not None:
            outcome = res_mod.write_zero_resolution(
                con,
                iso3=decision.iso3,
                year=year,
                month=month,
                hazard=HAZARD,
                evidence_of_absence={
                    **decision.evidence_of_absence,
                    "source_fetches": run.fetches,
                    "sources_unavailable": unavailable,
                },
                rulebook=rulebook,
                today=today,
                run_type=run_type,
            )
        else:
            outcome = res_mod.write_reconciliation(
                con,
                decision.as_reconciliation(),
                rulebook,
                today=today,
                run_type=run_type,
            )
        if outcome == res_mod.WRITE_FROZEN_SKIP:
            run.frozen_skipped += 1
        _ledger_decision(
            decision, outcome=outcome, unavailable=unavailable, run_type=run_type
        )

    _log_run(run, unavailable)
    return run


#: Why a drought cell carries no row. PENDING is the calendar's doing;
#: INCONCLUSIVE means an indicator could not be read, which is fail-closed
#: and a different repair entirely.
_DROUGHT_NO_ROW_REASON = {
    STATUS_PENDING: cell_ledger.REASON_PENDING,
    STATUS_INCONCLUSIVE: cell_ledger.REASON_INCONCLUSIVE,
}


def _ledger_decision(
    decision: DroughtDecision,
    *,
    outcome: str,
    unavailable: list[str],
    run_type: str,
) -> None:
    """Record one assessed drought cell. Never raises."""

    if not cell_ledger.enabled():
        return
    reason = _DROUGHT_NO_ROW_REASON.get(decision.status)
    if outcome == res_mod.WRITE_FROZEN_SKIP:
        reason = cell_ledger.REASON_FROZEN
    indicators = (decision.provenance.get("decision") or {}).get("indicators")
    cell_ledger.record_cell(
        stage=cell_ledger.STAGE_DROUGHT,
        iso3=decision.iso3, hazard=HAZARD, ym=decision.ym,
        triggered=decision.triggered,
        trigger_source=decision.trigger_source,
        status=decision.status,
        value=decision.value,
        rule_fired=decision.rule_fired,
        flagged=decision.flagged,
        provisional=decision.provisional,
        write_outcome=outcome,
        reason_code=reason,
        rungs_unavailable=list(unavailable),
        answering_rung="ipc" if decision.candidate is not None else None,
        detail={
            "flags": list(decision.flags),
            "indicators": indicators,
            "trigger_detail": decision.trigger_detail,
        },
        run_type=run_type,
    )


def _log_run(run: DroughtRun, unavailable: list[str]) -> None:
    LOG.info(
        "[drought] %s: %d cells -> %d values, %d zeros, %d no-data, %d pending, "
        "%d inconclusive, %d flagged, %d provisional, %d frozen-skipped",
        run.ym, run.cells, run.resolved_value, run.resolved_zero, run.no_data,
        run.pending, run.inconclusive, run.flagged, run.provisional,
        run.frozen_skipped,
    )
    if run.inconclusive:
        LOG.warning(
            "[drought] %s: %d cells were inconclusive because a required drought "
            "indicator could not be read — those cells have no row, which is not "
            "the same as a zero",
            run.ym, run.inconclusive,
        )
    if run.failed_cells:
        LOG.error(
            "[drought] %s: %d cell(s) raised and were NOT decided this run "
            "(re-run to retry them): %s",
            run.ym, len(run.failed_cells), "; ".join(run.failed_cells[:10]),
        )
    if unavailable:
        LOG.warning(
            "[drought] %s: ran with unavailable sources %s — rows from this run "
            "record that those sources were unreadable, not empty",
            run.ym, ",".join(unavailable),
        )
