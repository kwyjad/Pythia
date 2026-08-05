# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The ladder: pick one figure from many, deterministically.

**Hard rule 2 lives here.** This module contains no LLM call, no network
I/O, and no randomness — it is a pure function of (candidates, national
population, rulebook, today). Given the same inputs it always returns the
same answer, and the answer always says which rule produced it.

The decision, in order:

1. **Walk the ladder.** Take the first rung in ``rulebook.ladder`` that
   has a candidate. That rung's figure is the answer. Nothing below it can
   overturn it — a lower rung disagreeing is a reason to FLAG, never a
   reason to substitute.
2. **Label lower bounds.** If the winning rung is in
   ``lower_bound_rungs`` the value stands but is published as a floor, in
   both ``rule_fired`` and provenance.
3. **Apply the ceilings.** GDACS modelled exposure
   (``sanity.ceiling_multiplier`` x exposure) and, when
   ``sanity.population_cap`` is on, the national population. A figure
   above either is kept — the ladder decided it — and flagged. The machine
   does not quietly rewrite a source's number.
4. **Detect wild conflict.** Adjacent populated rungs differing by more
   than ``conflict_detection.order_of_magnitude_factor`` flag the row.
5. **No candidate?** It depends on the clock. Past the freeze deadline,
   ``NO_DATA``, flagged, with the rungs consulted recorded so the gap is
   legible. BEFORE it, ``PENDING`` — no row is written at all, because the
   sources have not finished reporting. EM-DAT alone routinely lands weeks
   after an event, so declaring "no data" the week of a flood would record
   our own impatience as a fact about the world.

Everything above is `conflict_rule: ladder_with_flag`: the ladder's answer
survives every check; checks only ever raise a flag.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any

from resolver.hazard_resolution.candidates import (
    Candidate,
    ceiling_candidates,
    ladder_candidates,
)
from resolver.hazard_resolution.rulebook import Rulebook
from resolver.hazard_resolution.rules import (
    is_lower_bound_rung,
    is_provisional,
    orders_of_magnitude_apart,
    within_population_cap,
    within_sanity_ceiling,
)

LOG = logging.getLogger(__name__)

STATUS_RESOLVED_VALUE = "RESOLVED_VALUE"
STATUS_NO_DATA = "NO_DATA"

#: Not a stored status — ``haz_resolutions.status`` has no such value, and
#: the CHECK constraint would reject it. It means "no row yet": the cell is
#: triggered, no rung has a figure, and the freeze deadline has not passed,
#: so the machine waits instead of recording an absence of knowledge as
#: knowledge of absence.
STATUS_PENDING = "PENDING"

# rule_fired vocabulary — stable strings, queried by downstream analysis.
RULE_LADDER = "ladder:{rung}"
RULE_LADDER_LOWER_BOUND = "ladder:{rung}:lower_bound"
RULE_NO_CANDIDATE = "no_data:no_candidate_on_any_rung"
RULE_PENDING = "pending:awaiting_sources_before_freeze"

FLAG_CEILING_EXCEEDED = "ceiling_exceeded"
FLAG_POPULATION_EXCEEDED = "population_cap_exceeded"
FLAG_RUNG_CONFLICT = "adjacent_rung_order_of_magnitude"
FLAG_NO_CANDIDATE = "no_candidate_past_freeze"


@dataclass
class Reconciliation:
    """The ladder's verdict for one country-month-hazard."""

    iso3: str
    ym: str
    hazard: str
    status: str
    value: float | None
    rule_fired: str
    flagged: bool
    provisional: bool
    flags: list[str] = field(default_factory=list)
    winner: Candidate | None = None
    lower_bound: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)


def _best_on_rung(candidates: list[Candidate], rung: str) -> Candidate | None:
    """The candidate a rung contributes when it has several.

    Within one rung the machine takes the LARGEST stated figure. Rationale:
    multiple records on one rung for one country-month are near-always
    partial reports of the same event (separate field reports, separate
    admin areas), and revisions of a people-affected figure move upward as
    assessment proceeds — the same reasoning behind the resolver's
    "latest informed report wins" rule. Ties break on source_ref so the
    choice is stable across runs.

    A rung whose candidates carry ``preference_rank`` has already ordered
    itself and that order wins — ``reliefweb_extracted`` is the case: the
    rulebook ranks its figures by attributed authority then recency, and
    "largest wins" would quietly overrule that with whichever body quoted
    the biggest number. Still deterministic, still rulebook-driven; the
    rank was computed upstream by a pure function.
    """

    on_rung = [c for c in candidates if c.source == rung]
    if not on_rung:
        return None
    ranked = [c for c in on_rung if c.preference_rank is not None]
    if ranked:
        return min(ranked, key=lambda c: (int(c.preference_rank), str(c.source_ref)))
    return max(on_rung, key=lambda c: (c.value, str(c.source_ref)))


def _ceiling(candidates: list[Candidate]) -> float | None:
    """The GDACS exposure ceiling for this cell, or None if unknown.

    Several overlapping events each bound the month; the largest exposure
    is the binding one, since any of them could account for the figure.
    """

    ceilings = [c.value for c in ceiling_candidates(candidates)]
    return max(ceilings) if ceilings else None


def reconcile(
    *,
    iso3: str,
    ym: str,
    hazard: str,
    candidates: list[Candidate],
    rulebook: Rulebook,
    national_population: float | None = None,
    today: dt.date | None = None,
) -> Reconciliation:
    """Resolve one country-month-hazard from its candidates. Pure function."""

    year, month = (int(p) for p in ym.split("-"))
    provisional = is_provisional(year, month, rulebook, today=today)
    ladder = [str(rung) for rung in rulebook.get("ladder")]

    usable = ladder_candidates(candidates)
    exposure_ceiling = _ceiling(candidates)

    # Which rungs actually have something, in ladder order.
    populated: list[tuple[str, Candidate]] = []
    for rung in ladder:
        best = _best_on_rung(usable, rung)
        if best is not None:
            populated.append((rung, best))

    consulted = {
        "ladder": ladder,
        "rungs_populated": [rung for rung, _ in populated],
        "rungs_empty": [rung for rung in ladder if rung not in {r for r, _ in populated}],
        "candidates": [c.provenance() for c in candidates],
        "ceiling": {
            "source": rulebook.get("sanity.ceiling_source"),
            "multiplier": rulebook.get("sanity.ceiling_multiplier"),
            "exposed_population": exposure_ceiling,
            "national_population": national_population,
            "population_cap_enabled": bool(rulebook.get("sanity.population_cap")),
        },
        "conflict_rule": rulebook.get("conflict_rule"),
        "event_attribution": rulebook.get("event_attribution"),
    }

    # --- No rung has a figure. ---
    if not populated:
        # Before the freeze deadline this is impatience, not a finding:
        # the sources have not finished reporting. Write nothing.
        if provisional:
            return Reconciliation(
                iso3=iso3,
                ym=ym,
                hazard=hazard,
                status=STATUS_PENDING,
                value=None,
                rule_fired=RULE_PENDING,
                flagged=False,
                provisional=True,
                provenance={
                    "rule_fired": RULE_PENDING,
                    "decision": consulted,
                    "note": (
                        "hazard detected, no rung has reported a figure yet, and "
                        "the cell has not frozen — waiting rather than recording "
                        "NO_DATA"
                    ),
                },
            )
        return Reconciliation(
            iso3=iso3,
            ym=ym,
            hazard=hazard,
            status=STATUS_NO_DATA,
            value=None,
            rule_fired=RULE_NO_CANDIDATE,
            flagged=True,
            provisional=False,
            flags=[FLAG_NO_CANDIDATE],
            provenance={
                "source": "ladder",
                "source_record_ids": [],
                "source_urls": [],
                "retrieved_at": None,
                "rule_fired": RULE_NO_CANDIDATE,
                "decision": consulted,
                "note": (
                    "hazard detected but no rung stated a people-affected figure "
                    "by the freeze deadline; flagged for human review"
                ),
            },
        )

    # --- Rung 1: the ladder decides. ---
    winning_rung, winner = populated[0]
    lower_bound = is_lower_bound_rung(winning_rung, rulebook)
    rule_fired = (
        RULE_LADDER_LOWER_BOUND if lower_bound else RULE_LADDER
    ).format(rung=winning_rung)

    flags: list[str] = []

    # --- Sanity ceilings. Flag, never rewrite. ---
    if not within_sanity_ceiling(winner.value, exposure_ceiling, rulebook):
        flags.append(FLAG_CEILING_EXCEEDED)
        LOG.warning(
            "[reconcile] %s/%s/%s: %s figure %.0f exceeds GDACS exposure ceiling "
            "%.0f — keeping the ladder's answer, flagging",
            iso3, hazard, ym, winning_rung, winner.value, exposure_ceiling or 0.0,
        )
    if not within_population_cap(winner.value, national_population, rulebook):
        flags.append(FLAG_POPULATION_EXCEEDED)
        LOG.warning(
            "[reconcile] %s/%s/%s: %s figure %.0f exceeds national population "
            "%.0f — keeping the ladder's answer, flagging",
            iso3, hazard, ym, winning_rung, winner.value, national_population or 0.0,
        )

    # --- Adjacent-rung disagreement. ---
    conflicts: list[dict[str, Any]] = []
    for (upper_rung, upper), (lower_rung, lower) in zip(populated, populated[1:]):
        if orders_of_magnitude_apart(upper.value, lower.value, rulebook):
            conflicts.append(
                {
                    "upper_rung": upper_rung,
                    "upper_value": upper.value,
                    "lower_rung": lower_rung,
                    "lower_value": lower.value,
                    "factor": rulebook.get(
                        "conflict_detection.order_of_magnitude_factor"
                    ),
                }
            )
    if conflicts:
        flags.append(FLAG_RUNG_CONFLICT)
        LOG.warning(
            "[reconcile] %s/%s/%s: %d adjacent-rung conflict(s) beyond the "
            "order-of-magnitude factor — keeping the ladder's answer, flagging",
            iso3, hazard, ym, len(conflicts),
        )

    provenance = {
        "source": winner.source,
        "source_record_ids": [winner.source_ref],
        "source_urls": [u for u in {c.doc_url for c in candidates} if u],
        "retrieved_at": winner.retrieved_at,
        "rule_fired": rule_fired,
        "winning_rung": winning_rung,
        "value_is_lower_bound": lower_bound,
        "stated_by": winner.stated_by,
        "event_span": {"start": winner.span_start, "end": winner.span_end},
        "decision": {**consulted, "conflicts": conflicts, "flags": flags},
    }
    if lower_bound:
        provenance["note"] = (
            "value is a DISPLACEMENT figure and therefore a LOWER BOUND on "
            "people affected, not an estimate of it"
        )

    return Reconciliation(
        iso3=iso3,
        ym=ym,
        hazard=hazard,
        status=STATUS_RESOLVED_VALUE,
        value=float(winner.value),
        rule_fired=rule_fired,
        flagged=bool(flags),
        provisional=provisional,
        flags=flags,
        winner=winner,
        lower_bound=lower_bound,
        provenance=provenance,
    )
