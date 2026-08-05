# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The acceptance report — did the machine actually do what it promised?

    python -m resolver.hazard_resolution.acceptance [--db PATH] [--out report.md]

One page of markdown, for a human, covering the last ``--months`` fully
frozen months: the resolution rate per hazard against the IFRC-GO-only
baseline it replaces, how many rows are flagged, where the answers came
from, the DFO calibration cross-check, and a reproducible random sample of
20 zeros and 20 values with their evidence.

**This module measures; it never tunes.** No threshold in ``rulebook.yaml``
is read here except to describe the window, and nothing about the report
feeds back into a resolution. If the machine falls short of the 80% target
the report says so, in its own section, with the shortfall per hazard —
because a target met by moving a threshold until it was met is not
evidence of anything.

**The denominator is the honest one.** The rate is computed over CELLS
ASSESSED — every ``haz_triggers`` row in the window, i.e. every
country-month-hazard the machine actually looked at — not over the rows it
managed to write. Dividing resolutions by resolutions would score 100%
against a machine that answered one cell. Cells that produced no row at
all (a fail-closed INCONCLUSIVE, a coverage-gated zero) count against the
rate, and are reported separately so the reason is visible.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from resolver.hazard_resolution.base_rates import last_frozen_month, months_back_window
from resolver.hazard_resolution.rulebook import (
    HAZARD_CODE_BY_RULEBOOK_NAME,
    Rulebook,
)
from resolver.hazard_resolution.schema import ensure_haz_schema

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

#: The published rate of the path this machine replaces: PA questions
#: resolving from IFRC GO alone, ~4% a month (see docs/montandon_assessment.md).
#: Quoted for context only — the report also RECOMPUTES an IFRC-GO-only rate
#: from the machine's own candidate table, which is the comparison that can
#: actually be audited.
DOCUMENTED_IFRC_GO_BASELINE_PCT = 4.0

#: The programme target: this share of assessed cells resolving to a number
#: or a credible zero. A goal to measure against, never a threshold to tune to.
TARGET_RESOLUTION_PCT = 80.0

DEFAULT_MONTHS = 12
DEFAULT_SAMPLE_SIZE = 20
DEFAULT_SEED = 20260805


@dataclass
class HazardRates:
    """One hazard's resolution performance over the window."""

    hazard: str
    cells_assessed: int = 0
    resolved_value: int = 0
    resolved_zero: int = 0
    no_data: int = 0
    no_row: int = 0
    flagged: int = 0
    lower_bound: int = 0
    ifrc_go_only_cells: int = 0

    @property
    def resolved(self) -> int:
        return self.resolved_value + self.resolved_zero

    @property
    def rate_pct(self) -> float:
        return 100.0 * self.resolved / self.cells_assessed if self.cells_assessed else 0.0

    @property
    def baseline_pct(self) -> float:
        return (
            100.0 * self.ifrc_go_only_cells / self.cells_assessed
            if self.cells_assessed
            else 0.0
        )

    @property
    def gap_pct(self) -> float:
        """Percentage points short of the target (0 when the target is met)."""

        return max(0.0, TARGET_RESOLUTION_PCT - self.rate_pct)


@dataclass
class AcceptanceResult:
    """Everything the report renders."""

    first_ym: str
    last_ym: str
    months: int
    rates: dict[str, HazardRates] = field(default_factory=dict)
    provenance_mix: dict[str, dict[str, int]] = field(default_factory=dict)
    flag_counts: dict[str, int] = field(default_factory=dict)
    zero_samples: list[dict[str, Any]] = field(default_factory=list)
    value_samples: list[dict[str, Any]] = field(default_factory=list)
    cross_check: Any = None
    seed: int = DEFAULT_SEED

    @property
    def overall(self) -> HazardRates:
        total = HazardRates(hazard="ALL")
        for rates in self.rates.values():
            total.cells_assessed += rates.cells_assessed
            total.resolved_value += rates.resolved_value
            total.resolved_zero += rates.resolved_zero
            total.no_data += rates.no_data
            total.no_row += rates.no_row
            total.flagged += rates.flagged
            total.lower_bound += rates.lower_bound
            total.ifrc_go_only_cells += rates.ifrc_go_only_cells
        return total


def _ym_bounds(first_ym: str, last_ym: str) -> tuple[int, int, int, int]:
    first_year, first_month = (int(p) for p in first_ym.split("-"))
    last_year, last_month = (int(p) for p in last_ym.split("-"))
    return first_year, first_month, last_year, last_month


#: A window predicate over (year, month) columns, as SQL. Written once
#: because the report joins four tables on the same window and a
#: copy-pasted comparison that drifted would silently change a denominator.
_WINDOW_SQL = """
    (year * 12 + month) BETWEEN (? * 12 + ?) AND (? * 12 + ?)
"""


def _loads(provenance_json: Any) -> dict[str, Any]:
    try:
        loaded = json.loads(provenance_json) if provenance_json else {}
    except (TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _provenances(
    con: "duckdb.DuckDBPyConnection",
    *,
    hazard: str | None,
    bounds: list[int],
    status: str | None = None,
    flagged_only: bool = False,
) -> list[dict[str, Any]]:
    """Parsed ``provenance_json`` for every matching resolution row."""

    clauses = [_WINDOW_SQL]
    params: list[Any] = list(bounds)
    if hazard is not None:
        clauses.insert(0, "hazard = ?")
        params.insert(0, hazard)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if flagged_only:
        clauses.append("COALESCE(flagged, FALSE)")
    rows = con.execute(
        f"SELECT provenance_json FROM haz_resolutions WHERE {' AND '.join(clauses)}",
        params,
    ).fetchall()
    return [_loads(row[0]) for row in rows]


def compute_rates(
    con: "duckdb.DuckDBPyConnection",
    *,
    first_ym: str,
    last_ym: str,
    hazards: Sequence[str],
) -> dict[str, HazardRates]:
    """Resolution rate per hazard over the window."""

    bounds = list(_ym_bounds(first_ym, last_ym))
    out: dict[str, HazardRates] = {}

    for hazard in hazards:
        rates = HazardRates(hazard=hazard)
        rates.cells_assessed = int(
            con.execute(
                f"SELECT COUNT(*) FROM haz_triggers WHERE hazard = ? AND {_WINDOW_SQL}",
                [hazard, *bounds],
            ).fetchone()[0]
        )
        for status, value_flag, count in con.execute(
            f"""
            SELECT status, COALESCE(flagged, FALSE), COUNT(*)
            FROM haz_resolutions
            WHERE hazard = ? AND {_WINDOW_SQL}
            GROUP BY status, COALESCE(flagged, FALSE)
            """,
            [hazard, *bounds],
        ).fetchall():
            count = int(count)
            if status == "RESOLVED_VALUE":
                rates.resolved_value += count
            elif status == "RESOLVED_ZERO":
                rates.resolved_zero += count
            else:
                rates.no_data += count
            if value_flag:
                rates.flagged += count

        # Counted in Python rather than with a SQL LIKE over the JSON blob:
        # a substring probe would depend on the separator json.dumps happens
        # to emit, and would start silently answering zero if that ever
        # changed. This is the same parse the provenance mix does.
        rates.lower_bound = sum(
            1
            for provenance in _provenances(
                con,
                hazard=hazard,
                bounds=bounds,
                status="RESOLVED_VALUE",
            )
            if provenance.get("value_is_lower_bound")
        )

        rows_written = rates.resolved + rates.no_data
        rates.no_row = max(0, rates.cells_assessed - rows_written)

        # The baseline this machine replaces, recomputed from its own
        # evidence: cells where the IFRC GO rung alone had a figure. That is
        # what the pre-machine path could have resolved, and unlike the
        # quoted ~4% it can be audited from this database.
        rates.ifrc_go_only_cells = int(
            con.execute(
                f"""
                SELECT COUNT(DISTINCT (iso3 || '|' || year || '|' || month))
                FROM haz_impact_candidates
                WHERE hazard = ? AND source = 'ifrc_go' AND {_WINDOW_SQL}
                """,
                [hazard, *bounds],
            ).fetchone()[0]
        )
        out[hazard] = rates
    return out


def compute_provenance_mix(
    con: "duckdb.DuckDBPyConnection",
    *,
    first_ym: str,
    last_ym: str,
    hazards: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Which source produced each hazard's answers, over the window."""

    bounds = list(_ym_bounds(first_ym, last_ym))
    mix: dict[str, dict[str, int]] = {}
    for hazard in hazards:
        counts: dict[str, int] = {}
        # Keyed by "status/source", because a source that produced 40 zeros
        # and a source that produced 40 figures are not doing the same job,
        # and one pooled count per source cannot tell them apart.
        for status in ("RESOLVED_VALUE", "RESOLVED_ZERO", "NO_DATA"):
            for provenance in _provenances(
                con, hazard=hazard, bounds=bounds, status=status
            ):
                key = f"{status}/{provenance.get('source') or 'unknown'}"
                counts[key] = counts.get(key, 0) + 1
        mix[hazard] = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return mix


def compute_flag_counts(
    con: "duckdb.DuckDBPyConnection", *, first_ym: str, last_ym: str
) -> dict[str, int]:
    """How many rows carry each individual review flag.

    ``flagged`` is one boolean but the reasons are several, and "37 rows
    flagged" tells a reviewer nothing about whether to look at ceilings, at
    population caps, or at rungs that disagree.
    """

    bounds = list(_ym_bounds(first_ym, last_ym))
    counts: dict[str, int] = {}
    for provenance in _provenances(
        con, hazard=None, bounds=bounds, flagged_only=True
    ):
        flags = (provenance.get("decision") or {}).get("flags") or []
        for flag in flags or ["unspecified"]:
            counts[str(flag)] = counts.get(str(flag), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def sample_resolutions(
    con: "duckdb.DuckDBPyConnection",
    *,
    status: str,
    first_ym: str,
    last_ym: str,
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """A reproducible random sample of resolutions, with their evidence.

    Seeded rather than ``ORDER BY random()``: a reviewer who queries a row
    must be able to regenerate the same page, and a report whose sample
    changes on every run cannot be cited in a review.
    """

    bounds = list(_ym_bounds(first_ym, last_ym))
    rows = con.execute(
        f"""
        SELECT iso3, year, month, hazard, value, rule_fired,
               COALESCE(flagged, FALSE), COALESCE(provisional, FALSE),
               COALESCE(run_type, 'live'), provenance_json
        FROM haz_resolutions
        WHERE status = ? AND {_WINDOW_SQL}
        ORDER BY iso3, hazard, year, month
        """,
        [status, *bounds],
    ).fetchall()
    if not rows:
        return []

    picked = (
        rows if len(rows) <= n else random.Random(seed).sample(rows, n)
    )
    picked = sorted(picked, key=lambda r: (r[0], r[3], r[1], r[2]))

    out: list[dict[str, Any]] = []
    for iso3, year, month, hazard, value, rule, flagged, provisional, run_type, prov_json in picked:
        provenance = _loads(prov_json)
        out.append(
            {
                "iso3": str(iso3),
                "ym": f"{int(year):04d}-{int(month):02d}",
                "hazard": str(hazard),
                "value": value,
                "rule_fired": str(rule),
                "flagged": bool(flagged),
                "provisional": bool(provisional),
                "run_type": str(run_type),
                "source": provenance.get("source"),
                "source_record_ids": provenance.get("source_record_ids") or [],
                "source_urls": (provenance.get("source_urls") or [])[:3],
                "retrieved_at": provenance.get("retrieved_at"),
                "stated_by": provenance.get("stated_by"),
                "value_is_lower_bound": bool(provenance.get("value_is_lower_bound")),
                "note": provenance.get("note"),
                "evidence_of_absence": _summarise_absence(
                    provenance.get("evidence_of_absence") or {}
                ),
            }
        )
    return out


def _summarise_absence(evidence: dict[str, Any]) -> dict[str, Any] | None:
    """Condense a zero's evidence to the lines a reviewer needs.

    The full block is on the row; this is the one-line version — what was
    checked, and what it found — so the sample stays readable.
    """

    if not evidence:
        return None
    sweep = evidence.get("reliefweb") or {}
    indicators = evidence.get("indicators") or {}
    summary: dict[str, Any] = {}
    if sweep:
        summary["reliefweb_hits"] = sweep.get("total_hits")
        summary["reliefweb_silent"] = sweep.get("silent")
    for detector in ("ibtracs", "gdacs"):
        block = evidence.get(detector) or {}
        query = block.get("query") or {}
        if block:
            summary[detector] = {
                "records": block.get("total_records") or block.get("total_storms"),
                "qualifying_global": query.get("n_points_qualifying")
                or query.get("n_events_qualifying_global"),
            }
    if indicators:
        summary["indicators"] = {
            "verdict": indicators.get("verdict") or indicators.get("state"),
            "readings": len(indicators.get("readings") or []),
        }
    if evidence.get("ipc"):
        summary["ipc"] = "analyses read (see the row's provenance for the windows)"
    return summary or None


def months_window(
    rulebook: Rulebook, months: int, today: dt.date | None = None
) -> tuple[str, str]:
    """The last ``months`` fully frozen months, as (first_ym, last_ym)."""

    last = last_frozen_month(rulebook, today=today)
    return months_back_window(last, months), last


def build_result(
    con: "duckdb.DuckDBPyConnection",
    rulebook: Rulebook,
    *,
    months: int = DEFAULT_MONTHS,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    today: dt.date | None = None,
    hazards: Sequence[str] | None = None,
    with_cross_check: bool = True,
) -> AcceptanceResult:
    """Gather every number the report renders."""

    ensure_haz_schema(con)
    first_ym, last_ym = months_window(rulebook, months, today=today)
    targets = list(hazards or sorted(HAZARD_CODE_BY_RULEBOOK_NAME.values()))

    result = AcceptanceResult(
        first_ym=first_ym, last_ym=last_ym, months=months, seed=seed
    )
    result.rates = compute_rates(
        con, first_ym=first_ym, last_ym=last_ym, hazards=targets
    )
    result.provenance_mix = compute_provenance_mix(
        con, first_ym=first_ym, last_ym=last_ym, hazards=targets
    )
    result.flag_counts = compute_flag_counts(con, first_ym=first_ym, last_ym=last_ym)
    result.zero_samples = sample_resolutions(
        con, status="RESOLVED_ZERO", first_ym=first_ym, last_ym=last_ym,
        n=sample_size, seed=seed,
    )
    result.value_samples = sample_resolutions(
        con, status="RESOLVED_VALUE", first_ym=first_ym, last_ym=last_ym,
        n=sample_size, seed=seed + 1,
    )
    if with_cross_check:
        from resolver.hazard_resolution import dfo as dfo_mod

        result.cross_check = dfo_mod.cross_check(con, rulebook)
    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value:.1f}%"


def _fmt_value(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value)


def _render_rates(result: AcceptanceResult) -> list[str]:
    lines = [
        "## Resolution rate",
        "",
        f"Window: **{result.first_ym} .. {result.last_ym}** "
        f"({result.months} fully frozen months). The denominator is *cells "
        "assessed* — every country-month-hazard with a detection verdict — so "
        "a cell that produced no row at all counts against the rate.",
        "",
        "| hazard | cells assessed | resolved | rate | value | zero | no-data | no row | IFRC-GO-only baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for hazard, rates in sorted(result.rates.items()):
        lines.append(
            f"| {hazard} | {rates.cells_assessed:,} | {rates.resolved:,} | "
            f"**{_pct(rates.rate_pct)}** | {rates.resolved_value:,} | "
            f"{rates.resolved_zero:,} | {rates.no_data:,} | {rates.no_row:,} | "
            f"{_pct(rates.baseline_pct)} |"
        )
    total = result.overall
    lines.append(
        f"| **all** | {total.cells_assessed:,} | {total.resolved:,} | "
        f"**{_pct(total.rate_pct)}** | {total.resolved_value:,} | "
        f"{total.resolved_zero:,} | {total.no_data:,} | {total.no_row:,} | "
        f"{_pct(total.baseline_pct)} |"
    )
    lines += [
        "",
        f"The *IFRC-GO-only baseline* column is recomputed from this database: "
        f"the share of assessed cells where the IFRC GO rung alone held a "
        f"figure. The documented rate for the path this machine replaces is "
        f"~{DOCUMENTED_IFRC_GO_BASELINE_PCT:.0f}% of PA questions per month.",
        "",
    ]
    return lines


def _render_gap(result: AcceptanceResult) -> list[str]:
    total = result.overall
    lines = ["## Gap against the target", ""]
    # A hazard with no assessed cells was never RUN in this window. Listing
    # it as "0% — short by 80 points" invents a failure out of an absence,
    # which is the same error the rate's denominator is careful to avoid.
    not_assessed = sorted(h for h, r in result.rates.items() if r.cells_assessed == 0)
    short = {
        h: r for h, r in result.rates.items() if r.cells_assessed > 0 and r.gap_pct > 0
    }
    if total.cells_assessed == 0:
        lines += [
            "No cells were assessed in this window, so there is no rate to "
            "report. Run the backcast (or a live month) before reading this "
            "section as a result.",
            "",
        ]
        return lines
    if not_assessed:
        lines += [
            f"Not assessed in this window (no detection verdicts, so no rate): "
            f"**{', '.join(not_assessed)}**. Run the backcast, or a live month, "
            "for these hazards before reading a rate for them.",
            "",
        ]
    if not short:
        lines += [
            f"Every hazard that ran meets the {TARGET_RESOLUTION_PCT:.0f}% target "
            f"(overall {_pct(total.rate_pct)}).",
            "",
        ]
        return lines
    lines.append(
        f"Target is {TARGET_RESOLUTION_PCT:.0f}% of assessed cells resolving to a "
        f"number or a credible zero. Overall the machine reaches "
        f"{_pct(total.rate_pct)}. Short by hazard:"
    )
    lines.append("")
    lines.append(
        "| hazard | rate | short by | NO_DATA (no rung reported) | no row (fail-closed / coverage-gated) |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for hazard, rates in sorted(short.items(), key=lambda kv: -kv[1].gap_pct):
        # Both buckets are reported rather than one being named "largest":
        # they have different causes and different fixes, and with similar
        # counts picking a winner would point a reader at the wrong one.
        lines.append(
            f"| {hazard} | {_pct(rates.rate_pct)} | {rates.gap_pct:.1f} pp | "
            f"{rates.no_data:,} | {rates.no_row:,} |"
        )
    lines += [
        "",
        "*These numbers are what the shipped rulebook produces. They are "
        "reported, not closed: moving a threshold until the target is met "
        "would change the number without changing what the machine knows.*",
        "",
    ]
    return lines


def _render_provenance(result: AcceptanceResult) -> list[str]:
    lines = [
        "## Provenance mix",
        "",
        "Which source produced each hazard's answers, split by what kind of "
        "answer it was. `detection:absence` is a zero resting on evidence of "
        "absence; `ladder` is a NO_DATA where the ladder ran and no rung "
        "reported.",
        "",
        "| hazard | status | source | rows |",
        "|---|---|---|---:|",
    ]
    any_rows = False
    for hazard, counts in sorted(result.provenance_mix.items()):
        for key, count in counts.items():
            any_rows = True
            status, _, source = key.partition("/")
            lines.append(f"| {hazard} | {status} | `{source}` | {count:,} |")
    if not any_rows:
        lines.append("| — | — | *no resolutions in window* | 0 |")
    lines.append("")

    total = result.overall
    lines += [
        f"Of {total.resolved_value:,} resolved values, **{total.lower_bound:,}** "
        "are displacement figures and therefore LOWER BOUNDS on people "
        "affected, not estimates of it.",
        "",
    ]
    return lines


def _render_flags(result: AcceptanceResult) -> list[str]:
    total = result.overall
    lines = [
        "## Flagged for human review",
        "",
        f"**{total.flagged:,}** rows are flagged.",
        "",
    ]
    if result.flag_counts:
        lines += ["| flag | rows |", "|---|---:|"]
        for flag, count in result.flag_counts.items():
            lines.append(f"| `{flag}` | {count:,} |")
        lines.append("")
    return lines


def _render_cross_check(result: AcceptanceResult) -> list[str]:
    check = result.cross_check
    lines = ["## Calibration cross-check — Dartmouth Flood Observatory", ""]
    if check is None:
        lines += ["_Not run._", ""]
        return lines
    if not getattr(check, "available", False):
        lines += [
            f"_Unavailable: {getattr(check, 'reason', 'unknown')}._",
            "",
            "The cross-check is calibration only — it never triggers anything "
            "and never supplies a resolution value — so its absence affects no "
            "resolved row.",
            "",
        ]
        return lines

    lines += [
        f"Flood-months per year per country: the machine's rate over "
        f"**{check.machine_window}** against DFO's independent archive over "
        f"**{check.dfo_window}** ({check.n_events:,} archive events). The two "
        "windows do not overlap by design — a comparison against the years the "
        "machine already resolved would confirm nothing.",
        "",
        f"{len(check.rows):,} countries compared; **{len(check.diverging):,}** "
        f"differ by more than {check.divergence_factor:.0f}x.",
        "",
    ]
    if check.diverging:
        lines += [
            "| iso3 | machine (flood-months/yr) | DFO (flood-months/yr) | DFO/machine |",
            "|---|---:|---:|---:|",
        ]
        worst = sorted(
            check.diverging,
            key=lambda r: (r.ratio if r.ratio is not None else float("inf")),
            reverse=True,
        )[:15]
        for row in worst:
            ratio = "∞ (machine sees none)" if row.ratio is None else f"{row.ratio:.1f}x"
            lines.append(
                f"| {row.iso3} | {row.machine_p:.2f} | {row.dfo_p:.2f} | {ratio} |"
            )
        lines += [
            "",
            "A DFO rate far above the machine's is a country where flood "
            "detection is likely missing events. This is a pointer for a human, "
            "not a correction — nothing here changes a resolved value.",
            "",
        ]
    return lines


def _render_sample(title: str, samples: list[dict[str, Any]], *, zeros: bool) -> list[str]:
    lines = [f"## {title}", ""]
    if not samples:
        lines += ["_None in the window._", ""]
        return lines
    for entry in samples:
        head = f"**{entry['iso3']} / {entry['hazard']} / {entry['ym']}**"
        marks = []
        if entry["flagged"]:
            marks.append("FLAGGED")
        if entry["provisional"]:
            marks.append("provisional")
        if entry["value_is_lower_bound"]:
            marks.append("LOWER BOUND")
        if entry["run_type"] != "live":
            marks.append(entry["run_type"])
        if marks:
            head += " — " + ", ".join(marks)
        lines.append(f"- {head}")
        if not zeros:
            lines.append(f"  - value: **{_fmt_value(entry['value'])}** people")
        lines.append(f"  - rule: `{entry['rule_fired']}`")
        lines.append(f"  - source: `{entry['source']}`")
        if entry["stated_by"]:
            lines.append(f"  - stated by: {entry['stated_by']}")
        if entry["source_record_ids"]:
            ids = ", ".join(str(i) for i in entry["source_record_ids"][:3])
            lines.append(f"  - records: {ids}")
        if entry["retrieved_at"]:
            lines.append(f"  - retrieved: {entry['retrieved_at']}")
        if entry["evidence_of_absence"]:
            lines.append(
                f"  - evidence of absence: `{json.dumps(entry['evidence_of_absence'])}`"
            )
        for url in entry["source_urls"]:
            lines.append(f"  - {url}")
        if entry["note"]:
            lines.append(f"  - note: {entry['note']}")
    lines.append("")
    return lines


def render_report(result: AcceptanceResult) -> str:
    """Render the whole report as markdown."""

    generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = result.overall
    lines = [
        "# People-affected resolution machine — acceptance report",
        "",
        f"Generated {generated} · window {result.first_ym}..{result.last_ym} · "
        f"sample seed `{result.seed}` (re-run with the same seed for the same sample)",
        "",
        f"**Headline: {_pct(total.rate_pct)} of {total.cells_assessed:,} assessed "
        f"country-month-hazard cells resolved to a number or a credible zero"
        + (
            f", against a target of {TARGET_RESOLUTION_PCT:.0f}% and a documented "
            f"IFRC-GO-only baseline of ~{DOCUMENTED_IFRC_GO_BASELINE_PCT:.0f}%.**"
        ),
        "",
    ]
    lines += _render_rates(result)
    lines += _render_gap(result)
    lines += _render_provenance(result)
    lines += _render_flags(result)
    lines += _render_cross_check(result)
    lines += _render_sample(
        f"Sample: {len(result.zero_samples)} zeros", result.zero_samples, zeros=True
    )
    lines += _render_sample(
        f"Sample: {len(result.value_samples)} values", result.value_samples, zeros=False
    )
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="haz-acceptance", description=__doc__)
    parser.add_argument("--db", default=os.getenv("RESOLVER_DB_URL", ""))
    parser.add_argument(
        "--months", type=int, default=DEFAULT_MONTHS,
        help=f"How many fully frozen months to measure (default {DEFAULT_MONTHS})",
    )
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help=f"Zeros and values to sample, each (default {DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Sampling seed — the same seed reproduces the same report",
    )
    parser.add_argument("--out", default=None, help="Write the report here (default: stdout)")
    parser.add_argument(
        "--no-cross-check", action="store_true",
        help="Skip the DFO calibration section",
    )
    parser.add_argument("--log-level", default=os.getenv("RESOLVER_LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    from resolver.db.duckdb_io import close_db, get_db
    from resolver.hazard_resolution.rulebook import load_rulebook

    rulebook = load_rulebook()
    con = get_db(args.db or None)
    try:
        result = build_result(
            con,
            rulebook,
            months=args.months,
            sample_size=args.sample_size,
            seed=args.seed,
            with_cross_check=not args.no_cross_check,
        )
        report = render_report(result)
    finally:
        close_db(con)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report)
        LOG.info("[acceptance] wrote %s", args.out)
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
