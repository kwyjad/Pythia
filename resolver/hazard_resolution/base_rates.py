# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Base rates computed from the backcast — occurrence and severity.

Two products, both derived from tables the machine already wrote, both
deterministic, and neither of them consulting any source of its own:

``haz_base_rates_occurrence``
    Per (iso3, hazard, calendar month): the share of backcast years in
    which that month TRIGGERED. Read from ``haz_triggers``, over each
    hazard's full backcast depth (``rulebook.backcast.<hazard>`` onward).

``haz_base_rates_severity``
    Per (iso3, hazard): quantiles of the RESOLVED_VALUE amounts from
    ``severity_base_rate_window_start`` onward, with ``n_events`` and the
    provenance mix — the share of those resolutions contributed by each
    ladder rung, per year.

**Why the denominators are what they are.** Occurrence counts YEARS
ASSESSED, not years in the calendar: a country-month with no trigger row
was never looked at, and counting it as a quiet month would manufacture a
low base rate out of an ingestion gap — the same mistake the coverage gate
exists to prevent one cell at a time. ``n_years`` travels with every row so
a thin denominator is visible rather than implied, and
``base_rates.occurrence.min_years`` refuses to publish a rate below a floor
(one observed November is not a base rate, and a p_occurrence of 1.0 from a
single year reads as certainty).

**Why the provenance mix matters more than it looks.** A severity
distribution assembled mostly from IDMC displacement figures is a
distribution of LOWER BOUNDS, not of people-affected figures; one assembled
from EM-DAT is a distribution of settled post-event assessments. The mix,
including ``lower_bound_share``, is what lets a consumer tell those apart —
so it is stored beside the quantiles rather than derived on demand and
forgotten.

Nothing here is a source of truth of its own: drop both tables and a re-run
rebuilds them exactly, because every input is already on disk.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from resolver.hazard_resolution.rulebook import (
    HAZARD_CODE_BY_RULEBOOK_NAME,
    Rulebook,
)
from resolver.hazard_resolution.rules import freeze_deadline
from resolver.hazard_resolution.schema import ensure_haz_schema

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

STATUS_RESOLVED_VALUE = "RESOLVED_VALUE"

#: The quantiles ``haz_base_rates_severity`` stores, in column order.
QUANTILES: tuple[tuple[str, float], ...] = (
    ("q10", 0.10),
    ("q25", 0.25),
    ("q50", 0.50),
    ("q75", 0.75),
    ("q90", 0.90),
)


def quantile(values: list[float], q: float) -> float | None:
    """Linear-interpolated quantile of ``values`` (the numpy/DuckDB default).

    Written out rather than delegated so the definition is pinned by this
    repo's own tests: a base rate that quietly changes because a dependency
    changed its interpolation default would be very hard to notice and very
    easy to act on.
    """

    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(q)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def last_frozen_month(rulebook: Rulebook, today: dt.date | None = None) -> str:
    """The most recent ``YYYY-MM`` whose freeze deadline has already passed.

    The backcast's upper bound, and the acceptance report's: a month that
    can still change is not a month to draw a base rate from.
    """

    reference = today or dt.datetime.now(dt.timezone.utc).date()
    year, month = reference.year, reference.month
    # Walk back from the current month until one has frozen. The freeze
    # window is 60 days by default, so this terminates in three or four
    # steps; the loop bound is generous insurance against a huge freeze_days.
    for _ in range(400):
        if reference > freeze_deadline(year, month, rulebook):
            return f"{year:04d}-{month:02d}"
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    raise ValueError(  # pragma: no cover - freeze_days would have to exceed ~33 years
        "no frozen month found; freeze_days is implausibly large"
    )


def months_back_window(last_ym: str, months: int) -> str:
    """The first ``YYYY-MM`` of a window of ``months`` months ending at ``last_ym``."""

    if months < 1:
        raise ValueError(f"months must be >= 1, got {months!r}")
    year, month = (int(p) for p in last_ym.split("-"))
    index = (year * 12 + month - 1) - (int(months) - 1)
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def backcast_window(
    hazard: str, rulebook: Rulebook, today: dt.date | None = None
) -> tuple[str, str]:
    """(first, last) ``YYYY-MM`` of a hazard's backcast, inclusive."""

    name = _rulebook_name(hazard)
    first_year = int(rulebook.get(f"backcast.{name}"))
    return f"{first_year:04d}-01", last_frozen_month(rulebook, today=today)


def _rulebook_name(hazard: str) -> str:
    for name, code in HAZARD_CODE_BY_RULEBOOK_NAME.items():
        if code == hazard:
            return name
    raise ValueError(
        f"unknown hazard code {hazard!r}; known: {sorted(HAZARD_CODE_BY_RULEBOOK_NAME.values())}"
    )


@dataclass
class BaseRateRun:
    """What one base-rate computation wrote."""

    kind: str
    rows_written: int = 0
    rows_skipped_thin: int = 0
    hazards: dict[str, int] = field(default_factory=dict)
    windows: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Occurrence
# ---------------------------------------------------------------------------


def compute_occurrence(
    con: "duckdb.DuckDBPyConnection",
    rulebook: Rulebook,
    *,
    hazards: list[str] | None = None,
    today: dt.date | None = None,
) -> BaseRateRun:
    """Rebuild ``haz_base_rates_occurrence`` from ``haz_triggers``."""

    ensure_haz_schema(con)
    run = BaseRateRun(kind="occurrence")
    min_years = int(rulebook.get("base_rates.occurrence.min_years"))
    targets = hazards or sorted(HAZARD_CODE_BY_RULEBOOK_NAME.values())

    for hazard in targets:
        first_ym, last_ym = backcast_window(hazard, rulebook, today=today)
        first_year = int(first_ym.split("-")[0])
        last_year, last_month = (int(p) for p in last_ym.split("-"))
        window = f"{first_ym}..{last_ym}"
        run.windows[hazard] = window

        # A year counts for a calendar month only if that cell was actually
        # assessed — i.e. it has a trigger row. The final partial year is
        # handled by the same rule, with no special case: its unassessed
        # months simply have no rows.
        rows = con.execute(
            """
            SELECT iso3, month,
                   COUNT(DISTINCT year) AS n_years,
                   COUNT(DISTINCT CASE WHEN triggered THEN year END) AS n_triggered
            FROM haz_triggers
            WHERE hazard = ?
              AND year >= ?
              AND (year < ? OR (year = ? AND month <= ?))
            GROUP BY iso3, month
            ORDER BY iso3, month
            """,
            [hazard, first_year, last_year, last_year, last_month],
        ).fetchall()

        written = 0
        skipped_before = run.rows_skipped_thin
        payload: list[list[Any]] = []
        for iso3, month, n_years, n_triggered in rows:
            if int(n_years) < min_years:
                run.rows_skipped_thin += 1
                continue
            payload.append(
                [
                    str(iso3),
                    hazard,
                    int(month),
                    float(n_triggered) / float(n_years),
                    int(n_years),
                    window,
                ]
            )
            written += 1

        try:
            con.execute("BEGIN TRANSACTION")
            con.execute("DELETE FROM haz_base_rates_occurrence WHERE hazard = ?", [hazard])
            for values in payload:
                con.execute(
                    """
                    INSERT INTO haz_base_rates_occurrence
                        (iso3, hazard, calendar_month, p_occurrence, n_years, source_window)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

        run.hazards[hazard] = written
        run.rows_written += written
        LOG.info(
            "[base_rates] occurrence %s: %d rows over %s (%d cells below "
            "min_years=%d were not published)",
            hazard, written, window, run.rows_skipped_thin - skipped_before, min_years,
        )

    return run


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------


def _provenance_mix(rows: list[tuple[int, str, bool]]) -> dict[str, Any]:
    """Share of resolutions by source, per year, plus the lower-bound share.

    ``rows`` is (year, source, is_lower_bound). The by-year breakdown is
    what shows a distribution's composition CHANGING — a country whose
    severity record shifts from EM-DAT to IDMC lower bounds has a
    distribution whose meaning drifted, and a single pooled mix would hide
    that behind an average.
    """

    by_year: dict[str, dict[str, float]] = {}
    counts: dict[int, dict[str, int]] = {}
    for year, source, _ in rows:
        counts.setdefault(int(year), {}).setdefault(str(source), 0)
        counts[int(year)][str(source)] += 1
    for year, sources in sorted(counts.items()):
        total = sum(sources.values())
        by_year[str(year)] = {
            source: round(n / total, 4) for source, n in sorted(sources.items())
        }

    overall: dict[str, int] = {}
    for _, source, _ in rows:
        overall[str(source)] = overall.get(str(source), 0) + 1
    total = sum(overall.values()) or 1
    n_lower_bound = sum(1 for _, _, lb in rows if lb)

    return {
        "by_year": by_year,
        "overall": {source: round(n / total, 4) for source, n in sorted(overall.items())},
        # The share of the distribution that is a DISPLACEMENT figure, and
        # therefore a lower bound on people affected rather than an estimate
        # of it. A high share means these quantiles sit below the truth.
        "lower_bound_share": round(n_lower_bound / total, 4),
        "n_lower_bound": n_lower_bound,
        "n_values": len(rows),
    }


def compute_severity(
    con: "duckdb.DuckDBPyConnection",
    rulebook: Rulebook,
    *,
    hazards: list[str] | None = None,
    today: dt.date | None = None,
) -> BaseRateRun:
    """Rebuild ``haz_base_rates_severity`` from ``haz_resolutions``."""

    ensure_haz_schema(con)
    run = BaseRateRun(kind="severity")
    window_start = int(rulebook.get("severity_base_rate_window_start"))
    min_events = int(rulebook.get("base_rates.severity.min_events"))
    targets = hazards or sorted(HAZARD_CODE_BY_RULEBOOK_NAME.values())
    window_end = int(last_frozen_month(rulebook, today=today).split("-")[0])

    for hazard in targets:
        run.windows[hazard] = f"{window_start}..{window_end}"
        # Provisional rows are excluded: a pre-freeze value is real but
        # still revisable, and a quantile that moves when a live month is
        # revised is not a base rate — the stored window claims a frozen
        # record, so only frozen (non-provisional) values may enter it.
        rows = con.execute(
            """
            SELECT iso3, year, value, provenance_json
            FROM haz_resolutions
            WHERE hazard = ? AND status = ? AND year >= ? AND value IS NOT NULL
              AND COALESCE(provisional, FALSE) = FALSE
            ORDER BY iso3, year
            """,
            [hazard, STATUS_RESOLVED_VALUE, window_start],
        ).fetchall()

        per_country: dict[str, list[tuple[float, int, str, bool]]] = {}
        for iso3, year, value, provenance_json in rows:
            try:
                provenance = json.loads(provenance_json) if provenance_json else {}
            except (TypeError, ValueError):
                provenance = {}
            per_country.setdefault(str(iso3), []).append(
                (
                    float(value),
                    int(year),
                    str(provenance.get("source") or "unknown"),
                    bool(provenance.get("value_is_lower_bound")),
                )
            )

        written = 0
        skipped_before = run.rows_skipped_thin
        payload: list[list[Any]] = []
        for iso3, entries in sorted(per_country.items()):
            if len(entries) < min_events:
                run.rows_skipped_thin += 1
                continue
            values = [value for value, _, _, _ in entries]
            quantiles = [quantile(values, q) for _, q in QUANTILES]
            mix = _provenance_mix([(y, s, lb) for _, y, s, lb in entries])
            payload.append(
                [
                    iso3,
                    hazard,
                    *quantiles,
                    len(entries),
                    window_start,
                    window_end,
                    json.dumps(mix),
                ]
            )
            written += 1

        try:
            con.execute("BEGIN TRANSACTION")
            con.execute("DELETE FROM haz_base_rates_severity WHERE hazard = ?", [hazard])
            for values in payload:
                con.execute(
                    """
                    INSERT INTO haz_base_rates_severity
                        (iso3, hazard, q10, q25, q50, q75, q90,
                         n_events, window_start, window_end, provenance_mix_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

        run.hazards[hazard] = written
        run.rows_written += written
        LOG.info(
            "[base_rates] severity %s: %d countries over %d..%d (%d below "
            "min_events=%d were not published)",
            hazard, written, window_start, window_end,
            run.rows_skipped_thin - skipped_before, min_events,
        )

    return run


def compute_all(
    con: "duckdb.DuckDBPyConnection",
    rulebook: Rulebook,
    *,
    hazards: list[str] | None = None,
    today: dt.date | None = None,
) -> dict[str, BaseRateRun]:
    """Rebuild both base-rate tables."""

    # First flip the provisional label off rows whose freeze deadline has
    # passed — values untouched, only the revisability label moves to match
    # the calendar. Without this, the oldest month of every live trailing
    # window (written 1-2 days before its own deadline, then out of the
    # window) stayed provisional forever and compute_severity's
    # non-provisional filter excluded it from the quantiles for good. Both
    # callers of the base rates (the nightly backcast and resolver_update
    # Phase 2.5) route through here, so one call site covers both.
    from resolver.hazard_resolution.resolutions import finalize_frozen_provisionals

    finalize_frozen_provisionals(con, today=today)

    return {
        "occurrence": compute_occurrence(con, rulebook, hazards=hazards, today=today),
        "severity": compute_severity(con, rulebook, hazards=hazards, today=today),
    }


def main(argv: list[str] | None = None) -> int:
    """``python -m resolver.hazard_resolution.base_rates [--db PATH]``"""

    import argparse

    from resolver.db.duckdb_io import close_db, get_db
    from resolver.hazard_resolution.rulebook import load_rulebook

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None, help="DuckDB path or duckdb:/// URL")
    parser.add_argument(
        "--hazard",
        nargs="*",
        default=None,
        choices=sorted(HAZARD_CODE_BY_RULEBOOK_NAME.values()),
        help="Hazard codes to recompute (default: all)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    rulebook = load_rulebook()
    con = get_db(args.db)
    try:
        runs = compute_all(con, rulebook, hazards=args.hazard)
    finally:
        close_db(con)

    for kind, run in runs.items():
        print(f"[base-rates] {kind}: {run.rows_written} rows written", end="")
        if run.rows_skipped_thin:
            print(f" ({run.rows_skipped_thin} too thin to publish)", end="")
        print()
        for hazard, count in sorted(run.hazards.items()):
            print(f"  {hazard}: {count} rows | window {run.windows.get(hazard, '?')}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
