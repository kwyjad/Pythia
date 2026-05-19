# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Quantify the coverage gap between Pythia questions and resolution data.

The /performance dashboard surfaces low PA-metric resolution rates (e.g.
FL/PA ≈ 4%, DR/PA = 0%, ACE/PA ≈ 2%). Before investing in connector
improvements (PA imputation from collateral fields, IFRC Appeals
endpoint, GDACS fallback, etc.) we need to know which of three causes
dominates:

  1. Connector does not see the event at all  — fixable only by a richer
     or different source.
  2. Connector sees the event but lacks the PA field — fixable by
     imputation, Appeals endpoint, or fallback to another publisher.
  3. Question is too new for any horizon to be inside the calendar
     cutoff yet — not a coverage problem at all, just time.

This script counts, per (hazard, metric) group, how many questions fall
in each bucket. It also breaks the matched rows down by publisher so we
can see whether the gap is IFRC-specific or general.

Usage::

    PYTHIA_DB_URL=duckdb:///path/to/resolver.duckdb \
        python -m scripts.diagnose_pa_resolution_coverage [--csv out.csv]

Markdown tables go to stdout. CSV is optional.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence

from pythia.db.schema import connect

LOGGER = logging.getLogger(__name__)

# Mirrors pythia/tools/compute_resolutions.py — kept in sync by hand so the
# diagnostic counts the same rows the resolution pipeline would actually use.
# If those filters change, mirror them here.
METRIC_FILTERS_SQL: dict[str, str] = {
    "PA": "LOWER(metric) IN ('affected','people_affected','pa','displaced')",
    "FATALITIES": "LOWER(metric) = 'fatalities'",
    "EVENT_OCCURRENCE": "LOWER(metric) = 'event_occurrence'",
    "PHASE3PLUS_IN_NEED": "LOWER(metric) = 'phase3plus_in_need'",
}

NUM_HORIZONS = 6


@dataclass
class CoverageRow:
    hazard_code: str
    metric: str
    total: int
    events_in_facts: int
    pa_eligible_in_facts: int

    @property
    def gap_event_no_pa(self) -> int:
        return self.events_in_facts - self.pa_eligible_in_facts

    @property
    def event_capture_rate(self) -> float:
        return self.events_in_facts / self.total if self.total else 0.0

    @property
    def pa_capture_rate(self) -> float:
        return self.pa_eligible_in_facts / self.total if self.total else 0.0


@dataclass
class PublisherRow:
    hazard_code: str
    metric: str
    publisher: str
    pa_eligible_questions: int


def _calendar_cutoff_ym(today: date | None = None) -> str:
    """The same cutoff compute_resolutions uses: previous complete month."""
    if today is None:
        today = date.today()
    first_of_this_month = today.replace(day=1)
    last_day_prev = first_of_this_month.replace(
        year=first_of_this_month.year if first_of_this_month.month > 1
        else first_of_this_month.year - 1,
        month=first_of_this_month.month - 1 if first_of_this_month.month > 1 else 12,
        day=1,
    )
    return last_day_prev.strftime("%Y-%m")


def _coverage_query(conn, metric: str, hazard_filter: str = "") -> list[CoverageRow]:
    """Return coverage rows for the given metric. Joins each question's 6
    horizons against facts_resolved on (iso3, hazard, ym) and counts matches."""
    filt = METRIC_FILTERS_SQL[metric]
    sql = f"""
        WITH q AS (
            SELECT
                question_id,
                UPPER(hazard_code) AS hazard_code,
                UPPER(metric) AS metric,
                iso3,
                CAST(window_start_date AS DATE) AS window_start_date
            FROM questions
            WHERE COALESCE(is_test, FALSE) = FALSE
              AND COALESCE(status, '') != 'retired'
              AND UPPER(metric) = ?
              {hazard_filter}
        ),
        q_months AS (
            SELECT
                q.question_id,
                q.hazard_code,
                q.metric,
                q.iso3,
                strftime(q.window_start_date + (h - 1) * INTERVAL 1 MONTH, '%Y-%m') AS ym
            FROM q, generate_series(1, {NUM_HORIZONS}) AS s(h)
        ),
        hits_any AS (
            SELECT DISTINCT q.question_id
            FROM q_months q
            JOIN facts_resolved f
              ON f.iso3 = q.iso3
             AND UPPER(f.hazard_code) = q.hazard_code
             AND f.ym = q.ym
        ),
        hits_pa AS (
            SELECT DISTINCT q.question_id
            FROM q_months q
            JOIN facts_resolved f
              ON f.iso3 = q.iso3
             AND UPPER(f.hazard_code) = q.hazard_code
             AND f.ym = q.ym
            WHERE {filt.replace('metric', 'f.metric')}
        )
        SELECT
            q.hazard_code,
            q.metric,
            COUNT(DISTINCT q.question_id) AS total,
            COUNT(DISTINCT ha.question_id) AS events_in_facts,
            COUNT(DISTINCT hp.question_id) AS pa_in_facts
        FROM q
        LEFT JOIN hits_any ha ON ha.question_id = q.question_id
        LEFT JOIN hits_pa hp ON hp.question_id = q.question_id
        GROUP BY q.hazard_code, q.metric
        ORDER BY q.hazard_code
    """
    rows = conn.execute(sql, [metric]).fetchall()
    return [CoverageRow(hc, m, t, e, p) for hc, m, t, e, p in rows]


def _publisher_breakdown(conn, metric: str) -> list[PublisherRow]:
    """For each (hazard, metric) and publisher, how many distinct questions
    have at least one PA-eligible matching fact?"""
    filt = METRIC_FILTERS_SQL[metric]
    sql = f"""
        WITH q AS (
            SELECT
                question_id,
                UPPER(hazard_code) AS hazard_code,
                UPPER(metric) AS metric,
                iso3,
                CAST(window_start_date AS DATE) AS window_start_date
            FROM questions
            WHERE COALESCE(is_test, FALSE) = FALSE
              AND COALESCE(status, '') != 'retired'
              AND UPPER(metric) = ?
        ),
        q_months AS (
            SELECT
                q.question_id, q.hazard_code, q.metric, q.iso3,
                strftime(q.window_start_date + (h - 1) * INTERVAL 1 MONTH, '%Y-%m') AS ym
            FROM q, generate_series(1, {NUM_HORIZONS}) AS s(h)
        )
        SELECT
            q.hazard_code,
            q.metric,
            COALESCE(UPPER(f.publisher), '(null)') AS publisher,
            COUNT(DISTINCT q.question_id) AS n_questions
        FROM q_months q
        JOIN facts_resolved f
          ON f.iso3 = q.iso3
         AND UPPER(f.hazard_code) = q.hazard_code
         AND f.ym = q.ym
        WHERE {filt.replace('metric', 'f.metric')}
        GROUP BY q.hazard_code, q.metric, COALESCE(UPPER(f.publisher), '(null)')
        ORDER BY q.hazard_code, COALESCE(UPPER(f.publisher), '(null)')
    """
    rows = conn.execute(sql, [metric]).fetchall()
    return [PublisherRow(hc, m, pub, n) for hc, m, pub, n in rows]


def _render_coverage_table(rows: Sequence[CoverageRow]) -> str:
    lines = [
        "| Hazard | Metric | Total | Events in facts | PA-eligible | Event capture % | PA capture % | Gap (event but no PA) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.hazard_code} | {r.metric} | {r.total} | {r.events_in_facts} | "
            f"{r.pa_eligible_in_facts} | {r.event_capture_rate * 100:.1f}% | "
            f"{r.pa_capture_rate * 100:.1f}% | {r.gap_event_no_pa} |"
        )
    return "\n".join(lines)


def _render_publisher_table(rows: Sequence[PublisherRow]) -> str:
    lines = [
        "| Hazard | Metric | Publisher | PA-eligible questions matched |",
        "|---|---|---|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.hazard_code} | {r.metric} | {r.publisher} | {r.pa_eligible_questions} |"
        )
    return "\n".join(lines)


def _write_csv(
    path: str,
    coverage: Iterable[CoverageRow],
    publisher: Iterable[PublisherRow],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow([
            "section", "hazard_code", "metric", "publisher",
            "total", "events_in_facts", "pa_eligible_in_facts",
            "gap_event_no_pa", "event_capture_rate", "pa_capture_rate",
            "pa_eligible_questions",
        ])
        for r in coverage:
            w.writerow([
                "coverage", r.hazard_code, r.metric, "",
                r.total, r.events_in_facts, r.pa_eligible_in_facts,
                r.gap_event_no_pa,
                f"{r.event_capture_rate:.4f}",
                f"{r.pa_capture_rate:.4f}",
                "",
            ])
        for r in publisher:
            w.writerow([
                "publisher", r.hazard_code, r.metric, r.publisher,
                "", "", "", "", "", "", r.pa_eligible_questions,
            ])


def run_diagnostic(conn) -> tuple[list[CoverageRow], list[PublisherRow]]:
    coverage: list[CoverageRow] = []
    publisher: list[PublisherRow] = []
    for metric in METRIC_FILTERS_SQL:
        coverage.extend(_coverage_query(conn, metric))
        publisher.extend(_publisher_breakdown(conn, metric))
    return coverage, publisher


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Quantify PA-resolution coverage gap by (hazard, metric).",
    )
    parser.add_argument(
        "--csv",
        help="Optional path to also write machine-readable CSV output.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    conn = connect(read_only=True)
    try:
        coverage, publisher = run_diagnostic(conn)
    finally:
        conn.close()

    print(f"# PA resolution coverage diagnostic")
    print(f"\nCalendar cutoff (= previous complete month): **{_calendar_cutoff_ym()}**")
    print(
        "\nA question's earliest horizon falls in `window_start_date` month; "
        "its 6 horizons span the next 6 months. A row in `facts_resolved` is "
        "matched per `(iso3, hazard_code, ym)`. \"PA-eligible\" matches use "
        "the same metric filters as `pythia/tools/compute_resolutions.py`."
    )
    print("\n## Per-(hazard, metric) coverage\n")
    print(_render_coverage_table(coverage))
    print("\n## PA-eligible match counts by publisher\n")
    print(_render_publisher_table(publisher))

    if args.csv:
        _write_csv(args.csv, coverage, publisher)
        print(f"\n_CSV written to {args.csv}_")

    return 0


if __name__ == "__main__":
    sys.exit(main())
