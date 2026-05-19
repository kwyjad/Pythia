# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for scripts/diagnose_pa_resolution_coverage.py — fixture-based
coverage scenarios on a synthetic DB."""

from __future__ import annotations

from datetime import date

import duckdb
import pytest

from scripts.diagnose_pa_resolution_coverage import (
    METRIC_FILTERS_SQL,
    run_diagnostic,
)


def _setup_db(tmp_path):
    db_path = tmp_path / "diag.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE questions (
            question_id TEXT,
            hazard_code TEXT,
            metric TEXT,
            iso3 TEXT,
            window_start_date DATE,
            status TEXT,
            is_test BOOLEAN DEFAULT FALSE
        );
        """
    )
    con.execute(
        """
        CREATE TABLE facts_resolved (
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            ym TEXT,
            value DOUBLE,
            publisher TEXT
        );
        """
    )
    return con


@pytest.fixture
def coverage_db(tmp_path):
    """Three FL/PA questions covering the three coverage scenarios, plus an
    ACE/FATALITIES question to confirm the metric filter is correctly scoped.

    All questions have window_start_date='2026-01-01' so horizons are
    Jan..Jun 2026 — comfortably in the past so they'd be resolvable.
    """
    con = _setup_db(tmp_path)
    con.executemany(
        """
        INSERT INTO questions(question_id, hazard_code, metric, iso3,
                              window_start_date, status)
        VALUES (?, ?, ?, ?, ?, 'active')
        """,
        [
            # FL/PA — full coverage: IFRC row with metric='affected' for one horizon
            ("FL_FULL", "FL", "PA", "PAK", date(2026, 1, 1)),
            # FL/PA — events_but_no_pa: IFRC row exists but only fatalities
            ("FL_EVENT_ONLY", "FL", "PA", "BGD", date(2026, 1, 1)),
            # FL/PA — nothing at all in facts
            ("FL_MISSING", "FL", "PA", "ZZZ", date(2026, 1, 1)),
            # ACE/FATALITIES — has an ACLED fatality row for one horizon
            ("ACE_FAT", "ACE", "FATALITIES", "PAK", date(2026, 1, 1)),
            # Retired question — must NOT be counted
            ("FL_RETIRED", "FL", "PA", "PAK", date(2026, 1, 1)),
        ],
    )
    # Mark FL_RETIRED retired
    con.execute("UPDATE questions SET status='retired' WHERE question_id='FL_RETIRED'")

    con.executemany(
        """
        INSERT INTO facts_resolved(iso3, hazard_code, metric, ym, value, publisher)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            # PAK/FL has an affected row for Jan 2026 → FL_FULL is PA-eligible
            ("PAK", "FL", "affected", "2026-01", 50000.0, "IFRC"),
            # BGD/FL has only fatalities for Jan 2026 → FL_EVENT_ONLY is
            # "event captured but no PA field"
            ("BGD", "FL", "fatalities", "2026-01", 12.0, "IFRC"),
            # ACE/FATALITIES match for PAK
            ("PAK", "ACE", "fatalities", "2026-01", 30.0, "ACLED"),
            # Extra noise row that should NOT contribute to FL/PA counts
            ("PAK", "TC", "affected", "2026-01", 999.0, "IFRC"),
        ],
    )
    yield con
    con.close()


def test_metric_filters_in_sync_with_compute_resolutions():
    """Sanity check that the four metric filters we use mirror the strings
    in compute_resolutions.py. If those drift, this test fails so we notice."""
    assert METRIC_FILTERS_SQL["PA"] == (
        "LOWER(metric) IN ('affected','people_affected','pa','displaced')"
    )
    assert METRIC_FILTERS_SQL["FATALITIES"] == "LOWER(metric) = 'fatalities'"
    assert METRIC_FILTERS_SQL["EVENT_OCCURRENCE"] == "LOWER(metric) = 'event_occurrence'"
    assert METRIC_FILTERS_SQL["PHASE3PLUS_IN_NEED"] == "LOWER(metric) = 'phase3plus_in_need'"


def test_coverage_buckets(coverage_db):
    coverage, publisher = run_diagnostic(coverage_db)

    by_key = {(r.hazard_code, r.metric): r for r in coverage}

    fl_pa = by_key[("FL", "PA")]
    # 3 FL/PA questions counted (retired excluded). 2 have a matching fact row;
    # 1 of those has a PA-eligible metric (FL_FULL).
    assert fl_pa.total == 3
    assert fl_pa.events_in_facts == 2  # FL_FULL + FL_EVENT_ONLY
    assert fl_pa.pa_eligible_in_facts == 1  # only FL_FULL has 'affected'
    assert fl_pa.gap_event_no_pa == 1  # FL_EVENT_ONLY — the imputation-target population

    ace_fat = by_key[("ACE", "FATALITIES")]
    assert ace_fat.total == 1
    assert ace_fat.events_in_facts == 1
    assert ace_fat.pa_eligible_in_facts == 1  # ACE/FATALITIES filter matches 'fatalities'

    # Publisher breakdown shows IFRC for FL/PA, ACLED for ACE/FATALITIES
    by_pub = {(r.hazard_code, r.metric, r.publisher): r.pa_eligible_questions
              for r in publisher}
    assert by_pub.get(("FL", "PA", "IFRC")) == 1
    assert by_pub.get(("ACE", "FATALITIES", "ACLED")) == 1


def test_retired_questions_excluded(coverage_db):
    coverage, _ = run_diagnostic(coverage_db)
    fl_pa = next(r for r in coverage if (r.hazard_code, r.metric) == ("FL", "PA"))
    # If retired were counted, total would be 4 not 3.
    assert fl_pa.total == 3


def test_empty_db(tmp_path):
    """No questions, no facts → empty output, no crash."""
    con = _setup_db(tmp_path)
    coverage, publisher = run_diagnostic(con)
    assert coverage == []
    assert publisher == []
    con.close()
