# Pythia / Copyright (c) 2025 Kevin Wyjad
"""Tests for build_ensemble_scores_export metric_family separation.

Binary EVENT_OCCURRENCE Brier (0-1) and multiclass SPD Brier (0-2) are on
different scales and must never be averaged together. The export carries a
``metric_family`` column so consumers can group by family before aggregating.
"""

from __future__ import annotations

import duckdb
import pytest

from resolver.query.downloads import build_ensemble_scores_export


@pytest.fixture()
def con():
    c = duckdb.connect(":memory:")
    c.execute(
        """
        CREATE TABLE questions (
            question_id TEXT,
            hs_run_id TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            target_month TEXT,
            track INTEGER
        )
        """
    )
    c.execute(
        """
        CREATE TABLE scores (
            question_id TEXT,
            horizon_m INTEGER,
            metric TEXT,
            score_type TEXT,
            model_name TEXT,
            value DOUBLE,
            is_test BOOLEAN
        )
        """
    )
    # One binary EVENT_OCCURRENCE question and one multiclass SPD question.
    c.execute(
        "INSERT INTO questions VALUES "
        "('AFG_DR_EVENT_OCCURRENCE_2026-05','run1','AFG','DR','EVENT_OCCURRENCE','2026-10',1),"
        "('AFG_ACE_FATALITIES_2026-05','run1','AFG','ACE','FATALITIES','2026-10',1)"
    )
    c.execute(
        "INSERT INTO scores VALUES "
        "('AFG_DR_EVENT_OCCURRENCE_2026-05',1,'EVENT_OCCURRENCE','brier','ensemble_mean_v2',0.0004,FALSE),"
        "('AFG_ACE_FATALITIES_2026-05',1,'FATALITIES','brier','ensemble_mean_v2',0.85,FALSE)"
    )
    yield c
    c.close()


def test_export_has_metric_family_column(con):
    df = build_ensemble_scores_export(con, "ensemble_mean")
    assert not df.empty
    # Column present and positioned immediately after `metric`.
    cols = list(df.columns)
    assert "metric_family" in cols
    assert cols.index("metric_family") == cols.index("metric") + 1
    # metric_family adds one column to the export template (58 -> 59).
    assert len(cols) == 59


def test_metric_family_values_match_metric(con):
    df = build_ensemble_scores_export(con, "ensemble_mean")
    fam = dict(zip(df["metric"], df["metric_family"]))
    assert fam["EVENT_OCCURRENCE"] == "binary"
    assert fam["FATALITIES"] == "spd"
