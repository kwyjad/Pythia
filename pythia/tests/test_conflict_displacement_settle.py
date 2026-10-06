# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Conflict displacement: when a month may resolve, and what the loop learns.

On the 5 October 2026 release 64 of 96 ACE/PA resolutions were zero-defaults
and 60 of them were TRAILING: months IDMC had not reported yet. The zeros were
scored and reached family_recalibration (factors at the clip limit on four of
six buckets, in apply mode), calibration weights, advice and centroids.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest

from pythia.tools import base_rate_spd as brs


def _months(first: str, last: str) -> list[str]:
    out, ym = [], first
    while ym <= last:
        out.append(ym)
        ym = brs._add_months(ym, 1)
    return out


REGULAR = {m: 1000.0 for m in _months("2025-01", "2026-06") if m not in ("2026-02",)}
LATER = date(2027, 1, 1)


# --- the rule ---------------------------------------------------------------


def test_a_month_is_unknown_until_it_has_settled_reported_or_not():
    assert brs.CONFLICT_SETTLE_DAYS == 90
    # 2026-06 ended 30 June; 90 days later is 28 September.
    assert brs.resolve_conflict_month(REGULAR, "2026-06", date(2026, 9, 27)) == (
        None, brs.CONFLICT_STATUS_UNSETTLED,
    )
    assert brs.resolve_conflict_month(REGULAR, "2026-06", date(2026, 9, 28)) == (
        1000.0, brs.CONFLICT_STATUS_REPORTED,
    )


def test_a_regular_reporters_bracketed_gap_is_zero():
    assert brs.resolve_conflict_month(REGULAR, "2026-02", LATER) == (0.0, brs.CONFLICT_STATUS_QUIET)


def test_a_trailing_month_is_unknown_and_looked_at_again():
    """The 5 October fault: 2026-09 for SDN, MMR, SOM ... all read zero."""
    assert brs.resolve_conflict_month(REGULAR, "2026-07", LATER) == (None, brs.CONFLICT_STATUS_TRAILING)
    later_report = dict(REGULAR, **{"2026-08": 50.0})
    assert brs.resolve_conflict_month(later_report, "2026-07", LATER) == (0.0, brs.CONFLICT_STATUS_QUIET)


def test_an_irregular_reporters_missing_month_never_resolves_to_zero():
    afg = {"2025-02": 7369.0, "2025-03": 96.0, "2025-10": 160800.0, "2026-02": 272515.0}
    assert not brs.conflict_regular_reporter(afg, "2025-11")
    assert brs.resolve_conflict_month(afg, "2025-11", LATER) == (None, brs.CONFLICT_STATUS_IRREGULAR)
    # ...while its reported months still resolve.
    assert brs.resolve_conflict_month(afg, "2025-10", LATER) == (160800.0, brs.CONFLICT_STATUS_REPORTED)


def test_a_held_out_month_is_neither_a_value_nor_a_zero():
    assert brs.resolve_conflict_month(REGULAR, "2025-06", LATER, held={"2025-06"}) == (
        None, brs.CONFLICT_STATUS_HELD,
    )
    assert brs.resolve_conflict_month(REGULAR, "2026-02", LATER, held={"2026-02"}) == (
        None, brs.CONFLICT_STATUS_HELD,
    )


def test_the_regular_reporter_rule_counts_the_twelve_months_before():
    eight = {m: 1.0 for m in _months("2025-01", "2025-08")}
    assert brs.conflict_regular_reporter(eight, "2026-01")       # 2025-01..2025-12: 8
    assert not brs.conflict_regular_reporter(eight, "2026-02")   # 2025-02..2026-01: 7


# --- readers ----------------------------------------------------------------


def _db(path: Path | None = None):
    con = duckdb.connect(str(path) if path else ":memory:")
    con.execute(
        "CREATE TABLE facts_resolved (ym TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, "
        "series_semantics TEXT, value DOUBLE, publisher TEXT, source_id TEXT)"
    )
    return con


def _row(con, ym, iso3, value, metric="new_displacements"):
    con.execute(
        "INSERT INTO facts_resolved VALUES (?, ?, 'ACE', ?, 'new', ?, 'IDMC', 'idmc')",
        [ym, iso3, metric, value],
    )


def test_held_rows_are_read_as_held_and_never_as_values():
    con = _db()
    for m, v in REGULAR.items():
        _row(con, m, "PSE", v)
    _row(con, "2025-06", "PSE", 54_037_759, metric=brs.CONFLICT_DISPLACEMENT_HELD_METRIC)
    reported, held = brs.conflict_displacement_series(con)
    assert held == {"PSE": {"2025-06"}}
    assert brs.conflict_displacement_value(con, "PSE", "2025-06", today=LATER) is None
    assert brs.conflict_displacement_status(con, "PSE", "2025-06", today=LATER)[1] == (
        brs.CONFLICT_STATUS_HELD
    )


def test_the_anchor_reads_the_series_as_it_stood_on_the_forecast_day():
    con = _db()
    for m, v in REGULAR.items():
        _row(con, m, "SDN", v)
    # A window starting 2026-07 was forecast on 13 June 2026: a month had
    # settled by then if it ended on or before 15 March, so 2026-02 is the
    # newest the forecaster could read.
    _probs, _src, detail = brs.base_rate_spd(con, "SDN", "ACE", "PA", "2026-07")
    assert detail["known_at"] == "2026-06-13"
    assert detail["n_months_quiet"] == 1          # 2026-02, bracketed by 2026-03
    assert detail["n_months_reported"] == 13      # 2025-01..2026-01
    assert detail["n_months_unknown"] == 36 - 14  # before 2025 and after 2026-02


def test_the_prompt_rows_stop_at_the_latest_settled_month():
    con = _db()
    for m, v in REGULAR.items():
        _row(con, m, "SDN", v)
    out = brs.conflict_displacement_settled_rows(con, "SDN", date(2026, 10, 13))
    assert out["latest_settled_month"] == "2026-06"  # ended 30 June + 90 days
    assert out["regular_reporter"] is True
    assert out["rows"][-1] == ("2026-06", 1000.0)
    assert ("2026-02", 0.0) in out["rows"]


def test_the_prompt_block_names_the_settled_month_and_prints_no_trend_for_an_irregular_reporter():
    from forecaster.history_loaders import _format_base_rate_for_prompt

    trajectory = brs.conflict_trajectory([("2025-02", 7369.0), ("2025-03", 96.0), ("2025-10", 160800.0), ("2026-02", 272515.0)], "IDMC")
    trajectory.update({
        "latest_settled_month": "2026-07", "regular_reporter": False, "n_reported_12m": 2,
        "trend_pct": None, "trend_direction": None,
        "trend_note": "no trend: IDMC reported this country in 2 of the last 12 months, too irregularly for one",
    })
    summary = {
        "type": "conflict_trajectory", "as_of_ym": "2026-10",
        "fatalities": brs.conflict_trajectory([], "ACLED"), "displacements": trajectory,
    }
    text = _format_base_rate_for_prompt(summary, "AFG", "ACE", metric="PA")
    assert "Latest settled month: 2026-07" in text
    assert "not yet settled and are UNKNOWN, not quiet" in text
    assert "2 of the last 12 months" in text
    assert "%" not in text.split("Displacement")[1].split("Fatalities")[0]


# --- resolution, scores and guards -----------------------------------------


def test_scores_with_no_resolution_behind_them_are_removed():
    from pythia.tools.compute_scores import purge_orphan_scores

    con = duckdb.connect()
    con.execute("CREATE TABLE resolutions (question_id TEXT, horizon_m INT)")
    con.execute("CREATE TABLE scores (question_id TEXT, horizon_m INT, model_name TEXT)")
    con.execute("INSERT INTO resolutions VALUES ('q', 1)")
    con.execute("INSERT INTO scores VALUES ('q', 1, 'm'), ('q', 6, 'm'), ('__gone', 1, 'm')")
    assert purge_orphan_scores(con) == {"scores": 2}
    assert con.execute("SELECT * FROM scores").fetchall() == [("q", 1, "m")]


def test_a_group_mostly_zero_defaults_is_named_and_events_are_exempt():
    from pythia.tools.compute_resolutions import mostly_zero_default_groups

    counts = {"ACE/PA": (32, 64), "TC/EVENT_OCCURRENCE": (10, 23), "FL/PA": (9, 0)}
    assert mostly_zero_default_groups(counts) == ["ACE/PA"]


def test_the_reset_audits_then_deletes_every_ace_pa_row(tmp_path):
    from scripts.reset_conflict_displacement import reset

    con = duckdb.connect()
    con.execute("CREATE TABLE questions (question_id TEXT, hazard_code TEXT, metric TEXT)")
    con.execute("INSERT INTO questions VALUES ('a', 'ACE', 'PA'), ('f', 'ACE', 'FATALITIES')")
    con.execute("CREATE TABLE resolutions (question_id TEXT, horizon_m INT, source_desc TEXT)")
    con.execute("INSERT INTO resolutions VALUES ('a', 1, 'zero_default'), ('f', 1, 'x')")
    con.execute("CREATE TABLE scores (question_id TEXT, horizon_m INT)")
    con.execute("INSERT INTO scores VALUES ('a', 1), ('f', 1)")
    for table in ("family_recalibration", "calibration_weights", "calibration_advice", "bucket_centroids"):
        con.execute(f"CREATE TABLE {table} (hazard_code TEXT, metric TEXT, v DOUBLE)")
        con.execute(f"INSERT INTO {table} VALUES ('ACE', 'PA', 2.0), ('ACE', 'FATALITIES', 1.0), ('*', 'PA', 0.5)")
    plan = reset(con, audit_dir=tmp_path, audit_only=True)
    assert plan["tables"]["resolutions"] == 1 and plan["tables"]["family_recalibration"] == 1
    assert con.execute("SELECT COUNT(*) FROM resolutions").fetchone()[0] == 2
    assert (tmp_path / "family_recalibration.csv").read_text().count("\n") == 2
    reset(con)
    assert con.execute("SELECT question_id FROM resolutions").fetchall() == [("f",)]
    assert con.execute("SELECT question_id FROM scores").fetchall() == [("f",)]
    for table in ("family_recalibration", "calibration_weights", "calibration_advice", "bucket_centroids"):
        left = con.execute(f"SELECT hazard_code, metric FROM {table} ORDER BY 1, 2").fetchall()
        assert left == [("*", "PA"), ("ACE", "FATALITIES")]
    assert reset(con)["tables"]["resolutions"] == 0
