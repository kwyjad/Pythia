# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""The run a report compares with, and how persistence is counted (Oct 2026).

The 1 Oct 2026 report compared itself with fc_1789641908, a 13-question test
run of 17 Sept, because the previous-run lookup filtered on the QUESTIONS'
is_test and same-epoch test runs forecast production questions. And with two
production reports in September, a risk flagged since August would read "4
consecutive runs" in October across three months.
"""

from __future__ import annotations

import json

import pytest

duckdb = pytest.importorskip("duckdb")

from interpreter import persistence
from scripts.ai_bundle import build_current_run_bundle as b


@pytest.fixture()
def con():
    c = duckdb.connect(":memory:")
    c.execute("CREATE TABLE questions (question_id TEXT, window_start_date DATE, is_test BOOLEAN)")
    c.execute("CREATE TABLE forecasts_raw (run_id TEXT, question_id TEXT, is_test BOOLEAN)")
    for qid, ws in [("Q10", "2026-10-01"), ("Q11", "2026-11-01")]:
        c.execute("INSERT INTO questions VALUES (?, ?, FALSE)", [qid, ws])
    for run, qid, test in [
        ("fc_1788237725", "Q10", False),   # 1 Sept, production, epoch 10
        ("fc_1789534892", "Q10", False),   # 15 Sept, production, epoch 10
        ("fc_1789641908", "Q10", True),    # 17 Sept, TEST run on a production question
        ("fc_1790831584", "Q11", False),   # 1 Oct, production, epoch 11
    ]:
        c.execute("INSERT INTO forecasts_raw VALUES (?, ?, ?)", [run, qid, test])
    return c


def test_previous_run_is_the_latest_production_run_of_the_previous_epoch(con):
    """No earlier run in its own epoch: the latest production run of the one
    before, never the later test run of that epoch."""
    assert b._previous_run_id(con, "fc_1790831584", False) == "fc_1789534892"
    # The first run of the record has nothing to compare with.
    assert b._previous_run_id(con, "fc_1788237725", False) is None


def test_a_same_epoch_rerun_is_compared_with_the_run_it_supersedes(con):
    """The 13 Oct 2026 run opens epoch 2026-11, the 1 Oct run's epoch, and
    re-asks its questions: it is compared with 1 Oct, not with 15 Sept. Test
    runs in the epoch (2, 6, 7 Oct) are passed over."""
    for run, test in [("fc_1790952430", True), ("fc_1791380544", True),
                      ("fc_1791860000", False)]:
        con.execute("INSERT INTO forecasts_raw VALUES (?, 'Q11', ?)", [run, test])
    assert b._previous_run_id(con, "fc_1791860000", False) == "fc_1790831584"
    # The same rule one epoch back: 15 Sept superseded 1 Sept.
    assert b._previous_run_id(con, "fc_1789534892", False) == "fc_1788237725"
    # A test run may compare with test runs when asked to.
    assert b._previous_run_id(con, "fc_1791860000", True) == "fc_1791380544"


def test_the_later_report_of_a_month_replaces_the_earlier_one(con):
    """13 Oct 2026 replaces 1 Oct as October's report: the 13 Oct report
    itself reads persistence from September back, and a November report
    reads October's months from 13 Oct alone."""
    con.execute(
        "CREATE TABLE interpretations (kind TEXT, run_id TEXT, hs_run_id TEXT, status TEXT, "
        "content_json TEXT, created_at TIMESTAMP, version INTEGER, is_test BOOLEAN)"
    )
    key = ("SOM", "DR", "PA")
    flag = json.dumps({"attention": [{"iso3": "SOM", "hazard_code": "DR", "metric": "PA"}]})
    none = json.dumps({"attention": []})
    for run, hs, at, content in [
        ("fc_b", "hs_20260915T000000", "2026-09-16", flag),
        ("fc_d", "hs_20261001T000000", "2026-10-01", flag),
        ("fc_e", "hs_20261013T000000", "2026-10-14", none),
    ]:
        con.execute(
            "INSERT INTO interpretations VALUES ('combined', ?, ?, 'ok', ?, ?, 1, FALSE)",
            [run, hs, content, at],
        )
    # Building the 13 Oct report: October is its own month, so 1 Oct is not
    # a previous report; persistence counts September back.
    reports = b._previous_reports(con, include_test=False, before_month="2026-10",
                                  exclude_run_id="fc_e")
    assert [r["run_id"] for r in reports] == ["fc_b"]
    months = [(r["month_label"], r["flagged_keys"]) for r in reports]
    assert persistence.consecutive_months(key, months, "2026-10") == 2
    # A November report: October is 13 Oct's report, which did not flag SOM.
    reports = b._previous_reports(con, include_test=False, before_month="2026-11")
    assert [r["run_id"] for r in reports] == ["fc_e", "fc_b"]
    months = [(r["month_label"], r["flagged_keys"]) for r in reports]
    assert persistence.consecutive_months(key, months, "2026-11") == 1


def test_reports_are_one_per_month_and_never_this_runs_own(con):
    con.execute(
        "CREATE TABLE interpretations (kind TEXT, run_id TEXT, hs_run_id TEXT, status TEXT, "
        "content_json TEXT, created_at TIMESTAMP, version INTEGER, is_test BOOLEAN)"
    )
    flag = json.dumps({"attention": [{"iso3": "SOM", "hazard_code": "DR", "metric": "PA"}]})
    for run, hs, at in [
        ("fc_a", "hs_20260801T000000", "2026-08-01"),
        ("fc_b", "hs_20260901T000000", "2026-09-01"),
        ("fc_c", "hs_20260915T000000", "2026-09-16"),
        ("fc_d", "hs_20261001T000000", "2026-10-01"),
    ]:
        con.execute(
            "INSERT INTO interpretations VALUES ('combined', ?, ?, 'ok', ?, ?, 1, FALSE)",
            [run, hs, flag, at],
        )
    reports = b._previous_reports(
        con, include_test=False, before_month="2026-10", exclude_run_id="fc_d",
    )
    assert [r["month_label"] for r in reports] == ["2026-09", "2026-08"]
    assert reports[0]["run_id"] == "fc_c"
    months = [(r["month_label"], r["flagged_keys"]) for r in reports]
    assert persistence.consecutive_months(("SOM", "DR", "PA"), months, "2026-10") == 3


def test_a_missing_month_breaks_the_run():
    key = ("SOM", "DR", "PA")
    assert persistence.consecutive_months(key, [("2026-08", [key])], "2026-10") == 1
    assert persistence.persistence_phrase(3) == "flagged for 3 consecutive months"


def test_the_track_counts_partition_the_countries_forecast():
    """Oct 2026: "carried 70 ... 52 ... Another 35" read as 87 of 70."""

    questions = [
        {"iso3": "SDN", "track": 1}, {"iso3": "SDN", "track": 2},  # both tracks
        {"iso3": "ETH", "track": 1},
        {"iso3": "KEN", "track": 2},
    ]
    summary = b._build_run_summary(duckdb.connect(":memory:"), None, questions, [])
    assert summary["countries_with_questions"] == 3
    assert summary["countries_track1"] == 2
    assert summary["countries_track2"] == 2
    assert summary["countries_track2_only"] == 1
    assert summary["countries_track1"] + summary["countries_track2_only"] == 3
