# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""IDMC conflict displacement is its own series (Oct 2026).

Every IDMC row reached facts_resolved as hazard IDU, all causes summed, so
the ACE/PA prompts showed China's 7.3 million typhoon evacuees as conflict
displacement and no ACE/PA question resolved.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from resolver.ingestion import idmc_conflict as ic
from resolver.transform.adapters.idmc import IDMCAdapter


def _rec(iso3, dtype, figure, start, rid="1"):
    return {
        "id": rid, "iso3": iso3, "displacement_type": dtype, "figure": figure,
        "displacement_start_date": start, "displacement_end_date": start,
    }


RECORDS = [
    _rec("SDN", "Conflict", 9000, "2026-08-03", "1"),
    _rec("SDN", "Conflict", 1000, "2026-08-20", "2"),
    _rec("SDN", "Conflict", 500, "2026-09-30", "3"),
    _rec("CHN", "Disaster", 7_305_385, "2026-07-12", "4"),
    _rec("PHL", "Disaster", 1_743_994, "2026-09-02", "5"),
    _rec("COD", "Other", 40, "2026-08-01", "6"),
    _rec("AFG", "", 12, "2026-08-01", "7"),
    _rec("SOM", "Conflict", 300, "2026-10-04", "8"),   # month in progress
    _rec("ETH", "Conflict", 70, "2022-01-04", "9"),    # before the window
]


def test_causes_are_read_from_displacement_type():
    assert ic.cause_of({"displacement_type": "Conflict"}) == "conflict"
    assert ic.cause_of({"displacement_type": "Disaster"}) == "disaster"
    assert ic.cause_of({"displacement_type": "Other"}) == "other"
    assert ic.cause_of({}) == "untyped"


def test_the_window_ends_with_the_last_complete_month():
    assert ic.month_window(dt.date(2026, 11, 11), 36) == ("2023-11", "2026-10")
    assert ic.month_window(dt.date(2026, 10, 5), 3) == ("2026-07", "2026-09")


def test_only_conflict_is_summed_and_every_other_cause_is_counted():
    first, last = ic.month_window(dt.date(2026, 10, 5), 36)
    flows, report = ic.conflict_monthly_flows(RECORDS, first, last)
    assert flows.to_dict("records") == [
        {"iso3": "SDN", "ym": "2026-08", "value": 10000.0, "metric": ic.METRIC},
        {"iso3": "SDN", "ym": "2026-09", "value": 500.0, "metric": ic.METRIC},
    ]
    assert report["excluded_people"] == {
        "disaster": 7_305_385 + 1_743_994, "other": 40.0, "untyped": 12.0,
    }
    assert report["excluded_records"] == {"disaster": 2, "other": 1, "untyped": 1}
    assert report["conflict_records_dropped"] == {"conflict_out_of_window": 2}
    assert report["first_conflict_month_served"] == "2022-01"
    assert report["months_per_country"] == {"min": 2, "median": 2.0, "max": 2}


def test_a_figure_is_attributed_to_the_month_its_displacement_started():
    rec = _rec("SDN", "Conflict", 800, "2026-07-29", "x")
    rec["displacement_end_date"] = "2026-08-04"
    flows, _ = ic.conflict_monthly_flows([rec], "2026-01", "2026-09")
    assert flows.to_dict("records") == [
        {"iso3": "SDN", "ym": "2026-07", "value": 800.0, "metric": ic.METRIC}
    ]


def test_run_writes_a_typed_staging_file_the_adapter_turns_into_ace_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("IDMC_API_KEY", "secret-client-id")
    seen = {}

    def _get(url, params, timeout):
        seen["params"] = params
        return RECORDS

    staging, diag = tmp_path / "idmc", tmp_path / "diag"
    rc = ic.run(months=36, today=dt.date(2026, 10, 5), get=_get,
                staging_dir=staging, diagnostics_dir=diag)
    assert rc == 0
    assert seen["params"] == {"client_id": "secret-client-id"}
    frame = pd.read_csv(staging / "flow.csv", dtype=str)
    assert set(frame["displacement_type"]) == {"conflict"}
    canonical = IDMCAdapter("idmc").normalize(tmp_path)
    assert set(canonical["hazard_code"]) == {"ACE"}
    assert set(canonical["iso3"]) == {"SDN"}
    assert sorted(canonical["value"]) == [500.0, 10000.0]
    exclusions = json.loads((diag / "conflict_exclusions.json").read_text())
    assert exclusions["excluded_records"]["disaster"] == 2
    summary = json.loads((diag / "summary.json").read_text())
    assert summary["counts"]["written"] == 2


def test_an_unreadable_source_is_not_an_empty_series(tmp_path, monkeypatch):
    monkeypatch.delenv("IDMC_API_KEY", raising=False)
    monkeypatch.delenv("IDMC_HELIX_CLIENT_ID", raising=False)
    assert ic.run(today=dt.date(2026, 10, 5), get=lambda *a: RECORDS,
                  staging_dir=tmp_path / "s", diagnostics_dir=tmp_path / "d") == 1

    monkeypatch.setenv("IDMC_HELIX_CLIENT_ID", "abc123")

    def _boom(url, params, timeout):
        raise RuntimeError("403 Forbidden for client_id=abc123")

    assert ic.run(today=dt.date(2026, 10, 5), get=_boom,
                  staging_dir=tmp_path / "s", diagnostics_dir=tmp_path / "d") == 1
    summary = json.loads((tmp_path / "d" / "summary.json").read_text())
    assert "abc123" not in json.dumps(summary)


# --- Oct 2026: figures that cannot be monthly counts of people -------------


def test_a_figure_idmc_does_not_recommend_for_totals_is_dropped_and_counted():
    keep = _rec("PSE", "Conflict", 1000, "2026-03-02", "a")
    keep["role"] = "Recommended figure"
    tri = _rec("PSE", "Conflict", 50_000_000, "2026-03-03", "b")
    tri["role"] = "Triangulation"
    flows, report = ic.conflict_monthly_flows([keep, tri], "2026-01", "2026-09")
    assert flows.to_dict("records") == [
        {"iso3": "PSE", "ym": "2026-03", "value": 1000.0, "metric": ic.METRIC}
    ]
    assert report["conflict_records_dropped"] == {"conflict_not_recommended_role": 1}
    assert report["conflict_people_not_recommended"] == 50_000_000


def test_a_record_spanning_more_than_a_month_is_held_out_of_every_month_it_touches():
    rec = _rec("IRN", "Conflict", 6_000_000, "2025-06-13", "war")
    rec["displacement_end_date"] = "2025-08-20"
    short = _rec("IRN", "Conflict", 300, "2025-07-02", "x")
    flows, report = ic.conflict_monthly_flows([rec, short], "2025-01", "2025-12")
    by_metric = {(r["ym"], r["metric"]): r["value"] for r in flows.to_dict("records")}
    assert by_metric == {
        ("2025-06", ic.METRIC_HELD): 6_000_000.0,
        ("2025-07", ic.METRIC_HELD): 6_000_000.0,
        ("2025-08", ic.METRIC_HELD): 6_000_000.0,
    }
    assert report["conflict_records_dropped"]["conflict_long_span_held"] == 1
    assert report["rows"] == 0 and report["rows_held"] == 3


def test_a_country_month_above_the_population_is_held_out_and_named():
    big = _rec("PSE", "Conflict", 54_037_759, "2023-10-07", "x")
    flows, report = ic.conflict_monthly_flows(
        [big], "2023-01", "2023-12", population={"PSE": 5_000_000.0},
    )
    assert flows.to_dict("records") == [
        {"iso3": "PSE", "ym": "2023-10", "value": 54_037_759.0, "metric": ic.METRIC_HELD}
    ]
    assert report["over_population"] == [
        {"iso3": "PSE", "ym": "2023-10", "total": 54_037_759.0, "population": 5_000_000.0}
    ]


def test_held_rows_reach_the_staging_file_under_their_own_metric():
    frame = pd.DataFrame([
        {"iso3": "PSE", "ym": "2023-10", "value": 9.0, "metric": ic.METRIC_HELD},
        {"iso3": "PSE", "ym": "2023-11", "value": 4.0, "metric": ic.METRIC},
    ])
    out = ic.staging_frame(frame)
    assert list(out["metric"]) == [ic.METRIC_HELD, ic.METRIC]


def test_the_population_table_loads():
    pop = ic.load_population()
    assert pop.get("PSE", 0) > 1_000_000


def test_the_load_replaces_the_staged_window_so_a_held_month_loses_its_old_figure():
    import duckdb

    from resolver.tools.load_and_derive import replace_conflict_displacement_window

    con = duckdb.connect()
    con.execute(
        "CREATE TABLE facts_resolved (ym VARCHAR, iso3 VARCHAR, hazard_code VARCHAR, "
        "metric VARCHAR, value DOUBLE, publisher VARCHAR)"
    )
    con.execute(
        "INSERT INTO facts_resolved VALUES "
        "('2023-10','PSE','ACE','new_displacements',54037759,'IDMC'),"
        "('2022-01','PSE','ACE','new_displacements',5,'IDMC'),"
        "('2023-10','PSE','ACE','fatalities',9,'ACLED')"
    )
    canonical = pd.DataFrame([{
        "hazard_code": "ACE", "metric": "new_displacements_held", "source": "idmc",
        "as_of_date": "2023-10-31",
    }, {
        "hazard_code": "ACE", "metric": "new_displacements", "source": "idmc",
        "as_of_date": "2026-09-30",
    }])
    counts = replace_conflict_displacement_window(con, canonical)
    assert counts["facts_resolved"] == 1
    left = con.execute("SELECT ym, metric FROM facts_resolved ORDER BY ym, metric").fetchall()
    assert left == [("2022-01", "new_displacements"), ("2023-10", "fatalities")]


def test_an_unread_idmc_replaces_nothing():
    import duckdb

    from resolver.tools.load_and_derive import replace_conflict_displacement_window

    con = duckdb.connect()
    con.execute("CREATE TABLE facts_resolved (ym VARCHAR, hazard_code VARCHAR, metric VARCHAR, publisher VARCHAR)")
    con.execute("INSERT INTO facts_resolved VALUES ('2026-01','ACE','new_displacements','IDMC')")
    empty = pd.DataFrame(columns=["hazard_code", "metric", "source", "as_of_date"])
    assert replace_conflict_displacement_window(con, empty) == {"facts_resolved": 0, "facts_deltas": 0}
    assert con.execute("SELECT COUNT(*) FROM facts_resolved").fetchone()[0] == 1
