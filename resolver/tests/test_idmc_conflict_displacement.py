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
        {"iso3": "SDN", "ym": "2026-08", "value": 10000.0},
        {"iso3": "SDN", "ym": "2026-09", "value": 500.0},
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
    assert flows.to_dict("records") == [{"iso3": "SDN", "ym": "2026-07", "value": 800.0}]


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
