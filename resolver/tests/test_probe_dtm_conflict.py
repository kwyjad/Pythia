# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The DTM conflict probe's analysis, on records shaped like DTM API v3's."""

import datetime as dt

from tools.probe_dtm_conflict import (
    analyse_country,
    cadence,
    classify_flow_stock,
    extract_records,
    hdx_summary,
    interesting_keys,
    lag_days,
    reason_split,
    record_date,
    scrub,
    summarise,
    walk_keys,
)

TODAY = dt.date(2026, 10, 7)


def _rec(date, idps, reason="Conflict", round_no=1, iso3="SOM", **extra):
    return {
        "id": 1, "operation": "x", "admin0Name": "Somalia", "admin0Pcode": iso3,
        "numPresentIdpInd": idps, "reportingDate": date, "yearReportingDate": int(date[:4]),
        "monthReportingDate": int(date[5:7]), "roundNumber": round_no,
        "displacementReason": reason, **extra,
    }


def test_records_are_found_in_any_envelope():
    rows = [_rec("2026-01-15T00:00:00", 10)]
    assert extract_records({"isSuccess": True, "result": rows}) == rows
    assert extract_records({"data": {"items": rows}}) == rows
    assert extract_records(rows) == rows
    assert extract_records({"meta": {"x": 1}, "deep": [{"a": rows}]}) == rows
    assert extract_records("nope") == []


def test_key_walk_finds_the_fields_that_matter():
    keys = walk_keys({"result": [_rec("2026-01-15", 5)]})
    picked = interesting_keys(keys)
    assert "reportingDate" in picked
    assert "displacementReason" in picked
    assert "numPresentIdpInd" in picked
    assert "admin0Pcode" in picked
    assert "operation" not in picked


def test_record_date_prefers_the_reporting_date_then_year_and_month():
    assert record_date(_rec("2026-03-31T00:00:00", 1)) == dt.date(2026, 3, 31)
    assert record_date({"yearReportingDate": 2025, "monthReportingDate": 7}) == dt.date(2025, 7, 1)
    assert record_date({"x": 1}) is None


def test_cadence_counts_distinct_reports_in_the_window_and_the_median_gap():
    dates = [dt.date(2026, 1, 1), dt.date(2026, 4, 1), dt.date(2026, 4, 1),
             dt.date(2026, 7, 1), dt.date(2020, 1, 1)]  # 2020 is outside 36 months
    cad = cadence(dates, today=TODAY, months=36)
    assert cad["n_reports"] == 3
    assert cad["median_gap_days"] == 90.5  # gaps of 90 and 91 days
    assert cad["months_with_a_report"] == 3
    assert cad["last"] == "2026-07-01"
    assert cad["days_since_last"] == (TODAY - dt.date(2026, 7, 1)).days


def test_lag_is_reporting_date_minus_the_period_date_where_both_exist():
    rows = [
        _rec("2026-02-15", 1, dataCollectionEndDate="2026-01-31"),  # 15 days
        _rec("2026-05-20", 1, dataCollectionEndDate="2026-04-30"),  # 20 days
        _rec("2026-06-01", 1),  # no period date
    ]
    lag = lag_days(rows)
    assert lag["n"] == 2
    assert lag["median_days"] == 17.5
    assert lag["field_pairs"] == {"reportingDate - dataCollectionEndDate": 2}


def test_present_idps_read_as_a_stock_and_new_arrivals_as_a_flow():
    assert classify_flow_stock([_rec("2026-01-01", 100)])["verdict"] == "stock"
    flow = classify_flow_stock([{"newDisplacements": 40, "roundNumber": 3, "id": 9}])
    assert flow["verdict"] == "flow"
    assert flow["flow_fields"] == {"newDisplacements": 1}
    both = classify_flow_stock([_rec("2026-01-01", 100, numNewIdpArrivals=7)])
    assert both["verdict"] == "both"
    assert classify_flow_stock([{"operation": "x"}])["verdict"] == "unknown"


def test_the_reason_field_separates_conflict_from_disaster():
    rows = [_rec("2026-01-01", 1, "Conflict"), _rec("2026-02-01", 1, "Natural disaster"),
            _rec("2026-03-01", 1, "Conflict"), _rec("2026-04-01", 1, None)]
    split = reason_split(rows)
    assert split["fields"] == {"displacementReason": 4}
    assert split["classes"] == {"conflict": 2, "disaster": 1, "blank": 1}
    assert split["separable"] == "yes"
    assert reason_split([{"numPresentIdpInd": 3}])["separable"] == "no_reason_field"
    assert reason_split([_rec("2026-01-01", 1, "Flood")])["separable"] == "no_conflict_values"


def test_country_analysis_counts_only_conflict_records_when_a_reason_exists():
    rows = [_rec("2026-01-15", 10, "Conflict", 1), _rec("2026-04-15", 20, "Conflict", 2),
            _rec("2026-05-15", 30, "Flood", 3)]
    out = analyse_country(rows, today=TODAY)
    assert out["covered"] is True
    assert out["conflict_records"] == 2
    assert out["conflict_filter"] == "by_reason"
    assert out["cadence"]["n_reports"] == 2
    assert out["rounds"] == ["1", "2"]
    assert out["flow_or_stock"]["verdict"] == "stock"

    no_reason = analyse_country([{"reportingDate": "2026-01-01", "numPresentIdpInd": 5}], today=TODAY)
    assert no_reason["conflict_filter"] == "no_reason_field_all_records"
    assert no_reason["covered"] is True

    assert analyse_country([], today=TODAY)["covered"] is False


def test_summary_names_covered_and_uncovered_countries():
    per = {
        "SOM": analyse_country([_rec("2026-01-15", 10)], today=TODAY),
        "PER": analyse_country([], today=TODAY),
    }
    s = summarise(per)
    assert s["countries_covered"] == ["SOM"]
    assert s["countries_not_covered"] == ["PER"]
    assert s["flow_or_stock_verdicts"] == {"stock": 1}


def test_hdx_summary_reads_groups_and_dates():
    pkgs = [{"name": "dtm-som", "title": "Somalia DTM", "organization": {"name": "iom"},
             "groups": [{"name": "som"}], "dataset_date": "[2024-01-01T00:00:00 TO 2026-06-30T23:59:59]",
             "metadata_modified": "2026-07-02", "resources": [{"format": "CSV"}, {"format": "XLSX"}]}]
    out = hdx_summary(pkgs)
    assert out["countries"] == {"SOM": 1}
    assert out["packages"][0]["formats"] == ["CSV", "XLSX"]
    assert out["packages"][0]["organization"] == "iom"


def test_the_key_never_survives_in_a_recorded_text():
    assert "sekret123" not in scrub("401 for key sekret123", "sekret123")
    assert scrub("nothing to hide", "") == "nothing to hide"


def test_a_spent_budget_stops_asking_and_names_what_was_skipped(monkeypatch):
    """The first run made ~85 requests at a 90-second timeout, was killed at
    the step's 30-minute cap, and wrote no report at all."""
    import datetime as dt

    from tools import probe_dtm_conflict as probe

    calls = []

    class _Resp:
        status_code = 200
        url = "https://example.test"
        headers = {"Content-Type": "application/json"}
        text = "{}"

        def json(self):
            return {"result": []}

    def fake_get(url, **kw):
        calls.append(url)
        return _Resp()

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    now = [0.0]
    rec = probe.Recorder("", delay=0.0, budget_sec=10.0, clock=lambda: now[0])

    def tick(url, **kw):
        now[0] += 4.0  # each request costs 4 seconds of the 10-second budget
        return fake_get(url, **kw)

    monkeypatch.setattr(requests, "get", tick)
    out = probe.probe_dtm(rec, ["AFG", "SDN", "SOM", "ETH"], today=dt.date(2026, 10, 7),
                          months=12, sample_admin_levels=0)
    assert len(calls) == 3  # the two catalogue routes, then one country
    assert out["skipped_for_budget"] == ["SDN", "SOM", "ETH"]
    assert probe.REQUEST_TIMEOUT_SEC <= 30
