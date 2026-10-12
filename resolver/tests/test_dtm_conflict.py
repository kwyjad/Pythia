# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""DTM conflict stock and the lower bound on new displacement, network-free."""

from __future__ import annotations

import datetime as dt

import duckdb
import pytest

from resolver.ingestion import dtm_conflict as dtm


def _rec(date, idps, reason="Conflict", operation="Mobility Tracking", iso3="SDN", round_no=1):
    return {
        "id": 1, "operation": operation, "admin0Name": "Sudan", "admin0Pcode": iso3,
        "numPresentIdpInd": idps, "reportingDate": f"{date}T00:00:00",
        "roundNumber": round_no, "displacementReason": reason,
    }


def _bounds(records, iso3="SDN", gap=35):
    series, excluded = dtm.conflict_series(records, iso3)
    bounds, counts = dtm.derive_lower_bounds(series, max_gap_days=gap)
    return bounds, counts, excluded


def test_an_increase_between_close_rounds_goes_to_the_later_rounds_month():
    bounds, counts, _ = _bounds([_rec("2026-02-20", 1000), _rec("2026-03-10", 1600)])
    assert set(bounds) == {"2026-03"}
    assert bounds["2026-03"]["value"] == 600
    assert counts == {"pairs_used": 1, "pairs_skipped_gap": 0}
    pair = bounds["2026-03"]["pairs"][0]
    assert pair["from"] == "2026-02-20" and pair["to"] == "2026-03-10" and pair["gap_days"] == 18


def test_a_fall_in_the_stock_is_a_bound_of_zero_never_negative():
    bounds, _, _ = _bounds([_rec("2026-03-01", 1000), _rec("2026-03-20", 400)])
    assert bounds["2026-03"]["value"] == 0


def test_never_across_a_gap_longer_than_35_days():
    bounds, counts, _ = _bounds([_rec("2026-01-01", 1000), _rec("2026-02-06", 9000)])
    assert bounds == {}
    assert counts == {"pairs_used": 0, "pairs_skipped_gap": 1}


def test_exactly_35_days_is_allowed():
    bounds, _, _ = _bounds([_rec("2026-01-01", 1000), _rec("2026-02-05", 1500)])
    assert bounds["2026-02"]["value"] == 500


def test_close_pairs_ending_in_one_month_add_up_and_are_never_apportioned():
    bounds, _, _ = _bounds([
        _rec("2026-02-25", 100), _rec("2026-03-08", 300), _rec("2026-03-20", 250),
        _rec("2026-03-31", 450),
    ])
    assert set(bounds) == {"2026-03"}
    assert bounds["2026-03"]["value"] == 200 + 0 + 200
    assert len(bounds["2026-03"]["pairs"]) == 3


def test_operations_are_never_added_the_month_takes_the_largest():
    bounds, _, _ = _bounds([
        _rec("2026-03-01", 100, operation="A"), _rec("2026-03-20", 300, operation="A"),
        _rec("2026-03-02", 1000, operation="B"), _rec("2026-03-25", 1100, operation="B"),
    ])
    assert bounds["2026-03"]["value"] == 200
    assert bounds["2026-03"]["operation"] == "A"
    assert bounds["2026-03"]["operations"] == {"A": 200, "B": 100}


def test_only_conflict_reason_records_count():
    _, _, excluded = _bounds([
        _rec("2026-03-01", 100), _rec("2026-03-01", 900, reason="Natural disaster"),
        {"admin0Pcode": "SDN", "numPresentIdpInd": 5, "reportingDate": "2026-03-01"},
    ])
    assert excluded == {"not_conflict": 1, "no_reason_field": 1}
    series, _ = dtm.conflict_series([_rec("2026-03-01", 100),
                                     _rec("2026-03-01", 50, reason="Armed conflict")], "SDN")
    assert series["Mobility Tracking"][dt.date(2026, 3, 1)]["stock"] == 150


def test_records_for_another_country_are_left_out():
    _, _, excluded = _bounds([_rec("2026-03-01", 100, iso3="TCD")])
    assert excluded == {"other_country": 1}


def test_store_and_read_back():
    con = duckdb.connect(":memory:")
    series, _ = dtm.conflict_series(
        [_rec("2026-02-20", 1000), _rec("2026-03-10", 1600)], "SDN")
    bounds, _ = dtm.derive_lower_bounds(series)
    stored = dtm.store_country(con, "SDN", series, bounds, max_gap_days=35)
    assert stored == {"rounds": 2, "lower_bound_months": 1}
    row = dtm.dtm_lower_bound(con, "SDN", "2026-03")
    assert row["value"] == 600 and row["value_type"] == "displaced_lower_bound"
    assert dtm.dtm_lower_bound(con, "SDN", "2026-02") is None
    # A rerun rewrites the country's bounds, never appends.
    dtm.store_country(con, "SDN", series, {}, max_gap_days=35)
    assert dtm.dtm_lower_bounds(con, "SDN") == {}


def test_the_key_travels_in_a_header_and_errors_are_scrubbed():
    seen = {}

    def get(url, params, headers, timeout):
        seen.update(url=url, params=params, headers=headers)
        return 401, None, "denied for key sekret-key-value"

    records, error = dtm.fetch_admin0("SDN", dt.date(2024, 1, 1), dt.date(2026, 10, 1),
                                      "sekret-key-value", get=get)
    assert records == []
    assert "sekret" not in error and "REDACTED" in error
    assert seen["headers"]["Ocp-Apim-Subscription-Key"] == "sekret-key-value"
    assert "sekret" not in seen["url"] and "sekret" not in str(seen["params"])


def test_run_costs_one_country_per_failure():
    con = duckdb.connect(":memory:")

    def get(url, params, headers, timeout):
        if params["Admin0Pcode"] == "COD":
            raise ConnectionError("reset")
        return 200, {"isSuccess": True, "result": [
            _rec("2026-02-20", 1000), _rec("2026-03-10", 1600)]}, ""

    summary = dtm.run(con, ["SDN", "COD"], key="k", get=get, delay=0,
                      today=dt.date(2026, 10, 1))
    assert summary["countries_with_bounds"] == ["SDN"]
    assert "COD" in summary["fetch_errors"]
    assert summary["lower_bound_rows"] == 1


@pytest.mark.parametrize("gap,expected", [(30, set()), (35, {"2026-02"})])
def test_the_gap_rule_follows_the_argument(gap, expected):
    bounds, _, _ = _bounds([_rec("2026-01-01", 1), _rec("2026-02-05", 2)], gap=gap)
    assert set(bounds) == expected
