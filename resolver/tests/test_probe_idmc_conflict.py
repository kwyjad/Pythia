# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The IDMC conflict probe's analysis, on records shaped like the feed's."""

from tools.probe_idmc_conflict import _scrub, analyse


def _rec(iso3, start, end, figure, role="Recommended figure", created="2026-03-20", dtype="Conflict"):
    return {
        "iso3": iso3, "displacement_type": dtype, "role": role, "figure": figure,
        "displacement_start_date": start, "displacement_end_date": end,
        "created_at": created, "event_name": "e",
    }


def test_lag_is_measured_from_the_end_of_the_start_month():
    report = analyse([
        _rec("SDN", "2026-01-05", "2026-01-06", 10, created="2026-02-10"),  # 10 days
        _rec("SDN", "2026-01-05", "2026-01-06", 10, created="2026-04-01"),  # 60 days
        _rec("SDN", "2026-01-05", "2026-01-06", 10, dtype="Disaster", created="2027-01-01"),
    ])
    lag = report["lag"]["created_at"]
    assert lag["n"] == 2
    assert lag["share_within_30d"] == 0.5
    assert lag["share_within_60d"] == 1.0


def test_roles_spans_and_composition_are_reported():
    report = analyse([
        _rec("PSE", "2023-10-07", "2023-12-31", 1_000_000),
        _rec("PSE", "2023-10-08", "2023-10-09", 500, role="Triangulation"),
        _rec("LBN", "2024-09-01", "2024-09-02", 7),
        _rec("LBN", "2026-03-01", "2026-03-02", 9),
    ])
    assert report["roles"] == {"Recommended figure": 3, "Triangulation": 1}
    assert report["multi_month_records"] == 1
    assert report["records_over_31_days"] == 1
    pse = report["composition"][0]
    assert pse["records"] == 2
    assert pse["people_by_role"]["Triangulation"] == 500
    assert report["composition"][-1]["ym"] == "2026-03"


def test_the_client_id_never_survives_in_an_error_text():
    assert "abc123" not in _scrub("GET https://x/?client_id=abc123 failed", "abc123")


def test_the_settle_curve_counts_recommended_people_by_arrival():
    from tools.probe_idmc_conflict import settle_curve

    curve = settle_curve([
        _rec("SDN", "2025-01-05", "2025-01-06", 100, created="2025-02-10"),   # 10 days
        _rec("SDN", "2025-01-08", "2025-01-09", 300, created="2025-05-01"),   # 90 days
        _rec("SDN", "2025-01-08", "2025-01-09", 9999, role="Triangulation", created="2025-02-01"),
        _rec("MLI", "2025-02-08", "2025-02-09", 100, created="2025-06-30"),  # 122 days
    ])
    assert curve["country_months"] == 2
    assert curve["share_of_people_arrived"]["15d"] == 0.2
    assert curve["share_of_people_arrived"]["90d"] == 0.8
    assert curve["share_of_country_months_with_a_first_report"]["15d"] == 0.5
    assert curve["share_of_country_months_with_a_first_report"]["180d"] == 1.0
