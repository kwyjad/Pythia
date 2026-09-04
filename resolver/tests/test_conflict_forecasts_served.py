# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""A stale vintage is flagged on the row, and no reader serves a past month.

CAST's December 2025 vintage had target months January to May 2026, all
in the past by September, and the readers selected "lead months 1 to 3"
— which meant January to March. The flag lives on the row; the readers
filter on the target month; and lead_months is never the filter.
"""

from __future__ import annotations

from datetime import date

import pytest

pytest.importorskip("duckdb")


@pytest.fixture()
def db(tmp_path, monkeypatch):
    from pythia.db.schema import ensure_schema

    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{tmp_path / 'cf.duckdb'}")
    ensure_schema()
    return tmp_path / "cf.duckdb"


def _insert(rows):
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        for r in rows:
            con.execute(
                "INSERT INTO conflict_forecasts (source, iso3, hazard_code, metric, "
                "lead_months, value, forecast_issue_date, target_month, model_version) "
                "VALUES (?, ?, 'AC', ?, ?, ?, ?, ?, 'v')",
                list(r),
            )
    finally:
        con.close()


def _first_of_this_month() -> date:
    return date.today().replace(day=1)


def _add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    return date(d.year + y, m + 1, 1)


def test_stale_flags_are_stamped_on_every_row_from_each_sources_threshold(db):
    from pythia.db.schema import connect
    from resolver.tools.fetch_conflict_forecasts import stamp_vintage_flags

    _insert([
        ("ACLED_CAST", "SOM", "cast_total_events", 1, 10.0, date(2025, 12, 1), date(2026, 1, 1)),
        ("VIEWS", "SOM", "views_predicted_fatalities", 1, 5.0, date(2026, 8, 1), date(2026, 9, 1)),
    ])
    con = connect(read_only=False)
    try:
        stale = stamp_vintage_flags(con, today=date(2026, 9, 4))
        rows = con.execute(
            "SELECT source, is_stale_vintage, vintage_age_days FROM conflict_forecasts ORDER BY source"
        ).fetchall()
    finally:
        con.close()
    assert rows == [("ACLED_CAST", True, 277), ("VIEWS", False, 34)]
    assert stale == {"ACLED_CAST": 1, "VIEWS": 0}


def test_restamping_moves_with_the_calendar(db):
    from pythia.db.schema import connect
    from resolver.tools.fetch_conflict_forecasts import stamp_vintage_flags

    _insert([("VIEWS", "SOM", "views_predicted_fatalities", 1, 5.0, date(2026, 8, 1), date(2026, 9, 1))])
    con = connect(read_only=False)
    try:
        stamp_vintage_flags(con, today=date(2026, 9, 4))
        assert con.execute("SELECT is_stale_vintage FROM conflict_forecasts").fetchone()[0] is False
        stamp_vintage_flags(con, today=date(2026, 10, 4))
        assert con.execute("SELECT is_stale_vintage FROM conflict_forecasts").fetchone()[0] is True
    finally:
        con.close()


def test_no_reader_serves_a_row_whose_target_month_has_passed(db):
    """A frozen CAST vintage yields NO cast block; VIEWS keeps its future months only."""

    from horizon_scanner.conflict_forecasts import load_conflict_forecasts

    this_month = _first_of_this_month()
    last_month = _add_months(this_month, -1)
    _insert([
        # CAST: every target month in the past.
        ("ACLED_CAST", "SOM", "cast_total_events", 1, 10.0, _add_months(this_month, -8), _add_months(this_month, -7)),
        ("ACLED_CAST", "SOM", "cast_total_events", 2, 11.0, _add_months(this_month, -8), _add_months(this_month, -6)),
        # VIEWS: one past month, two current/future months.
        ("VIEWS", "SOM", "views_predicted_fatalities", 1, 5.0, _add_months(this_month, -2), last_month),
        ("VIEWS", "SOM", "views_predicted_fatalities", 2, 6.0, _add_months(this_month, -2), this_month),
        ("VIEWS", "SOM", "views_predicted_fatalities", 3, 7.0, _add_months(this_month, -2), _add_months(this_month, 1)),
        # conflictforecast.org: a 3-month risk whose target has passed.
        ("conflictforecast_org", "SOM", "cf_armed_conflict_risk_3m", 3, 0.4, _add_months(this_month, -6), _add_months(this_month, -3)),
    ])

    served = load_conflict_forecasts("SOM")
    assert served is not None
    assert "cast_total" not in served                  # nothing current to serve
    assert "cf_risk_3m" not in served
    leads = sorted(e["lead_months"] for e in served["views_fatalities"])
    assert leads == [2, 3]                             # the past month is gone, lead is not the filter


def test_the_served_predicate_is_on_target_month_not_lead(db):
    from horizon_scanner.conflict_forecasts import SERVED_TARGET_FILTER_SQL

    assert "target_month" in SERVED_TARGET_FILTER_SQL
    assert "lead_months" not in SERVED_TARGET_FILTER_SQL
