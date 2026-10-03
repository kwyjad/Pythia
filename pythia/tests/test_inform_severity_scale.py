# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""INFORM Severity: the right scale and one quantity in the trend (Oct 2026).

The prompt printed "9.2/5.0". The index runs 0 to 10 (the API, 2026-10-03:
Afghanistan 9.2, category numeric 5). The trend table held the country-log's
component indicators beside the index: Afghanistan 652230.0, 1.82 and 9.2.
"""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from pythia import acaps


def test_the_prompt_prints_the_ten_point_scale():
    text = acaps.format_inform_severity_for_prompt({
        "iso3": "AFG", "severity_score": 9.2, "severity_category": "Very High",
        "snapshot_date": "Sep2026", "impact_score": 9.578423686825111,
        "conditions_score": 9.1, "complexity_score": 8.4,
    })
    assert "Overall: 9.2/10 (Very High)" in text
    assert "Impact 9.6/10" in text
    assert "/5" not in text
    spd = acaps.format_inform_severity_for_spd({"iso3": "AFG", "severity_score": 9.2})
    assert "9.2/10" in spd


def test_deltas_compare_like_snapshots_only():
    trend = [
        {"date": "2026-06-01", "score": 8.6},
        {"date": "2026-08-01", "score": 9.0},
        {"date": "2024-01-29", "score": 652230.0},  # a component indicator
    ]
    d1, d3 = acaps.severity_deltas(9.2, "Sep2026", trend)
    assert d1 == pytest.approx(0.2)
    assert d3 == pytest.approx(0.6)
    # No snapshot a month back: no delta, not a difference against the last row.
    d1, d3 = acaps.severity_deltas(9.2, "Sep2026", [{"date": "2026-06-01", "score": 8.6}])
    assert d1 is None and d3 == pytest.approx(0.6)


def test_store_purges_mixed_rows_and_keeps_the_index(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{tmp_path / 'p.duckdb'}")
    from pythia.db.schema import connect, ensure_schema

    con = connect(read_only=False)
    ensure_schema(con)
    con.execute(
        "INSERT INTO acaps_inform_severity_trend (iso3, snapshot_date, score, fetched_at) VALUES "
        "('AFG', '2019-01-01', 3.0166, 'x'), ('AFG', '2024-01-29', 652230.0, 'x')"
    )
    con.close()
    acaps.store_inform_severity("AFG", {
        "iso3": "AFG", "severity_score": 9.2, "snapshot_date": "Sep2026",
        "trend_6m": [
            {"date": "2026-08-01", "score": 9.0},
            {"date": "2026-07-01", "score": 652230.0},
        ],
    })
    con = connect(read_only=False)
    rows = con.execute(
        "SELECT snapshot_date, score, source FROM acaps_inform_severity_trend ORDER BY 1"
    ).fetchall()
    con.close()
    assert rows == [("2026-08-01", 9.0, "monthly_snapshot")]
    loaded = acaps.load_inform_severity("AFG")
    assert loaded["delta_1m"] == pytest.approx(0.2)
