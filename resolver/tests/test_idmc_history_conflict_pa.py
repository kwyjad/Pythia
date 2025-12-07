from __future__ import annotations

import duckdb
import pytest

from forecaster import cli as forecaster_cli  # type: ignore


@pytest.mark.db
def test_idmc_history_for_conflict_pa(monkeypatch, tmp_path):
    db_path = tmp_path / "idmc_conflict_pa.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE facts_deltas (
                ym DATE,
                iso3 TEXT,
                hazard_code TEXT,
                metric TEXT,
                series_semantics TEXT,
                value_new DOUBLE
            )
            """
        )
        for month in range(1, 9):
            con.execute(
                """
                INSERT INTO facts_deltas (ym, iso3, hazard_code, metric, series_semantics, value_new)
                VALUES (DATE '2024-0'||?||'-01', 'ETH', 'ACE', 'idp_displacement_flow_idmc', 'new', ?)
                """,
                [month, 1000.0 * month],
            )
    finally:
        con.close()

    def fake_connect(read_only: bool = False):
        return duckdb.connect(str(db_path))

    monkeypatch.setattr(forecaster_cli, "connect", fake_connect)

    summary = forecaster_cli._build_history_summary("ETH", "ACE", "PA")
    assert summary["source"] == "IDMC"
    assert summary["history_length_months"] == 8
    assert summary["recent_max"] == pytest.approx(8000.0)
    assert summary["recent_mean"] == pytest.approx(sum(range(3000, 9000, 1000)) / 6)
    assert len(summary["last_6m_values"]) == 6
