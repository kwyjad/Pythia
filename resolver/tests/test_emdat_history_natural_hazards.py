# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import duckdb
import pytest

from forecaster import cli as forecaster_cli  # type: ignore
from resolver.cli import emdat_to_duckdb as emdat_cli


@pytest.mark.db
def test_emdat_history_for_drought_pa(monkeypatch, tmp_path):
    db_path = tmp_path / "emdat_drought.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE emdat_pa (
                iso3 TEXT,
                shock_type TEXT,
                ym DATE,
                pa DOUBLE,
                source_id TEXT
            )
            """
        )
        con.execute(
            """
            INSERT INTO emdat_pa (iso3, shock_type, ym, pa, source_id) VALUES
                ('ETH', 'drought', DATE '2024-01-01', 10000.0, 'ev1'),
                ('ETH', 'drought', DATE '2024-02-01', 20000.0, 'ev2')
            """
        )
    finally:
        con.close()

    def fake_connect(read_only: bool = False):
        return duckdb.connect(str(db_path))

    monkeypatch.setattr(forecaster_cli, "connect", fake_connect)

    summary = forecaster_cli._build_history_summary("ETH", "DR", "PA")
    assert summary["source"] == "EM-DAT"
    assert summary["history_length_months"] == 2
    assert summary["recent_max"] == pytest.approx(20000.0)
    assert summary["last_6m_values"][-1]["value"] == pytest.approx(20000.0)


@pytest.mark.slow
def test_emdat_backfill_populates_emdat_pa_and_history_summary(tmp_path, monkeypatch):
    """Ensure EM-DAT backfill populates emdat_pa and surfaces in history summary."""

    db_path = tmp_path / "resolver_emdat_test.duckdb"
    db_url = f"duckdb:///{db_path}"

    rc = emdat_cli.run_emdat_pa_backfill(
        from_year=2024,
        to_year=2024,
        db_url=db_url,
        iso3_list=["ETH"],
        network=False,
    )
    assert rc == 0

    def fake_connect(read_only: bool = True):
        return duckdb.connect(str(db_path))

    monkeypatch.setattr(forecaster_cli, "connect", fake_connect)

    summary = forecaster_cli._build_history_summary("ETH", "FL", "PA")

    assert summary["source"] == "EM-DAT"
    assert summary["history_length_months"] >= 1
    assert summary["recent_max"] is not None
    assert summary["last_6m_values"]
