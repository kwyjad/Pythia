# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ACE/PA base-rate contract.

This test asserted the pre-#816 behaviour, where ACE/PA routed to
_load_idmc_conflict_flow_history_summary and returned a flat dict keyed
`source`/`history_length_months`. #816 deliberately re-routed ACE/PA to
_build_conflict_base_rate so it renders a curated block instead of a raw JSON
dump, which changed the shape to a `conflict_trajectory` carrying both series.
The test kept asserting the old keys and has been failing on main ever since —
invisibly, because it lives under resolver/tests while the code it covers is in
forecaster/, and resolver-ci-fast does not trigger on forecaster/**.
"""

from __future__ import annotations

import duckdb
import pytest

from forecaster import cli as forecaster_cli  # type: ignore


def _seed(db_path):
    """facts_deltas with the columns the conflict builder actually selects."""
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
                value_new DOUBLE,
                source_id TEXT
            )
            """
        )
        for month in range(1, 9):
            con.execute(
                """
                INSERT INTO facts_deltas
                    (ym, iso3, hazard_code, metric, series_semantics, value_new, source_id)
                VALUES (?, 'ETH', 'ACE', 'idp_displacement_flow_idmc', 'new', ?, 'idmc')
                """,
                [f"2024-{month:02d}-01", 1000.0 * month],
            )
        con.execute(
            """
            CREATE TABLE acled_monthly_fatalities (
                ym TEXT, iso3 TEXT, fatalities DOUBLE
            )
            """
        )
    finally:
        con.close()


@pytest.mark.db
def test_ace_pa_returns_a_conflict_trajectory(monkeypatch, tmp_path):
    """ACE/PA must carry a `type`, or the prompt renderer falls back to a raw
    JSON dump — the exact regression #816 fixed."""
    db_path = tmp_path / "idmc_conflict_pa.duckdb"
    _seed(db_path)
    monkeypatch.setattr(forecaster_cli, "connect", lambda read_only=False: duckdb.connect(str(db_path)))

    summary = forecaster_cli._build_history_summary("ETH", "ACE", "PA")

    assert summary["type"] == "conflict_trajectory"
    assert "displacements" in summary and "fatalities" in summary


@pytest.mark.db
def test_ace_pa_displacement_series_comes_from_idmc(monkeypatch, tmp_path):
    """PA resolves against IDMC displacement, so that series must be populated."""
    db_path = tmp_path / "idmc_conflict_pa.duckdb"
    _seed(db_path)
    monkeypatch.setattr(forecaster_cli, "connect", lambda read_only=False: duckdb.connect(str(db_path)))

    disp = forecaster_cli._build_history_summary("ETH", "ACE", "PA")["displacements"]

    assert disp["source"] == "IDMC"
    assert disp["last_6m"], "IDMC displacement rows should have been picked up"
    assert disp["last_6m"][-1]["value"] == pytest.approx(8000.0)


@pytest.mark.db
def test_ace_fatalities_shares_the_same_builder(monkeypatch, tmp_path):
    """Both ACE metrics route through one trajectory builder (#816)."""
    db_path = tmp_path / "idmc_conflict_pa.duckdb"
    _seed(db_path)
    monkeypatch.setattr(forecaster_cli, "connect", lambda read_only=False: duckdb.connect(str(db_path)))

    assert (
        forecaster_cli._build_history_summary("ETH", "ACE", "FATALITIES")["type"]
        == "conflict_trajectory"
    )
