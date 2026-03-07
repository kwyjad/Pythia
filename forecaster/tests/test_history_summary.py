# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

import forecaster.cli as cli  # type: ignore


def test_build_history_summary_ifrc_uses_facts_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    """_build_history_summary should read IFRC PA from the facts_resolved table."""
    # Prepare an in-memory DuckDB with a minimal facts_resolved table.
    con = duckdb.connect(":memory:")

    con.execute(
        """
        CREATE TABLE facts_resolved (
            iso3 TEXT,
            hazard_code TEXT,
            ym DATE,
            metric TEXT,
            value DOUBLE,
            source_id TEXT
        )
        """
    )
    # Two months of synthetic PA data for ETH/FL.
    con.execute(
        """
        INSERT INTO facts_resolved (iso3, hazard_code, ym, metric, value, source_id) VALUES
            ('ETH', 'FL', DATE '2024-01-01', 'affected', 10.0, 'ifrc'),
            ('ETH', 'FL', DATE '2024-02-01', 'affected', 20.0, 'ifrc')
        """
    )

    # Monkeypatch cli.connect to use our in-memory DB, ignoring read_only.
    def fake_connect(read_only: bool = False):
        return con

    monkeypatch.setattr(cli, "connect", fake_connect)

    summary = cli._build_history_summary("ETH", "FL", "PA")

    assert summary["type"] == "seasonal_profile"
    assert summary["source"] == "IFRC"
    assert summary["years_of_data"] == 1
    # January and February should have observations
    assert summary["months"][1]["n_observations"] == 1
    assert summary["months"][1]["max"] == pytest.approx(10.0)
    assert summary["months"][2]["n_observations"] == 1
    assert summary["months"][2]["max"] == pytest.approx(20.0)
    # Months without data should be zero-filled
    assert summary["months"][3]["n_observations"] == 0


def test_build_history_summary_idmc_conflict_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """IDMC history should come from conflict flow metrics with hazard filter."""

    con = duckdb.connect(":memory:")

    # Need both tables since _build_conflict_base_rate queries both
    con.execute(
        """
        CREATE TABLE acled_monthly_fatalities (
            iso3 TEXT, month DATE, fatalities BIGINT,
            source TEXT, updated_at TIMESTAMP
        )
        """
    )

    con.execute(
        """
        CREATE TABLE facts_deltas (
            ym DATE,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            value_new DOUBLE,
            series_semantics TEXT,
            source_id TEXT
        )
        """
    )

    con.execute(
        """
        INSERT INTO facts_deltas (ym, iso3, hazard_code, metric, value_new, series_semantics, source_id) VALUES
            (DATE '2024-01-01', 'ETH', 'ACE', 'new_displacements', 1000.0, 'new', 'idmc'),
            (DATE '2024-02-01', 'ETH', 'ACE', 'new_displacements', 2000.0, 'new', 'idmc'),
            (DATE '2024-03-01', 'ETH', 'ACE', 'new_displacements', 2000.0, 'new', 'idmc'),
            (DATE '2024-03-01', 'ETH', 'ACE', 'new_displacements', 500.0, 'new', 'idmc')
        """
    )

    def fake_connect(read_only: bool = False):
        return con

    monkeypatch.setattr(cli, "connect", fake_connect)

    summary = cli._build_history_summary("ETH", "ACE", "PA")

    assert summary["type"] == "conflict_trajectory"
    # Displacements should have data
    disp = summary["displacements"]
    assert disp["source"] == "IDMC"
    assert disp["last_month"] is not None
    assert len(disp["last_6m"]) == 3
    assert disp["last_6m"][-1]["value"] == pytest.approx(2500.0)


def test_build_history_summary_di_has_no_base_rate() -> None:
    """DI should explicitly return no Resolver base rate."""

    summary = cli._build_history_summary("ETH", "DI", "PA")

    assert summary["type"] == "no_base_rate"
    assert summary["source"] == "NONE"
    assert "no resolver base rate" in summary["note"].lower()
