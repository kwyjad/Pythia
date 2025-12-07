from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

import forecaster.cli as cli  # type: ignore


def test_build_history_summary_emdat_uses_pa_column(monkeypatch: pytest.MonkeyPatch) -> None:
    """_build_history_summary should read EM-DAT PA from the `pa` column."""
    # Prepare an in-memory DuckDB with a minimal emdat_pa table.
    con = duckdb.connect(":memory:")

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
    # Two months of synthetic PA data for ETH/flood.
    con.execute(
        """
        INSERT INTO emdat_pa (iso3, shock_type, ym, pa, source_id) VALUES
            ('ETH', 'flood', DATE '2024-01-01', 10.0, 'src'),
            ('ETH', 'flood', DATE '2024-02-01', 20.0, 'src')
        """
    )

    # Monkeypatch cli.connect to use our in-memory DB, ignoring read_only.
    def fake_connect(read_only: bool = False):
        return con

    monkeypatch.setattr(cli, "connect", fake_connect)

    summary = cli._build_history_summary("ETH", "FL", "PA")

    assert summary["source"] == "EM-DAT"
    assert summary["history_length_months"] == 2
    # The most recent value should reflect the last row we inserted.
    assert summary["recent_max"] == pytest.approx(20.0)
    assert summary["last_6m_values"][-1]["value"] == pytest.approx(20.0)


def test_build_history_summary_idmc_conflict_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """IDMC history should come from conflict flow metrics with hazard filter."""

    con = duckdb.connect(":memory:")

    con.execute(
        """
        CREATE TABLE facts_deltas (
            ym DATE,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            value_new DOUBLE,
            series_semantics TEXT
        )
        """
    )

    con.execute(
        """
        INSERT INTO facts_deltas (ym, iso3, hazard_code, metric, value_new, series_semantics) VALUES
            (DATE '2024-01-01', 'ETH', 'ACE', 'new_displacements', 1000.0, 'new'),
            (DATE '2024-02-01', 'ETH', 'ACE', 'idp_displacement_new_dtm', 2000.0, 'new'),
            (DATE '2024-03-01', 'ETH', 'ACE', 'new_displacements', 2000.0, 'new'),
            (DATE '2024-03-01', 'ETH', 'ACE', 'idp_displacement_flow_idmc', 500.0, 'new')
        """
    )

    def fake_connect(read_only: bool = False):
        return con

    monkeypatch.setattr(cli, "connect", fake_connect)

    summary = cli._build_history_summary("ETH", "ACE", "PA")

    assert summary["source"] == "IDMC"
    assert summary["history_length_months"] == 3
    assert summary["recent_max"] == pytest.approx(2500.0)
    assert summary["last_6m_values"][-1]["value"] == pytest.approx(2500.0)


def test_build_history_summary_di_has_no_base_rate() -> None:
    """DI should explicitly return no Resolver base rate."""

    summary = cli._build_history_summary("ETH", "DI", "PA")

    assert summary["source"] == "NONE"
    assert summary["history_length_months"] == 0
    assert "no resolver base rate" in summary["notes"].lower()
