# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json

import pytest

duckdb = pytest.importorskip("duckdb")

from horizon_scanner import horizon_scanner as hs_mod


def test_hazard_catalog_excludes_conflict_aliases():
    catalog = hs_mod._build_hazard_catalog()
    banned = {"CONFLICT", "POLITICAL_VIOLENCE", "CIVIL_CONFLICT", "URBAN_CONFLICT"}
    assert not banned & set(catalog.keys())


def test_write_hs_triage_filters_unknown_hazards(monkeypatch, tmp_path):
    db_path = tmp_path / "hs.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE hs_triage (
            run_id TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            tier TEXT,
            triage_score DOUBLE,
            need_full_spd BOOLEAN,
            drivers_json TEXT,
            regime_shifts_json TEXT,
            data_quality_json TEXT,
            scenario_stub TEXT,
            is_test BOOLEAN DEFAULT FALSE
        );
        """
    )
    con.close()

    def fake_connect(read_only=False):  # noqa: ARG001 - signature parity only
        # Ignore read_only, exactly like pythia.db.schema.connect does: mixing
        # read-only and read-write connections to one path in a single process
        # raises DuckDB's "different configuration" error. _write_hs_triage
        # takes both kinds (the grounding gate reads, the writer writes), so a
        # read_only-honouring double made this test order-dependent.
        return duckdb.connect(str(db_path), read_only=False)

    monkeypatch.setattr(hs_mod, "pythia_connect", fake_connect)

    hs_mod._write_hs_triage(
        "run1",
        "ETH",
        {
            "hazards": {
                "ACE": {"tier": "priority", "triage_score": 0.8, "drivers": ["x"]},
                "CONFLICT": {"tier": "priority", "triage_score": 0.9},
            }
        },
    )

    with duckdb.connect(str(db_path), read_only=True) as con_check:
        rows = con_check.execute("SELECT hazard_code, drivers_json FROM hs_triage").fetchall()

    # Only the ACTIVE catalog is written: DI / CU / HW are silenced at the
    # hazard-catalog level (BLOCKED_HAZARDS in db_writer), so the pipeline
    # writes exactly ACE / DR / FL / TC per country.
    codes = {r[0] for r in rows}
    assert codes == {"ACE", "DR", "FL", "TC"}
    assert "CONFLICT" not in codes
    assert ("ACE", json.dumps(["x"])) in rows
    # Non-LLM hazards get empty driver lists
    for code in ("DR", "FL", "TC"):
        assert (code, "[]") in rows
