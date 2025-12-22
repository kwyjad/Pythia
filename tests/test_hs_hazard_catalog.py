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
            scenario_stub TEXT
        );
        """
    )
    con.close()

    def fake_connect(read_only=False):
        return duckdb.connect(str(db_path), read_only=read_only)

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

    assert rows == [("ACE", json.dumps(["x"]))]
