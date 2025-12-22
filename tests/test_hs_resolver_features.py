# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import pytest

duckdb = pytest.importorskip("duckdb")

import pythia.db.schema as schema_mod
from horizon_scanner import horizon_scanner as hs_mod


def test_build_resolver_features_handles_acled_month(monkeypatch, tmp_path):
    db_path = tmp_path / "resolver.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month TEXT, fatalities INTEGER);"
    )
    con.execute(
        "INSERT INTO acled_monthly_fatalities VALUES ('ETH','2025-01',10),('ETH','2025-02',20);"
    )
    con.execute(
        """
        CREATE TABLE facts_deltas (
            ym DATE,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            value_new DOUBLE,
            series_semantics TEXT
        );
        """
    )
    con.execute(
        """
        INSERT INTO facts_deltas (ym, iso3, hazard_code, metric, value_new, series_semantics) VALUES
            (DATE '2025-01-01', 'ETH', 'ACE', 'new_displacements', 50.0, 'new'),
            (DATE '2025-02-01', 'ETH', 'ACE', 'idp_displacement_flow_idmc', 75.0, 'new');
        """
    )
    con.execute(
        """
        CREATE TABLE emdat_pa (
            iso3 TEXT,
            shock_type TEXT,
            ym DATE,
            pa DOUBLE,
            source_id TEXT
        );
        """
    )
    con.execute(
        """
        INSERT INTO emdat_pa (iso3, shock_type, ym, pa, source_id) VALUES
            ('ETH', 'flood', DATE '2024-12-01', 10.0, 'src'),
            ('ETH', 'heat_wave', DATE '2025-01-01', 2.0, 'src');
        """
    )
    con.close()

    def fake_connect(read_only=False):
        return duckdb.connect(str(db_path), read_only=read_only)

    monkeypatch.setattr(schema_mod, "connect", fake_connect)

    feats = hs_mod._build_resolver_features_for_country("ETH")
    assert "conflict" in feats
    conf = feats["conflict"]
    assert conf["source"] == "ACLED"
    assert conf["history_length"] == 2
    assert conf["recent_max"] == 20

    disp = feats["displacement"]
    assert disp["source"] == "IDMC/DTM"
    assert disp["history_length"] == 2
    assert disp["recent_max"] == 75.0

    nat = feats["natural_hazards"]
    assert nat["FL"]["history_length"] == 1
    assert nat["HW"]["history_length"] == 1
    assert nat["DR"]["history_length"] == 0
