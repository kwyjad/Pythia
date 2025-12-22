# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import pytest

try:
    import duckdb
except ModuleNotFoundError:  # pragma: no cover - dependency may be absent in CI smoke runs
    pytest.skip("duckdb not installed", allow_module_level=True)

from horizon_scanner.horizon_scanner import _build_resolver_features_for_country


def test_build_resolver_features_handles_month_column(monkeypatch):
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month TEXT, fatalities INTEGER);"
    )
    con.execute(
        "INSERT INTO acled_monthly_fatalities VALUES ('ETH','2025-01', 10), ('ETH','2025-02', 20);"
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
            (DATE '2025-01-01', 'ETH', 'ACE', 'new_displacements', 100.0, 'new'),
            (DATE '2025-02-01', 'ETH', 'ACE', 'idp_displacement_new_dtm', 200.0, 'new'),
            (DATE '2025-03-01', 'ETH', 'ACE', 'idp_displacement_flow_idmc', 300.0, 'new');
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
            ('ETH', 'flood', DATE '2025-01-01', 20.0, 'src'),
            ('ETH', 'drought', DATE '2025-02-01', 5.0, 'src');
        """
    )

    import pythia.db.schema as schema_mod

    monkeypatch.setattr(schema_mod, "connect", lambda read_only=False: con)

    features = _build_resolver_features_for_country("ETH")

    assert "conflict" in features
    conflict = features["conflict"]
    assert conflict["history_length"] == 2
    assert conflict["recent_max"] == 20

    displacement = features["displacement"]
    assert displacement["history_length"] == 3
    assert displacement["recent_max"] == 300.0
    assert displacement["source"] == "IDMC/DTM"

    natural = features["natural_hazards"]
    assert set(natural.keys()) == {"FL", "DR", "TC", "HW"}
    assert natural["FL"]["history_length"] == 2
    assert natural["DR"]["history_length"] == 1
