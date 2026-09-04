# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The structured-data orchestrator reports the rows its writers wrote (G1).

``counts.written`` was a COUNTRY count for per-country sources and a row
count for self-storing ones, and no writer reported the rows it actually
inserted or replaced. The reconciliation compared that number against a
table delta, which an idempotent upsert never moves.
"""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from pythia.tools import ingest_structured_data as ingest  # noqa: E402


def test_store_all_sums_the_rows_each_store_reports(monkeypatch):
    import pythia.acaps as acaps

    monkeypatch.setattr(acaps, "store_risk_radar", lambda iso3, data: len(data["risks"]))
    stats = ingest._store_all(
        "acaps_risk_radar",
        {"SOM": {"risks": [1, 2, 3]}, "KEN": {"risks": [1]}},
    )
    assert stats["success"] == 2
    assert stats["rows_written"] == 4


def test_a_self_storing_source_reports_its_merge_count_through_the_sentinel():
    stats = ingest._store_all(
        "nmme_seasonal_forecasts", {"__nmme_done__": True, "__rows_written__": 2408},
    )
    assert stats["rows_written"] == 2408
    assert ingest._self_stored_rows({"__conflict_rows__": 546}) == 546
    assert ingest._self_stored_rows({"__gdelt_rows__": 752}) == 752
    assert ingest._self_stored_rows({"__pipeline_resolved_rows__": 17}) == 17
    assert ingest._self_stored_rows({"__nmme_done__": True}) == 0


def test_the_nmme_fetch_carries_the_upsert_count(monkeypatch):
    import resolver.tools.ingest_nmme as nmme

    monkeypatch.setattr(nmme, "main", lambda argv: {"rows_written": 2408, "rows_before": 4816, "rows_after": 4816})
    assert ingest._bulk_fetch_nmme() == {"__nmme_done__": True, "__rows_written__": 2408}


def test_a_real_store_returns_the_rows_it_wrote(tmp_path, monkeypatch):
    from pythia.acaps import store_risk_radar

    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{tmp_path / 'p.duckdb'}")
    data = {"risks": [{"risk_id": "r1", "title": "Drought"}, {"risk_id": "r2", "title": "Flood"}]}
    assert store_risk_radar("SOM", data) == 2
    # A re-store of the same risks is an idempotent replace: still two rows
    # reported, and the table holds two.
    assert store_risk_radar("SOM", data) == 2
    from pythia.db.schema import connect

    assert connect().execute("SELECT COUNT(*) FROM acaps_risk_radar").fetchone()[0] == 2
    assert store_risk_radar("SOM", {"risks": []}) == 0
