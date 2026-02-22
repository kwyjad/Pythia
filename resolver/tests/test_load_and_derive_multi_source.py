# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Integration test: load_and_derive with multi-source canonical data."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from resolver.tools.load_and_derive import (
    CANONICAL_COLUMNS,
    PeriodMonths,
    _load_into_db,
    _read_canonical_dir,
)
from resolver.db.duckdb_io import init_schema


@pytest.fixture()
def canonical_dir(tmp_path: Path) -> Path:
    """Create a canonical directory with multi-source CSVs."""

    canonical = tmp_path / "staging" / "2024Q1" / "canonical"
    canonical.mkdir(parents=True)

    # IFRC GO data: stock semantics (PA data) — should go to facts_resolved
    ifrc_csv = textwrap.dedent("""\
        event_id,country_name,iso3,hazard_code,hazard_label,hazard_class,metric,unit,as_of_date,value,series_semantics,source
        IFRC-SDN-FL-2024-01,Sudan,SDN,FL,Flood,natural,affected,persons,2024-01-31,5000.0,stock,ifrc_go
        IFRC-SDN-FL-2024-02,Sudan,SDN,FL,Flood,natural,affected,persons,2024-02-29,7500.0,stock,ifrc_go
        IFRC-ETH-DR-2024-01,Ethiopia,ETH,DR,Drought,natural,affected,persons,2024-01-31,12000.0,stock,ifrc_go
    """)
    (canonical / "ifrc_go.csv").write_text(ifrc_csv, encoding="utf-8")

    # ACLED data: new semantics (flow data) — should go to facts_deltas
    acled_csv = textwrap.dedent("""\
        event_id,country_name,iso3,hazard_code,hazard_label,hazard_class,metric,unit,as_of_date,value,series_semantics,source
        ACLED-SDN-ACE-2024-01,Sudan,SDN,ACE,Armed Conflict Escalation,conflict,fatalities,persons,2024-01-31,150.0,new,acled
        ACLED-SDN-ACE-2024-02,Sudan,SDN,ACE,Armed Conflict Escalation,conflict,fatalities,persons,2024-02-28,200.0,new,acled
        ACLED-MMR-CU-2024-01,Myanmar,MMR,CU,Civil Unrest,unrest,events,events,2024-01-31,47.0,new,acled
    """)
    (canonical / "acled.csv").write_text(acled_csv, encoding="utf-8")

    # IDMC data: new semantics (displacement flows)
    idmc_csv = textwrap.dedent("""\
        event_id,country_name,iso3,hazard_code,hazard_label,hazard_class,metric,unit,as_of_date,value,series_semantics,source
        ,Sudan,SDN,IDU,Internal Displacement,displacement,new_displacements,persons,2024-01-31,800.0,new,idmc
        ,Congo DR,COD,IDU,Internal Displacement,displacement,new_displacements,persons,2024-02-29,500.0,new,idmc
    """)
    (canonical / "idmc.csv").write_text(idmc_csv, encoding="utf-8")

    return canonical


@pytest.fixture()
def db_conn():
    conn = duckdb.connect(":memory:")
    init_schema(conn)
    yield conn
    conn.close()


class TestLoadCanonicalDir:
    def test_reads_all_sources(self, canonical_dir: Path) -> None:
        combined = _read_canonical_dir(canonical_dir)
        assert len(combined) == 8  # 3 IFRC + 3 ACLED + 2 IDMC
        assert set(combined.columns) >= set(CANONICAL_COLUMNS)

    def test_series_semantics_normalized(self, canonical_dir: Path) -> None:
        combined = _read_canonical_dir(canonical_dir)
        valid_semantics = {"stock", "new"}
        assert set(combined["series_semantics"].unique()) <= valid_semantics

    def test_ym_column_derived(self, canonical_dir: Path) -> None:
        combined = _read_canonical_dir(canonical_dir)
        assert "ym" in combined.columns
        yms = set(combined["ym"].unique())
        assert "2024-01" in yms
        assert "2024-02" in yms


class TestLoadIntoDb:
    def test_stock_goes_to_facts_resolved(
        self, canonical_dir: Path, db_conn
    ) -> None:
        canonical = _read_canonical_dir(canonical_dir)
        counts = _load_into_db(db_conn, canonical)
        assert counts["facts_resolved"] > 0
        resolved = db_conn.execute("SELECT * FROM facts_resolved").df()
        # IFRC rows are stock → should appear in facts_resolved
        ifrc_rows = resolved[resolved["source_id"] == "ifrc_go"]
        assert len(ifrc_rows) == 3

    def test_new_goes_to_facts_deltas(
        self, canonical_dir: Path, db_conn
    ) -> None:
        canonical = _read_canonical_dir(canonical_dir)
        counts = _load_into_db(db_conn, canonical)
        assert counts["facts_deltas"] > 0
        deltas = db_conn.execute("SELECT * FROM facts_deltas").df()
        # ACLED + IDMC rows are new → should appear in facts_deltas
        assert len(deltas) == 5  # 3 ACLED + 2 IDMC

    def test_facts_raw_contains_all_rows(
        self, canonical_dir: Path, db_conn
    ) -> None:
        canonical = _read_canonical_dir(canonical_dir)
        counts = _load_into_db(db_conn, canonical)
        assert counts["facts_raw"] == 8

    def test_acled_metric_name_in_deltas(
        self, canonical_dir: Path, db_conn
    ) -> None:
        """Verify ACLED fatalities appear with canonical metric name."""
        canonical = _read_canonical_dir(canonical_dir)
        _load_into_db(db_conn, canonical)
        deltas = db_conn.execute("SELECT * FROM facts_deltas").df()
        acled_fatalities = deltas[deltas["metric"] == "fatalities"]
        assert len(acled_fatalities) == 2

    def test_idmc_metric_name_in_deltas(
        self, canonical_dir: Path, db_conn
    ) -> None:
        """Verify IDMC displacement data uses expected metric name."""
        canonical = _read_canonical_dir(canonical_dir)
        _load_into_db(db_conn, canonical)
        deltas = db_conn.execute("SELECT * FROM facts_deltas").df()
        idmc_rows = deltas[deltas["metric"] == "new_displacements"]
        assert len(idmc_rows) == 2
