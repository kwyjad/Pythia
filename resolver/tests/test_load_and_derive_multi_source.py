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
    _derive_deltas,
    _load_into_db,
    _read_canonical_dir,
    delete_months,
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


class TestPartialConnectorLoadPreservesOtherSources:
    """Regression: loading a single connector must not wipe other sources.

    This simulates the scenario where the canonical DB already has emdat
    and idmc data, and then only IFRC data is loaded in a partial run.
    Pre-existing rows from emdat/idmc must survive.
    """

    def test_partial_load_preserves_facts_resolved(self, db_conn) -> None:
        """facts_resolved rows from other sources survive an IFRC-only load."""
        # Seed the DB with pre-existing emdat stock rows
        init_schema(db_conn)
        db_conn.execute(
            """
            INSERT INTO facts_resolved
                (ym, iso3, hazard_code, hazard_label, hazard_class, metric,
                 series_semantics, value, unit, as_of_date, source_id, event_id)
            VALUES
                ('2024-01', 'SDN', 'FL', 'Flood', 'natural', 'affected',
                 'stock', 3000, 'persons', '2024-01-31', 'emdat', 'EM-SDN-1'),
                ('2024-02', 'SDN', 'FL', 'Flood', 'natural', 'affected',
                 'stock', 4000, 'persons', '2024-02-28', 'emdat', 'EM-SDN-2')
            """
        )
        before = db_conn.execute(
            "SELECT COUNT(*) FROM facts_resolved"
        ).fetchone()[0]
        assert before == 2

        # Now load only IFRC canonical data (partial connector run)
        ifrc_csv = textwrap.dedent("""\
            event_id,country_name,iso3,hazard_code,hazard_label,hazard_class,metric,unit,as_of_date,value,series_semantics,source
            IFRC-SDN-FL-2024-01,Sudan,SDN,FL,Flood,natural,affected,persons,2024-01-31,5000.0,stock,ifrc_go
        """)
        canonical_dir = Path(db_conn.execute("SELECT current_setting('temp_directory')").fetchone()[0]) / "canonical"
        canonical_dir.mkdir(parents=True, exist_ok=True)
        (canonical_dir / "ifrc_go.csv").write_text(ifrc_csv, encoding="utf-8")
        canonical = _read_canonical_dir(canonical_dir)

        _load_into_db(db_conn, canonical, loaded_sources=["ifrc_go"])

        resolved = db_conn.execute("SELECT * FROM facts_resolved").df()
        emdat_rows = resolved[resolved["source_id"] == "emdat"]
        ifrc_rows = resolved[resolved["source_id"] == "ifrc_go"]

        # emdat rows must survive
        assert len(emdat_rows) == 2, (
            f"Expected 2 emdat rows to survive, got {len(emdat_rows)}"
        )
        # IFRC rows must be present
        assert len(ifrc_rows) == 1

    def test_partial_load_preserves_facts_deltas(self, db_conn) -> None:
        """facts_deltas rows from other sources survive an IFRC-only load."""
        init_schema(db_conn)
        # Seed with pre-existing idmc delta rows
        db_conn.execute(
            """
            INSERT INTO facts_deltas
                (ym, iso3, hazard_code, metric, value_new, value_stock,
                 series_semantics, as_of, source_id, first_observation,
                 rebase_flag, delta_negative_clamped)
            VALUES
                ('2024-01', 'SDN', 'IDU', 'new_displacements', 800, 800,
                 'new', '2024-01-31', 'idmc', 1, 0, 0),
                ('2024-02', 'COD', 'IDU', 'new_displacements', 500, 500,
                 'new', '2024-02-29', 'idmc', 1, 0, 0)
            """
        )
        before = db_conn.execute(
            "SELECT COUNT(*) FROM facts_deltas"
        ).fetchone()[0]
        assert before == 2

        # Derive deltas for only IFRC (no IFRC stock rows yet → 0 derived)
        period = PeriodMonths.from_label("2024Q1")
        derived = _derive_deltas(
            db_conn, period, allow_negatives=True, sources=["ifrc_go"]
        )
        assert derived == 0

        # Pre-existing idmc deltas must still be there
        after = db_conn.execute(
            "SELECT COUNT(*) FROM facts_deltas WHERE source_id = 'idmc'"
        ).fetchone()[0]
        assert after == 2, (
            f"Expected 2 idmc delta rows to survive, got {after}"
        )

    def test_delete_months_with_sources_only_deletes_matching(
        self, db_conn
    ) -> None:
        """delete_months(sources=...) leaves rows from other sources."""
        init_schema(db_conn)
        db_conn.execute(
            """
            INSERT INTO facts_resolved
                (ym, iso3, hazard_code, metric, series_semantics,
                 value, unit, as_of_date, source_id, event_id)
            VALUES
                ('2024-01', 'SDN', 'FL', 'affected', 'stock',
                 100, 'persons', '2024-01-31', 'emdat', 'E1'),
                ('2024-01', 'SDN', 'FL', 'affected', 'stock',
                 200, 'persons', '2024-01-31', 'ifrc_go', 'I1'),
                ('2024-01', 'SDN', 'IDU', 'new_displacements', 'stock',
                 300, 'persons', '2024-01-31', 'idmc', 'D1')
            """
        )
        delete_months(
            db_conn, "facts_resolved", ["2024-01"], sources=["ifrc_go"]
        )
        remaining = db_conn.execute(
            "SELECT source_id FROM facts_resolved ORDER BY source_id"
        ).df()
        assert sorted(remaining["source_id"].tolist()) == ["emdat", "idmc"]
