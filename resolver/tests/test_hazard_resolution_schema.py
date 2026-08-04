# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The haz_* migration: clean against a copy of the resolver DB, idempotent,
and enforcing the spec'd enum/uniqueness constraints at the DB level."""

from __future__ import annotations

import duckdb
import pytest

from resolver.db import duckdb_io
from resolver.hazard_resolution import migrate as haz_migrate
from resolver.hazard_resolution.schema import (
    RAW_SOURCES,
    ensure_haz_schema,
    haz_table_names,
    raw_table_name,
)


@pytest.fixture()
def resolver_db_copy(tmp_path):
    """A file-backed copy of the production resolver schema with a probe row."""

    path = tmp_path / "resolver_copy.duckdb"
    conn = duckdb.connect(str(path))
    duckdb_io.init_schema(conn)
    conn.execute(
        """
        INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, series_semantics, value)
        VALUES ('2026-01', 'SOM', 'FL', 'affected', 'stock', 12345)
        """
    )
    conn.close()
    return path


def _tables_in(path) -> set[str]:
    conn = duckdb.connect(str(path))
    try:
        return {row[0] for row in conn.execute("PRAGMA show_tables").fetchall()}
    finally:
        conn.close()


def test_migration_runs_cleanly_against_resolver_db_copy(resolver_db_copy):
    counts = haz_migrate.migrate(str(resolver_db_copy))

    expected = set(haz_table_names())
    assert set(counts) == expected
    assert all(count == 0 for count in counts.values())
    # One raw cache per source, population included.
    assert {raw_table_name(source) for source in RAW_SOURCES} <= expected

    tables = _tables_in(resolver_db_copy)
    assert expected <= tables

    # The existing resolver data is untouched.
    conn = duckdb.connect(str(resolver_db_copy))
    try:
        probe = conn.execute(
            "SELECT value FROM facts_resolved WHERE ym='2026-01' AND iso3='SOM'"
        ).fetchone()
    finally:
        conn.close()
    assert probe == (12345,)


def test_migration_is_idempotent(resolver_db_copy):
    first = haz_migrate.migrate(str(resolver_db_copy))

    # Write a row, re-run, and confirm nothing is dropped or duplicated.
    conn = duckdb.connect(str(resolver_db_copy))
    try:
        conn.execute(
            """
            INSERT INTO haz_triggers (iso3, year, month, hazard, triggered, trigger_source)
            VALUES ('PHL', 2026, 7, 'TC', TRUE, 'ibtracs')
            """
        )
    finally:
        conn.close()

    second = haz_migrate.migrate(str(resolver_db_copy))
    assert set(first) == set(second)
    assert second["haz_triggers"] == 1


def test_migrate_cli_entrypoint(resolver_db_copy, capsys):
    assert haz_migrate.main(["--db", str(resolver_db_copy)]) == 0
    out = capsys.readouterr().out
    assert "[haz-migrate] ok" in out
    assert "haz_resolutions" in out


def test_enum_and_uniqueness_constraints(tmp_path):
    path = tmp_path / "haz.duckdb"
    conn = duckdb.connect(str(path))
    try:
        ensure_haz_schema(conn)

        # Valid resolution row inserts fine.
        conn.execute(
            """
            INSERT INTO haz_resolutions (iso3, year, month, hazard, status, value, provenance_json, rule_fired)
            VALUES ('PHL', 2026, 7, 'TC', 'RESOLVED_VALUE', 80000, '{}', 'ladder:emdat')
            """
        )

        # The spec'd enums are enforced at the DB level.
        with pytest.raises(duckdb.Error, match="[Cc]onstraint"):
            conn.execute(
                """
                INSERT INTO haz_resolutions (iso3, year, month, hazard, status, provenance_json, rule_fired)
                VALUES ('PHL', 2026, 8, 'TC', 'MAYBE', '{}', 'x')
                """
            )
        with pytest.raises(duckdb.Error, match="[Cc]onstraint"):
            conn.execute(
                """
                INSERT INTO haz_impact_candidates (iso3, year, month, hazard, value, value_type, source)
                VALUES ('PHL', 2026, 7, 'TC', 5000, 'guessed', 'emdat')
                """
            )

        # One resolution per country-month-hazard.
        with pytest.raises(duckdb.Error, match="[Cc]onstraint"):
            conn.execute(
                """
                INSERT INTO haz_resolutions (iso3, year, month, hazard, status, provenance_json, rule_fired)
                VALUES ('PHL', 2026, 7, 'TC', 'NO_DATA', '{}', 'freeze_passed')
                """
            )

        # Hazard codes outside FL/DR/TC are rejected.
        with pytest.raises(duckdb.Error, match="[Cc]onstraint"):
            conn.execute(
                """
                INSERT INTO haz_triggers (iso3, year, month, hazard, triggered)
                VALUES ('PHL', 2026, 7, 'HW', FALSE)
                """
            )
    finally:
        conn.close()
