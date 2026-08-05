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


def test_provisional_column_exists_and_defaults_to_final():
    """Phase 2 added `provisional`; the resolver core has no such concept."""
    import duckdb

    from resolver.hazard_resolution.schema import ensure_haz_schema

    con = duckdb.connect(":memory:")
    ensure_haz_schema(con)
    columns = [d[0] for d in con.execute("SELECT * FROM haz_resolutions").description]
    assert "provisional" in columns

    con.execute(
        "INSERT INTO haz_resolutions "
        "(iso3, year, month, hazard, status, value, provenance_json, rule_fired) "
        "VALUES ('PHL', 2024, 3, 'FL', 'RESOLVED_ZERO', 0, '{}', 'r')"
    )
    assert con.execute("SELECT provisional FROM haz_resolutions").fetchone()[0] is False


def test_column_migration_widens_a_pre_phase2_table_idempotently():
    """A Phase-1 DB must gain `provisional` without losing its rows."""
    import duckdb

    from resolver.hazard_resolution.schema import _CORE_TABLE_DDL, ensure_haz_schema

    con = duckdb.connect(":memory:")
    legacy = _CORE_TABLE_DDL["haz_resolutions"].replace(
        "provisional BOOLEAN NOT NULL DEFAULT FALSE,\n", ""
    )
    assert "provisional" not in legacy
    con.execute(legacy)
    con.execute(
        "INSERT INTO haz_resolutions "
        "(iso3, year, month, hazard, status, value, provenance_json, rule_fired) "
        "VALUES ('PHL', 2013, 11, 'TC', 'RESOLVED_ZERO', 0, '{}', 'r')"
    )

    ensure_haz_schema(con)
    ensure_haz_schema(con)  # idempotent

    # The row survives, and the backfill leaves no NULL behind.
    assert con.execute(
        "SELECT iso3, provisional FROM haz_resolutions"
    ).fetchall() == [("PHL", False)]
    assert con.execute(
        "SELECT COUNT(*) FROM haz_resolutions WHERE provisional IS NULL"
    ).fetchone()[0] == 0


def test_a_fresh_connection_is_always_migrated_even_after_others_are_collected():
    """The ensure-memo must not alias a recycled connection.

    ensure_haz_schema memoizes per connection so a 200-country run does not
    re-run the DDL (and the migration's backfill scan) once per cell. Keying
    that memo on id(conn) is unsafe: CPython recycles the id of a collected
    connection, and an aliased id skips the migration on a genuinely new
    connection — leaving it without the `provisional` column. The memo is a
    WeakSet of the connection objects for exactly this reason.
    """
    import gc

    import duckdb

    from resolver.hazard_resolution.schema import _CORE_TABLE_DDL, ensure_haz_schema

    legacy_ddl = _CORE_TABLE_DDL["haz_resolutions"].replace(
        "provisional BOOLEAN NOT NULL DEFAULT FALSE,\n", ""
    )

    seen_ids = set()
    for _ in range(50):
        con = duckdb.connect(":memory:")
        seen_ids.add(id(con))
        con.execute(legacy_ddl)  # a pre-Phase-2 table on a brand-new connection
        ensure_haz_schema(con)
        columns = [d[0] for d in con.execute("SELECT * FROM haz_resolutions").description]
        assert "provisional" in columns, (
            "a fresh connection was skipped by the ensure-memo and never migrated"
        )
        con.close()
        del con
        gc.collect()

    # Sanity: the loop really did recycle ids, so the assertion above was
    # exercising the aliasing case rather than getting lucky.
    assert len(seen_ids) < 50
