# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""DuckDB schema for the hazard resolution machine (``haz_*`` tables).

All DDL is idempotent (``CREATE TABLE IF NOT EXISTS``), matching how
``resolver/db/schema.sql`` + ``duckdb_io.init_schema`` manage the core
resolver tables: re-running the migration against a live DB is always safe.

Conventions carried over from the rest of the repo:

* JSON payloads live in ``*_json TEXT`` columns (as ``reasoning_trace_json``
  et al. do in the Pythia DB) rather than a native JSON type.
* The spec's ENUM columns are ``TEXT`` with a ``CHECK`` constraint — DuckDB
  ENUM types make idempotent migrations awkward, CHECK gives the same
  DB-level enforcement.
* Every row records when it was created; raw-cache rows record when the
  source material was retrieved (provenance hard rule).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

#: One thin raw-cache table per upstream source. ``population`` has its own
#: shape (annual denominators); the rest share the generic record cache.
RAW_SOURCES = (
    "ibtracs",
    "gdacs",
    "emdat",
    "idmc_idu",
    "ifrc_go",
    "reliefweb_docs",
    "ipc",
    "population",
)

#: Hazards in scope for the resolution machine (repo hazard codes).
RESOLVABLE_HAZARDS = ("FL", "DR", "TC")

_HAZARD_CHECK = "CHECK (hazard IN ('FL','DR','TC'))"
_MONTH_CHECK = "CHECK (month BETWEEN 1 AND 12)"


def raw_table_name(source: str) -> str:
    if source not in RAW_SOURCES:
        raise ValueError(f"unknown raw source {source!r}; known: {list(RAW_SOURCES)}")
    return f"haz_raw_{source}"


def _generic_raw_ddl(table: str) -> str:
    # record_id is the source-native identifier (storm id, GLIDE/disno,
    # GDACS event id, IDU id, GO report id, ReliefWeb doc id, IPC analysis id).
    # content_hash lets a re-fetch of identical content be a no-op while a
    # revised record appends a new row — that is what haz_revisions audits.
    return f"""
    CREATE TABLE IF NOT EXISTS {table} (
        record_id TEXT NOT NULL,
        iso3 TEXT,
        ym TEXT,
        hazard TEXT,
        payload_json TEXT NOT NULL,
        content_hash TEXT,
        source_url TEXT,
        retrieved_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (record_id, content_hash)
    )
    """


_POPULATION_RAW_DDL = """
CREATE TABLE IF NOT EXISTS haz_raw_population (
    iso3 TEXT NOT NULL,
    year INTEGER NOT NULL,
    population BIGINT NOT NULL,
    source TEXT NOT NULL,
    product TEXT,
    download_date TEXT,
    source_url TEXT,
    notes TEXT,
    retrieved_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (iso3, year, source)
)
"""

_CORE_TABLE_DDL: dict[str, str] = {
    # Layer 1 output: did a qualifying hazard occur in this cell?
    # evidence_of_absence_json records what was checked when nothing occurred
    # (e.g. the ReliefWeb keyword sweep that confirmed silence).
    "haz_triggers": f"""
    CREATE TABLE IF NOT EXISTS haz_triggers (
        iso3 TEXT NOT NULL,
        year INTEGER NOT NULL,
        month INTEGER NOT NULL {_MONTH_CHECK},
        hazard TEXT NOT NULL {_HAZARD_CHECK},
        triggered BOOLEAN NOT NULL,
        trigger_source TEXT,
        trigger_detail_json TEXT,
        evidence_of_absence_json TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (iso3, year, month, hazard)
    )
    """,
    # Layer 2 input: every candidate figure the ladder can choose from.
    # extraction_model is NULL unless the figure came from LLM extraction
    # (which only ever transcribes stated figures, never estimates).
    "haz_impact_candidates": f"""
    CREATE TABLE IF NOT EXISTS haz_impact_candidates (
        iso3 TEXT NOT NULL,
        year INTEGER NOT NULL,
        month INTEGER NOT NULL {_MONTH_CHECK},
        hazard TEXT NOT NULL {_HAZARD_CHECK},
        value DOUBLE NOT NULL,
        value_type TEXT NOT NULL
            CHECK (value_type IN ('affected','displaced_lower_bound','exposed_ceiling')),
        source TEXT NOT NULL,
        source_ref TEXT,
        stated_by TEXT,
        doc_url TEXT,
        extraction_model TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (iso3, year, month, hazard, source, source_ref, value_type)
    )
    """,
    # The machine's answer per cell. provenance_json carries source, record
    # ids / doc URLs, retrieval timestamps and the rule that fired (hard rule
    # 1). frozen_at is stamped at month-end + freeze_days; frozen rows are
    # never reopened (hard rule 4).
    "haz_resolutions": f"""
    CREATE TABLE IF NOT EXISTS haz_resolutions (
        iso3 TEXT NOT NULL,
        year INTEGER NOT NULL,
        month INTEGER NOT NULL {_MONTH_CHECK},
        hazard TEXT NOT NULL {_HAZARD_CHECK},
        status TEXT NOT NULL
            CHECK (status IN ('RESOLVED_ZERO','RESOLVED_VALUE','NO_DATA')),
        value DOUBLE,
        provenance_json TEXT NOT NULL,
        rule_fired TEXT NOT NULL,
        flagged BOOLEAN NOT NULL DEFAULT FALSE,
        frozen_at TIMESTAMP,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (iso3, year, month, hazard)
    )
    """,
    # Post-freeze source changes, audit only — a revision never alters a
    # resolved value.
    "haz_revisions": """
    CREATE TABLE IF NOT EXISTS haz_revisions (
        iso3 TEXT NOT NULL,
        year INTEGER NOT NULL,
        month INTEGER NOT NULL,
        hazard TEXT NOT NULL,
        source TEXT NOT NULL,
        source_ref TEXT,
        old_value DOUBLE,
        new_value DOUBLE,
        detail_json TEXT,
        observed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "haz_base_rates_occurrence": f"""
    CREATE TABLE IF NOT EXISTS haz_base_rates_occurrence (
        iso3 TEXT NOT NULL,
        hazard TEXT NOT NULL {_HAZARD_CHECK},
        calendar_month INTEGER NOT NULL CHECK (calendar_month BETWEEN 1 AND 12),
        p_occurrence DOUBLE NOT NULL,
        n_years INTEGER NOT NULL,
        source_window TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (iso3, hazard, calendar_month)
    )
    """,
    "haz_base_rates_severity": f"""
    CREATE TABLE IF NOT EXISTS haz_base_rates_severity (
        iso3 TEXT NOT NULL,
        hazard TEXT NOT NULL {_HAZARD_CHECK},
        q10 DOUBLE,
        q25 DOUBLE,
        q50 DOUBLE,
        q75 DOUBLE,
        q90 DOUBLE,
        n_events INTEGER NOT NULL,
        window_start INTEGER,
        window_end INTEGER,
        provenance_mix_json TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (iso3, hazard)
    )
    """,
}

_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_haz_triggers_lookup ON haz_triggers (iso3, hazard, year, month)",
    "CREATE INDEX IF NOT EXISTS idx_haz_impact_candidates_lookup ON haz_impact_candidates (iso3, hazard, year, month)",
    "CREATE INDEX IF NOT EXISTS idx_haz_resolutions_lookup ON haz_resolutions (iso3, hazard, year, month)",
)


def haz_table_names() -> tuple[str, ...]:
    """Every table the hazard-resolution migration owns."""

    return tuple(raw_table_name(source) for source in RAW_SOURCES) + tuple(_CORE_TABLE_DDL)


def ensure_haz_schema(conn: "duckdb.DuckDBPyConnection") -> tuple[str, ...]:
    """Idempotently create every ``haz_*`` table and index; return the table names."""

    for source in RAW_SOURCES:
        table = raw_table_name(source)
        ddl = _POPULATION_RAW_DDL if source == "population" else _generic_raw_ddl(table)
        conn.execute(ddl)

    for ddl in _CORE_TABLE_DDL.values():
        conn.execute(ddl)

    for ddl in _INDEX_DDL:
        conn.execute(ddl)

    tables = haz_table_names()
    LOG.info("haz schema ensured | tables=%s", ", ".join(tables))
    return tables
