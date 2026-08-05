# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""DuckDB schema for the resolution machine's own tables.

The machine owns its DDL (same self-contained pattern as
``horizon_scanner/reliefweb.py``): every writer calls
:func:`ensure_schema` first, and all statements are
``CREATE TABLE IF NOT EXISTS`` so re-runs are no-ops.  Timestamps are
ISO-8601 UTC strings (VARCHAR), matching the newer connector tables.

Tables
------
haz_raw_ibtracs
    Raw IBTrACS track points (the columns detection needs, values
    unmodified apart from longitude normalisation to [-180, 180]).
haz_triggers
    Layer-1 detection outcome per (hazard, iso3, ym) — one row for BOTH
    triggered and non-triggered country-months, with the parameters and
    evidence that produced the verdict.
pa_resolutions
    The machine's resolution rows.  Hard rule 1: every row carries
    provenance (source, record ids / doc URLs, retrieval timestamp,
    rule fired).  Hard rule 4: rows whose freeze date has passed are
    never modified.
pa_resolution_revisions
    Append-only log of post-freeze observations (rule 4: revisions are
    logged, resolved values never change).
"""

from __future__ import annotations

from datetime import datetime, timezone

_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS haz_raw_ibtracs (
        sid          VARCHAR NOT NULL,   -- IBTrACS storm id
        season       INTEGER,
        basin        VARCHAR,
        storm_name   VARCHAR,
        iso_time     TIMESTAMP NOT NULL,
        lat          DOUBLE,
        lon          DOUBLE,             -- normalised to [-180, 180]
        usa_wind_kt  DOUBLE,             -- 1-min sustained (USA agencies)
        wmo_wind_kt  DOUBLE,             -- WMO reporting agency
        nature       VARCHAR,            -- TS/ET/DS/SS/NR/MX
        source_scope VARCHAR NOT NULL,   -- last3years | ALL
        source_url   VARCHAR,
        fetched_at   VARCHAR NOT NULL,   -- ISO-8601 UTC
        PRIMARY KEY (sid, iso_time)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS haz_triggers (
        hazard_code      VARCHAR NOT NULL,
        iso3             VARCHAR NOT NULL,
        ym               VARCHAR NOT NULL,    -- YYYY-MM
        triggered        BOOLEAN NOT NULL,
        trigger_source   VARCHAR NOT NULL,    -- ibtracs | reliefweb_sweep | none
        detail_json      VARCHAR,             -- storms, distances, params, sweep evidence
        rulebook_version VARCHAR,
        created_at       VARCHAR NOT NULL,    -- ISO-8601 UTC
        PRIMARY KEY (hazard_code, iso3, ym)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pa_resolutions (
        iso3              VARCHAR NOT NULL,
        hazard_code       VARCHAR NOT NULL,
        metric            VARCHAR NOT NULL,   -- rulebook resolution.metric
        ym                VARCHAR NOT NULL,   -- YYYY-MM
        status            VARCHAR NOT NULL,   -- RESOLVED_ZERO | RESOLVED_VALUE | NO_DATA
        value             DOUBLE,
        unit              VARCHAR,
        source            VARCHAR NOT NULL,   -- which layer/ladder rung produced this
        source_record_ids VARCHAR,            -- JSON list of source record ids
        source_urls       VARCHAR,            -- JSON list of document/query URLs
        rule_fired        VARCHAR NOT NULL,
        evidence_json     VARCHAR,            -- evidence_of_absence for zeros
        retrieved_at      VARCHAR NOT NULL,   -- when the evidence was retrieved
        resolved_at       VARCHAR NOT NULL,
        freeze_at         VARCHAR NOT NULL,   -- YYYY-MM-DD; row immutable from this date
        rulebook_version  VARCHAR,
        created_at        VARCHAR NOT NULL,
        PRIMARY KEY (iso3, hazard_code, metric, ym)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pa_resolution_revisions (
        iso3            VARCHAR NOT NULL,
        hazard_code     VARCHAR NOT NULL,
        metric          VARCHAR NOT NULL,
        ym              VARCHAR NOT NULL,
        observed_status VARCHAR,             -- what a re-run would have written
        observed_value  DOUBLE,
        source          VARCHAR,
        note            VARCHAR,
        evidence_json   VARCHAR,
        logged_at       VARCHAR NOT NULL     -- ISO-8601 UTC
    )
    """,
)


def ensure_schema(con) -> None:
    """Create the resolution-machine tables if they do not exist."""
    for ddl in _DDL:
        con.execute(ddl)


def utcnow_iso() -> str:
    """ISO-8601 UTC timestamp string, the machine's timestamp format."""
    return datetime.now(timezone.utc).isoformat()
