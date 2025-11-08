"""Database utilities for the resolver's DuckDB backend."""

from .duckdb_io import (
    DEFAULT_DB_URL,
    FACTS_DELTAS_KEY_COLUMNS,
    FACTS_RESOLVED_KEY_COLUMNS,
    UX_DELTAS,
    UX_RESOLVED,
    get_db,
    init_schema,
    upsert_dataframe,
    write_snapshot,
)

__all__ = [
    "DEFAULT_DB_URL",
    "FACTS_DELTAS_KEY_COLUMNS",
    "FACTS_RESOLVED_KEY_COLUMNS",
    "UX_DELTAS",
    "UX_RESOLVED",
    "get_db",
    "init_schema",
    "upsert_dataframe",
    "write_snapshot",
]
