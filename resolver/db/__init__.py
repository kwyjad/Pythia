# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

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
