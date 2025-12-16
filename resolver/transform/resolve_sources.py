# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Utilities to consolidate ``facts_resolved`` rows by source priority."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import yaml

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:  # pragma: no cover - silence by default
    LOGGER.addHandler(logging.NullHandler())

DEFAULT_PRIORITY_PATH = (
    Path(__file__).resolve().parents[1]
    / "ingestion"
    / "config"
    / "source_priority.yml"
)

_REQUIRED_GROUP_COLS: Sequence[str] = (
    "iso3",
    "hazard_code",
    "metric",
    "unit",
    "as_of_date",
)


def _load_priority(path: Path | None = None) -> list[str]:
    """Load an ordered list of preferred sources from YAML."""

    priority_path = path or DEFAULT_PRIORITY_PATH
    if not priority_path.exists():
        raise FileNotFoundError(
            f"Source priority config not found at {priority_path}"
        )
    with priority_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or []
    if isinstance(loaded, dict):
        # Support configs expressed as {priority: [..]} for flexibility.
        if "priority" in loaded and isinstance(loaded["priority"], list):
            loaded = loaded["priority"]
        else:
            loaded = list(loaded.values())
    if not isinstance(loaded, Iterable):
        raise TypeError(
            "Source priority config must be a sequence of identifiers"
        )
    priority = [str(item).strip() for item in loaded if str(item).strip()]
    if not priority:
        raise ValueError(
            f"Source priority config at {priority_path} is empty"
        )
    return priority


def _normalise_source(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def resolve_sources(
    conn,
    *,
    priority_path: Path | None = None,
    input_table: str = "facts_resolved",
    output_table: str | None = None,
) -> int:
    """Resolve multi-source rows in ``facts_resolved`` according to priority.

    Parameters
    ----------
    conn:
        DuckDB connection that already contains ``facts_resolved`` rows.
    priority_path:
        Optional path to the YAML priority list. Defaults to
        ``resolver/ingestion/config/source_priority.yml``.
    input_table:
        Table to read candidate rows from (defaults to ``facts_resolved``).
    output_table:
        Optional destination table name. When ``None`` (default) the input
        table is overwritten with the prioritized view.

    Returns
    -------
    int
        The number of rows written to the destination table.
    """

    priority = _load_priority(priority_path)
    normalised_priority = [_normalise_source(item) for item in priority]
    priority_map = {
        source: idx + 1 for idx, source in enumerate(normalised_priority)
    }
    fallback_rank = len(priority_map) + 1

    query = f"SELECT * FROM {input_table}"
    frame = conn.execute(query).df()
    if frame.empty:
        LOGGER.info("No rows in %s; skipping source resolution", input_table)
        return 0

    missing_cols = [col for col in _REQUIRED_GROUP_COLS if col not in frame]
    if missing_cols:
        raise KeyError(
            f"facts_resolved frame missing required columns: {missing_cols}"
        )

    source_col = "source_id" if "source_id" in frame.columns else "source"
    if source_col not in frame.columns:
        raise KeyError(
            "facts_resolved frame missing source column (expected source_id or source)"
        )

    normalised = frame[source_col].map(_normalise_source)
    frame["provenance_source"] = frame[source_col].fillna("").astype(str)
    frame["provenance_rank"] = normalised.map(priority_map).fillna(
        fallback_rank
    )
    frame["provenance_rank"] = frame["provenance_rank"].astype(int)

    sort_columns: list[str] = list(_REQUIRED_GROUP_COLS) + [
        "provenance_rank",
        "publication_date" if "publication_date" in frame.columns else "ym",
        source_col,
    ]
    existing_sort = [col for col in sort_columns if col in frame.columns]
    frame = frame.sort_values(existing_sort)

    dedup_keys = [col for col in _REQUIRED_GROUP_COLS if col in frame.columns]
    prioritized = frame.drop_duplicates(subset=dedup_keys, keep="first")

    target_table = output_table or input_table
    cols = list(frame.columns)
    temp_name = "tmp_resolved_priority"
    conn.register(temp_name, prioritized[cols])
    try:
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(f"DELETE FROM {target_table}")
            placeholders = ", ".join(cols)
            conn.execute(
                f"INSERT INTO {target_table} ({placeholders})"
                f" SELECT {placeholders} FROM {temp_name}"
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.unregister(temp_name)

    kept = len(prioritized)
    LOGGER.info(
        "Resolved %s rows for %s (from %s input rows)",
        kept,
        target_table,
        len(frame),
    )
    return kept


__all__ = ["resolve_sources"]
