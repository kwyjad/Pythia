"""DuckDB writer for normalized ODP time-series frames."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from resolver.common.logs import df_schema, get_logger
from resolver.db import duckdb_io
from resolver.ingestion import odp_series

LOGGER = get_logger(__name__)
_CANONICAL_COLUMNS = [
    "source_id",
    "iso3",
    "origin_iso3",
    "admin_name",
    "ym",
    "as_of_date",
    "metric",
    "series_semantics",
    "value",
    "unit",
    "extra",
]


def _ensure_odp_table_exists(conn, table: str = "odp_timeseries_raw") -> None:
    """Create the ODP table and natural-key index if missing."""

    ddl = f"""
    CREATE TABLE IF NOT EXISTS {table} (
        source_id TEXT,
        iso3 TEXT,
        origin_iso3 TEXT,
        admin_name TEXT,
        ym TEXT,
        as_of_date DATE,
        metric TEXT,
        series_semantics TEXT,
        value DOUBLE,
        unit TEXT,
        extra TEXT
    );
    """
    conn.execute(ddl)
    conn.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_natural_key
        ON {table} (source_id, iso3, origin_iso3, admin_name, ym, metric);
        """
    )
    LOGGER.debug("ODP DuckDB writer: ensured table exists", extra={"table": table})


def _ensure_canonical_odp_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with canonical ODP columns present."""

    if frame is None:
        return pd.DataFrame(columns=_CANONICAL_COLUMNS)
    result = frame.copy()
    for column in _CANONICAL_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    if "value" in result.columns:
        result["value"] = pd.to_numeric(result["value"], errors="coerce")
    if "series_semantics" in result.columns:
        result["series_semantics"] = result["series_semantics"].astype(str).str.strip().str.lower()
    return result[_CANONICAL_COLUMNS]


def write_odp_timeseries(
    df: pd.DataFrame,
    db_url: str,
    *,
    table: str = "odp_timeseries_raw",
    stats: odp_series.OdpPipelineStats | None = None,
) -> None:
    """Upsert ``df`` into ``table`` within the DuckDB database at ``db_url``."""

    source_frame = df if df is not None else pd.DataFrame(columns=_CANONICAL_COLUMNS)
    canonical = _ensure_canonical_odp_columns(source_frame)
    conn = duckdb_io.get_db(db_url)
    try:
        _ensure_odp_table_exists(conn, table=table)
        if canonical.empty:
            LOGGER.info("ODP DuckDB writer: no rows to write", extra={"table": table})
            if stats is not None:
                stats.notes.append("duckdb_created_no_rows")
            return
        keys = ["source_id", "iso3", "origin_iso3", "admin_name", "ym", "metric"]
        result = duckdb_io.upsert_dataframe(conn, table, canonical, keys=keys)
        LOGGER.info(
            "ODP DuckDB writer: upserted rows",
            extra={
                "table": table,
                "rows": len(canonical),
                "schema": df_schema(canonical),
                "rows_written": getattr(result, "rows_written", None),
                "rows_delta": getattr(result, "rows_delta", None),
            },
        )
    finally:
        duckdb_io.close_db(conn)


def build_and_write_odp_series(
    *,
    config_path: str | Path,
    normalizers_path: str | Path | None = None,
    db_url: str,
    fetch_html: Callable[[str], str] | None = None,
    fetch_json: Callable[[str], Any] | None = None,
    today: date | None = None,
    stats: odp_series.OdpPipelineStats | None = None,
) -> int:
    """Run discovery → normalization → DuckDB write for ODP series."""

    frame = odp_series.build_odp_frame(
        config_path=config_path,
        normalizers_path=normalizers_path,
        fetch_html=fetch_html,
        fetch_json=fetch_json,
        today=today,
        stats=stats,
    )
    write_odp_timeseries(frame, db_url, stats=stats)
    LOGGER.info(
        "ODP build+write complete",
        extra={"rows": len(frame), "db_url": db_url},
    )
    return int(len(frame))


__all__ = ["build_and_write_odp_series", "write_odp_timeseries"]
