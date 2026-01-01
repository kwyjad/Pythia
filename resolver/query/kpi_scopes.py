from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Optional

import duckdb

logger = logging.getLogger(__name__)


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    try:
        df = con.execute("PRAGMA show_tables").fetchdf()
    except Exception:
        return False
    if df.empty:
        return False
    first_col = df.columns[0]
    return df[first_col].astype(str).str.lower().eq(table.lower()).any()


def _table_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    try:
        df = con.execute(f"PRAGMA table_info('{table}')").fetchdf()
    except Exception:
        return set()
    if df.empty or "name" not in df.columns:
        return set()
    return set(df["name"].astype(str).str.lower().tolist())


def _pick_col(cols: set[str], candidates: list[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate.lower() in cols:
            return candidate
    return None


def _pick_timestamp_column(
    con: duckdb.DuckDBPyConnection, table: str, candidates: list[str]
) -> Optional[str]:
    if not _table_exists(con, table):
        return None
    cols = _table_columns(con, table)
    return _pick_col(cols, candidates)


def _parse_year_month(value: str) -> Optional[tuple[int, int]]:
    match = re.match(r"^(\d{4})-(\d{2})$", value.strip())
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2))
    if month < 1 or month > 12:
        return None
    return year, month


def _month_window(year: int, month: int) -> tuple[str, str]:
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)
    return start.isoformat(), end.isoformat()


def compute_countries_triaged_for_month_with_source(
    con: duckdb.DuckDBPyConnection, year_month: str
) -> tuple[int, Optional[str]]:
    parsed = _parse_year_month(year_month)
    if not parsed:
        logger.debug("countries_triaged: invalid year_month=%s", year_month)
        return 0, None
    start_iso, end_iso = _month_window(parsed[0], parsed[1])

    llm_count: Optional[int] = None
    if not _table_exists(con, "llm_calls"):
        logger.debug("countries_triaged: llm_calls table missing")
    else:
        llm_ts = _pick_timestamp_column(con, "llm_calls", ["created_at", "timestamp", "started_at"])
        llm_cols = _table_columns(con, "llm_calls")
        if not llm_ts:
            logger.debug("countries_triaged: llm_calls timestamp column missing")
        elif "phase" not in llm_cols or "iso3" not in llm_cols:
            logger.debug("countries_triaged: llm_calls missing phase or iso3 columns")
        else:
            try:
                row = con.execute(
                    f"""
                    SELECT COUNT(DISTINCT UPPER(iso3))
                    FROM llm_calls
                    WHERE phase = 'hs_triage'
                      AND {llm_ts} >= ?
                      AND {llm_ts} < ?
                      AND iso3 IS NOT NULL
                    """,
                    [start_iso, end_iso],
                ).fetchone()
                llm_count = int(row[0]) if row else 0
            except Exception:
                logger.debug("countries_triaged: llm_calls query failed", exc_info=True)

    if llm_count and llm_count > 0:
        return llm_count, "llm_calls.hs_triage"

    logger.debug("countries_triaged: hs_triage fallback used")
    if not _table_exists(con, "hs_triage"):
        logger.debug("countries_triaged: hs_triage table missing")
        return 0, None

    hs_cols = _table_columns(con, "hs_triage")
    if "created_at" not in hs_cols or "iso3" not in hs_cols:
        logger.debug("countries_triaged: hs_triage missing created_at or iso3 columns")
        return 0, None

    try:
        row = con.execute(
            """
            SELECT COUNT(DISTINCT UPPER(iso3))
            FROM hs_triage
            WHERE created_at >= ?
              AND created_at < ?
              AND iso3 IS NOT NULL
            """,
            [start_iso, end_iso],
        ).fetchone()
        return (int(row[0]) if row else 0), "hs_triage"
    except Exception:
        logger.debug("countries_triaged: hs_triage query failed", exc_info=True)
        return 0, None


def compute_countries_triaged_for_month(
    con: duckdb.DuckDBPyConnection, year_month: str
) -> int:
    count, _ = compute_countries_triaged_for_month_with_source(con, year_month)
    return count
