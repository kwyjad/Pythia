"""Thin DuckDB readers to support resolver selectors."""

from __future__ import annotations

from typing import Optional

import os


_DUCKDB_CONN_CACHE: dict[str, "duckdb.DuckDBPyConnection"] = {}


def get_shared_duckdb_conn(db_url: str | None):
    """Return a cached duckdb connection for the given file path."""
    import duckdb

    if not db_url:
        return None

    db_path = db_url.replace("duckdb:///", "", 1).strip()
    if not db_path:
        return None

    if db_path not in _DUCKDB_CONN_CACHE:
        _DUCKDB_CONN_CACHE[db_path] = duckdb.connect(database=db_path)

    return _DUCKDB_CONN_CACHE[db_path]


def _metric_case_sql() -> str:
    return (
        "CASE "
        "WHEN lower(metric) = lower(?) THEN 0 "
        "WHEN lower(metric) = 'in_need' THEN 1 "
        "WHEN lower(metric) = 'affected' THEN 2 "
        "ELSE 3 END"
    )


def fetch_deltas_point(
    conn,
    *,
    ym: str,
    iso3: str,
    hazard_code: str,
    cutoff: str,
    preferred_metric: str,
) -> Optional[dict]:
    """Return the latest delta row at or before ``cutoff`` for the request."""

    if conn is None:
        conn = get_shared_duckdb_conn(os.environ.get("RESOLVER_DB_URL"))

    if conn is None:
        return None

    metric_case = _metric_case_sql()
    query = f"""
        WITH ranked AS (
            SELECT
                ym,
                iso3,
                hazard_code,
                metric,
                value_new,
                value_stock,
                series_semantics,
                as_of,
                source_id,
                series,
                rebase_flag,
                first_observation,
                delta_negative_clamped,
                created_at,
                TRY_CAST(as_of AS DATE) AS as_of_parsed,
                {metric_case} AS metric_rank
            FROM facts_deltas
            WHERE ym = ?
              AND iso3 = ?
              AND hazard_code = ?
        )
        SELECT *
        FROM ranked
        WHERE as_of_parsed IS NULL OR as_of_parsed <= TRY_CAST(? AS DATE)
        ORDER BY metric_rank, as_of_parsed DESC NULLS LAST, created_at DESC
        LIMIT 1
    """


    df = conn.execute(
        query,
        [preferred_metric, ym, iso3, hazard_code, cutoff],
    ).fetch_df()
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def fetch_resolved_point(
    conn,
    *,
    ym: str,
    iso3: str,
    hazard_code: str,
    cutoff: str,
    preferred_metric: str,
) -> Optional[dict]:
    """Return the latest resolved row at or before ``cutoff`` for the request."""

    if conn is None:
        conn = get_shared_duckdb_conn(os.environ.get("RESOLVER_DB_URL"))

    if conn is None:
        return None

    metric_case = _metric_case_sql()
    query = f"""
        WITH ranked AS (
            SELECT
                ym,
                iso3,
                hazard_code,
                metric,
                value,
                unit,
                as_of,
                series_semantics,
                as_of_date,
                publication_date,
                publisher,
                source_id,
                source_type,
                source_url,
                doc_title,
                definition_text,
                precedence_tier,
                event_id,
                proxy_for,
                confidence,
                created_at,
                COALESCE(
                    TRY_CAST(NULLIF(as_of_date, '') AS DATE),
                    TRY_CAST(NULLIF(publication_date, '') AS DATE),
                    as_of
                ) AS as_of_parsed,
                {metric_case} AS metric_rank
            FROM facts_resolved
            WHERE ym = ?
              AND iso3 = ?
              AND hazard_code = ?
        )
        SELECT *
        FROM ranked
        WHERE as_of_parsed IS NULL OR as_of_parsed <= TRY_CAST(? AS DATE)
        ORDER BY metric_rank, as_of_parsed DESC NULLS LAST, created_at DESC
        LIMIT 1
    """

    df = conn.execute(
        query,
        [preferred_metric, ym, iso3, hazard_code, cutoff],
    ).fetch_df()
    if df.empty:
        return None
    return df.iloc[0].to_dict()
