"""Thin DuckDB readers to support resolver selectors."""

from __future__ import annotations

from typing import Optional

import logging
import os

from resolver.common import get_logger
from resolver.db.conn_shared import get_shared_duckdb_conn, normalize_duckdb_url

LOGGER = get_logger(__name__)
DEBUG_ENABLED = os.getenv("RESOLVER_DEBUG") == "1"
if DEBUG_ENABLED:
    LOGGER.setLevel(logging.DEBUG)


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

    resolved_path: str | None = None
    reused_label: str | None = None
    if conn is None:
        conn, resolved_path = get_shared_duckdb_conn(os.environ.get("RESOLVER_DB_URL"))
        reused_label = "shared"
    else:
        resolved_path = getattr(conn, "database", None) or normalize_duckdb_url(
            os.environ.get("RESOLVER_DB_URL", "")
        )
        reused_label = "external"

    if conn is None:
        return None

    if DEBUG_ENABLED:
        LOGGER.debug(
            "DuckDB fetch_deltas_point path=%s reused=%s ym=%s iso3=%s hazard=%s cutoff=%s preferred_metric=%s",
            resolved_path,
            reused_label,
            ym,
            iso3,
            hazard_code,
            cutoff,
            preferred_metric,
        )
        try:
            total = conn.execute("SELECT COUNT(*) FROM facts_deltas").fetchone()[0]
            triple = conn.execute(
                "SELECT COUNT(*) FROM facts_deltas WHERE ym = ? AND iso3 = ? AND hazard_code = ?",
                [ym, iso3, hazard_code],
            ).fetchone()[0]
            cutoff_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM facts_deltas
                WHERE ym = ?
                  AND iso3 = ?
                  AND hazard_code = ?
                  AND (TRY_CAST(as_of AS DATE) IS NULL OR TRY_CAST(as_of AS DATE) <= TRY_CAST(? AS DATE))
                """,
                [ym, iso3, hazard_code, cutoff],
            ).fetchone()[0]
            LOGGER.debug(
                "DuckDB fetch_deltas_point counts path=%s total=%s triple=%s cutoff=%s",
                resolved_path,
                total,
                triple,
                cutoff_count,
            )
        except Exception:  # pragma: no cover - diagnostics only
            LOGGER.debug("DuckDB fetch_deltas_point diagnostics failed", exc_info=True)

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


    params = [preferred_metric, ym, iso3, hazard_code, cutoff]
    if DEBUG_ENABLED:
        LOGGER.debug(
            "DuckDB fetch_deltas_point executing with params=%s path=%s",
            params,
            resolved_path,
        )
    df = conn.execute(query, params).fetch_df()
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

    resolved_path: str | None = None
    reused_label: str | None = None
    if conn is None:
        conn, resolved_path = get_shared_duckdb_conn(os.environ.get("RESOLVER_DB_URL"))
        reused_label = "shared"
    else:
        resolved_path = getattr(conn, "database", None) or normalize_duckdb_url(
            os.environ.get("RESOLVER_DB_URL", "")
        )
        reused_label = "external"

    if conn is None:
        return None

    if DEBUG_ENABLED:
        LOGGER.debug(
            "DuckDB fetch_resolved_point path=%s reused=%s ym=%s iso3=%s hazard=%s cutoff=%s preferred_metric=%s",
            resolved_path,
            reused_label,
            ym,
            iso3,
            hazard_code,
            cutoff,
            preferred_metric,
        )
        try:
            total = conn.execute("SELECT COUNT(*) FROM facts_resolved").fetchone()[0]
            triple = conn.execute(
                "SELECT COUNT(*) FROM facts_resolved WHERE ym = ? AND iso3 = ? AND hazard_code = ?",
                [ym, iso3, hazard_code],
            ).fetchone()[0]
            cutoff_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM facts_resolved
                WHERE ym = ?
                  AND iso3 = ?
                  AND hazard_code = ?
                  AND (
                        COALESCE(
                            TRY_CAST(NULLIF(as_of_date, '') AS DATE),
                            TRY_CAST(NULLIF(publication_date, '') AS DATE),
                            TRY_CAST(as_of AS DATE)
                        ) IS NULL
                        OR COALESCE(
                            TRY_CAST(NULLIF(as_of_date, '') AS DATE),
                            TRY_CAST(NULLIF(publication_date, '') AS DATE),
                            TRY_CAST(as_of AS DATE)
                        ) <= TRY_CAST(? AS DATE)
                  )
                """,
                [ym, iso3, hazard_code, cutoff],
            ).fetchone()[0]
            LOGGER.debug(
                "DuckDB fetch_resolved_point counts path=%s total=%s triple=%s cutoff=%s",
                resolved_path,
                total,
                triple,
                cutoff_count,
            )
        except Exception:  # pragma: no cover - diagnostics only
            LOGGER.debug("DuckDB fetch_resolved_point diagnostics failed", exc_info=True)

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

    params = [preferred_metric, ym, iso3, hazard_code, cutoff]
    if DEBUG_ENABLED:
        LOGGER.debug(
            "DuckDB fetch_resolved_point executing with params=%s path=%s",
            params,
            resolved_path,
        )
    df = conn.execute(query, params).fetch_df()
    if df.empty:
        return None
    return df.iloc[0].to_dict()
