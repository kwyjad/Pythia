"""Thin DuckDB readers to support resolver selectors."""

from __future__ import annotations

from typing import Optional


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
    import logging, duckdb as _duckdb
    _log = logging.getLogger(__name__)
    _log.warning("DBG fetch_deltas_point: ym=%s iso3=%s hazard=%s cutoff=%s pref=%s", ym, iso3, hazard_code, cutoff, preferred_metric)
    _log.warning("DBG duckdb version: %s", getattr(_duckdb, "__version__", "n/a"))
    c1 = conn.execute("SELECT COUNT(*) FROM facts_deltas WHERE ym=? AND iso3=? AND hazard_code=?", [ym, iso3, hazard_code]).fetchone()[0]
    c2 = conn.execute("SELECT COUNT(*) FROM facts_deltas").fetchone()[0]
    c3 = conn.execute("SELECT COUNT(*) FROM (SELECT TRY_CAST(as_of AS DATE) as as_of_parsed FROM facts_deltas WHERE ym=? AND iso3=? AND hazard_code=?) WHERE as_of_parsed IS NULL OR as_of_parsed <= TRY_CAST(? AS DATE)", [ym, iso3, hazard_code, cutoff]).fetchone()[0]
    _log.warning("DBG facts_deltas counts: match_keys=%s, total=%s, match_with_cutoff=%s", c1, c2, c3)

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
    """Return the latest resolved row at or before ``cutoff``."""

    metric_case = _metric_case_sql()
    query = f"""
        WITH ranked AS (
            SELECT
                ym,
                iso3,
                hazard_code,
                hazard_label,
                hazard_class,
                metric,
                value,
                unit,
                as_of,
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
                series_semantics,
                created_at,
                TRY_CAST(as_of_date AS DATE) AS as_of_parsed,
                TRY_CAST(publication_date AS DATE) AS publication_parsed,
                {metric_case} AS metric_rank
            FROM facts_resolved
            WHERE ym = ?
              AND iso3 = ?
              AND hazard_code = ?
              AND COALESCE(NULLIF(series_semantics, ''), 'stock') = 'stock'
        )
        SELECT *
        FROM ranked
        WHERE as_of_parsed IS NULL OR as_of_parsed <= TRY_CAST(? AS DATE)
        ORDER BY metric_rank, as_of_parsed DESC NULLS LAST, publication_parsed DESC NULLS LAST, created_at DESC
        LIMIT 1
    """
    df = conn.execute(
        query,
        [preferred_metric, ym, iso3, hazard_code, cutoff],
    ).fetch_df()
    if df.empty:
        return None
    return df.iloc[0].to_dict()
