"""Thin DuckDB readers to support resolver selectors."""

from __future__ import annotations

from typing import Optional

print("DBG db_reader import marker v1", flush=True)

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
    print(
        f"DBG fetch_deltas_point CALLED ym={ym} iso3={iso3} hazard={hazard_code} cutoff={cutoff} pref={preferred_metric}",
        flush=True,
    )
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

    # --- DEBUG START (temporary) ---
    print(f"DBG fetch_deltas_point: ym={ym} iso3={iso3} hazard={hazard_code} cutoff={cutoff} pref={preferred_metric}")
    try:
        import duckdb as _duckdb
        print("DBG duckdb version:", getattr(_duckdb, "__version__", "n/a"))
    except Exception as _e:
        print("DBG duckdb import failed:", _e)

    # counts
    c_total = conn.execute("SELECT COUNT(*) FROM facts_deltas").fetchone()[0]
    c_keys  = conn.execute(
        "SELECT COUNT(*) FROM facts_deltas WHERE ym=? AND iso3=? AND hazard_code=?",
        [ym, iso3, hazard_code]
    ).fetchone()[0]
    c_cut   = conn.execute(
        """
        SELECT COUNT(*) FROM (
        SELECT TRY_CAST(as_of AS DATE) AS as_of_parsed
        FROM facts_deltas
        WHERE ym=? AND iso3=? AND hazard_code=?
        ) WHERE as_of_parsed IS NULL OR as_of_parsed <= TRY_CAST(? AS DATE)
        """,
        [ym, iso3, hazard_code, cutoff]
    ).fetchone()[0]
    print(f"DBG facts_deltas counts: total={c_total} match_keys={c_keys} match_with_cutoff={c_cut}")

    # peek rows at each step
    rows_keys = conn.execute(
        "SELECT ym, iso3, hazard_code, metric, value_new, as_of FROM facts_deltas WHERE ym=? AND iso3=? AND hazard_code=?",
        [ym, iso3, hazard_code]
    ).fetchall()
    print("DBG rows(match_keys):", rows_keys)

    rows_cut = conn.execute(
        """
        SELECT ym, iso3, hazard_code, metric, value_new, as_of
        FROM (
        SELECT ym, iso3, hazard_code, metric, value_new, as_of, TRY_CAST(as_of AS DATE) AS as_of_parsed
        FROM facts_deltas
        WHERE ym=? AND iso3=? AND hazard_code=?
        )
        WHERE as_of_parsed IS NULL OR as_of_parsed <= TRY_CAST(? AS DATE)
        """,
        [ym, iso3, hazard_code, cutoff]
    ).fetchall()
    print("DBG rows(match_with_cutoff):", rows_cut)

    # show the params for the main query (order matters)
    print("DBG main params:", [preferred_metric, ym, iso3, hazard_code, cutoff])
    # --- DEBUG END ---

    df = conn.execute(
        query,
        [preferred_metric, ym, iso3, hazard_code, cutoff],
    ).fetch_df()
    if df.empty:
        return None
    return df.iloc[0].to_dict()
