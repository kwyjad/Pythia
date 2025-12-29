# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Compute per-question forecast summary stats for the API."""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:  # pragma: no cover - silence library default
    LOGGER.addHandler(logging.NullHandler())


def _table_exists(conn, table: str) -> bool:
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE LOWER(table_name) = LOWER(?)",
            [table],
        ).fetchone()
        return bool(row and row[0])
    except Exception:
        pass

    try:
        df = conn.execute("PRAGMA show_tables").fetchdf()
    except Exception:
        return False
    if df.empty:
        return False
    first_col = df.columns[0]
    return df[first_col].astype(str).str.lower().eq(table.lower()).any()


def _table_columns(conn, table: str) -> set[str]:
    try:
        df = conn.execute(f"PRAGMA table_info('{table}')").fetchdf()
    except Exception:
        return set()
    if df.empty or "name" not in df.columns:
        return set()
    return set(df["name"].astype(str).str.lower().tolist())


def _pick_column(columns: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate.lower() in columns:
            return candidate.lower()
    return None


def compute_questions_forecast_summary(
    conn,
    *,
    question_ids: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    if conn is None:
        return {}
    if not _table_exists(conn, "questions") or not _table_exists(conn, "forecasts_ensemble"):
        return {}

    q_cols = _table_columns(conn, "questions")
    if "question_id" not in q_cols or "hazard_code" not in q_cols or "metric" not in q_cols:
        LOGGER.warning("questions table missing required columns for forecast summary.")
        return {}

    fe_cols = _table_columns(conn, "forecasts_ensemble")
    status_col = _pick_column(fe_cols, ["status"])
    prob_col = _pick_column(fe_cols, ["probability", "p"])
    horizon_col = _pick_column(fe_cols, ["month_index", "horizon_m"])
    bucket_col = _pick_column(fe_cols, ["bucket_index", "class_bin"])
    ts_col = _pick_column(fe_cols, ["created_at", "timestamp"])
    run_col = _pick_column(fe_cols, ["run_id"])
    ev_value_col = _pick_column(fe_cols, ["ev_value"])

    if status_col:
        filter_expr = f"fe.{status_col} = 'ok'"
    elif prob_col and horizon_col and bucket_col:
        filter_expr = (
            f"fe.{prob_col} IS NOT NULL AND "
            f"fe.{horizon_col} IS NOT NULL AND "
            f"fe.{bucket_col} IS NOT NULL"
        )
    else:
        LOGGER.info("Skipping forecast summary; forecasts_ensemble columns missing.")
        return {}

    bc_exists = _table_exists(conn, "bucket_centroids")
    bc_cols = _table_columns(conn, "bucket_centroids") if bc_exists else set()
    bc_bucket_col = _pick_column(bc_cols, ["bucket_index", "class_bin"])
    bc_centroid_col = _pick_column(bc_cols, ["centroid", "ev"])
    bc_joinable = bool(
        bc_exists
        and bc_bucket_col
        and bc_centroid_col
        and prob_col
        and bucket_col
        and "hazard_code" in bc_cols
        and "metric" in bc_cols
    )

    base_alias = "scoped" if run_col and ts_col else "base"
    base_ev_expr = f"{base_alias}.{ev_value_col}" if ev_value_col else "NULL"
    if bc_joinable:
        bc_exact_alias = "bc_exact"
        bc_any_alias = "bc_any"
        bc_centroid_expr = (
            f"COALESCE({bc_exact_alias}.{bc_centroid_col}, {bc_any_alias}.{bc_centroid_col})"
        )
        eiv_expr = f"COALESCE({base_ev_expr}, {base_alias}.{prob_col} * {bc_centroid_expr})"
        bc_join = (
            "LEFT JOIN bucket_centroids bc_exact ON "
            f"bc_exact.hazard_code = {base_alias}.hazard_code "
            f"AND bc_exact.metric = {base_alias}.metric "
            f"AND bc_exact.{bc_bucket_col} = {base_alias}.{bucket_col} "
            "LEFT JOIN bucket_centroids bc_any ON "
            f"bc_any.hazard_code = '*' "
            f"AND bc_any.metric = {base_alias}.metric "
            f"AND bc_any.{bc_bucket_col} = {base_alias}.{bucket_col}"
        )
    else:
        if not ev_value_col:
            LOGGER.info("Bucket centroids unavailable; EIV totals will be null.")
        eiv_expr = base_ev_expr
        bc_join = ""

    horizon_expr = (
        f"MAX({base_alias}.{horizon_col}) AS horizon_max" if horizon_col else "NULL AS horizon_max"
    )
    forecast_date_expr = (
        f"STRFTIME(MAX({base_alias}.{ts_col}), '%Y-%m-%d') AS forecast_date"
        if ts_col
        else "NULL AS forecast_date"
    )
    eiv_total_expr = f"SUM({eiv_expr}) AS eiv_total"

    params: list[Any] = []
    question_filter = ""
    if question_ids is not None:
        if not question_ids:
            return {}
        placeholders = ", ".join(["?"] * len(question_ids))
        question_filter = f"AND q.question_id IN ({placeholders})"
        params.extend(question_ids)

    base_sql = f"""
        SELECT
          fe.*,
          q.hazard_code AS hazard_code,
          q.metric AS metric
        FROM forecasts_ensemble fe
        JOIN questions q ON q.question_id = fe.question_id
        WHERE {filter_expr}
        {question_filter}
    """

    try:
        if run_col and ts_col:
            sql = f"""
                WITH base AS (
                    {base_sql}
                ),
                latest_runs AS (
                    SELECT
                      question_id,
                      {run_col} AS run_id,
                      ROW_NUMBER() OVER (
                        PARTITION BY question_id
                        ORDER BY {ts_col} DESC
                      ) AS rn
                    FROM base
                    GROUP BY question_id, {run_col}, {ts_col}
                ),
                scoped AS (
                    SELECT base.*
                    FROM base
                    JOIN latest_runs lr
                      ON lr.question_id = base.question_id
                     AND lr.run_id = base.{run_col}
                    WHERE lr.rn = 1
                )
                SELECT
                  scoped.question_id AS question_id,
                  {forecast_date_expr},
                  {horizon_expr},
                  {eiv_total_expr}
                FROM scoped
                {bc_join}
                GROUP BY scoped.question_id
                ORDER BY scoped.question_id
            """
        else:
            sql = f"""
                WITH base AS (
                    {base_sql}
                )
                SELECT
                  base.question_id AS question_id,
                  {forecast_date_expr},
                  {horizon_expr},
                  {eiv_total_expr}
                FROM base
                {bc_join}
                GROUP BY base.question_id
                ORDER BY base.question_id
            """

        rows = conn.execute(sql, params).fetchall()
    except Exception:
        LOGGER.exception("Failed to compute questions forecast summary")
        return {}

    summary: dict[str, dict[str, Any]] = {}
    for question_id, forecast_date, horizon_max, eiv_total in rows or []:
        if not question_id:
            continue
        summary[str(question_id)] = {
            "forecast_date": forecast_date,
            "horizon_max": int(horizon_max) if horizon_max is not None else None,
            "eiv_total": float(eiv_total) if eiv_total is not None else None,
        }
    return summary
