# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Compute per-question forecast summary stats for the API."""

from __future__ import annotations

import logging
from typing import Any

from resolver.query import eiv_sql

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
    fe_has_hazard = "hazard_code" in fe_cols
    fe_has_metric = "metric" in fe_cols
    model_col = _pick_column(fe_cols, ["model_name"])

    def _log_summary_context(reason: str, *, bc_joinable: bool = False) -> None:
        LOGGER.info(
            "Skipping forecast summary (%s). Columns: prob=%s horizon=%s bucket=%s status=%s "
            "ts=%s run=%s bc_joinable=%s",
            reason,
            prob_col,
            horizon_col,
            bucket_col,
            status_col,
            ts_col,
            run_col,
            bc_joinable,
        )

    if not prob_col or not horizon_col or not bucket_col:
        _log_summary_context("forecasts_ensemble columns missing")
        return {}

    filter_bits = []
    if status_col:
        filter_bits.append(f"LOWER(CAST(fe.{status_col} AS VARCHAR)) = 'ok'")
    if prob_col:
        filter_bits.append(f"fe.{prob_col} IS NOT NULL")
    if horizon_col:
        filter_bits.append(f"fe.{horizon_col} IS NOT NULL")
    if bucket_col:
        filter_bits.append(f"fe.{bucket_col} IS NOT NULL")
    if not filter_bits:
        _log_summary_context("no usable filters")
        return {}
    filter_expr = " AND ".join(filter_bits)

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

    use_latest = bool(run_col and ts_col)
    base_alias = "filtered" if model_col else ("scoped" if use_latest else "base")
    metric_expr = f"{base_alias}.metric"
    hazard_expr = f"{base_alias}.hazard_code"
    bucket_expr = f"{base_alias}.bucket"
    bucket_is_label = bucket_col == "class_bin"

    if bc_joinable:
        bc_join, centroid_expr, _bucket_index_expr = eiv_sql.build_centroid_join(
            base_alias=base_alias,
            metric_expr=metric_expr,
            hazard_expr=hazard_expr,
            bucket_expr=bucket_expr,
            bucket_is_label=bucket_is_label,
            bc_bucket_col=bc_bucket_col,
            bc_centroid_col=bc_centroid_col,
        )
    else:
        bucket_index_expr = eiv_sql.bucket_index_expr(
            metric_expr, bucket_expr, bucket_is_label=bucket_is_label
        )
        centroid_expr = eiv_sql.fallback_centroid_expr(metric_expr, bucket_index_expr)
        bc_join = ""

    eiv_expr = f"{base_alias}.prob * {centroid_expr}"

    horizon_expr = f"MAX({base_alias}.horizon) AS horizon_max"
    forecast_date_expr = (
        f"STRFTIME(MAX({base_alias}.ts), '%Y-%m-%d') AS forecast_date" if ts_col else "NULL AS forecast_date"
    )

    params: list[Any] = []
    question_filter = ""
    if question_ids is not None:
        if not question_ids:
            return {}
        placeholders = ", ".join(["?"] * len(question_ids))
        question_filter = f"AND q.question_id IN ({placeholders})"
        params.extend(question_ids)

    hazard_expr = "q.hazard_code AS hazard_code"
    if fe_has_hazard:
        hazard_expr = "COALESCE(NULLIF(fe.hazard_code, ''), q.hazard_code) AS hazard_code"
    metric_expr = "q.metric AS metric"
    if fe_has_metric:
        metric_expr = "COALESCE(NULLIF(fe.metric, ''), q.metric) AS metric"

    base_select = [
        "fe.question_id AS question_id",
        hazard_expr,
        metric_expr,
        f"fe.{prob_col} AS prob",
        f"fe.{horizon_col} AS horizon",
        f"fe.{bucket_col} AS bucket",
    ]
    if model_col:
        base_select.append(f"fe.{model_col} AS model_name")
    if run_col:
        base_select.append(f"fe.{run_col} AS run_id")
    if ts_col:
        base_select.append(f"fe.{ts_col} AS ts")

    base_sql = f"""
        SELECT
          {", ".join(base_select)}
        FROM forecasts_ensemble fe
        JOIN questions q ON q.question_id = fe.question_id
        WHERE {filter_expr}
        {question_filter}
    """

    try:
        if run_col and ts_col:
            if model_col:
                sql = f"""
                    WITH base AS (
                        {base_sql}
                    ),
                    latest_runs AS (
                        SELECT
                          question_id,
                          run_id,
                          ROW_NUMBER() OVER (
                            PARTITION BY question_id
                            ORDER BY ts DESC
                          ) AS rn
                        FROM base
                        GROUP BY question_id, run_id, ts
                    ),
                    scoped AS (
                        SELECT base.*
                        FROM base
                        JOIN latest_runs lr
                          ON lr.question_id = base.question_id
                         AND lr.run_id = base.run_id
                        WHERE lr.rn = 1
                    ),
                    chosen_model AS (
                        SELECT
                          question_id,
                          CASE
                            WHEN SUM(CASE WHEN model_name = 'ensemble_bayesmc_v2' THEN 1 ELSE 0 END) > 0
                              THEN 'ensemble_bayesmc_v2'
                            WHEN SUM(CASE WHEN model_name = 'ensemble_mean_v2' THEN 1 ELSE 0 END) > 0
                              THEN 'ensemble_mean_v2'
                            ELSE MIN(model_name)
                          END AS chosen_model
                        FROM scoped
                        GROUP BY question_id
                    ),
                    filtered AS (
                        SELECT scoped.*
                        FROM scoped
                        JOIN chosen_model USING (question_id)
                        WHERE scoped.model_name = chosen_model.chosen_model
                    )
                    ,
                    monthly AS (
                        SELECT
                          filtered.question_id AS question_id,
                          filtered.horizon AS horizon,
                          SUM({eiv_expr}) AS eiv_month
                        FROM filtered
                        {bc_join}
                        GROUP BY filtered.question_id, filtered.horizon
                    ),
                    summary AS (
                        SELECT
                          question_id,
                          SUM(eiv_month) AS eiv_total,
                          MAX(eiv_month) AS eiv_peak
                        FROM monthly
                        GROUP BY question_id
                    ),
                    meta AS (
                        SELECT
                          filtered.question_id AS question_id,
                          {forecast_date_expr},
                          {horizon_expr}
                        FROM filtered
                        GROUP BY filtered.question_id
                    )
                    SELECT
                      meta.question_id AS question_id,
                      meta.forecast_date,
                      meta.horizon_max,
                      summary.eiv_total,
                      summary.eiv_peak
                    FROM meta
                    JOIN summary ON summary.question_id = meta.question_id
                    ORDER BY meta.question_id
                """
            else:
                sql = f"""
                    WITH base AS (
                        {base_sql}
                    ),
                    latest_runs AS (
                        SELECT
                          question_id,
                          run_id,
                          ROW_NUMBER() OVER (
                            PARTITION BY question_id
                            ORDER BY ts DESC
                          ) AS rn
                        FROM base
                        GROUP BY question_id, run_id, ts
                    ),
                    scoped AS (
                        SELECT base.*
                        FROM base
                        JOIN latest_runs lr
                          ON lr.question_id = base.question_id
                         AND lr.run_id = base.run_id
                        WHERE lr.rn = 1
                    )
                    ,
                    monthly AS (
                        SELECT
                          scoped.question_id AS question_id,
                          scoped.horizon AS horizon,
                          SUM({eiv_expr}) AS eiv_month
                        FROM scoped
                        {bc_join}
                        GROUP BY scoped.question_id, scoped.horizon
                    ),
                    summary AS (
                        SELECT
                          question_id,
                          SUM(eiv_month) AS eiv_total,
                          MAX(eiv_month) AS eiv_peak
                        FROM monthly
                        GROUP BY question_id
                    ),
                    meta AS (
                        SELECT
                          scoped.question_id AS question_id,
                          {forecast_date_expr},
                          {horizon_expr}
                        FROM scoped
                        GROUP BY scoped.question_id
                    )
                    SELECT
                      meta.question_id AS question_id,
                      meta.forecast_date,
                      meta.horizon_max,
                      summary.eiv_total,
                      summary.eiv_peak
                    FROM meta
                    JOIN summary ON summary.question_id = meta.question_id
                    ORDER BY meta.question_id
                """
        else:
            if model_col:
                sql = f"""
                    WITH base AS (
                        {base_sql}
                    ),
                    chosen_model AS (
                        SELECT
                          question_id,
                          CASE
                            WHEN SUM(CASE WHEN model_name = 'ensemble_bayesmc_v2' THEN 1 ELSE 0 END) > 0
                              THEN 'ensemble_bayesmc_v2'
                            WHEN SUM(CASE WHEN model_name = 'ensemble_mean_v2' THEN 1 ELSE 0 END) > 0
                              THEN 'ensemble_mean_v2'
                            ELSE MIN(model_name)
                          END AS chosen_model
                        FROM base
                        GROUP BY question_id
                    ),
                    filtered AS (
                        SELECT base.*
                        FROM base
                        JOIN chosen_model USING (question_id)
                        WHERE base.model_name = chosen_model.chosen_model
                    )
                    ,
                    monthly AS (
                        SELECT
                          filtered.question_id AS question_id,
                          filtered.horizon AS horizon,
                          SUM({eiv_expr}) AS eiv_month
                        FROM filtered
                        {bc_join}
                        GROUP BY filtered.question_id, filtered.horizon
                    ),
                    summary AS (
                        SELECT
                          question_id,
                          SUM(eiv_month) AS eiv_total,
                          MAX(eiv_month) AS eiv_peak
                        FROM monthly
                        GROUP BY question_id
                    ),
                    meta AS (
                        SELECT
                          filtered.question_id AS question_id,
                          {forecast_date_expr},
                          {horizon_expr}
                        FROM filtered
                        GROUP BY filtered.question_id
                    )
                    SELECT
                      meta.question_id AS question_id,
                      meta.forecast_date,
                      meta.horizon_max,
                      summary.eiv_total,
                      summary.eiv_peak
                    FROM meta
                    JOIN summary ON summary.question_id = meta.question_id
                    ORDER BY meta.question_id
                """
            else:
                sql = f"""
                    WITH base AS (
                        {base_sql}
                    ),
                    monthly AS (
                        SELECT
                          base.question_id AS question_id,
                          base.horizon AS horizon,
                          SUM({eiv_expr}) AS eiv_month
                        FROM base
                        {bc_join}
                        GROUP BY base.question_id, base.horizon
                    ),
                    summary AS (
                        SELECT
                          question_id,
                          SUM(eiv_month) AS eiv_total,
                          MAX(eiv_month) AS eiv_peak
                        FROM monthly
                        GROUP BY question_id
                    ),
                    meta AS (
                        SELECT
                          base.question_id AS question_id,
                          {forecast_date_expr},
                          {horizon_expr}
                        FROM base
                        GROUP BY base.question_id
                    )
                    SELECT
                      meta.question_id AS question_id,
                      meta.forecast_date,
                      meta.horizon_max,
                      summary.eiv_total,
                      summary.eiv_peak
                    FROM meta
                    JOIN summary ON summary.question_id = meta.question_id
                    ORDER BY meta.question_id
                """

        rows = conn.execute(sql, params).fetchall()
    except Exception:
        LOGGER.exception("Failed to compute questions forecast summary")
        _log_summary_context("query failed", bc_joinable=bc_joinable)
        return {}

    summary: dict[str, dict[str, Any]] = {}
    for question_id, forecast_date, horizon_max, eiv_total, eiv_peak in rows or []:
        if not question_id:
            continue
        summary[str(question_id)] = {
            "forecast_date": forecast_date,
            "horizon_max": int(horizon_max) if horizon_max is not None else None,
            "eiv_total": float(eiv_total) if eiv_total is not None else None,
            "eiv_peak": float(eiv_peak) if eiv_peak is not None else None,
        }
    return summary


def compute_questions_triage_summary(conn, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if conn is None:
        return {}
    if not _table_exists(conn, "hs_triage"):
        return {}

    hs_cols = _table_columns(conn, "hs_triage")
    run_col = _pick_column(hs_cols, ["run_id", "hs_run_id"])
    iso_col = _pick_column(hs_cols, ["iso3"])
    hazard_col = _pick_column(hs_cols, ["hazard_code"])
    tier_col = _pick_column(hs_cols, ["tier", "triage_tier"])
    score_col = _pick_column(hs_cols, ["triage_score", "score"])
    need_col = _pick_column(hs_cols, ["need_full_spd", "triage_need_full_spd"])
    created_col = _pick_column(hs_cols, ["created_at", "timestamp"])

    if not (run_col and iso_col and hazard_col and tier_col and score_col and need_col):
        return {}

    values: list[str] = []
    params: list[Any] = []
    for row in rows:
        question_id = row.get("question_id")
        run_id = row.get("hs_run_id") or row.get("run_id")
        iso3 = row.get("iso3")
        hazard_code = row.get("hazard_code")
        if not question_id or not run_id or not iso3 or not hazard_code:
            continue
        values.append("(?, ?, ?, ?)")
        params.extend([question_id, run_id, iso3, hazard_code])

    if not values:
        return {}

    created_expr = created_col or "NULL"
    triage_ts_expr = f"TRY_CAST({created_expr} AS TIMESTAMP)"
    sql = f"""
        WITH qlist(question_id, hs_run_id, iso3, hazard_code) AS (
            VALUES {", ".join(values)}
        ),
        triage_latest AS (
            SELECT
              {run_col} AS run_id,
              {iso_col} AS iso3,
              {hazard_col} AS hazard_code,
              {tier_col} AS triage_tier,
              {score_col} AS triage_score,
              {need_col} AS triage_need_full_spd,
              {triage_ts_expr} AS triage_ts,
              ROW_NUMBER() OVER (
                PARTITION BY {run_col}, {iso_col}, {hazard_col}
                ORDER BY {triage_ts_expr} DESC NULLS LAST
              ) AS rn
            FROM hs_triage
        )
        SELECT
          qlist.question_id,
          triage_latest.triage_score,
          triage_latest.triage_tier,
          triage_latest.triage_need_full_spd,
          STRFTIME(triage_latest.triage_ts, '%Y-%m-%d') AS triage_date
        FROM qlist
        LEFT JOIN triage_latest
          ON triage_latest.run_id = qlist.hs_run_id
         AND UPPER(triage_latest.iso3) = UPPER(qlist.iso3)
         AND UPPER(triage_latest.hazard_code) = UPPER(qlist.hazard_code)
         AND triage_latest.rn = 1
    """

    try:
        triage_rows = conn.execute(sql, params).fetchall()
    except Exception:
        LOGGER.exception("Failed to compute questions triage summary")
        return {}

    summary: dict[str, dict[str, Any]] = {}
    for (
        question_id,
        triage_score,
        triage_tier,
        triage_need_full_spd,
        triage_date,
    ) in triage_rows or []:
        if not question_id:
            continue
        summary[str(question_id)] = {
            "triage_score": float(triage_score) if triage_score is not None else None,
            "triage_tier": str(triage_tier) if triage_tier is not None else None,
            "triage_need_full_spd": (
                bool(triage_need_full_spd) if triage_need_full_spd is not None else None
            ),
            "triage_date": str(triage_date) if triage_date is not None else None,
        }
    return summary
