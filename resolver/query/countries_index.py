# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Compute country index stats from DuckDB for the API."""

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


def _base_rows(conn) -> dict[str, dict[str, Any]]:
    base = {}
    try:
        rows = conn.execute(
            """
            SELECT
              UPPER(iso3) AS iso3,
              COUNT(DISTINCT question_id) AS n_questions
            FROM questions
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchall()
    except Exception:
        LOGGER.exception("Failed to query base country rows")
        return base

    for iso3, n_questions in rows or []:
        if not iso3:
            continue
        base[str(iso3).upper()] = {
            "iso3": str(iso3).upper(),
            "n_questions": int(n_questions or 0),
            "n_forecasted": 0,
            "last_triaged": None,
            "last_forecasted": None,
        }
    return base


def _update_last_triaged(conn, base: dict[str, dict[str, Any]]) -> None:
    if not base:
        return
    if not _table_exists(conn, "hs_runs"):
        return

    q_cols = _table_columns(conn, "questions")
    h_cols = _table_columns(conn, "hs_runs")
    q_run_col = _pick_column(q_cols, ["hs_run_id", "run_id"])
    h_run_col = _pick_column(h_cols, ["hs_run_id", "run_id"])

    ts_expr = None
    if "created_at" in h_cols and "generated_at" in h_cols:
        ts_expr = "COALESCE(h.created_at, h.generated_at)"
    elif "created_at" in h_cols:
        ts_expr = "h.created_at"
    elif "generated_at" in h_cols:
        ts_expr = "h.generated_at"

    if not (q_run_col and h_run_col and ts_expr):
        LOGGER.info("Skipping last_triaged; hs_runs join columns missing.")
        return

    try:
        rows = conn.execute(
            f"""
            SELECT
              UPPER(q.iso3) AS iso3,
              STRFTIME(MAX({ts_expr}), '%Y-%m-%d') AS last_triaged
            FROM questions q
            JOIN hs_runs h ON q.{q_run_col} = h.{h_run_col}
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchall()
    except Exception:
        LOGGER.exception("Failed to compute last_triaged")
        return

    for iso3, last_triaged in rows or []:
        key = str(iso3).upper() if iso3 else None
        if key and key in base:
            base[key]["last_triaged"] = last_triaged


def _update_forecasts(conn, base: dict[str, dict[str, Any]]) -> None:
    if not base:
        return
    if not _table_exists(conn, "forecasts_ensemble"):
        return

    fe_cols = _table_columns(conn, "forecasts_ensemble")
    status_col = _pick_column(fe_cols, ["status"])
    prob_col = _pick_column(fe_cols, ["probability", "p"])
    horizon_col = _pick_column(fe_cols, ["month_index", "horizon_m"])
    bucket_col = _pick_column(fe_cols, ["bucket_index", "class_bin"])
    ts_col = _pick_column(fe_cols, ["created_at", "timestamp"])

    if status_col:
        filter_expr = f"fe.{status_col} = 'ok'"
    elif prob_col and horizon_col and bucket_col:
        filter_expr = (
            f"fe.{prob_col} IS NOT NULL AND "
            f"fe.{horizon_col} IS NOT NULL AND "
            f"fe.{bucket_col} IS NOT NULL"
        )
    else:
        LOGGER.info("Skipping forecast stats; forecasts_ensemble columns missing.")
        return

    last_expr = (
        f"STRFTIME(MAX(CASE WHEN {filter_expr} THEN fe.{ts_col} END), '%Y-%m-%d') "
        "AS last_forecasted"
        if ts_col
        else "NULL AS last_forecasted"
    )

    try:
        rows = conn.execute(
            f"""
            SELECT
              UPPER(q.iso3) AS iso3,
              COUNT(DISTINCT CASE WHEN {filter_expr} THEN fe.question_id END) AS n_forecasted,
              {last_expr}
            FROM questions q
            LEFT JOIN forecasts_ensemble fe ON fe.question_id = q.question_id
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchall()
    except Exception:
        LOGGER.exception("Failed to compute forecast stats")
        return

    for iso3, n_forecasted, last_forecasted in rows or []:
        key = str(iso3).upper() if iso3 else None
        if key and key in base:
            base[key]["n_forecasted"] = int(n_forecasted or 0)
            base[key]["last_forecasted"] = last_forecasted


def compute_countries_index(conn) -> list[dict[str, Any]]:
    if conn is None:
        return []
    if not _table_exists(conn, "questions"):
        return []

    q_cols = _table_columns(conn, "questions")
    if "iso3" not in q_cols or "question_id" not in q_cols:
        LOGGER.warning("questions table missing iso3/question_id columns.")
        return []

    base = _base_rows(conn)
    if not base:
        return []

    _update_last_triaged(conn, base)
    _update_forecasts(conn, base)

    return [base[key] for key in sorted(base)]
