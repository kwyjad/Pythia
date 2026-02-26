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

_INFO_FLAGS: set[str] = set()


def _info_once(key: str, message: str) -> None:
    if key in _INFO_FLAGS:
        return
    _INFO_FLAGS.add(key)
    LOGGER.info(message)

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


def _base_rows(
    conn,
    metric_scope: str | None = None,
    hs_run_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    base = {}
    try:
        clauses: list[str] = []
        params: list[str] = []
        if metric_scope:
            clauses.append("UPPER(metric) = ?")
            params.append(metric_scope.upper())
        if hs_run_id:
            clauses.append("hs_run_id = ?")
            params.append(hs_run_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = conn.execute(
            f"""
            SELECT
              UPPER(iso3) AS iso3,
              COUNT(DISTINCT question_id) AS n_questions
            FROM questions
            {where}
            GROUP BY 1
            ORDER BY 1
            """,
            params,
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
            "highest_rc_level": None,
            "highest_rc_score": None,
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


def _hs_run_id_for_month(conn, year_month: str) -> str | None:
    """Return the HS run ID whose timestamp falls within *year_month* (YYYY-MM).

    Prefers ``hs_runs``; falls back to ``hs_triage.created_at``.
    Returns the most-recent run in that month, or ``None``.
    """
    if _table_exists(conn, "hs_runs"):
        hs_cols = _table_columns(conn, "hs_runs")
        run_col = _pick_column(hs_cols, ["hs_run_id", "run_id"])
        if run_col:
            if "created_at" in hs_cols and "generated_at" in hs_cols:
                ts_expr = "COALESCE(created_at, generated_at)"
            elif "created_at" in hs_cols:
                ts_expr = "created_at"
            elif "generated_at" in hs_cols:
                ts_expr = "generated_at"
            else:
                ts_expr = None
            if ts_expr:
                row = conn.execute(
                    f"""
                    SELECT {run_col}
                    FROM hs_runs
                    WHERE STRFTIME({ts_expr}, '%Y-%m') = ?
                    ORDER BY {ts_expr} DESC NULLS LAST
                    LIMIT 1
                    """,
                    [year_month],
                ).fetchone()
                if row and row[0]:
                    return str(row[0])

    if _table_exists(conn, "hs_triage"):
        triage_cols = _table_columns(conn, "hs_triage")
        run_col = _pick_column(triage_cols, ["run_id", "hs_run_id"])
        if run_col and "created_at" in triage_cols:
            row = conn.execute(
                f"""
                SELECT DISTINCT {run_col}
                FROM hs_triage
                WHERE STRFTIME(created_at, '%Y-%m') = ?
                ORDER BY created_at DESC NULLS LAST
                LIMIT 1
                """,
                [year_month],
            ).fetchone()
            if row and row[0]:
                return str(row[0])

    return None


def _latest_hs_run_id(conn) -> str | None:
    if _table_exists(conn, "hs_runs"):
        hs_cols = _table_columns(conn, "hs_runs")
        run_col = _pick_column(hs_cols, ["hs_run_id", "run_id"])
        if run_col:
            if "created_at" in hs_cols and "generated_at" in hs_cols:
                ts_expr = "COALESCE(created_at, generated_at)"
            elif "created_at" in hs_cols:
                ts_expr = "created_at"
            elif "generated_at" in hs_cols:
                ts_expr = "generated_at"
            else:
                ts_expr = None
            if ts_expr:
                row = conn.execute(
                    f"""
                    SELECT {run_col}
                    FROM hs_runs
                    ORDER BY {ts_expr} DESC NULLS LAST
                    LIMIT 1
                    """
                ).fetchone()
                if row and row[0]:
                    return str(row[0])

    if not _table_exists(conn, "hs_triage"):
        return None

    triage_cols = _table_columns(conn, "hs_triage")
    run_col = _pick_column(triage_cols, ["run_id", "hs_run_id"])
    if not run_col:
        return None
    if "created_at" in triage_cols:
        order_expr = "created_at DESC NULLS LAST"
    else:
        order_expr = f"{run_col} DESC NULLS LAST"
    row = conn.execute(
        f"""
        SELECT {run_col}
        FROM hs_triage
        ORDER BY {order_expr}
        LIMIT 1
        """
    ).fetchone()
    if row and row[0]:
        return str(row[0])
    return None


def _update_highest_rc(
    conn,
    base: dict[str, dict[str, Any]],
    hs_run_id: str | None = None,
) -> None:
    """Populate ``highest_rc_level`` / ``highest_rc_score`` in *base*.

    When *hs_run_id* is given, RC data is taken from that single run (used when
    the caller has a specific run-month context).  When it is ``None``, RC data
    is joined through the ``questions`` table so each country gets RC values
    only from the run(s) that produced its questions.
    """
    if not base:
        return
    if not _table_exists(conn, "hs_triage"):
        return

    hs_cols = _table_columns(conn, "hs_triage")
    run_col = _pick_column(hs_cols, ["run_id", "hs_run_id"])
    iso_col = _pick_column(hs_cols, ["iso3"])
    level_col = _pick_column(hs_cols, ["regime_change_level"])
    score_col = _pick_column(hs_cols, ["regime_change_score"])

    if not (run_col and iso_col and level_col and score_col):
        _info_once(
            "highest_rc_missing_columns",
            "Skipping highest_rc fields; hs_triage missing regime_change columns.",
        )
        return

    try:
        if hs_run_id:
            # Single-run mode: take RC from the specified run only.
            rows = conn.execute(
                f"""
                SELECT
                  UPPER({iso_col}) AS iso3,
                  MAX({level_col}) AS highest_rc_level,
                  MAX({score_col}) AS highest_rc_score
                FROM hs_triage
                WHERE {run_col} = ?
                GROUP BY 1
                ORDER BY 1
                """,
                [hs_run_id],
            ).fetchall()
        else:
            # Per-country mode: join through questions so each country's RC
            # comes from the same run(s) that produced its questions.  This
            # avoids leaking RC data from a newer run onto countries whose
            # questions belong to an older run that predates RC scoring.
            rows = conn.execute(
                f"""
                SELECT
                  UPPER(ht.{iso_col}) AS iso3,
                  MAX(ht.{level_col}) AS highest_rc_level,
                  MAX(ht.{score_col}) AS highest_rc_score
                FROM hs_triage ht
                JOIN questions q
                  ON q.hs_run_id = ht.{run_col}
                 AND UPPER(q.iso3) = UPPER(ht.{iso_col})
                GROUP BY 1
                ORDER BY 1
                """,
            ).fetchall()
    except Exception:
        LOGGER.exception("Failed to compute highest_rc fields")
        return

    for iso3, highest_rc_level, highest_rc_score in rows or []:
        key = str(iso3).upper() if iso3 else None
        if key and key in base:
            base[key]["highest_rc_level"] = (
                int(highest_rc_level) if highest_rc_level is not None else None
            )
            base[key]["highest_rc_score"] = (
                float(highest_rc_score) if highest_rc_score is not None else None
            )


def compute_countries_index(
    conn,
    metric_scope: str | None = None,
    year_month: str | None = None,
) -> list[dict[str, Any]]:
    if conn is None:
        return []
    if not _table_exists(conn, "questions"):
        return []

    q_cols = _table_columns(conn, "questions")
    if "iso3" not in q_cols or "question_id" not in q_cols:
        LOGGER.warning("questions table missing iso3/question_id columns.")
        return []

    # Resolve the HS run for the requested month (or latest).
    hs_run_id: str | None = None
    if year_month:
        hs_run_id = _hs_run_id_for_month(conn, year_month)
    if not hs_run_id:
        hs_run_id = _latest_hs_run_id(conn)

    # Filter base rows by metric and run.
    run_filter = hs_run_id if year_month else None
    base = _base_rows(conn, metric_scope=metric_scope, hs_run_id=run_filter)
    if not base:
        return []

    _update_last_triaged(conn, base)
    _update_forecasts(conn, base)
    # When a specific month is requested, use that run's RC data.
    # Otherwise let _update_highest_rc join through questions so each
    # country's RC comes from the run(s) that produced its questions.
    _update_highest_rc(conn, base, hs_run_id=hs_run_id if year_month else None)

    return [base[key] for key in sorted(base)]
