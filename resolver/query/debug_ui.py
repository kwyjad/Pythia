# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""DuckDB query helpers for the debug UI."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import pandas as pd

from resolver.query.downloads import _load_country_registry, _load_triage_models

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:  # pragma: no cover - silence library default
    LOGGER.addHandler(logging.NullHandler())

HAZARD_LABELS = {
    "ACE": "Armed Conflict",
    "DI": "Displacement Inflow",
    "DR": "Drought",
    "FL": "Flood",
    "HW": "Heatwave",
    "TC": "Tropical Cyclone",
}


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


def _rows_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    sanitized = df.where(pd.notnull(df), None)
    return sanitized.to_dict(orient="records")


def _count_distinct_pair(iso_col: str, hazard_col: str) -> str:
    return (
        "COUNT(DISTINCT COALESCE(UPPER({iso}), '') || '|' || "
        "COALESCE(UPPER({hazard}), ''))"
    ).format(iso=iso_col, hazard=hazard_col)


def _list_hs_runs_with_debug(
    conn, limit: int = 50
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    debug: dict[str, Any] = {"source_table": None, "columns": {}, "notes": []}

    table = None
    phase_filter = None
    if _table_exists(conn, "hs_triage"):
        table = "hs_triage"
    elif _table_exists(conn, "llm_calls"):
        table = "llm_calls"
        phase_filter = "hs_triage"

    if not table:
        debug["notes"].append("hs_triage_unavailable")
        return [], debug

    columns = _table_columns(conn, table)
    run_col = _pick_column(columns, ["run_id", "hs_run_id"])
    iso_col = _pick_column(columns, ["iso3", "country_iso3"])
    hazard_col = _pick_column(columns, ["hazard_code", "hazard"])
    ts_col = _pick_column(columns, ["created_at", "timestamp"])
    phase_col = _pick_column(columns, ["phase"]) if phase_filter else None

    debug["source_table"] = table
    debug["columns"] = {
        "run_id": run_col,
        "iso3": iso_col,
        "hazard_code": hazard_col,
        "timestamp": ts_col,
        "phase": phase_col,
    }

    if not run_col:
        debug["notes"].append("run_id_missing")
        return [], debug

    triage_date_expr = "NULL AS triage_date"
    triage_year_expr = "NULL AS triage_year"
    triage_month_expr = "NULL AS triage_month"
    order_expr = f"{run_col} DESC"
    if ts_col:
        triage_date_expr = f"STRFTIME(MAX({ts_col}), '%Y-%m-%d') AS triage_date"
        triage_year_expr = f"CAST(STRFTIME(MAX({ts_col}), '%Y') AS INTEGER) AS triage_year"
        triage_month_expr = (
            f"CAST(STRFTIME(MAX({ts_col}), '%m') AS INTEGER) AS triage_month"
        )
        order_expr = f"MAX({ts_col}) DESC NULLS LAST"

    if iso_col:
        countries_expr = f"COUNT(DISTINCT UPPER({iso_col})) AS countries_triaged"
    else:
        countries_expr = "NULL AS countries_triaged"

    if iso_col and hazard_col:
        hazards_expr = f"{_count_distinct_pair(iso_col, hazard_col)} AS hazards_triaged"
    else:
        hazards_expr = "COUNT(*) AS hazards_triaged"

    sql = f"""
        SELECT
          {run_col} AS run_id,
          {triage_date_expr},
          {triage_year_expr},
          {triage_month_expr},
          {countries_expr},
          {hazards_expr}
        FROM {table}
    """
    params: list[Any] = []
    if phase_filter:
        if not phase_col:
            debug["notes"].append("phase_missing")
            return [], debug
        sql += f" WHERE {phase_col} = ?"
        params.append(phase_filter)

    sql += f" GROUP BY 1 ORDER BY {order_expr} LIMIT ?"
    params.append(limit)

    try:
        df = conn.execute(sql, params).fetchdf()
    except Exception:
        LOGGER.exception("Failed to list hs runs")
        return [], debug

    return _rows_from_df(df), debug


def list_hs_runs(conn, limit: int = 50) -> list[dict[str, Any]]:
    rows, _ = _list_hs_runs_with_debug(conn, limit=limit)
    return rows


def _get_hs_triage_rows_with_debug(
    conn,
    run_id: str,
    iso3: str | None = None,
    hazard_code: str | None = None,
    limit: int = 500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    debug: dict[str, Any] = {"columns": {}, "notes": []}
    if not _table_exists(conn, "hs_triage"):
        debug["notes"].append("hs_triage_missing")
        return [], debug

    columns = _table_columns(conn, "hs_triage")
    run_col = _pick_column(columns, ["run_id", "hs_run_id"])
    iso_col = _pick_column(columns, ["iso3", "country_iso3"])
    hazard_col = _pick_column(columns, ["hazard_code", "hazard"])
    score_col = _pick_column(columns, ["triage_score", "score"])
    tier_col = _pick_column(columns, ["tier", "triage_tier"])
    ts_col = _pick_column(columns, ["created_at", "timestamp"])

    debug["columns"] = {
        "run_id": run_col,
        "iso3": iso_col,
        "hazard_code": hazard_col,
        "triage_score": score_col,
        "tier": tier_col,
        "timestamp": ts_col,
    }

    if not run_col:
        debug["notes"].append("run_id_missing")
        return [], debug

    created_expr = f"{ts_col} AS created_at" if ts_col else "NULL AS created_at"
    score_expr = f"{score_col} AS triage_score" if score_col else "NULL AS triage_score"
    tier_expr = f"{tier_col} AS tier" if tier_col else "NULL AS tier"
    hazard_expr = f"{hazard_col} AS hazard_code" if hazard_col else "NULL AS hazard_code"
    iso_expr = f"UPPER({iso_col}) AS iso3" if iso_col else "NULL AS iso3"

    sql = f"""
        SELECT
          {run_col} AS run_id,
          {iso_expr},
          {hazard_expr},
          {score_expr},
          {tier_expr},
          {created_expr}
        FROM hs_triage
        WHERE {run_col} = ?
    """
    params: list[Any] = [run_id]
    if iso3 and iso_col:
        sql += f" AND UPPER({iso_col}) = ?"
        params.append(iso3.upper())
    if hazard_code and hazard_col:
        sql += f" AND {hazard_col} = ?"
        params.append(hazard_code)

    if ts_col:
        sql += f" ORDER BY {ts_col} DESC NULLS LAST"
    else:
        sql += " ORDER BY 1"

    sql += " LIMIT ?"
    params.append(limit)

    try:
        df = conn.execute(sql, params).fetchdf()
    except Exception:
        LOGGER.exception("Failed to query hs_triage rows")
        return [], debug

    return _rows_from_df(df), debug


def get_hs_triage_rows(
    conn,
    run_id: str,
    iso3: str | None = None,
    hazard_code: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    rows, _ = _get_hs_triage_rows_with_debug(
        conn, run_id=run_id, iso3=iso3, hazard_code=hazard_code, limit=limit
    )
    return rows


def _get_hs_triage_llm_calls_with_debug(
    conn,
    run_id: str,
    iso3: str | None = None,
    hazard_code: str | None = None,
    limit: int = 200,
    preview_chars: int = 800,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    debug: dict[str, Any] = {"columns": {}, "notes": []}
    if not _table_exists(conn, "llm_calls"):
        debug["notes"].append("llm_calls_missing")
        return [], debug

    columns = _table_columns(conn, "llm_calls")
    run_col = _pick_column(columns, ["hs_run_id", "run_id"])
    ts_col = _pick_column(columns, ["created_at", "timestamp"])
    iso_col = _pick_column(columns, ["iso3", "country_iso3"])
    hazard_col = _pick_column(columns, ["hazard_code", "hazard"])
    phase_col = _pick_column(columns, ["phase"])
    response_col = _pick_column(columns, ["response_text", "response", "completion"])
    parse_col = _pick_column(columns, ["parse_error"])
    model_col = _pick_column(columns, ["model_id", "model_name"])
    provider_col = _pick_column(columns, ["provider"])

    debug["columns"] = {
        "run_id": run_col,
        "timestamp": ts_col,
        "iso3": iso_col,
        "hazard_code": hazard_col,
        "phase": phase_col,
        "response_text": response_col,
        "parse_error": parse_col,
        "model_id": model_col,
        "provider": provider_col,
    }

    if not run_col:
        debug["notes"].append("run_id_missing")
        return [], debug
    if not phase_col:
        debug["notes"].append("phase_missing")
        return [], debug

    created_expr = f"{ts_col} AS created_at" if ts_col else "NULL AS created_at"
    response_expr = (
        f"SUBSTR({response_col}, 1, {int(preview_chars)}) AS response_preview"
        if response_col
        else "NULL AS response_preview"
    )
    parse_expr = f"{parse_col} AS parse_error" if parse_col else "NULL AS parse_error"
    model_expr = f"{model_col} AS model_id" if model_col else "NULL AS model_id"
    provider_expr = f"{provider_col} AS provider" if provider_col else "NULL AS provider"
    iso_expr = f"UPPER({iso_col}) AS iso3" if iso_col else "NULL AS iso3"
    hazard_expr = f"{hazard_col} AS hazard_code" if hazard_col else "NULL AS hazard_code"

    sql = f"""
        SELECT
          {created_expr},
          {iso_expr},
          {hazard_expr},
          {model_expr},
          {provider_expr},
          {parse_expr},
          {response_expr}
        FROM llm_calls
        WHERE {phase_col} = 'hs_triage'
          AND {run_col} = ?
    """
    params: list[Any] = [run_id]

    if iso3 and iso_col:
        sql += f" AND UPPER({iso_col}) = ?"
        params.append(iso3.upper())
    if hazard_code and hazard_col:
        sql += f" AND {hazard_col} = ?"
        params.append(hazard_code)

    if ts_col:
        sql += f" ORDER BY {ts_col} DESC NULLS LAST"
    else:
        sql += " ORDER BY 1"

    sql += " LIMIT ?"
    params.append(limit)

    try:
        df = conn.execute(sql, params).fetchdf()
    except Exception:
        LOGGER.exception("Failed to query hs_triage llm_calls")
        return [], debug

    return _rows_from_df(df), debug


def get_hs_triage_llm_calls(
    conn,
    run_id: str,
    iso3: str | None = None,
    hazard_code: str | None = None,
    limit: int = 200,
    preview_chars: int = 800,
) -> list[dict[str, Any]]:
    rows, _ = _get_hs_triage_llm_calls_with_debug(
        conn,
        run_id=run_id,
        iso3=iso3,
        hazard_code=hazard_code,
        limit=limit,
        preview_chars=preview_chars,
    )
    return rows


_TRIAGE_SCORE_PATTERN = re.compile(
    r"triage_score\"?\s*[:=]\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _coerce_hazard_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score != score or score < 0 or score > 1:
        return None
    return score


def _extract_hazard_scores(response_text: str | None) -> dict[str, float]:
    if not response_text:
        return {}
    try:
        payload = json.loads(response_text.strip())
    except json.JSONDecodeError:
        return {}
    except TypeError:
        return {}

    scores: dict[str, float] = {}

    def record(code: Any, value: Any) -> None:
        if code is None:
            return
        code_str = str(code).strip().upper()
        if not code_str:
            return
        score = _coerce_hazard_score(value)
        if score is None:
            return
        scores[code_str] = score

    def handle_item(item: Any) -> None:
        if isinstance(item, dict):
            code = item.get("hazard_code") or item.get("code")
            score = item.get("triage_score") if "triage_score" in item else item.get("score")
            record(code, score)

    if isinstance(payload, list):
        for item in payload:
            handle_item(item)
        return scores

    if isinstance(payload, dict):
        hazards = payload.get("hazards")
        if isinstance(hazards, list):
            for item in hazards:
                handle_item(item)
            return scores
        results = payload.get("results")
        if isinstance(results, list):
            for item in results:
                handle_item(item)
            return scores

        for key, value in payload.items():
            if isinstance(value, dict):
                score = value.get("triage_score") if "triage_score" in value else value.get("score")
                record(key, score)
            else:
                record(key, value)
    return scores


def _extract_triage_score(value: Any, response_text: str | None) -> float | None:
    if value is not None:
        try:
            score = float(value)
            if score == score:
                return score
        except (TypeError, ValueError):
            pass
    if not response_text:
        return None
    match = _TRIAGE_SCORE_PATTERN.search(response_text)
    if not match:
        return None
    try:
        score = float(match.group(1))
    except ValueError:
        return None
    return score if score == score else None


def get_hs_triage_all(
    conn,
    run_id: str,
    iso3: str | None = None,
    hazard_code: str | None = None,
    limit: int = 2000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "parsed_scores": 0,
        "null_scores": 0,
        "total_calls": 0,
        "avg_from_hs_triage_score": 0,
        "hazard_code_normalized": True,
        "rows_returned": 0,
        "rows_with_avg": 0,
        "calls_grouped_by_iso3": 0,
        "countries_with_two_calls": 0,
        "countries_with_one_call": 0,
        "hazard_scores_extracted": 0,
        "hazard_scores_missing": 0,
        "score_avg_from_calls": 0,
        "score_avg_from_hs_triage": 0,
    }
    if not _table_exists(conn, "hs_triage"):
        diagnostics["notes"] = ["hs_triage_missing"]
        return [], diagnostics

    columns = _table_columns(conn, "hs_triage")
    run_col = _pick_column(columns, ["run_id", "hs_run_id"])
    iso_col = _pick_column(columns, ["iso3", "country_iso3"])
    hazard_col = _pick_column(columns, ["hazard_code", "hazard"])
    score_col = _pick_column(columns, ["triage_score", "score"])
    tier_col = _pick_column(columns, ["tier", "triage_tier"])
    ts_col = _pick_column(columns, ["created_at", "timestamp"])

    if not run_col:
        diagnostics["notes"] = ["run_id_missing"]
        return [], diagnostics

    created_expr = f"{ts_col} AS created_at" if ts_col else "NULL AS created_at"
    score_expr = f"{score_col} AS triage_score" if score_col else "NULL AS triage_score"
    tier_expr = f"{tier_col} AS tier" if tier_col else "NULL AS tier"
    hazard_expr = (
        f"UPPER({hazard_col}) AS hazard_code" if hazard_col else "NULL AS hazard_code"
    )
    iso_expr = f"UPPER({iso_col}) AS iso3" if iso_col else "NULL AS iso3"

    sql = f"""
        SELECT
          {run_col} AS run_id,
          {iso_expr},
          {hazard_expr},
          {score_expr},
          {tier_expr},
          {created_expr}
        FROM hs_triage
        WHERE {run_col} = ?
    """
    params: list[Any] = [run_id]
    if iso3 and iso_col:
        sql += f" AND UPPER({iso_col}) = ?"
        params.append(iso3.upper())
    if hazard_code and hazard_col:
        sql += f" AND UPPER({hazard_col}) = ?"
        params.append(hazard_code.upper())
    if ts_col:
        sql += f" ORDER BY {ts_col} DESC NULLS LAST"
    else:
        sql += " ORDER BY 1"
    sql += " LIMIT ?"
    params.append(limit)

    try:
        base_df = conn.execute(sql, params).fetchdf()
    except Exception:
        LOGGER.exception("Failed to query hs_triage rows")
        diagnostics["notes"] = ["hs_triage_query_failed"]
        return [], diagnostics

    if base_df.empty:
        return [], diagnostics

    country_map = _load_country_registry()
    model_map = _load_triage_models(conn)

    llm_scores: dict[str, list[dict[str, float]]] = {}
    if _table_exists(conn, "llm_calls"):
        llm_cols = _table_columns(conn, "llm_calls")
        llm_run_col = _pick_column(llm_cols, ["hs_run_id", "run_id"])
        llm_iso_col = _pick_column(llm_cols, ["iso3", "country_iso3"])
        llm_phase_col = _pick_column(llm_cols, ["phase"])
        llm_ts_col = _pick_column(llm_cols, ["created_at", "timestamp"])
        llm_response_col = _pick_column(llm_cols, ["response_text", "response", "completion"])
        llm_parse_col = _pick_column(llm_cols, ["parse_error"])

        if llm_run_col and llm_phase_col:
            created_expr = f"{llm_ts_col} AS created_at" if llm_ts_col else "NULL AS created_at"
            response_expr = (
                f"SUBSTR({llm_response_col}, 1, 2000) AS response_text"
                if llm_response_col
                else "NULL AS response_text"
            )
            iso_expr = f"UPPER({llm_iso_col}) AS iso3" if llm_iso_col else "NULL AS iso3"
            parse_expr = f"{llm_parse_col} AS parse_error" if llm_parse_col else "NULL AS parse_error"
            llm_sql = f"""
                SELECT
                  {created_expr},
                  {iso_expr},
                  {response_expr},
                  {parse_expr}
                FROM llm_calls
                WHERE {llm_phase_col} = 'hs_triage'
                  AND {llm_run_col} = ?
            """
            llm_params: list[Any] = [run_id]
            if iso3 and llm_iso_col:
                llm_sql += f" AND UPPER({llm_iso_col}) = ?"
                llm_params.append(iso3.upper())
            if llm_ts_col:
                llm_sql += f" ORDER BY {llm_ts_col} DESC NULLS LAST"
            else:
                llm_sql += " ORDER BY 1"
            llm_sql += " LIMIT ?"
            llm_params.append(limit * 6)

            try:
                llm_df = conn.execute(llm_sql, llm_params).fetchdf()
            except Exception:
                LOGGER.exception("Failed to query llm_calls triage scores")
                llm_df = None

            if llm_df is not None and not llm_df.empty:
                for row in llm_df.to_dict(orient="records"):
                    iso_val = row.get("iso3")
                    if not iso_val:
                        continue
                    scores_for_iso = llm_scores.setdefault(str(iso_val), [])
                    if len(scores_for_iso) >= 2:
                        continue
                    diagnostics["total_calls"] += 1
                    hazard_scores = _extract_hazard_scores(row.get("response_text"))
                    diagnostics["hazard_scores_extracted"] += len(hazard_scores)
                    scores_for_iso.append(hazard_scores)

    rows: list[dict[str, Any]] = []
    for row in base_df.to_dict(orient="records"):
        iso_val = row.get("iso3")
        hazard_val = row.get("hazard_code")
        hazard_key = str(hazard_val).upper() if hazard_val is not None else ""
        score_candidates = llm_scores.get(str(iso_val), [])
        score_1 = (
            score_candidates[0].get(hazard_key) if len(score_candidates) > 0 else None
        )
        score_2 = (
            score_candidates[1].get(hazard_key) if len(score_candidates) > 1 else None
        )
        score_values = [score for score in (score_1, score_2) if score is not None]
        score_avg = sum(score_values) / len(score_values) if score_values else None
        base_score = row.get("triage_score")
        if score_avg is None and base_score is not None:
            try:
                fallback_score = float(base_score)
            except (TypeError, ValueError):
                fallback_score = None
            if fallback_score is not None and fallback_score == fallback_score:
                score_avg = fallback_score
                diagnostics["avg_from_hs_triage_score"] += 1
                diagnostics["score_avg_from_hs_triage"] += 1
        if score_avg is not None and score_values:
            diagnostics["score_avg_from_calls"] += 1
        if score_avg is not None:
            diagnostics["rows_with_avg"] += 1
        created_at = row.get("created_at")
        if hasattr(created_at, "date"):
            triage_date = created_at.date().isoformat()
        elif created_at:
            triage_date = str(created_at).split(" ")[0]
        else:
            triage_date = None
        rows.append(
            {
                "triage_date": triage_date,
                "run_id": row.get("run_id"),
                "iso3": iso_val,
                "hazard_code": hazard_val,
                "hazard_label": HAZARD_LABELS.get(hazard_key, hazard_val),
                "country": country_map.get(str(iso_val).upper(), iso_val) if iso_val else None,
                "triage_tier": row.get("tier"),
                "triage_model": model_map.get(run_id),
                "triage_score_1": score_1,
                "triage_score_2": score_2,
                "triage_score_avg": score_avg,
            }
        )

    rows.sort(
        key=lambda item: (
            (item.get("iso3") or ""),
            (item.get("triage_date") or ""),
        )
    )
    calls_grouped = len(llm_scores)
    diagnostics["calls_grouped_by_iso3"] = calls_grouped
    diagnostics["countries_with_two_calls"] = sum(
        1 for scores in llm_scores.values() if len(scores) >= 2
    )
    diagnostics["countries_with_one_call"] = sum(
        1 for scores in llm_scores.values() if len(scores) == 1
    )
    expected_pairs = 0
    extracted_pairs = 0
    for item in rows:
        score_candidates = llm_scores.get(str(item.get("iso3")), [])
        expected_pairs += min(2, len(score_candidates))
        extracted_pairs += sum(
            1
            for value in (item.get("triage_score_1"), item.get("triage_score_2"))
            if value is not None
        )
    diagnostics["hazard_scores_missing"] = max(expected_pairs - extracted_pairs, 0)
    diagnostics["parsed_scores"] = diagnostics["hazard_scores_extracted"]
    diagnostics["null_scores"] = diagnostics["hazard_scores_missing"]
    diagnostics["rows_returned"] = len(rows)
    return rows, diagnostics


def get_country_run_summary(conn, run_id: str, iso3: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "run_id": run_id,
        "iso3": iso3.upper(),
        "hazards_triaged": None,
        "questions_generated": None,
        "questions_forecasted": None,
        "notes": [],
        "diagnostics": {},
    }

    iso3_upper = iso3.upper()

    if _table_exists(conn, "hs_triage"):
        h_cols = _table_columns(conn, "hs_triage")
        run_col = _pick_column(h_cols, ["run_id", "hs_run_id"])
        iso_col = _pick_column(h_cols, ["iso3", "country_iso3"])
        if run_col and iso_col:
            try:
                row = conn.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM hs_triage
                    WHERE {run_col} = ? AND UPPER({iso_col}) = ?
                    """,
                    [run_id, iso3_upper],
                ).fetchone()
                summary["hazards_triaged"] = int(row[0]) if row else 0
            except Exception:
                LOGGER.exception("Failed to count hs_triage rows")
                summary["notes"].append("hs_triage_count_failed")
        else:
            summary["notes"].append("hs_triage_columns_missing")
    else:
        summary["notes"].append("hs_triage_missing")

    if _table_exists(conn, "questions"):
        q_cols = _table_columns(conn, "questions")
        run_col = _pick_column(q_cols, ["hs_run_id", "run_id"])
        iso_col = _pick_column(q_cols, ["iso3", "country_iso3"])
        if run_col and iso_col:
            try:
                row = conn.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM questions
                    WHERE {run_col} = ? AND UPPER({iso_col}) = ?
                    """,
                    [run_id, iso3_upper],
                ).fetchone()
                summary["questions_generated"] = int(row[0]) if row else 0
            except Exception:
                LOGGER.exception("Failed to count questions")
                summary["notes"].append("questions_count_failed")
        else:
            summary["notes"].append("questions_columns_missing")
    else:
        summary["notes"].append("questions_missing")

    if _table_exists(conn, "forecasts_ensemble"):
        fe_cols = _table_columns(conn, "forecasts_ensemble")
        fe_run_col = _pick_column(fe_cols, ["hs_run_id", "run_id"])
        question_col = _pick_column(fe_cols, ["question_id"])
        if not question_col:
            summary["notes"].append("forecasts_question_id_missing")
        else:
            q_cols = _table_columns(conn, "questions") if _table_exists(conn, "questions") else set()
            q_run_col = _pick_column(q_cols, ["hs_run_id", "run_id"])
            q_iso_col = _pick_column(q_cols, ["iso3", "country_iso3"])

            if fe_run_col and q_iso_col:
                summary["diagnostics"]["forecasts_source"] = "forecasts_table_direct"
                try:
                    row = conn.execute(
                        f"""
                        SELECT COUNT(DISTINCT fe.{question_col})
                        FROM forecasts_ensemble fe
                        JOIN questions q ON q.question_id = fe.{question_col}
                        WHERE fe.{fe_run_col} = ? AND UPPER(q.{q_iso_col}) = ?
                        """,
                        [run_id, iso3_upper],
                    ).fetchone()
                    summary["questions_forecasted"] = int(row[0]) if row else 0
                except Exception:
                    LOGGER.exception("Failed to count forecasts (run_id join)")
                    summary["notes"].append("forecasts_count_failed")
            elif q_run_col and q_iso_col:
                summary["diagnostics"]["forecasts_source"] = "fallback_join_via_questions"
                summary["diagnostics"]["forecasts_run_id_missing_fallback"] = True
                LOGGER.info(
                    "Debug summary forecasts fallback run_id=%s iso3=%s source=%s",
                    run_id,
                    iso3_upper,
                    summary["diagnostics"]["forecasts_source"],
                )
                try:
                    row = conn.execute(
                        f"""
                        SELECT COUNT(DISTINCT fe.{question_col})
                        FROM forecasts_ensemble fe
                        JOIN questions q ON q.question_id = fe.{question_col}
                        WHERE q.{q_run_col} = ? AND UPPER(q.{q_iso_col}) = ?
                        """,
                        [run_id, iso3_upper],
                    ).fetchone()
                    summary["questions_forecasted"] = int(row[0]) if row else 0
                except Exception:
                    LOGGER.exception("Failed to count forecasts (questions join)")
                    summary["notes"].append("forecasts_count_failed")
            else:
                summary["notes"].append("forecasts_columns_missing")
    else:
        summary["notes"].append("forecasts_missing")

    return summary


__all__ = [
    "list_hs_runs",
    "get_hs_triage_rows",
    "get_hs_triage_llm_calls",
    "get_hs_triage_all",
    "get_country_run_summary",
    "_list_hs_runs_with_debug",
    "_get_hs_triage_rows_with_debug",
    "_get_hs_triage_llm_calls_with_debug",
]
