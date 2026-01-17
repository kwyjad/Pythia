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

_FENCED_JSON_PATTERN = re.compile(r"```json\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _coerce_hazard_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score != score or score < 0 or score > 1:
        return None
    return score


def _extract_json_candidate(text: str | None) -> str | None:
    if not text:
        return None
    match = _FENCED_JSON_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    brace_match = re.search(r"\{[\s\S]*\}", text)
    bracket_match = re.search(r"\[[\s\S]*\]", text)
    if brace_match and bracket_match:
        brace_text = brace_match.group(0)
        bracket_text = bracket_match.group(0)
        return brace_text if len(brace_text) >= len(bracket_text) else bracket_text
    if brace_match:
        return brace_match.group(0)
    if bracket_match:
        return bracket_match.group(0)
    return None


def _as_float01(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.endswith("%"):
                parsed = float(cleaned[:-1].strip()) / 100
            else:
                parsed = float(cleaned)
        else:
            parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    if parsed < 0 or parsed > 1:
        return None
    return parsed


def _normalize_score_mapping(
    mapping: dict[Any, Any],
) -> tuple[dict[str, float], set[str], set[str]]:
    scores: dict[str, float] = {}
    hazards_seen: set[str] = set()
    invalid_scores: set[str] = set()
    for key, value in mapping.items():
        if key is None:
            continue
        code_str = str(key).strip().upper()
        if not code_str:
            continue
        hazards_seen.add(code_str)
        score = _as_float01(value)
        if score is None:
            if value is not None:
                invalid_scores.add(code_str)
            continue
        scores[code_str] = score
    return scores, hazards_seen, invalid_scores


def _extract_hazard_scores_with_diagnostics(
    response_text: str | None,
) -> tuple[dict[str, float], set[str], set[str], bool]:
    candidate = _extract_json_candidate(response_text)
    if not candidate:
        return {}, set(), set(), False
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return {}, set(), set(), False
    except TypeError:
        return {}, set(), set(), False

    scores: dict[str, float] = {}
    hazards_seen: set[str] = set()
    invalid_scores: set[str] = set()

    def record(code: Any, value: Any) -> None:
        if code is None:
            return
        code_str = str(code).strip().upper()
        if not code_str:
            return
        hazards_seen.add(code_str)
        score = _as_float01(value)
        if score is None:
            if value is not None:
                invalid_scores.add(code_str)
            return
        scores[code_str] = score

    def handle_item(item: Any) -> None:
        if isinstance(item, dict):
            code = item.get("hazard_code") or item.get("code")
            score = item.get("triage_score") if "triage_score" in item else item.get("score")
            record(code, score)

    def handle_mapping(mapping: dict[Any, Any]) -> None:
        for key, value in mapping.items():
            if isinstance(value, dict):
                score = value.get("triage_score") if "triage_score" in value else value.get("score")
                if score is not None:
                    record(key, score)
                else:
                    handle_item(value)
            else:
                record(key, value)

    if isinstance(payload, list):
        for item in payload:
            handle_item(item)
        return scores, hazards_seen, invalid_scores, True

    if isinstance(payload, dict):
        hazards = payload.get("hazards")
        if isinstance(hazards, dict):
            handle_mapping(hazards)
            return scores, hazards_seen, invalid_scores, True
        if isinstance(hazards, list):
            for item in hazards:
                handle_item(item)
            return scores, hazards_seen, invalid_scores, True
        results = payload.get("results")
        if isinstance(results, dict):
            handle_mapping(results)
            return scores, hazards_seen, invalid_scores, True
        if isinstance(results, list):
            for item in results:
                handle_item(item)
            return scores, hazards_seen, invalid_scores, True

        handle_mapping(payload)
    return scores, hazards_seen, invalid_scores, True


def _extract_hazard_scores(response_text: str | None) -> dict[str, float]:
    scores, _, _, _ = _extract_hazard_scores_with_diagnostics(response_text)
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
        "invalid_score_hazards": 0,
        "rows_with_invalid_score_value": 0,
        "notes": [],
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
    rc_prob_col = _pick_column(columns, ["regime_change_likelihood"])
    rc_dir_col = _pick_column(columns, ["regime_change_direction"])
    rc_mag_col = _pick_column(columns, ["regime_change_magnitude"])
    rc_score_col = _pick_column(columns, ["regime_change_score"])

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
    rc_prob_expr = (
        f"{rc_prob_col} AS regime_change_likelihood"
        if rc_prob_col
        else "NULL AS regime_change_likelihood"
    )
    rc_dir_expr = (
        f"{rc_dir_col} AS regime_change_direction"
        if rc_dir_col
        else "NULL AS regime_change_direction"
    )
    rc_mag_expr = (
        f"{rc_mag_col} AS regime_change_magnitude"
        if rc_mag_col
        else "NULL AS regime_change_magnitude"
    )
    rc_score_expr = (
        f"{rc_score_col} AS regime_change_score"
        if rc_score_col
        else "NULL AS regime_change_score"
    )

    if not (rc_prob_col and rc_dir_col and rc_mag_col and rc_score_col):
        diagnostics["notes"].append("hs_triage_rc_columns_missing")

    sql = f"""
        SELECT
          {run_col} AS run_id,
          {iso_expr},
          {hazard_expr},
          {score_expr},
          {tier_expr},
          {rc_prob_expr},
          {rc_dir_expr},
          {rc_mag_expr},
          {rc_score_expr},
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

    llm_calls: dict[str, list[dict[str, Any]]] = {}
    call_status_counts: dict[str, int] = {
        "ok": 0,
        "parse_error": 0,
        "response_missing": 0,
        "parse_failed": 0,
        "no_call": 0,
    }
    why_null_counts: dict[str, int] = {}
    if _table_exists(conn, "llm_calls"):
        llm_cols = _table_columns(conn, "llm_calls")
        llm_run_col = _pick_column(llm_cols, ["hs_run_id", "run_id"])
        llm_iso_col = _pick_column(llm_cols, ["iso3", "country_iso3"])
        llm_phase_col = _pick_column(llm_cols, ["phase"])
        llm_ts_col = _pick_column(llm_cols, ["created_at", "timestamp"])
        llm_response_col = _pick_column(llm_cols, ["response_text", "response", "completion"])
        llm_parse_col = _pick_column(llm_cols, ["parse_error"])
        llm_model_col = _pick_column(llm_cols, ["model_id", "model_name"])
        llm_provider_col = _pick_column(llm_cols, ["provider"])
        llm_hazard_col = _pick_column(llm_cols, ["hazard_code", "hazard"])
        llm_status_col = _pick_column(llm_cols, ["status"])
        llm_error_type_col = _pick_column(llm_cols, ["error_type"])
        llm_error_message_col = _pick_column(llm_cols, ["error_message"])
        llm_error_text_col = _pick_column(llm_cols, ["error_text"])
        llm_scores_json_col = _pick_column(llm_cols, ["hazard_scores_json"])
        llm_scores_parse_ok_col = _pick_column(llm_cols, ["hazard_scores_parse_ok"])
        llm_response_format_col = _pick_column(llm_cols, ["response_format"])

        if llm_run_col and llm_phase_col and llm_iso_col:
            created_expr = f"{llm_ts_col} AS created_at" if llm_ts_col else "NULL AS created_at"
            response_expr = (
                f"{llm_response_col} AS response_text"
                if llm_response_col
                else "NULL AS response_text"
            )
            iso_expr = f"UPPER({llm_iso_col}) AS iso3" if llm_iso_col else "NULL AS iso3"
            call_tag_expr = (
                f"{llm_hazard_col} AS call_tag" if llm_hazard_col else "NULL AS call_tag"
            )
            parse_expr = f"{llm_parse_col} AS parse_error" if llm_parse_col else "NULL AS parse_error"
            model_expr = f"{llm_model_col} AS model_id" if llm_model_col else "NULL AS model_id"
            provider_expr = f"{llm_provider_col} AS provider" if llm_provider_col else "NULL AS provider"
            status_expr = f"{llm_status_col} AS status" if llm_status_col else "NULL AS status"
            error_type_expr = (
                f"{llm_error_type_col} AS error_type" if llm_error_type_col else "NULL AS error_type"
            )
            error_message_expr = (
                f"{llm_error_message_col} AS error_message"
                if llm_error_message_col
                else "NULL AS error_message"
            )
            error_text_expr = (
                f"{llm_error_text_col} AS error_text"
                if llm_error_text_col
                else "NULL AS error_text"
            )
            scores_json_expr = (
                f"{llm_scores_json_col} AS hazard_scores_json"
                if llm_scores_json_col
                else "NULL AS hazard_scores_json"
            )
            scores_parse_ok_expr = (
                f"{llm_scores_parse_ok_col} AS hazard_scores_parse_ok"
                if llm_scores_parse_ok_col
                else "NULL AS hazard_scores_parse_ok"
            )
            response_format_expr = (
                f"{llm_response_format_col} AS response_format"
                if llm_response_format_col
                else "NULL AS response_format"
            )
            llm_where = f"{llm_phase_col} = 'hs_triage' AND {llm_run_col} = ?"
            pass_order_expr = "3"
            if llm_hazard_col:
                pass_order_expr = (
                    f"CASE"
                    f" WHEN {llm_hazard_col} IS NULL THEN 3"
                    f" WHEN UPPER({llm_hazard_col}) = 'PASS_1' THEN 1"
                    f" WHEN UPPER({llm_hazard_col}) = 'PASS_2' THEN 2"
                    f" ELSE 3"
                    f" END"
                )
            ts_order_expr = f"{llm_ts_col} DESC NULLS LAST" if llm_ts_col else "1"
            llm_sql = f"""
                SELECT
                  {created_expr},
                  {iso_expr},
                  {call_tag_expr},
                  {response_expr},
                  {parse_expr},
                  {model_expr},
                  {provider_expr},
                  {status_expr},
                  {error_type_expr},
                  {error_message_expr},
                  {error_text_expr},
                  {scores_json_expr},
                  {scores_parse_ok_expr},
                  {response_format_expr}
                FROM (
                  SELECT
                    *,
                    ROW_NUMBER() OVER (
                      PARTITION BY UPPER({llm_iso_col})
                      ORDER BY {pass_order_expr}, {ts_order_expr}
                    ) AS rn
                  FROM llm_calls
                  WHERE {llm_where}
                ) ranked_calls
                WHERE rn <= 2
            """
            llm_params: list[Any] = [run_id]
            if iso3 and llm_iso_col:
                llm_sql = llm_sql.replace(
                    llm_where,
                    f"{llm_where} AND UPPER({llm_iso_col}) = ?",
                )
                llm_params.append(iso3.upper())
            llm_sql += " ORDER BY iso3"

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
                    calls_for_iso = llm_calls.setdefault(str(iso_val), [])
                    if len(calls_for_iso) >= 2:
                        continue
                    diagnostics["total_calls"] += 1
                    response_text = row.get("response_text")
                    parse_error = row.get("parse_error")
                    response_text_present = bool(
                        response_text is not None and str(response_text).strip()
                    )
                    parse_error_present = bool(parse_error is not None and str(parse_error).strip())
                    status_raw = row.get("status")
                    status_value = str(status_raw).strip().lower() if status_raw else ""
                    error_type = row.get("error_type")
                    error_message = row.get("error_message")
                    error_text = row.get("error_text")
                    error_text_value = str(error_text).strip() if error_text else ""
                    scores_json = row.get("hazard_scores_json")
                    scores_parse_ok_value = row.get("hazard_scores_parse_ok")
                    scores_parse_ok = False
                    if scores_parse_ok_value is True:
                        scores_parse_ok = True
                    elif scores_parse_ok_value is not None:
                        scores_parse_ok = str(scores_parse_ok_value).strip().lower() in {
                            "true",
                            "1",
                            "yes",
                        }
                    hazard_scores: dict[str, float] = {}
                    hazards_seen: set[str] = set()
                    invalid_scores: set[str] = set()
                    json_parsed_ok = False
                    if scores_parse_ok and scores_json:
                        try:
                            parsed_scores = json.loads(scores_json)
                        except (TypeError, ValueError):
                            parsed_scores = None
                        if isinstance(parsed_scores, dict):
                            hazard_scores, hazards_seen, invalid_scores = _normalize_score_mapping(
                                parsed_scores
                            )
                            json_parsed_ok = True
                    if not hazard_scores and status_value != "error":
                        (
                            hazard_scores,
                            hazards_seen,
                            invalid_scores,
                            json_parsed_ok,
                        ) = _extract_hazard_scores_with_diagnostics(response_text)
                    structured_error_present = any(
                        value
                        for value in (status_raw, error_type, error_message)
                        if value is not None and str(value).strip()
                    )
                    legacy_error_status = ""
                    if not structured_error_present and error_text_value:
                        lowered_error = error_text_value.lower()
                        if "disabled after" in lowered_error:
                            legacy_error_status = "error:provider_disabled"
                        elif "read timed out" in lowered_error or "timeout" in lowered_error:
                            legacy_error_status = "error:timeout"
                        elif "triage parse failed" in lowered_error or "jsondecodeerror" in lowered_error:
                            legacy_error_status = "parse_error"
                        else:
                            legacy_error_status = "error:provider_error"
                    if status_value == "error":
                        error_type_str = (
                            str(error_type).strip().lower() if error_type else "unknown"
                        )
                        call_status = f"error:{error_type_str}"
                        hazard_scores = {}
                        hazards_seen = set()
                        invalid_scores = set()
                    elif legacy_error_status:
                        call_status = legacy_error_status
                        hazard_scores = {}
                        hazards_seen = set()
                        invalid_scores = set()
                    elif parse_error_present:
                        call_status = "parse_error"
                    elif hazard_scores:
                        call_status = "ok"
                    elif not response_text_present:
                        call_status = "response_missing"
                    else:
                        call_status = "parse_failed"
                    call_status_counts[call_status] = call_status_counts.get(call_status, 0) + 1
                    diagnostics["hazard_scores_extracted"] += len(hazard_scores)
                    diagnostics["invalid_score_hazards"] += len(invalid_scores)
                    calls_for_iso.append(
                        {
                            "created_at": row.get("created_at"),
                            "call_tag": row.get("call_tag"),
                            "model_id": row.get("model_id"),
                            "provider": row.get("provider"),
                            "status": status_raw,
                            "error_type": error_type,
                            "error_message": error_message,
                            "error_text": error_text,
                            "response_format": row.get("response_format"),
                            "parse_error_present": parse_error_present,
                            "response_text_present": response_text_present,
                            "hazard_scores": hazard_scores,
                            "hazards_seen": hazards_seen,
                            "invalid_scores": invalid_scores,
                            "call_status": call_status,
                        }
                    )

    iso_values = {
        str(iso_val)
        for iso_val in base_df["iso3"].dropna().astype(str).tolist()
        if str(iso_val).strip()
    }
    rows: list[dict[str, Any]] = []

    def select_calls(calls: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if not calls:
            return None, None
        tagged = {}
        for call in calls:
            tag = call.get("call_tag")
            if tag is None:
                continue
            tag_value = str(tag).strip().upper()
            if tag_value in {"PASS_1", "PASS_2"}:
                tagged[tag_value] = call
        call_1 = tagged.get("PASS_1")
        call_2 = tagged.get("PASS_2")
        if call_1 or call_2:
            remaining = [call for call in calls if call is not call_1 and call is not call_2]
            if call_1 is None and remaining:
                call_1 = remaining[0]
                remaining = remaining[1:]
            if call_2 is None and remaining:
                call_2 = remaining[0]
            return call_1, call_2
        call_1 = calls[0] if len(calls) > 0 else None
        call_2 = calls[1] if len(calls) > 1 else None
        return call_1, call_2

    for row in base_df.to_dict(orient="records"):
        iso_val = row.get("iso3")
        hazard_val = row.get("hazard_code")
        hazard_key = str(hazard_val).upper() if hazard_val is not None else ""
        call_candidates = llm_calls.get(str(iso_val), [])
        call_1, call_2 = select_calls(call_candidates)
        score_1 = (
            call_1.get("hazard_scores", {}).get(hazard_key) if call_1 else None
        )
        score_2 = (
            call_2.get("hazard_scores", {}).get(hazard_key) if call_2 else None
        )
        score_values = [score for score in (score_1, score_2) if score is not None]
        score_avg = sum(score_values) / len(score_values) if score_values else None
        score_avg_source = "calls" if score_values else "none"
        base_score = row.get("triage_score")
        if score_avg is None and base_score is not None:
            try:
                fallback_score = float(base_score)
            except (TypeError, ValueError):
                fallback_score = None
            if fallback_score is not None and fallback_score == fallback_score:
                score_avg = fallback_score
                score_avg_source = "hs_triage_fallback"
                diagnostics["avg_from_hs_triage_score"] += 1
                diagnostics["score_avg_from_hs_triage"] += 1
        if score_avg is not None and score_values:
            diagnostics["score_avg_from_calls"] += 1
        if score_avg is not None:
            diagnostics["rows_with_avg"] += 1
        is_null_avg = score_avg is None
        created_at = row.get("created_at")
        if hasattr(created_at, "date"):
            triage_date = created_at.date().isoformat()
        elif created_at:
            triage_date = str(created_at).split(" ")[0]
        else:
            triage_date = None
        call_1_status = call_1.get("call_status") if call_1 else "no_call"
        call_2_status = call_2.get("call_status") if call_2 else "no_call"

        def hazard_reason(call: dict[str, Any] | None) -> str:
            if not call:
                return "no_call"
            hazard_scores = call.get("hazard_scores") or {}
            hazards_seen = call.get("hazards_seen") or set()
            invalid_scores = call.get("invalid_scores") or set()
            if hazard_key in hazard_scores:
                return "ok"
            if hazard_key in invalid_scores:
                return "invalid_score_value"
            status = call.get("call_status")
            if status != "ok":
                return str(status)
            if hazard_key in hazards_seen:
                return "hazard_missing_in_response"
            return "hazard_missing_in_response"

        reason_1 = hazard_reason(call_1)
        reason_2 = hazard_reason(call_2)
        def is_non_ok(status: str) -> bool:
            return status.startswith("error:") or status in {
                "parse_failed",
                "parse_error",
                "response_missing",
            }

        why_null = ""
        if not (score_1 is not None and score_2 is not None):
            if call_1_status == "no_call" and call_2_status == "no_call":
                why_null = "no_calls"
            elif reason_1 == "invalid_score_value" or reason_2 == "invalid_score_value":
                why_null = "invalid_score_value"
            elif is_non_ok(call_1_status) and is_non_ok(call_2_status):
                why_null = f"call_failures:{call_1_status},{call_2_status}"
            elif (
                call_1_status == "ok"
                and reason_1 == "hazard_missing_in_response"
                and call_2_status == "no_call"
            ):
                why_null = "hazard_missing_in_call1"
            elif (
                call_1_status == "ok"
                and reason_1 == "hazard_missing_in_response"
                and call_2_status == "ok"
                and reason_2 == "hazard_missing_in_response"
            ):
                why_null = "hazard_missing_in_both_calls"
            elif (
                call_1_status == "ok"
                and reason_1 == "hazard_missing_in_response"
                and call_2_status == "ok"
                and reason_2 == "ok"
            ):
                why_null = "hazard_missing_in_call1"
            elif (
                call_1_status == "ok"
                and reason_1 == "ok"
                and call_2_status == "ok"
                and reason_2 == "hazard_missing_in_response"
            ):
                why_null = "hazard_missing_in_call2"
            elif is_non_ok(call_1_status) and call_2_status == "no_call":
                why_null = f"call_failures:{call_1_status},no_call"
            elif is_non_ok(call_2_status) and call_1_status == "no_call":
                why_null = f"call_failures:no_call,{call_2_status}"
        if why_null:
            why_null_counts[why_null] = why_null_counts.get(why_null, 0) + 1
            if why_null == "invalid_score_value":
                diagnostics["rows_with_invalid_score_value"] += 1
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
                "triage_score_avg_source": score_avg_source,
                "is_null_avg": is_null_avg,
                "regime_change_likelihood": row.get("regime_change_likelihood"),
                "regime_change_direction": row.get("regime_change_direction"),
                "regime_change_magnitude": row.get("regime_change_magnitude"),
                "regime_change_score": row.get("regime_change_score"),
                "call_1_status": call_1_status,
                "call_2_status": call_2_status,
                "why_null": why_null,
            }
        )

    rows.sort(
        key=lambda item: (
            (item.get("iso3") or ""),
            (item.get("triage_date") or ""),
        )
    )
    calls_grouped = len(llm_calls)
    diagnostics["calls_grouped_by_iso3"] = calls_grouped
    diagnostics["countries_with_two_calls"] = sum(
        1 for scores in llm_calls.values() if len(scores) >= 2
    )
    diagnostics["countries_with_one_call"] = sum(
        1 for scores in llm_calls.values() if len(scores) == 1
    )
    expected_pairs = 0
    extracted_pairs = 0
    for item in rows:
        call_candidates = llm_calls.get(str(item.get("iso3")), [])
        expected_pairs += min(2, len(call_candidates))
        extracted_pairs += sum(
            1
            for value in (item.get("triage_score_1"), item.get("triage_score_2"))
            if value is not None
        )
    diagnostics["hazard_scores_missing"] = max(expected_pairs - extracted_pairs, 0)
    diagnostics["parsed_scores"] = diagnostics["hazard_scores_extracted"]
    diagnostics["null_scores"] = diagnostics["hazard_scores_missing"]
    if iso_values:
        missing_calls = 0
        for iso_val in iso_values:
            count = len(llm_calls.get(str(iso_val), []))
            missing_calls += max(2 - count, 0)
        call_status_counts["no_call"] += missing_calls
    diagnostics["call_status_counts"] = call_status_counts
    diagnostics["why_null_counts"] = why_null_counts
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
