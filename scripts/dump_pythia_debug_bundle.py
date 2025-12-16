# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import duckdb
from resolver.db import duckdb_io

try:
    from forecaster.prompts import _bucket_labels_for_question  # type: ignore
except ImportError:  # pragma: no cover - optional helper
    _bucket_labels_for_question = None


def _fetch_llm_rows(
    con: duckdb.DuckDBPyConnection,
    query: str,
    params: list[Any],
) -> list[dict[str, Any]]:
    cur = con.execute(query, params)
    rows = cur.fetchall()
    desc = cur.description or []
    col_names = [d[0] for d in desc]
    return [dict(zip(col_names, row)) for row in rows]


def _build_usage_json_from_row(row: dict[str, Any]) -> str:
    # If there's already a usage_json column, prefer it
    usage_raw = row.get("usage_json")
    if isinstance(usage_raw, str) and usage_raw.strip():
        return usage_raw

    # Otherwise synthesize a usage dict from known numeric columns if they exist
    usage: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "elapsed_ms",
        "cost_usd",
        "cost",
        "input_cost_usd",
        "output_cost_usd",
        "total_cost_usd",
    ):
        if key in row and row[key] is not None:
            usage[key] = row[key]

    return json.dumps(usage) if usage else "{}"


def _load_bucket_centroids_for_question(
    con: duckdb.DuckDBPyConnection,
    hazard_code: str,
    metric: str,
    bucket_labels: List[str],
) -> List[float]:
    """
    Return centroids (aligned to bucket_labels) for a hazard/metric.
    Prefer DB bucket_centroids; fallback to default PA/FATALITIES values.
    """

    hz = (hazard_code or "").upper()
    m = (metric or "").upper()

    try:
        rows = con.execute(
            """
            SELECT bucket_index, centroid
            FROM bucket_centroids
            WHERE hazard_code = ? AND metric = ?
            ORDER BY bucket_index
            """,
            [hz, m],
        ).fetchall()
    except Exception:
        rows = []

    if rows:
        centroids: List[float] = [0.0] * len(bucket_labels)
        for idx, centroid in rows:
            # bucket_index may be 0- or 1-based; normalise to 0-based
            i = int(idx) - 1 if int(idx) > 0 else int(idx)
            if 0 <= i < len(centroids):
                centroids[i] = float(centroid or 0.0)
        return centroids

    if m == "FATALITIES":
        return [0.0, 15.0, 62.0, 300.0, 700.0]
    return [0.0, 30_000.0, 150_000.0, 375_000.0, 700_000.0]


def _get_bucket_labels_for_question(question: Dict[str, Any]) -> List[str]:
    explicit = question.get("bucket_labels") or question.get("class_bins")
    if isinstance(explicit, list) and len(explicit) > 0:
        return [str(x) for x in explicit]

    if _bucket_labels_for_question is not None:
        return list(_bucket_labels_for_question(question))

    metric = (question.get("metric") or "").upper()
    if metric == "FATALITIES":
        return ["<5", "5-<25", "25-<100", "100-<500", ">=500"]
    return ["<10k", "10k-<50k", "50k-<250k", "250k-<500k", ">=500k"]


def _load_ensemble_spd_for_question(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    question_id: str,
    centroids: list[float],
) -> Dict[int, Dict[str, Any]]:
    """
    Return {month_index: {"probs": [p1..p5], "ev_value": ev or None}}.

    Falls back to computing EV from bucket probabilities and provided centroids
    when the database does not contain an ev_value.
    """

    rows = con.execute(
        """
        SELECT month_index, bucket_index, probability, ev_value
        FROM forecasts_ensemble
        WHERE run_id = ? AND question_id = ?
          AND month_index IS NOT NULL
          AND bucket_index IS NOT NULL
        ORDER BY month_index, bucket_index
        """,
        [run_id, question_id],
    ).fetchall()

    bucket_count = len(centroids) if centroids else 5
    by_month: Dict[int, Dict[str, Any]] = {}
    for month_idx, bucket_idx, prob, ev_value in rows:
        if month_idx is None or bucket_idx is None:
            continue
        m = int(month_idx)
        b = int(bucket_idx)
        entry = by_month.setdefault(m, {"probs": [0.0] * bucket_count, "ev_value": None})
        if 1 <= b <= bucket_count:
            entry["probs"][b - 1] = float(prob or 0.0)
        elif 0 <= b < bucket_count:
            entry["probs"][b] = float(prob or 0.0)

        if ev_value is not None:
            entry["ev_value"] = float(ev_value)
    for entry in by_month.values():
        if entry.get("ev_value") is not None:
            continue
        probs = entry.get("probs") or []
        n = min(len(probs), len(centroids))
        ev_calc = 0.0
        for i in range(n):
            ev_calc += float(probs[i]) * float(centroids[i])
        entry["ev_value"] = ev_calc
    return by_month


def _load_triage_entry(
    con: duckdb.DuckDBPyConnection,
    hs_run_id: str | None,
    iso3: str,
    hazard_code: str,
    cache: dict[tuple[str, str, str], dict[str, Any] | None] | None = None,
) -> dict[str, Any] | None:
    key = (hs_run_id or "", (iso3 or "").upper(), (hazard_code or "").upper())
    if cache is not None and key in cache:
        return cache[key]

    if not hs_run_id:
        if cache is not None:
            cache[key] = None
        return None

    rows = _fetch_llm_rows(
        con,
        """
        SELECT *
        FROM hs_triage
        WHERE run_id = ? AND iso3 = ? AND hazard_code = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        [hs_run_id, iso3, hazard_code],
    )
    entry = rows[0] if rows else None
    if cache is not None:
        cache[key] = entry
    return entry


def _scenario_expected(
    hazard_code: str | None, metric: str | None, hs_entry: dict[str, Any] | None
) -> tuple[bool, str]:
    hz = (hazard_code or "").upper()
    tier = str((hs_entry or {}).get("tier") or "").lower()

    enabled_hazards = {"ACE", "DI"}
    watchlist_hazards = {"DR", "FL"}

    if hz in watchlist_hazards or tier == "watchlist":
        return False, "watchlist"
    if hz not in enabled_hazards:
        return False, "hazard_not_enabled"
    if tier and tier != "priority":
        return False, f"triage_tier:{tier}"
    return True, ""


def _load_scenario_call_count(
    con: duckdb.DuckDBPyConnection, run_id: str, question_id: str
) -> int:
    row = con.execute(
        """
        SELECT COUNT(*)
        FROM llm_calls
        WHERE run_id = ?
          AND question_id = ?
          AND (
              LOWER(COALESCE(call_type, '')) LIKE 'scenario%'
           OR LOWER(COALESCE(phase, '')) LIKE 'scenario%'
          )
        """,
        [run_id, question_id],
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump unified Pythia v2 debug bundle (HS, Research, SPD, Scenario) for one run.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("PYTHIA_DB_URL", "duckdb:///data/resolver.duckdb"),
        help="DuckDB URL (e.g. duckdb:///data/resolver.duckdb)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Forecast run_id (e.g. fc_17648...). If omitted, use latest fc_* in forecasts_ensemble.",
    )
    parser.add_argument(
        "--output-dir",
        default="debug",
        help="Directory to write the markdown bundle into (default: debug/).",
    )
    return parser.parse_args()


def _resolve_db_path(db_url: str) -> str:
    if db_url.startswith("duckdb:///"):
        return db_url[len("duckdb:///") :]
    return db_url


def _select_run_id(con: duckdb.DuckDBPyConnection, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    row = con.execute(
        """
        SELECT run_id
        FROM forecasts_ensemble
        WHERE run_id LIKE 'fc_%'
        ORDER BY COALESCE(created_at, CURRENT_TIMESTAMP) DESC, run_id DESC
        LIMIT 1
        """,
    ).fetchone()
    return row[0] if row and row[0] else None


def _load_questions_for_run(con: duckdb.DuckDBPyConnection, run_id: str) -> list[dict[str, Any]]:
    rows = con.execute(
        """
        SELECT DISTINCT
            q.question_id,
            q.hs_run_id,
            q.iso3,
            q.hazard_code,
            q.metric,
            q.target_month,
            q.window_start_date,
            q.window_end_date,
            q.wording
        FROM questions q
        JOIN forecasts_ensemble fe
          ON fe.question_id = q.question_id
         AND fe.run_id = ?
        WHERE q.status = 'active'
        ORDER BY q.iso3, q.hazard_code, q.metric, q.question_id
        """,
        [run_id],
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        (
            qid,
            hs_run_id,
            iso3,
            hazard_code,
            metric,
            target_month,
            window_start_date,
            window_end_date,
            wording,
        ) = row
        out.append(
            {
                "question_id": qid,
                "hs_run_id": hs_run_id,
                "iso3": iso3,
                "hazard_code": hazard_code,
                "metric": metric,
                "target_month": target_month,
                "window_start_date": window_start_date,
                "window_end_date": window_end_date,
                "wording": wording,
            }
        )
    return out


def _load_llm_calls_for_question(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    question_id: str,
    iso3: str,
    hazard_code: str,
    hs_run_id: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Return phase -> call dict for this question/run.

    This function is robust to schema differences in llm_calls: it uses
    SELECT * and only reads columns that exist.
    """
    calls: Dict[str, Dict[str, Any]] = {}

    rows = _fetch_llm_rows(
        con,
        """
        SELECT *
        FROM llm_calls
        WHERE run_id = ?
          AND question_id = ?
          AND phase IN ('research_v2', 'spd_v2', 'scenario_v2')
        ORDER BY COALESCE(timestamp, CURRENT_TIMESTAMP) DESC
        """,
        [run_id, question_id],
    )

    for row in rows:
        phase = row.get("phase")
        if not phase or phase in calls:
            continue

        usage_json = _build_usage_json_from_row(row)
        calls[phase] = {
            "call_type": row.get("call_type"),
            "phase": phase,
            "provider": row.get("provider"),
            "model_id": row.get("model_id") or row.get("model"),
            "temperature": row.get("temperature"),
            "run_id": row.get("run_id"),
            "question_id": row.get("question_id"),
            "iso3": row.get("iso3"),
            "hazard_code": row.get("hazard_code"),
            "metric": row.get("metric"),
            "prompt_text": row.get("prompt_text") or "",
            "response_text": row.get("response_text") or "",
            "error_text": row.get("error_text") or "",
            "usage_json": usage_json,
        }

    hs_rows: list[dict[str, Any]] = []

    if hs_run_id:
        hs_rows = _fetch_llm_rows(
            con,
            """
            SELECT *
            FROM llm_calls
            WHERE phase = 'hs_triage'
              AND hs_run_id = ?
            ORDER BY COALESCE(timestamp, CURRENT_TIMESTAMP) DESC
            LIMIT 1
            """,
            [hs_run_id],
        )

    if not hs_rows:
        hs_rows = _fetch_llm_rows(
            con,
            """
            SELECT *
            FROM llm_calls
            WHERE phase = 'hs_triage'
              AND iso3 = ?
              AND hazard_code = ?
            ORDER BY COALESCE(timestamp, CURRENT_TIMESTAMP) DESC
            LIMIT 1
            """,
            [iso3, hazard_code],
        )

    if not hs_rows:
        hs_rows = _fetch_llm_rows(
            con,
            """
            SELECT *
            FROM llm_calls
            WHERE phase = 'hs_triage'
              AND iso3 = ?
            ORDER BY COALESCE(timestamp, CURRENT_TIMESTAMP) DESC
            LIMIT 1
            """,
            [iso3],
        )

    if hs_rows:
        row = hs_rows[0]
        usage_json = _build_usage_json_from_row(row)
        calls["hs_triage"] = {
            "call_type": row.get("call_type"),
            "phase": row.get("phase"),
            "provider": row.get("provider"),
            "model_id": row.get("model_id") or row.get("model"),
            "temperature": row.get("temperature"),
            "run_id": row.get("run_id"),
            "question_id": row.get("question_id"),
            "iso3": row.get("iso3"),
            "hazard_code": row.get("hazard_code"),
            "metric": row.get("metric"),
            "prompt_text": row.get("prompt_text") or "",
            "response_text": row.get("response_text") or "",
            "error_text": row.get("error_text") or "",
            "usage_json": usage_json,
        }
    else:
        triage_rows = _fetch_llm_rows(
            con,
            """
            SELECT *
            FROM hs_triage
            WHERE run_id = ?
              AND iso3 = ?
              AND hazard_code = ?
            ORDER BY COALESCE(created_at, CURRENT_TIMESTAMP) DESC
            LIMIT 1
            """,
            [hs_run_id or run_id, iso3, hazard_code],
        )

        if triage_rows:
            triage = triage_rows[0]
            triage_payload = {
                "tier": triage.get("tier"),
                "triage_score": triage.get("triage_score"),
                "need_full_spd": triage.get("need_full_spd"),
                "drivers": triage.get("drivers_json"),
                "regime_shifts": triage.get("regime_shifts_json"),
                "data_quality": triage.get("data_quality_json"),
                "scenario_stub": triage.get("scenario_stub"),
            }

            for key in ("drivers", "regime_shifts", "data_quality"):
                raw_val = triage_payload.get(key)
                if isinstance(raw_val, str):
                    try:
                        triage_payload[key] = json.loads(raw_val)
                    except Exception:
                        continue

            calls["hs_triage"] = {
                "call_type": "hs_triage_fallback",
                "phase": "hs_triage",
                "provider": None,
                "model_id": None,
                "temperature": None,
                "run_id": triage.get("run_id") or hs_run_id or run_id,
                "question_id": question_id,
                "iso3": triage.get("iso3") or iso3,
                "hazard_code": triage.get("hazard_code") or hazard_code,
                "metric": None,
                "prompt_text": "HS triage (from hs_triage table; no llm_calls row)",
                "response_text": json.dumps(triage_payload, ensure_ascii=False, indent=2),
                "error_text": "",
                "usage_json": "{}",
            }

    return calls


def _resolve_hs_run_id_for_forecast(
    con: duckdb.DuckDBPyConnection, run_id: str
) -> str | None:
    try:
        row = con.execute(
            """
            SELECT q.hs_run_id
            FROM forecasts_ensemble fe
            JOIN questions q ON fe.question_id = q.question_id
            WHERE fe.run_id = ?
              AND q.hs_run_id IS NOT NULL
              AND q.hs_run_id <> ''
            ORDER BY COALESCE(fe.created_at, CURRENT_TIMESTAMP) DESC
            LIMIT 1
            """,
            [run_id],
        ).fetchone()
        if row and row[0]:
            return str(row[0])
    except Exception:
        return None
    return None


def _aggregate_usage_by_phase(
    con: duckdb.DuckDBPyConnection, run_id: str, hs_run_id: str | None
) -> dict[str, dict[str, float]]:
    """
    Aggregate token and cost usage by phase for a given forecast run.

    We:
      - Include all llm_calls rows where run_id = <run_id>
      - Optionally include HS triage calls (phase='hs_triage'), even if their run_id is NULL
        or not equal to <run_id>, by matching on phase alone.

    We DO NOT join to questions here to avoid schema assumptions such as q.run_id.
    """
    # Load all calls for this run (research_v2, spd_v2, scenario_v2, etc.)
    params: list[Any] = [run_id]
    if hs_run_id:
        query = """
            SELECT *
            FROM llm_calls
            WHERE run_id = ?
               OR (phase = 'hs_triage' AND hs_run_id = ?)
        """
        params.append(hs_run_id)
    else:
        query = """
            SELECT *
            FROM llm_calls
            WHERE run_id = ?
               OR (phase = 'hs_triage')
        """

    rows = _fetch_llm_rows(con, query, params)

    out: dict[str, dict[str, float]] = {}
    for row in rows:
        phase = row.get("phase") or "unknown"
        usage_raw = row.get("usage_json") or "{}"
        try:
            usage = json.loads(usage_raw)
        except Exception:
            usage = {}

        phase_acc = out.setdefault(
            phase,
            {
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "total_tokens": 0.0,
                "total_cost_usd": 0.0,
            },
        )
        phase_acc["prompt_tokens"] += float(usage.get("prompt_tokens") or 0.0)
        phase_acc["completion_tokens"] += float(usage.get("completion_tokens") or 0.0)
        phase_acc["total_tokens"] += float(usage.get("total_tokens") or 0.0)
        # For backwards compatibility, accept either total_cost_usd or cost_usd
        phase_acc["total_cost_usd"] += float(
            usage.get("total_cost_usd") or usage.get("cost_usd") or 0.0
        )

    return out


def _load_triage_tier(
    con: duckdb.DuckDBPyConnection, hs_run_id: str, iso3: str, hazard_code: str
) -> str | None:
    row = con.execute(
        """
        SELECT tier
        FROM hs_triage
        WHERE run_id = ? AND iso3 = ? AND hazard_code = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        [hs_run_id, iso3, hazard_code],
    ).fetchone()
    return row[0] if row and row[0] else None


def _load_spd_status(con: duckdb.DuckDBPyConnection, run_id: str, question_id: str) -> str | None:
    row = con.execute(
        """
        SELECT status
        FROM forecasts_ensemble
        WHERE run_id = ? AND question_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        [run_id, question_id],
    ).fetchone()
    return row[0] if row and row[0] else None


def _append_stage_block(lines: List[str], phase: str, call: Dict[str, Any] | None) -> None:
    if call is None:
        lines.append(f"_No LLM call recorded for phase `{phase}`._")
        return

    lines.append("")
    lines.append("##### Metadata")
    lines.append("")
    lines.append(f"- Phase: `{call.get('phase')}`")
    lines.append(f"- Provider: `{call.get('provider')}`")
    lines.append(f"- Model: `{call.get('model_id')}`")
    if call.get("temperature") is not None:
        lines.append(f"- Temperature: `{call.get('temperature')}`")
    lines.append(f"- Run ID: `{call.get('run_id')}`")
    if call.get("question_id"):
        lines.append(f"- Question ID: `{call.get('question_id')}`")
    if call.get("iso3"):
        lines.append(f"- ISO3: `{call.get('iso3')}`")
    if call.get("hazard_code"):
        lines.append(f"- Hazard: `{call.get('hazard_code')}`")
    if call.get("metric"):
        lines.append(f"- Metric: `{call.get('metric')}`")

    usage_raw = call.get("usage_json") or "{}"
    try:
        usage = json.loads(usage_raw)
    except Exception:
        usage = {}
    if usage:
        lines.append("")
        lines.append("##### Usage / Cost")
        lines.append("")
        ordered_keys = [
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_cost_usd",
            "output_cost_usd",
            "total_cost_usd",
            "cost_usd",
            "elapsed_ms",
        ]
        for key in ordered_keys:
            if key in usage and usage[key] is not None:
                lines.append(f"- {key}: `{usage[key]}`")

        for key, val in usage.items():
            if key in ordered_keys:
                continue
            lines.append(f"- {key}: `{val}`")

    lines.append("")
    lines.append("##### Prompt")
    lines.append("")
    lines.append("```text")
    lines.append(call.get("prompt_text") or "")
    lines.append("```")

    lines.append("")
    lines.append("##### Output")
    lines.append("")
    lines.append("```text")
    lines.append(call.get("response_text") or "")
    lines.append("```")

    lines.append("")
    lines.append("##### Errors / Failure Notes")
    lines.append("")
    error_text = call.get("error_text") or ""
    if error_text.strip():
        lines.append(error_text.strip())
    else:
        lines.append("_No error reported for this call._")


def build_debug_bundle_markdown(
    con: duckdb.DuckDBPyConnection,
    db_url: str,
    run_id: str,
    questions: list[dict[str, Any]],
) -> str:
    lines: List[str] = []

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    hs_run_id_for_costs = _resolve_hs_run_id_for_forecast(con, run_id)
    usage_by_phase = _aggregate_usage_by_phase(con, run_id, hs_run_id_for_costs)
    triage_cache: dict[tuple[str, str, str], dict[str, Any] | None] = {}

    hazards = sorted(
        {(q.get("hazard_code") or "").upper() for q in questions if q.get("hazard_code")}
    )
    iso3s = sorted({(q.get("iso3") or "").upper() for q in questions if q.get("iso3")})
    metrics = sorted({(q.get("metric") or "").upper() for q in questions if q.get("metric")})
    hs_run_ids = sorted({q.get("hs_run_id") for q in questions if q.get("hs_run_id")})

    scenario_status_rows: list[dict[str, Any]] = []
    for q in questions:
        qid = q.get("question_id")
        iso3 = q.get("iso3") or ""
        hz = q.get("hazard_code") or ""
        metric = q.get("metric") or ""
        hs_run_id = q.get("hs_run_id") or hs_run_id_for_costs or run_id
        triage_entry = _load_triage_entry(con, hs_run_id, iso3, hz, cache=triage_cache)
        triage_tier = (triage_entry or {}).get("tier")
        expected, reason = _scenario_expected(hz, metric, triage_entry)
        call_count = _load_scenario_call_count(con, run_id, str(qid))
        if call_count > 0:
            status = "generated"
        elif not expected:
            status = f"skipped_by_design: {reason or 'not_expected'}"
        else:
            status = "missing_unexpected"
        scenario_status_rows.append(
            {
                "question_id": qid,
                "iso3": iso3,
                "hazard_code": hz,
                "metric": metric,
                "hs_run_id": hs_run_id,
                "triage_tier": triage_tier,
                "status": status,
                "expected": expected,
                "call_count": call_count,
            }
        )
    scenario_status_by_qid = {row["question_id"]: row for row in scenario_status_rows}

    lines.append(f"# Pythia v2 Debug Bundle — Run {run_id}")
    lines.append("")
    lines.append(f"_Generated at {now}_")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append(f"- Database URL: `{db_url}`")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Question types included (by hazard_code, metric): {len(questions)}")
    lines.append(f"- Hazards present: {', '.join(hazards) if hazards else '(none)'}")
    lines.append(f"- ISO3s present: {', '.join(iso3s) if iso3s else '(none)'}")
    lines.append(f"- Metrics present: {', '.join(metrics) if metrics else '(none)'}")
    lines.append(f"- Linked HS run IDs: {', '.join(hs_run_ids) if hs_run_ids else '(none)'}")
    lines.append("")

    total_prompt = sum(v["prompt_tokens"] for v in usage_by_phase.values())
    total_completion = sum(v["completion_tokens"] for v in usage_by_phase.values())
    total_tokens = sum(v["total_tokens"] for v in usage_by_phase.values())
    total_cost = sum(v["total_cost_usd"] for v in usage_by_phase.values())

    lines.append("### 1.1 Token & Cost Summary")
    lines.append("")
    lines.append(f"- Total prompt tokens (all phases): `{int(total_prompt)}`")
    lines.append(f"- Total completion tokens (all phases): `{int(total_completion)}`")
    lines.append(f"- Total tokens (all phases): `{int(total_tokens)}`")
    lines.append(f"- Total cost (USD, all phases): `{total_cost:.4f}`")
    lines.append("")

    lines.append("### 1.2 Cost by Phase")
    lines.append("")
    lines.append("| phase | prompt_tokens | completion_tokens | total_tokens | total_cost_usd |")
    lines.append("|-------|---------------|-------------------|--------------|----------------|")
    for phase, agg in sorted(usage_by_phase.items()):
        lines.append(
            f"| {phase} | {int(agg['prompt_tokens'])} | {int(agg['completion_tokens'])} | "
            f"{int(agg['total_tokens'])} | {agg['total_cost_usd']:.4f} |"
        )
    lines.append("")

    lines.append("### 1.3 Scenario status (per question)")
    lines.append("")
    lines.append(
        "| question_id | iso3 | hazard | metric | hs_run_id | triage_tier | scenario_status | calls_logged |"
    )
    lines.append("|-------------|------|--------|--------|----------|-------------|-----------------|--------------|")
    for row in sorted(
        scenario_status_rows,
        key=lambda r: (
            str(r.get("iso3") or ""),
            str(r.get("hazard_code") or ""),
            str(r.get("metric") or ""),
            str(r.get("question_id") or ""),
        ),
    ):
        lines.append(
            f"| {row.get('question_id')} | {row.get('iso3')} | {row.get('hazard_code')} | "
            f"{row.get('metric')} | {row.get('hs_run_id')} | {row.get('triage_tier') or ''} | "
            f"{row.get('status')} | {row.get('call_count')} |"
        )
    lines.append("")

    lines.append("## 2. Question Types")
    lines.append("")

    for idx, question in enumerate(questions, start=1):
        qid = question.get("question_id")
        hs_run_id = question.get("hs_run_id")
        iso3_val = question.get("iso3")
        hz_val = question.get("hazard_code")
        metric_val = question.get("metric")
        target_month = question.get("target_month")
        window_start_date = question.get("window_start_date")
        window_end_date = question.get("window_end_date")
        wording = question.get("wording")

        triage_entry = _load_triage_entry(
            con, hs_run_id or hs_run_id_for_costs or run_id, iso3_val, hz_val, cache=triage_cache
        )
        triage_tier = (triage_entry or {}).get("tier") or _load_triage_tier(
            con, hs_run_id or run_id, iso3_val, hz_val
        )
        spd_status = _load_spd_status(con, run_id, qid)
        scenario_meta = scenario_status_by_qid.get(qid) or {}

        section_label = f"{iso3_val} / {hz_val} / {metric_val}"
        lines.append(f"### 2.{idx} {section_label} (question_id={qid})")
        lines.append("")
        lines.append(f"- ISO3: `{iso3_val}`")
        lines.append(f"- Hazard: `{hz_val}`")
        lines.append(f"- Metric: `{metric_val}`")
        lines.append(f"- HS run_id: `{hs_run_id or 'N/A'}`")
        lines.append(f"- Triaged tier: `{triage_tier or 'N/A'}`")
        lines.append(f"- SPD ensemble status: `{spd_status or 'N/A'}`")
        lines.append(
            f"- Scenario status: `{scenario_meta.get('status', 'unknown')}` "
            f"(calls logged: {scenario_meta.get('call_count', 0)}, "
            f"expected: {'yes' if scenario_meta.get('expected') else 'no'})"
        )
        lines.append(f"- Target month: `{target_month}`")
        lines.append(f"- Window: `{window_start_date}` → `{window_end_date}`")
        lines.append(f"- Wording: {wording}")
        lines.append("")

        calls = _load_llm_calls_for_question(
            con, run_id, qid, iso3_val, hz_val, hs_run_id
        )

        lines.append(f"#### 2.{idx}.1 Horizon Scanner (HS)")
        _append_stage_block(lines, "hs_triage", calls.get("hs_triage"))
        lines.append("")

        lines.append(f"#### 2.{idx}.2 Research (Research v2)")
        _append_stage_block(lines, "research_v2", calls.get("research_v2"))
        lines.append("")

        lines.append(f"#### 2.{idx}.3 Forecaster (SPD v2)")
        _append_stage_block(lines, "spd_v2", calls.get("spd_v2"))
        lines.append("")

        lines.append(f"#### 2.{idx}.4 Scenarios (Scenario v2)")
        scenario_details: list[str] = []
        if scenario_meta:
            scenario_details.append(f"status={scenario_meta.get('status')}")
            scenario_details.append(f"calls={scenario_meta.get('call_count')}")
            scenario_details.append(f"expected={'yes' if scenario_meta.get('expected') else 'no'}")
            if scenario_meta.get("triage_tier"):
                scenario_details.append(f"triage_tier={scenario_meta.get('triage_tier')}")
        if scenario_details:
            lines.append(f"_Scenario diagnostics: {', '.join(scenario_details)}_")
            lines.append("")
        _append_stage_block(lines, "scenario_v2", calls.get("scenario_v2"))
        lines.append("")

        lines.append(f"#### 2.{idx}.5 Ensemble SPD & EV (post-BayesMC)")
        lines.append("")

        q_dict = {
            "question_id": qid,
            "iso3": iso3_val,
            "hazard_code": hz_val,
            "metric": metric_val,
            "wording": wording,
        }
        bucket_labels = _get_bucket_labels_for_question(q_dict)
        centroids = _load_bucket_centroids_for_question(con, hz_val, metric_val, bucket_labels)

        lines.append("##### Buckets & Centroids")
        lines.append("")
        lines.append("| index | bucket_label | centroid |")
        lines.append("|-------|--------------|----------|")
        for i, label in enumerate(bucket_labels):
            centroid_val = centroids[i] if i < len(centroids) else 0.0
            lines.append(f"| {i + 1} | {label} | {centroid_val} |")
        lines.append("")

        ensemble_rows = con.execute(
            """
            SELECT month_index, bucket_index, probability, ev_value, status, human_explanation
            FROM forecasts_ensemble
            WHERE run_id = ? AND question_id = ?
            ORDER BY month_index, bucket_index
            """,
            [run_id, qid],
        ).fetchall()

        if not ensemble_rows:
            lines.append("_No ensemble SPD rows found for this question/run._")
            lines.append("")
        else:
            statuses = {row[4] for row in ensemble_rows if row}
            if statuses == {"no_forecast"}:
                reason = (ensemble_rows[0][5] or "unknown") if ensemble_rows else "unknown"
                lines.append("_SPD status: `no_forecast`._")
                lines.append(f"_SPD failure reason: {reason}_")
                lines.append("")
                lines.append("_No ensemble SPD rows found for this question/run._")
                lines.append("")
            else:
                ensemble = _load_ensemble_spd_for_question(con, run_id, qid, centroids)
                lines.append("##### Ensemble SPD and EV by Month")
                lines.append("")
                bucket_count = max(len(bucket_labels), 1)
                lines.append(
                    "| month_index | "
                    + " | ".join(f"p(bucket {i + 1})" for i in range(bucket_count))
                    + " | EV (units of centroid) |"
                )
                lines.append("|------------|" + "|".join(["--------------"] * (bucket_count + 1)) + "|")
                for month_idx in sorted(ensemble.keys()):
                    entry = ensemble[month_idx]
                    probs = entry.get("probs") or [0.0] * bucket_count
                    ev_val = entry.get("ev_value")
                    prob_cells = " | ".join(f"{p:.3f}" for p in probs[:bucket_count])
                    ev_cell = f"{ev_val:.1f}" if ev_val is not None else ""
                    lines.append(f"| {month_idx} | {prob_cells} | {ev_cell} |")
                lines.append("")

                lines.append("##### EV Calculation Notes")
                lines.append("")
                lines.append(
                    "For each month, the expected value (EV) is computed as:\n"
                    "- EV = sum_{i=1..5} p_i * centroid_i\n"
                    "where centroid_i is the representative value for bucket i (from `bucket_centroids` or defaults), "
                    "and p_i are the ensemble bucket probabilities after Bayes-MC aggregation."
                )
                lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    db_url = args.db
    db_path = _resolve_db_path(db_url)
    if db_path not in {":memory:"}:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    con = duckdb_io.get_db(db_url)
    try:
        run_id = _select_run_id(con, args.run_id)
        if not run_id:
            print("No fc_* run_id found in forecasts_ensemble; nothing to debug.")
            return

        questions = _load_questions_for_run(con, run_id)
        if not questions:
            print(f"No active questions found for run_id={run_id}.")
            return

        markdown = build_debug_bundle_markdown(con, db_url, run_id, questions)
    finally:
        duckdb_io.close_db(con)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pytia_debug_bundle__{run_id}.md"
    out_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote Pythia debug bundle to {out_path}")


if __name__ == "__main__":
    main()
