from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import duckdb
from resolver.db import duckdb_io


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


def _load_question_types(
    con: duckdb.DuckDBPyConnection, run_id: str
) -> List[Tuple[str, str, str, str]]:
    rows = con.execute(
        """
        WITH q_run AS (
            SELECT DISTINCT
                q.question_id,
                q.iso3,
                q.hazard_code,
                q.metric
            FROM questions q
            JOIN forecasts_ensemble fe
              ON fe.question_id = q.question_id
             AND fe.run_id = ?
            WHERE q.status = 'active'
        ),
        ranked AS (
            SELECT
                question_id,
                iso3,
                hazard_code,
                metric,
                ROW_NUMBER() OVER (
                    PARTITION BY hazard_code, metric
                    ORDER BY iso3, question_id
                ) AS rn
            FROM q_run
        )
        SELECT question_id, iso3, hazard_code, metric
        FROM ranked
        WHERE rn = 1
        ORDER BY hazard_code, metric, iso3, question_id
        """,
        [run_id],
    ).fetchall()
    return [(qid, iso3, hz, metric) for (qid, iso3, hz, metric) in rows]


def _load_llm_calls_for_question(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    question_id: str,
    iso3: str,
    hazard_code: str,
) -> Dict[str, Dict[str, Any]]:
    calls: Dict[str, Dict[str, Any]] = {}

    rows = con.execute(
        """
        SELECT
            call_type,
            phase,
            provider,
            model_id,
            temperature,
            run_id,
            question_id,
            iso3,
            hazard_code,
            metric,
            prompt_text,
            response_text,
            error_text,
            usage_json
        FROM llm_calls
        WHERE run_id = ?
          AND question_id = ?
          AND phase IN ('research_v2', 'spd_v2', 'scenario_v2')
        ORDER BY created_at DESC
        """,
        [run_id, question_id],
    ).fetchall()

    for (
        call_type,
        phase,
        provider,
        model_id,
        temperature,
        run_id2,
        qid2,
        iso3_val,
        hz_val,
        metric,
        prompt_text,
        response_text,
        error_text,
        usage_json,
    ) in rows:
        if phase and phase not in calls:
            calls[phase] = {
                "call_type": call_type,
                "phase": phase,
                "provider": provider,
                "model_id": model_id,
                "temperature": temperature,
                "run_id": run_id2,
                "question_id": qid2,
                "iso3": iso3_val,
                "hazard_code": hz_val,
                "metric": metric,
                "prompt_text": prompt_text or "",
                "response_text": response_text or "",
                "error_text": error_text or "",
                "usage_json": usage_json or "{}",
            }

    hs_row = con.execute(
        """
        SELECT
            call_type,
            phase,
            provider,
            model_id,
            temperature,
            run_id,
            question_id,
            iso3,
            hazard_code,
            metric,
            prompt_text,
            response_text,
            error_text,
            usage_json
        FROM llm_calls
        WHERE phase = 'hs_triage'
          AND iso3 = ?
          AND hazard_code = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        [iso3, hazard_code],
    ).fetchone()

    if hs_row:
        (
            call_type,
            phase,
            provider,
            model_id,
            temperature,
            run_id2,
            qid2,
            iso3_val,
            hz_val,
            metric,
            prompt_text,
            response_text,
            error_text,
            usage_json,
        ) = hs_row
        calls["hs_triage"] = {
            "call_type": call_type,
            "phase": phase,
            "provider": provider,
            "model_id": model_id,
            "temperature": temperature,
            "run_id": run_id2,
            "question_id": qid2,
            "iso3": iso3_val,
            "hazard_code": hz_val,
            "metric": metric,
            "prompt_text": prompt_text or "",
            "response_text": response_text or "",
            "error_text": error_text or "",
            "usage_json": usage_json or "{}",
        }

    return calls


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
        for key, val in usage.items():
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
    question_types: List[Tuple[str, str, str, str]],
) -> str:
    lines: List[str] = []

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    lines.append(f"# Pythia v2 Debug Bundle — Run {run_id}")
    lines.append("")
    lines.append(f"_Generated at {now}_")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append(f"- Database URL: `{db_url}`")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Question types included (by hazard_code, metric): {len(question_types)}")
    lines.append("")

    lines.append("## 2. Question Types")
    lines.append("")

    for idx, (question_id, iso3, hazard_code, metric) in enumerate(question_types, start=1):
        q_row = con.execute(
            """
            SELECT
                question_id, hs_run_id, iso3, hazard_code, metric,
                target_month, window_start_date, window_end_date, wording
            FROM questions
            WHERE question_id = ?
            """,
            [question_id],
        ).fetchone()
        if not q_row:
            continue

        (
            qid,
            hs_run_id,
            iso3_val,
            hz_val,
            metric_val,
            target_month,
            window_start_date,
            window_end_date,
            wording,
        ) = q_row

        triage_tier = _load_triage_tier(con, hs_run_id or run_id, iso3_val, hz_val)
        spd_status = _load_spd_status(con, run_id, qid)

        section_label = f"{iso3_val} / {hz_val} / {metric_val}"
        lines.append(f"### 2.{idx} {section_label} (question_id={qid})")
        lines.append("")
        lines.append(f"- ISO3: `{iso3_val}`")
        lines.append(f"- Hazard: `{hz_val}`")
        lines.append(f"- Metric: `{metric_val}`")
        lines.append(f"- HS run_id: `{hs_run_id or 'N/A'}`")
        lines.append(f"- Triaged tier: `{triage_tier or 'N/A'}`")
        lines.append(f"- SPD ensemble status: `{spd_status or 'N/A'}`")
        lines.append(f"- Target month: `{target_month}`")
        lines.append(f"- Window: `{window_start_date}` → `{window_end_date}`")
        lines.append(f"- Wording: {wording}")
        lines.append("")

        calls = _load_llm_calls_for_question(con, run_id, qid, iso3_val, hz_val)

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
        _append_stage_block(lines, "scenario_v2", calls.get("scenario_v2"))
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

        question_types = _load_question_types(con, run_id)
        if not question_types:
            print(f"No active questions found for run_id={run_id}.")
            return

        markdown = build_debug_bundle_markdown(con, db_url, run_id, question_types)
    finally:
        duckdb_io.close_db(con)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pytia_debug_bundle__{run_id}.md"
    out_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote Pythia debug bundle to {out_path}")


if __name__ == "__main__":
    main()
