# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pythia.db.schema import connect

from .hs_utils import load_hs_triage_entry
from .llm_logging import log_forecaster_llm_call
from .prompts import build_scenario_prompt

LOG = logging.getLogger(__name__)

MAX_SCENARIO_WORKERS = int(os.getenv("FORECASTER_SCENARIO_MAX_WORKERS", "6"))
SCENARIO_TIMEOUT_SECONDS = int(os.getenv("FORECASTER_SCENARIO_TIMEOUT_SECONDS", "120"))


def _safe_json_loads_scenario(text: str) -> Any:
    """
    Best-effort JSON loader for scenario LLM responses.

    - Strips ``` / ```json fences if present.
    - Tries to parse the whole string.
    - If that fails, tries the first {...} block.
    Raises json.JSONDecodeError if all attempts fail.
    """

    if text is None:
        raise json.JSONDecodeError("Empty text", "", 0)

    s = str(text).strip()

    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].lstrip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = s[start : end + 1]
            return json.loads(candidate)
        raise

SPD_CLASS_BINS_PA = [
    "<10k",
    "10k-<50k",
    "50k-<250k",
    "250k-<500k",
    ">=500k",
]

SPD_CLASS_BINS_FATALITIES = [
    "<5",
    "5-<25",
    "25-<100",
    "100-<500",
    ">=500",
]


def _build_spd_summary_from_ensemble(
    ensemble_spd: Dict[str, Any], metric: str
) -> Dict[str, Any]:
    bucket_labels = (
        SPD_CLASS_BINS_FATALITIES if (metric or "").upper() == "FATALITIES" else SPD_CLASS_BINS_PA
    )

    per_month_summary: Dict[str, Any] = {}
    per_month_int: Dict[int, List[float]] = {}
    for month_key, probs in ensemble_spd.items():
        if not month_key.startswith("month_"):
            continue
        try:
            month_index = int(month_key.split("_")[1])
        except Exception:
            continue
        padded_probs = list(probs) if isinstance(probs, list) else []
        if len(padded_probs) < len(bucket_labels):
            padded_probs.extend([0.0] * (len(bucket_labels) - len(padded_probs)))
        padded_probs = padded_probs[: len(bucket_labels)]
        per_month_int[month_index] = padded_probs

    if not per_month_int:
        return {}

    for mi, probs in sorted(per_month_int.items()):
        max_idx = probs.index(max(probs)) if probs else 0
        per_month_summary[str(mi)] = {
            "probs": probs,
            "bucket_label_max": bucket_labels[max_idx] if bucket_labels else "",
            "prob_max": probs[max_idx] if probs else 0.0,
        }

    agg_probs = [0.0] * len(bucket_labels)
    month_count = max(1, len(per_month_int))
    for probs in per_month_int.values():
        for idx, val in enumerate(probs):
            agg_probs[idx] += float(val or 0.0) / month_count

    max_idx = agg_probs.index(max(agg_probs)) if agg_probs else 0
    alt_idx: Optional[int] = None
    alt_prob = 0.0
    for idx, prob in enumerate(agg_probs):
        if idx == max_idx or prob < 0.05:
            continue
        if alt_idx is None or prob > alt_prob:
            alt_idx, alt_prob = idx, prob

    return {
        "bucket_labels": bucket_labels,
        "per_month": per_month_summary,
        "bucket_max": {
            "bucket_label": bucket_labels[max_idx] if bucket_labels else "",
            "probability": agg_probs[max_idx] if agg_probs else 0.0,
        },
        "bucket_alt": None
        if alt_idx is None
        else {
            "bucket_label": bucket_labels[alt_idx] if bucket_labels else "",
            "probability": float(alt_prob),
        },
    }


async def _run_scenario_for_question(
    run_id: str,
    question_row: Dict[str, Any],
    ensemble_spd: Dict[str, Any],
    rationale_by_key: Dict[Tuple[str, str, str], str],
) -> None:
    qid = question_row["question_id"]
    iso3 = question_row["iso3"]
    hz = question_row["hazard_code"]
    metric = question_row["metric"]
    hs_run_id = question_row.get("hs_run_id") or run_id

    triage = load_hs_triage_entry(hs_run_id, iso3, hz)
    tier = (triage.get("tier") or "").lower() if triage else ""
    if tier != "priority":
        LOG.info(
            "Skipping scenario for %s (%s/%s/%s) because triage tier=%r is not priority",
            qid,
            iso3,
            hz,
            metric,
            tier,
        )
        return

    spd_summary = _build_spd_summary_from_ensemble(ensemble_spd, metric)
    if not spd_summary:
        LOG.info("No ensemble SPD for %s; skipping scenario", qid)
        return

    rationale_key = (iso3, hz, metric)
    rationale_text = rationale_by_key.get(rationale_key, "")
    prompt = build_scenario_prompt(
        run_id=run_id,
        question={
            "iso3": iso3,
            "hazard_code": hz,
            "metric": metric,
            "wording": question_row.get("wording") or question_row.get("title") or "",
            "forecaster_rationale": rationale_text,
        },
        ensemble_spd=spd_summary,
        hs_triage_entry=triage or {},
    )

    from forecaster.providers import call_chat_ms, ModelSpec  # local import to avoid cycles

    ms = ModelSpec(
        name="Gemini",
        provider="google",
        model_id=os.getenv("PYTHIA_SCENARIO_MODEL_ID", "gemini-3-pro-preview"),
        active=True,
        purpose="scenario_v2",
    )

    start = time.time()

    async def _call() -> Tuple[str, Dict[str, Any], str]:
        return await call_chat_ms(
            ms,
            prompt,
            temperature=0.4,
            prompt_key="scenario.v2",
            prompt_version="1.0.0",
            component="ScenarioWriter",
            run_id=run_id,
        )

    text: str = ""
    usage: Dict[str, Any] = {}
    error: Optional[str] = None
    try:
        text, usage, error = await asyncio.wait_for(
            _call(),
            timeout=SCENARIO_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        usage = {"elapsed_ms": SCENARIO_TIMEOUT_SECONDS * 1000}
        await log_forecaster_llm_call(
            run_id=run_id,
            question_id=qid,
            iso3=iso3,
            hazard_code=hz,
            metric=metric,
            model_spec=ms,
            prompt_text=prompt,
            response_text=text,
            usage=usage,
            error_text=f"Scenario timeout after {SCENARIO_TIMEOUT_SECONDS}s",
            phase="scenario_v2",
            hs_run_id=hs_run_id,
        )
        LOG.warning(
            "Scenario LLM timeout for %s (%s/%s/%s) after %ss",
            qid,
            iso3,
            hz,
            metric,
            SCENARIO_TIMEOUT_SECONDS,
        )
        return
    except Exception as exc:
        usage = {"elapsed_ms": int((time.time() - start) * 1000)}
        await log_forecaster_llm_call(
            run_id=run_id,
            question_id=qid,
            iso3=iso3,
            hazard_code=hz,
            metric=metric,
            model_spec=ms,
            prompt_text=prompt,
            response_text=text,
            usage=usage,
            error_text=f"{type(exc).__name__}: {exc}",
            phase="scenario_v2",
            hs_run_id=hs_run_id,
        )
        LOG.exception("Scenario LLM call failed for %s", qid)
        return

    elapsed_ms = int((time.time() - start) * 1000)
    usage = dict(usage or {})
    usage.setdefault("elapsed_ms", elapsed_ms)

    await log_forecaster_llm_call(
        run_id=run_id,
        question_id=qid,
        iso3=iso3,
        hazard_code=hz,
        metric=metric,
        model_spec=ms,
        prompt_text=prompt,
        response_text=text or "",
        usage=usage,
        error_text=str(error) if error else None,
        phase="scenario_v2",
        hs_run_id=hs_run_id,
    )

    if error or not text or not text.strip():
        LOG.warning("Scenario LLM returned error/empty for %s: %s", qid, error)
        return

    try:
        scenario = _safe_json_loads_scenario(text)
    except json.JSONDecodeError as exc:
        raw_dir = Path("debug/scenarios_raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{run_id}__{qid}.txt"
        raw_path.write_text(text or "", encoding="utf-8")
        LOG.error(
            "Invalid scenario JSON for %s: %s (saved to %s)",
            qid,
            exc,
            raw_path,
        )
        return

    primary = scenario.get("primary") if isinstance(scenario, dict) else None
    if not isinstance(primary, dict):
        LOG.warning("Scenario JSON for %s missing 'primary'; skipping scenario write", qid)
        return

    alternative = scenario.get("alternative") if isinstance(scenario, dict) else None

    def _render_structured_scenario_text(s: Dict[str, Any]) -> str:
        lines: list[str] = []

        context_bullets = s.get("context") or []
        needs = s.get("needs") or {}
        ops_bullets = s.get("operational_impacts") or []

        lines.append("Context")
        for b in context_bullets:
            if b:
                lines.append(f"- {b}")

        lines.append("")
        lines.append("Humanitarian Needs")
        for sector in ["WASH", "Health", "Nutrition", "Protection", "Education", "Shelter", "FoodSecurity"]:
            sector_bullets = needs.get(sector) or []
            lines.append(f"- {sector}:")
            for sb in sector_bullets:
                if sb:
                    lines.append(f"  - {sb}")

        lines.append("")
        lines.append("Operational Impacts")
        for b in ops_bullets:
            if b:
                lines.append(f"- {b}")

        return "\n".join(lines)

    con = connect(read_only=False)
    try:
        con.execute(
            """
            INSERT INTO scenarios (
                run_id, iso3, hazard_code, metric,
                scenario_type, bucket_label, probability, text, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                run_id,
                iso3,
                hz,
                metric,
                "primary",
                primary.get("bucket_label") or "",
                float(primary.get("probability") or 0.0),
                _render_structured_scenario_text(primary),
            ],
        )

        if isinstance(alternative, dict):
            con.execute(
                """
                INSERT INTO scenarios (
                    run_id, iso3, hazard_code, metric,
                    scenario_type, bucket_label, probability, text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [
                    run_id,
                    iso3,
                    hz,
                    metric,
                    "alternative",
                    alternative.get("bucket_label") or "",
                    float(alternative.get("probability") or 0.0),
                    _render_structured_scenario_text(alternative),
                ],
            )
    finally:
        con.close()


def run_scenarios_for_run(run_id: str) -> None:
    con = connect()
    try:
        rows = con.execute(
            """
            SELECT
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
            GROUP BY
                q.question_id,
                q.hs_run_id,
                q.iso3,
                q.hazard_code,
                q.metric,
                q.target_month,
                q.window_start_date,
                q.window_end_date,
                q.wording
            """,
            [run_id],
        ).fetchall()
    finally:
        con.close()

    questions: List[Dict[str, Any]] = []
    for row in rows:
        (
            question_id,
            hs_run_id,
            iso3,
            hazard_code,
            metric,
            target_month,
            window_start_date,
            window_end_date,
            wording,
        ) = row
        questions.append(
            {
                "question_id": question_id,
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

    async def _run_scenarios_async() -> None:
        con2 = connect()
        try:
            ensemble_rows = con2.execute(
                """
                SELECT question_id, month_index, bucket_index, probability
                FROM forecasts_ensemble
                WHERE run_id = ? AND status = 'ok'
                """,
                [run_id],
            ).fetchall()

            rationale_rows = con2.execute(
                """
                SELECT iso3, hazard_code, metric, human_explanation
                FROM forecasts_ensemble
                WHERE run_id = ? AND status = 'ok'
                  AND human_explanation IS NOT NULL
                  AND human_explanation <> ''
                ORDER BY created_at DESC
                """,
                [run_id],
            ).fetchall()
        finally:
            con2.close()

        ensemble_by_q: Dict[str, Dict[str, Any]] = {}
        for qid, month_idx, bucket_idx, prob in ensemble_rows:
            q_map = ensemble_by_q.setdefault(qid, {})
            month_key = f"month_{int(month_idx)}"
            bucket_list = q_map.setdefault(month_key, [0.0] * 5)
            try:
                bucket_index_int = int(bucket_idx)
            except Exception:
                continue
            if 1 <= bucket_index_int <= len(bucket_list):
                bucket_list[bucket_index_int - 1] = float(prob or 0.0)

        rationale_by_key: Dict[Tuple[str, str, str], str] = {}
        for iso3, hazard_code, metric, human_explanation in rationale_rows:
            key = (iso3, hazard_code, metric)
            if key not in rationale_by_key:
                rationale_by_key[key] = human_explanation

        sem = asyncio.Semaphore(MAX_SCENARIO_WORKERS)

        async def _worker(q: Dict[str, Any]) -> None:
            qid = q["question_id"]
            ensemble_spd = ensemble_by_q.get(qid) or {}
            if not ensemble_spd:
                LOG.info("No ensemble SPD for %s; skipping scenario", qid)
                return
            async with sem:
                await _run_scenario_for_question(run_id, q, ensemble_spd, rationale_by_key)

        await asyncio.gather(*(_worker(q) for q in questions))

    asyncio.run(_run_scenarios_async())
