from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from pythia.db.schema import connect

from .hs_utils import load_hs_triage_entry
from .prompts import build_scenario_prompt
from .providers import GEMINI_MODEL_ID, ModelSpec, call_chat_ms

LOG = logging.getLogger(__name__)

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


async def _call_scenario_model(prompt_text: str) -> Tuple[str, Dict[str, Any], str]:
    ms = ModelSpec(name="Gemini", provider="google", model_id=GEMINI_MODEL_ID, active=True)
    return await call_chat_ms(
        ms,
        prompt_text,
        temperature=0.25,
        prompt_key="scenario.v1",
        prompt_version="1.0.0",
        component="ScenarioWriter",
    )


def _load_forecaster_rationale(
    con,
    run_id: str,
    iso3: str,
    hazard_code: str,
    metric: str,
) -> str:
    row = con.execute(
        """
        SELECT human_explanation
        FROM forecasts_ensemble
        WHERE run_id = ?
          AND iso3 = ?
          AND hazard_code = ?
          AND metric = ?
          AND status = 'ok'
          AND human_explanation IS NOT NULL
          AND human_explanation <> ''
        ORDER BY created_at DESC
        LIMIT 1
        """,
        [run_id, iso3, hazard_code, metric],
    ).fetchone()
    return row[0] if row else ""


def _build_spd_summary(
    con,
    run_id: str,
    iso3: str,
    hazard_code: str,
    metric: str,
) -> Dict[str, Any]:
    bucket_labels = (
        SPD_CLASS_BINS_FATALITIES if (metric or "").upper() == "FATALITIES" else SPD_CLASS_BINS_PA
    )

    rows = con.execute(
        """
        SELECT month_index, bucket_index, probability
        FROM forecasts_ensemble
        WHERE run_id = ?
          AND iso3 = ?
          AND hazard_code = ?
          AND metric = ?
          AND status = 'ok'
        ORDER BY month_index, bucket_index
        """,
        [run_id, iso3, hazard_code, metric],
    ).fetchall()

    if not rows:
        return {}

    per_month: Dict[int, List[float]] = {}
    for month_index, bucket_index, probability in rows:
        if bucket_index is None or month_index is None:
            continue
        probs = per_month.setdefault(int(month_index), [0.0] * len(bucket_labels))
        if 0 <= int(bucket_index) < len(probs):
            probs[int(bucket_index)] = float(probability or 0.0)

    per_month_summary: Dict[str, Any] = {}
    for mi, probs in sorted(per_month.items()):
        max_idx = probs.index(max(probs)) if probs else 0
        per_month_summary[str(mi)] = {
            "probs": probs,
            "bucket_label_max": bucket_labels[max_idx] if bucket_labels else "",
            "prob_max": probs[max_idx] if probs else 0.0,
        }

    agg_probs = [0.0] * len(bucket_labels)
    month_count = max(1, len(per_month))
    for probs in per_month.values():
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


def run_scenarios_for_run(run_id: str) -> None:
    con = connect(read_only=False)
    try:
        rows = con.execute(
            """
            SELECT DISTINCT f.iso3, f.hazard_code, f.metric, q.hs_run_id
            FROM forecasts_ensemble AS f
            LEFT JOIN questions AS q
              ON q.question_id = f.question_id
            WHERE f.run_id = ? AND f.status = 'ok'
            ORDER BY f.iso3, f.hazard_code, f.metric
            """,
            [run_id],
        ).fetchall()

        for iso3, hz, metric, hs_run_id in rows:
            spd_summary = _build_spd_summary(con, run_id, iso3, hz, metric)
            if not spd_summary:
                LOG.warning("No SPD summary for %s %s %s; skipping scenarios", iso3, hz, metric)
                continue

            hs_entry = load_hs_triage_entry(hs_run_id, iso3, hz) if hs_run_id else {}
            rationale = _load_forecaster_rationale(con, run_id, iso3, hz, metric)
            scenario_stub = hs_entry.get("scenario_stub", "") if hs_entry else ""

            prompt = build_scenario_prompt(
                iso3=iso3,
                hazard_code=hz,
                metric=metric,
                spd_summary=spd_summary,
                hs_triage_entry=hs_entry,
                scenario_stub=scenario_stub,
                forecaster_rationale=rationale,
            )

            try:
                text, usage, error = asyncio.run(_call_scenario_model(prompt))
            except Exception as exc:
                LOG.exception("Scenario writer call failed for %s %s %s: %s", iso3, hz, metric, exc)
                continue

            if error:
                LOG.warning("Scenario writer error for %s %s %s: %s", iso3, hz, metric, error)
                continue

            try:
                scenarios = json.loads(text)
            except json.JSONDecodeError:
                LOG.warning("Invalid scenario JSON for %s %s %s", iso3, hz, metric)
                continue

            primary = scenarios.get("primary") if isinstance(scenarios, dict) else None
            alt = scenarios.get("alternative") if isinstance(scenarios, dict) else None

            if primary:
                con.execute(
                    """
                    INSERT INTO scenarios
                      (run_id, iso3, hazard_code, metric, scenario_type, bucket_label, probability, text)
                    VALUES (?, ?, ?, ?, 'primary', ?, ?, ?)
                    """,
                    [
                        run_id,
                        iso3,
                        hz,
                        metric,
                        primary.get("bucket_label") or "",
                        float(primary.get("probability") or 0.0),
                        primary.get("text") or "",
                    ],
                )

            if alt:
                con.execute(
                    """
                    INSERT INTO scenarios
                      (run_id, iso3, hazard_code, metric, scenario_type, bucket_label, probability, text)
                    VALUES (?, ?, ?, ?, 'alternative', ?, ?, ?)
                    """,
                    [
                        run_id,
                        iso3,
                        hz,
                        metric,
                        alt.get("bucket_label") or "",
                        float(alt.get("probability") or 0.0),
                        alt.get("text") or "",
                    ],
                )
    finally:
        con.close()
