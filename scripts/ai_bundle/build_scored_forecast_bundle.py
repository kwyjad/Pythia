# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Build the scored-forecast analysis bundle (one zip, AI-consumable).

Runs at the calibration terminus (compute_calibration_pythia.yml), where the
canonical DB contains BOTH the forecast-time reasoning (llm_calls prompts and
raw responses, forecasts_raw traces, hs_triage rationale, grounding packs)
AND the realized outcomes (resolutions, scores, calibration outputs). For
every scored question it emits a full reasoning→outcome record, plus flat
score tables, case studies, a run digest, and an ANALYST_GUIDE.md written
for the consuming AI.

Usage:
    python -m scripts.ai_bundle.build_scored_forecast_bundle \
        --db "$PYTHIA_DB_URL" --out-dir ai_bundle

The builder is read-only with respect to pipeline semantics and must never
fail the calibration workflow — wire it with continue-on-error.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from scripts.ai_bundle.common import (
    column_exists,
    open_db,
    resolve_db_path,
    rows_as_dicts,
    safe_json_loads,
    size_guard,
    table_exists,
    write_bundle_zip,
    write_csv,
    write_json,
    write_manifest,
)
from scripts.ai_bundle.guides import build_analyst_guide, build_question_record_schema_md

LOGGER = logging.getLogger(__name__)

AGGREGATE_MODEL_NAMES = {"ensemble_mean_v2", "ensemble_bayesmc_v2", "track2_flash", "sibyl"}
# Ranking preference for case-study selection (mirrors the dashboard's
# STANDARD_MODEL_PREFERENCE, mean-first because binary scores only exist
# under the mean ensemble).
RANKING_MODEL_PREFERENCE = ("ensemble_mean_v2", "ensemble_bayesmc_v2", "track2_flash")

DIGEST_BUDGET_KB = 200
BRIEFING_BUDGET_KB = 300
# Warn-only ceiling for the whole zip (GitHub artifact limits are the real
# constraint; the per-file guards only cover digest/briefing).
_ZIP_WARN_MB = 500


def _score_family(metric: str | None) -> str:
    return "binary" if (metric or "").upper() == "EVENT_OCCURRENCE" else "spd"


def _test_clause(con, table: str, alias: str, include_test: bool) -> str:
    """is_test filter fragment, guarded on column existence."""
    if include_test or not column_exists(con, table, "is_test"):
        return ""
    return f" AND COALESCE({alias}.is_test, FALSE) = FALSE"


def _truncate(text: Any, n: int) -> str:
    s = str(text or "")
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------------------
# Bucket helpers (realized-bucket mapping)
# ---------------------------------------------------------------------------


def _thresholds_for_metric(con, metric: str) -> list[float] | None:
    """Bucket lower bounds for a metric — pythia.buckets first, then the
    bucket_definitions table, else None (realized-bucket columns skipped)."""
    try:
        from pythia.buckets import thresholds_for

        t = thresholds_for(metric)
        if t:
            return list(t)
    except Exception:  # noqa: BLE001
        pass
    if table_exists(con, "bucket_definitions"):
        rows = rows_as_dicts(
            con,
            "SELECT bucket_index, lower_bound FROM bucket_definitions "
            "WHERE UPPER(metric) = ? ORDER BY bucket_index",
            [metric.upper()],
        )
        if rows:
            return [float(r["lower_bound"] or 0.0) for r in rows]
    return None


def _realized_bucket(con, metric: str, value: float | None) -> int | None:
    """1-based bucket index the resolved value falls into."""
    if value is None:
        return None
    if _score_family(metric) == "binary":
        return 1 if value >= 0.5 else 2
    thresholds = _thresholds_for_metric(con, metric)
    if not thresholds:
        return None
    idx = 0
    for i, lower in enumerate(thresholds):
        if value >= lower:
            idx = i
        else:
            break
    return idx + 1


# ---------------------------------------------------------------------------
# Trace quality (recomputed — never persisted at forecast time)
# ---------------------------------------------------------------------------


def _recompute_trace_quality(
    members: list[dict[str, Any]], hazard_code: str, metric: str
) -> None:
    """Attach trace_quality to each member dict in place (best-effort).

    Uses forecaster.trace_validation on raw_calls reconstructed from the
    stored reasoning traces. base_rate_summary is empty at bundle time, so
    the prior check returns a neutral score — the guide documents this.
    """
    try:
        from forecaster.trace_validation import validate_reasoning_traces
    except Exception:  # noqa: BLE001
        return
    raw_calls = [
        {
            "model_spec": SimpleNamespace(name=m.get("model_name")),
            "reasoning_trace": m.get("reasoning_trace"),
        }
        for m in members
    ]
    try:
        results = validate_reasoning_traces(raw_calls, {}, hazard_code, metric)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("trace validation failed: %s", exc)
        return
    for member, result in zip(members, results):
        member["trace_quality"] = result


# ---------------------------------------------------------------------------
# Per-question record
# ---------------------------------------------------------------------------


def _forecast_run_id_for_question(con, qid: str, include_test: bool) -> str | None:
    """The forecaster run whose forecasts were scored for this question."""
    if column_exists(con, "scores", "run_id"):
        rows = rows_as_dicts(
            con,
            "SELECT run_id, COUNT(*) AS n FROM scores WHERE question_id = ? "
            "AND run_id IS NOT NULL AND run_id <> '' GROUP BY run_id ORDER BY n DESC",
            [qid],
        )
        if rows:
            return str(rows[0]["run_id"])
    if table_exists(con, "forecasts_ensemble"):
        # Honor include_test in the fallback: a production question can carry
        # a NEWER test-run forecast row (same-epoch reuse), and picking it
        # would join the record's prompts/members to the wrong run.
        test_clause = ""
        if not include_test and column_exists(con, "forecasts_ensemble", "is_test"):
            test_clause = " AND COALESCE(is_test, FALSE) = FALSE"
        rows = rows_as_dicts(
            con,
            "SELECT run_id FROM forecasts_ensemble WHERE question_id = ? "
            f"{test_clause} ORDER BY created_at DESC LIMIT 1",
            [qid],
        )
        if rows:
            return str(rows[0]["run_id"])
    return None


def _load_members(con, qid: str, run_id: str | None) -> list[dict[str, Any]]:
    """Ensemble-member rows from forecasts_raw, deduped from the 6×K bucket
    rows down to one entry per model."""
    if not table_exists(con, "forecasts_raw"):
        return []
    params: list[Any] = [qid]
    run_filter = ""
    if run_id:
        run_filter = " AND run_id = ?"
        params.append(run_id)
    rows = rows_as_dicts(
        con,
        "SELECT model_name, ok, status, elapsed_ms, cost_usd, prompt_tokens, "
        "completion_tokens, total_tokens, spd_json, human_explanation, "
        "reasoning_trace_json FROM forecasts_raw "
        f"WHERE question_id = ?{run_filter} ORDER BY model_name, month_index, bucket_index",
        params,
    )
    members: dict[str, dict[str, Any]] = {}
    for r in rows:
        name = str(r.get("model_name") or "")
        if not name or name in AGGREGATE_MODEL_NAMES or name in members:
            continue
        members[name] = {
            "model_name": name,
            "ok": r.get("ok"),
            "status": r.get("status"),
            "elapsed_ms": r.get("elapsed_ms"),
            "cost_usd": r.get("cost_usd"),
            "prompt_tokens": r.get("prompt_tokens"),
            "completion_tokens": r.get("completion_tokens"),
            "total_tokens": r.get("total_tokens"),
            "spd": safe_json_loads(r.get("spd_json")),
            "human_explanation": r.get("human_explanation"),
            "reasoning_trace": safe_json_loads(r.get("reasoning_trace_json")),
        }
    return list(members.values())


def _load_prompt_and_responses(
    con, qid: str, run_id: str | None, include_test: bool
) -> tuple[str | None, str | None, dict[str, dict[str, Any]]]:
    """(spd_prompt, prompt_source, {model_name: {response_text, prompt_text,
    provider, model_id, error_text}}) from llm_calls."""
    if not table_exists(con, "llm_calls"):
        return None, None, {}
    params: list[Any] = [qid]
    run_filter = ""
    if run_id:
        run_filter = " AND run_id = ?"
        params.append(run_id)
    rows = rows_as_dicts(
        con,
        "SELECT model_name, provider, model_id, phase, prompt_text, response_text, "
        "error_text FROM llm_calls WHERE question_id = ? "
        f"AND phase IN ('spd_v2', 'binary_v2'){run_filter}"
        + _test_clause(con, "llm_calls", "llm_calls", include_test)
        + " ORDER BY timestamp",
        params,
    )
    by_model: dict[str, dict[str, Any]] = {}
    prompt_counts: Counter[str] = Counter()
    prompt_source = None
    for r in rows:
        name = str(r.get("model_name") or "")
        prompt = r.get("prompt_text") or ""
        if prompt:
            prompt_counts[prompt] += 1
            prompt_source = r.get("phase")
        if name and name not in by_model:
            by_model[name] = {
                "provider": r.get("provider"),
                "model_id": r.get("model_id"),
                "prompt_text": prompt,
                "response_text": r.get("response_text"),
                "error_text": r.get("error_text"),
            }
    spd_prompt = prompt_counts.most_common(1)[0][0] if prompt_counts else None
    return spd_prompt, prompt_source, by_model


def _load_ensemble_grids(con, qid: str, run_id: str | None) -> dict[str, dict[str, Any]]:
    if not table_exists(con, "forecasts_ensemble"):
        return {}
    params: list[Any] = [qid]
    run_filter = ""
    if run_id:
        run_filter = " AND run_id = ?"
        params.append(run_id)
    rows = rows_as_dicts(
        con,
        "SELECT model_name, month_index, bucket_index, probability, ev_value, "
        "weights_profile, status FROM forecasts_ensemble "
        f"WHERE question_id = ?{run_filter} ORDER BY model_name, month_index, bucket_index",
        params,
    )
    grids: dict[str, dict[str, Any]] = {}
    for r in rows:
        name = str(r.get("model_name") or "")
        g = grids.setdefault(
            name,
            {"months": {}, "ev_by_month": {}, "weights_profile": r.get("weights_profile"),
             "status": r.get("status")},
        )
        mi = str(r.get("month_index"))
        g["months"].setdefault(mi, {})[str(r.get("bucket_index"))] = r.get("probability")
        if r.get("ev_value") is not None:
            g["ev_by_month"][mi] = r.get("ev_value")
    return grids


def build_question_record(
    con,
    q: dict[str, Any],
    *,
    include_test: bool,
    include_sibyl_trials: bool,
    run_id: str | None = None,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the full reasoning record for one question.

    ``run_id`` pins the forecaster run explicitly (the current-run bundle
    knows it); when omitted it is resolved from scores/forecasts_ensemble
    (the scored bundle's behaviour, unchanged).

    ``extras`` is merged into the record at the top level after everything
    else is assembled — the attribution bundle passes its ``attribution``
    block this way so all three bundles keep ONE record shape instead of a
    forked builder. Keys already in the record win; extras never overwrite.
    """
    qid = str(q["question_id"])
    iso3 = q.get("iso3")
    hz = q.get("hazard_code")
    metric = str(q.get("metric") or "")
    hs_run_id = q.get("hs_run_id")
    if run_id is None:
        run_id = _forecast_run_id_for_question(con, qid, include_test)

    record: dict[str, Any] = {
        "question": q,
        "forecast_run_id": run_id,
        "score_family": _score_family(metric),
    }

    # --- HS context ---------------------------------------------------------
    if table_exists(con, "hs_triage") and hs_run_id:
        rows = rows_as_dicts(
            con,
            "SELECT tier, triage_score, need_full_spd, drivers_json, "
            "data_quality_json, scenario_stub, regime_change_likelihood, "
            "regime_change_magnitude, regime_change_score, regime_change_level, "
            "regime_change_direction, regime_change_window, regime_change_json, "
            "track FROM hs_triage WHERE run_id = ? AND iso3 = ? AND hazard_code = ?",
            [hs_run_id, iso3, hz],
        )
        if rows:
            t = rows[0]
            rc_json = safe_json_loads(t.get("regime_change_json")) or {}
            record["regime_change"] = {
                "likelihood": t.get("regime_change_likelihood"),
                "magnitude": t.get("regime_change_magnitude"),
                "score": t.get("regime_change_score"),
                "level": t.get("regime_change_level"),
                "direction": t.get("regime_change_direction"),
                "window": t.get("regime_change_window"),
                "rationale_bullets": rc_json.get("rationale_bullets"),
                "trigger_signals": rc_json.get("trigger_signals"),
            }
            record["triage"] = {
                "tier": t.get("tier"),
                "triage_score": t.get("triage_score"),
                "need_full_spd": t.get("need_full_spd"),
                "track": t.get("track"),
                "drivers": safe_json_loads(t.get("drivers_json")),
                "data_quality": safe_json_loads(t.get("data_quality_json")),
                "scenario_stub": t.get("scenario_stub"),
            }

    if table_exists(con, "hs_hazard_tail_packs") and hs_run_id:
        packs = rows_as_dicts(
            con,
            "SELECT query, report_markdown, structural_context, "
            "recent_signals_json, sources_json, grounded FROM hs_hazard_tail_packs "
            "WHERE hs_run_id = ? AND iso3 = ? AND hazard_code = ? ORDER BY query",
            [hs_run_id, iso3, hz],
        )
        if packs:
            record["grounding"] = [
                {
                    "kind": (
                        "rc_grounding"
                        if str(p.get("query") or "").startswith("rc_grounding")
                        else "triage_grounding"
                        if str(p.get("query") or "").startswith("triage_grounding")
                        else "unknown"
                    ),
                    "query": p.get("query"),
                    "grounded": p.get("grounded"),
                    "report_markdown": p.get("report_markdown"),
                    "structural_context": p.get("structural_context"),
                    "recent_signals": safe_json_loads(p.get("recent_signals_json")),
                    "sources": safe_json_loads(p.get("sources_json")),
                }
                for p in packs
            ]

    if table_exists(con, "hs_adversarial_checks") and hs_run_id:
        adv = rows_as_dicts(
            con,
            "SELECT rc_level, net_assessment, summary, payload_json, sources_json, "
            "model_id FROM hs_adversarial_checks "
            "WHERE hs_run_id = ? AND iso3 = ? AND hazard_code = ?",
            [hs_run_id, iso3, hz],
        )
        if adv:
            a = adv[0]
            record["adversarial"] = {
                "rc_level": a.get("rc_level"),
                "net_assessment": a.get("net_assessment"),
                "summary": a.get("summary"),
                "payload": safe_json_loads(a.get("payload_json")),
                "sources": safe_json_loads(a.get("sources_json")),
                "model_id": a.get("model_id"),
            }

    # --- Forecast-time reasoning -------------------------------------------
    spd_prompt, prompt_source, calls_by_model = _load_prompt_and_responses(
        con, qid, run_id, include_test
    )
    record["spd_prompt"] = spd_prompt
    record["spd_prompt_source"] = prompt_source

    members = _load_members(con, qid, run_id)
    for m in members:
        call = calls_by_model.get(m["model_name"]) or {}
        m["provider"] = call.get("provider")
        m["model_id"] = call.get("model_id")
        m["response_text"] = call.get("response_text")
        m["error_text"] = call.get("error_text")
        member_prompt = call.get("prompt_text")
        if member_prompt and spd_prompt and member_prompt != spd_prompt:
            m["sent_prompt_override"] = member_prompt
    _recompute_trace_quality(members, str(hz or ""), metric)
    record["members"] = members

    record["ensemble"] = _load_ensemble_grids(con, qid, run_id)

    if table_exists(con, "calibration_weights"):
        weights = rows_as_dicts(
            con,
            "SELECT as_of_month, model_name, weight, n_questions, avg_brier "
            "FROM calibration_weights WHERE hazard_code = ? AND metric = ? "
            "AND as_of_month = (SELECT MAX(as_of_month) FROM calibration_weights "
            "WHERE hazard_code = ? AND metric = ?) ORDER BY model_name",
            [hz, metric, hz, metric],
        )
        if weights:
            record["weights_applied"] = weights

    # --- Outcome + scores ---------------------------------------------------
    # Guarded: the current-run bundle builds from a DB where these tables may
    # not exist yet (nothing has resolved); the scored bundle always has them.
    if table_exists(con, "resolutions"):
        has_source_desc = column_exists(con, "resolutions", "source_desc")
        resolution_rows = rows_as_dicts(
            con,
            "SELECT horizon_m, observed_month, value, source_snapshot_ym"
            + (", source_desc" if has_source_desc else "")
            + " FROM resolutions WHERE question_id = ? ORDER BY horizon_m",
            [qid],
        )
        for r in resolution_rows:
            r["realized_bucket"] = _realized_bucket(con, metric, r.get("value"))
        resolved_horizons = {int(r["horizon_m"]) for r in resolution_rows if r.get("horizon_m") is not None}
        record["outcome"] = {
            "resolutions": resolution_rows,
            "unresolved_horizons": sorted(set(range(1, 7)) - resolved_horizons),
        }

    if table_exists(con, "scores"):
        record["scores"] = rows_as_dicts(
            con,
            "SELECT horizon_m, model_name, score_type, value FROM scores "
            "WHERE question_id = ? ORDER BY model_name, score_type, horizon_m",
            [qid],
        )

    # --- Sibyl cross-reference ---------------------------------------------
    if table_exists(con, "sibyl_forecasts"):
        sib = rows_as_dicts(
            con,
            "SELECT sibyl_run_id, status, skip_reason, k, aggregation, "
            "volatility_score, js_divergence_vs_standard, js_divergence_inter_trial, "
            "cost_usd, opus_cost_usd, brave_cost_usd, leakage_json, "
            "pooled_quantiles_json, trials_json FROM sibyl_forecasts "
            "WHERE question_id = ? ORDER BY created_at DESC LIMIT 1",
            [qid],
        )
        if sib:
            s = sib[0]
            record["sibyl"] = {
                "sibyl_run_id": s.get("sibyl_run_id"),
                "status": s.get("status"),
                "skip_reason": s.get("skip_reason"),
                "k": s.get("k"),
                "aggregation": s.get("aggregation"),
                "volatility_score": s.get("volatility_score"),
                "js_divergence_vs_standard": s.get("js_divergence_vs_standard"),
                "js_divergence_inter_trial": s.get("js_divergence_inter_trial"),
                "cost_usd": s.get("cost_usd"),
                "leakage": safe_json_loads(s.get("leakage_json")),
                "pooled_quantiles": safe_json_loads(s.get("pooled_quantiles_json")),
            }
            if include_sibyl_trials:
                record["sibyl"]["trials"] = safe_json_loads(s.get("trials_json"))
            else:
                record["sibyl"]["trials_available"] = bool(s.get("trials_json"))

    if extras:
        for key, value in extras.items():
            record.setdefault(key, value)

    return record


# ---------------------------------------------------------------------------
# Flat tables
# ---------------------------------------------------------------------------


def _emit_scores_flat(con, out_dir: Path, qids: list[str]) -> None:
    has_source_desc = column_exists(con, "resolutions", "source_desc")
    has_run_id = column_exists(con, "scores", "run_id")
    rows = rows_as_dicts(
        con,
        "SELECT s.question_id, q.iso3, q.hazard_code, UPPER(q.metric) AS metric, "
        "CASE WHEN UPPER(q.metric) = 'EVENT_OCCURRENCE' THEN 'binary' ELSE 'spd' END "
        "AS score_family, s.horizon_m, s.model_name, s.score_type, s.value, "
        "r.value AS resolved_value, r.observed_month, r.source_snapshot_ym"
        + (", r.source_desc" if has_source_desc else ", NULL AS source_desc")
        + (", s.run_id" if has_run_id else ", NULL AS run_id")
        + " FROM scores s JOIN questions q ON q.question_id = s.question_id "
        "LEFT JOIN resolutions r ON r.question_id = s.question_id "
        "AND r.horizon_m = s.horizon_m "
        "WHERE s.question_id IN (SELECT UNNEST(?::VARCHAR[]))"
        " ORDER BY s.question_id, s.model_name, s.score_type, s.horizon_m",
        [qids],
    )
    write_csv(
        out_dir / "scores_flat.csv",
        [
            "question_id", "iso3", "hazard_code", "metric", "score_family",
            "horizon_m", "model_name", "score_type", "value", "resolved_value",
            "observed_month", "source_snapshot_ym", "source_desc", "run_id",
        ],
        rows,
    )


def _emit_forecast_vs_outcome(con, out_dir: Path, qids: list[str]) -> None:
    if not table_exists(con, "forecasts_ensemble") or not table_exists(con, "resolutions"):
        return
    rows = rows_as_dicts(
        con,
        "SELECT fe.question_id, q.iso3, q.hazard_code, UPPER(q.metric) AS metric, "
        "fe.model_name, r.horizon_m, r.value AS resolved_value, "
        "fe.bucket_index, fe.probability, fe.ev_value "
        "FROM forecasts_ensemble fe "
        "JOIN questions q ON q.question_id = fe.question_id "
        "JOIN resolutions r ON r.question_id = fe.question_id "
        "AND r.horizon_m = fe.month_index "
        "WHERE fe.question_id IN (SELECT UNNEST(?::VARCHAR[])) "
        "ORDER BY fe.question_id, fe.model_name, r.horizon_m, fe.bucket_index",
        [qids],
    )
    # Collapse bucket rows → one row per (question, model, horizon).
    grouped: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        key = (r["question_id"], r["model_name"], r["horizon_m"])
        g = grouped.setdefault(
            key,
            {
                "question_id": r["question_id"],
                "iso3": r["iso3"],
                "hazard_code": r["hazard_code"],
                "metric": r["metric"],
                "model_name": r["model_name"],
                "horizon_m": r["horizon_m"],
                "resolved_value": r["resolved_value"],
                "ev_value": r.get("ev_value"),
                "_probs": {},
            },
        )
        g["_probs"][int(r["bucket_index"] or 0)] = float(r["probability"] or 0.0)

    out_rows: list[dict[str, Any]] = []
    for g in grouped.values():
        probs = g.pop("_probs")
        realized = _realized_bucket(con, g["metric"], g["resolved_value"])
        modal = max(probs, key=probs.get) if probs else None
        g["realized_bucket"] = realized
        g["p_realized_bucket"] = probs.get(realized) if realized is not None else None
        g["modal_bucket"] = modal
        g["p_modal_bucket"] = probs.get(modal) if modal is not None else None
        out_rows.append(g)
    out_rows.sort(key=lambda r: (r["question_id"], r["model_name"], r["horizon_m"]))
    write_csv(
        out_dir / "forecast_vs_outcome.csv",
        [
            "question_id", "iso3", "hazard_code", "metric", "model_name",
            "horizon_m", "resolved_value", "realized_bucket", "p_realized_bucket",
            "modal_bucket", "p_modal_bucket", "ev_value",
        ],
        out_rows,
    )


def _attach_skill(rows: list[dict[str, Any]]) -> None:
    """Attach skill-vs-climatology columns to rollup rows in place.

    skill = 1 − (model_mean / climatology_mean), per
    (hazard, metric, score_family, score_type) — NEVER across score types or
    families. Positive = beat the base rate; 0 = matched; negative = lost.
    Climatology comes from the `__ext_climatology` reference rows written by
    pythia/tools/score_baselines.py; where no climatology row exists for a
    group (no base-rate anchor for that pair), skill stays empty.
    """
    clim_mean: dict[tuple, float] = {}
    for r in rows:
        if r.get("model_name") == "__ext_climatology" and r.get("mean_value") is not None:
            key = (r.get("hazard_code"), r.get("metric"),
                   r.get("score_family"), r.get("score_type"))
            clim_mean[key] = float(r["mean_value"])
    for r in rows:
        key = (r.get("hazard_code"), r.get("metric"),
               r.get("score_family"), r.get("score_type"))
        clim = clim_mean.get(key)
        r["climatology_mean"] = clim
        if clim is not None and clim > 0 and r.get("mean_value") is not None:
            r["skill_vs_climatology"] = round(1.0 - float(r["mean_value"]) / clim, 4)
        else:
            r["skill_vs_climatology"] = None


def _emit_rollups(con, out_dir: Path, qids: list[str]) -> list[dict[str, Any]]:
    rows = rows_as_dicts(
        con,
        "SELECT q.hazard_code, UPPER(q.metric) AS metric, "
        "CASE WHEN UPPER(q.metric) = 'EVENT_OCCURRENCE' THEN 'binary' ELSE 'spd' END "
        "AS score_family, s.model_name, s.score_type, COUNT(*) AS n_samples, "
        "COUNT(DISTINCT s.question_id) AS n_questions, AVG(s.value) AS mean_value, "
        "MEDIAN(s.value) AS median_value "
        "FROM scores s JOIN questions q ON q.question_id = s.question_id "
        "WHERE s.question_id IN (SELECT UNNEST(?::VARCHAR[])) "
        "GROUP BY 1, 2, 3, 4, 5 ORDER BY 3, 1, 2, 5, 8",
        [qids],
    )
    _attach_skill(rows)
    write_csv(
        out_dir / "rollups.csv",
        [
            "hazard_code", "metric", "score_family", "model_name", "score_type",
            "n_samples", "n_questions", "mean_value", "median_value",
            "climatology_mean", "skill_vs_climatology",
        ],
        rows,
    )
    return rows


def _emit_calibration(con, out_dir: Path) -> list[dict[str, Any]]:
    """calibration_weights.csv with weight movement vs the previous vintage."""
    movement_rows: list[dict[str, Any]] = []
    if table_exists(con, "calibration_weights"):
        rows = rows_as_dicts(
            con,
            "SELECT as_of_month, hazard_code, metric, model_name, weight, "
            "n_questions, avg_brier FROM calibration_weights "
            "ORDER BY hazard_code, metric, model_name, as_of_month",
        )
        by_key: dict[tuple, list[dict[str, Any]]] = {}
        for r in rows:
            by_key.setdefault((r["hazard_code"], r["metric"], r["model_name"]), []).append(r)
        for (hz, metric, model), vintages in sorted(by_key.items()):
            current = vintages[-1]
            previous = vintages[-2] if len(vintages) > 1 else None
            movement_rows.append(
                {
                    "hazard_code": hz,
                    "metric": metric,
                    "model_name": model,
                    "as_of_month": current.get("as_of_month"),
                    "weight": current.get("weight"),
                    "previous_as_of_month": previous.get("as_of_month") if previous else None,
                    "previous_weight": previous.get("weight") if previous else None,
                    "weight_delta": (
                        (current.get("weight") or 0) - (previous.get("weight") or 0)
                        if previous
                        else None
                    ),
                    "n_questions": current.get("n_questions"),
                    "avg_brier": current.get("avg_brier"),
                }
            )
        write_csv(
            out_dir / "calibration_weights.csv",
            [
                "hazard_code", "metric", "model_name", "as_of_month", "weight",
                "previous_as_of_month", "previous_weight", "weight_delta",
                "n_questions", "avg_brier",
            ],
            movement_rows,
        )

    if table_exists(con, "calibration_advice"):
        advice_rows = rows_as_dicts(
            con,
            "SELECT as_of_month, hazard_code, metric, model_name, advice, "
            "findings_json FROM calibration_advice "
            "WHERE as_of_month = (SELECT MAX(as_of_month) FROM calibration_advice) "
            "ORDER BY hazard_code, metric, model_name",
        )
        lines = ["# Calibration Advice (latest vintage)", ""]
        for r in advice_rows:
            lines.append(
                f"## {r.get('hazard_code')} / {r.get('metric')} / {r.get('model_name')}"
            )
            lines.append("")
            lines.append(str(r.get("advice") or "_(no advice)_"))
            findings = safe_json_loads(r.get("findings_json"))
            if findings:
                import json as _json

                lines.append("")
                lines.append("```json")
                lines.append(_json.dumps(findings, indent=1, default=str))
                lines.append("```")
            lines.append("")
        (out_dir / "calibration_advice.md").write_text("\n".join(lines), encoding="utf-8")

    if table_exists(con, "eiv_scores"):
        eiv = rows_as_dicts(con, "SELECT * FROM eiv_scores")
        if eiv:
            write_csv(out_dir / "eiv_scores.csv", list(eiv[0].keys()), eiv)
    return movement_rows


# ---------------------------------------------------------------------------
# Index, ranking, digest, briefing
# ---------------------------------------------------------------------------


def _question_summary(record: dict[str, Any]) -> dict[str, Any]:
    q = record["question"]
    members = record.get("members") or []
    tq = [
        (m.get("trace_quality") or {}).get("trace_quality_score")
        for m in members
        if isinstance(m.get("trace_quality"), dict)
    ]
    tq = [t for t in tq if isinstance(t, (int, float))]
    score_by_type: dict[str, list[float]] = {}
    ranking_model = None
    for pref in RANKING_MODEL_PREFERENCE:
        if any(s.get("model_name") == pref for s in record.get("scores") or []):
            ranking_model = pref
            break
    for s in record.get("scores") or []:
        if s.get("model_name") == ranking_model and s.get("value") is not None:
            score_by_type.setdefault(str(s.get("score_type")), []).append(float(s["value"]))
    mean_scores = {k: sum(v) / len(v) for k, v in score_by_type.items() if v}
    return {
        "question_id": q.get("question_id"),
        "iso3": q.get("iso3"),
        "hazard_code": q.get("hazard_code"),
        "metric": q.get("metric"),
        "score_family": record.get("score_family"),
        "track": q.get("track"),
        "target_month": q.get("target_month"),
        "tier": (record.get("triage") or {}).get("tier"),
        "triage_score": (record.get("triage") or {}).get("triage_score"),
        "rc_level": (record.get("regime_change") or {}).get("level"),
        "rc_score": (record.get("regime_change") or {}).get("score"),
        "n_members": len(members),
        "avg_trace_quality": round(sum(tq) / len(tq), 4) if tq else None,
        "n_horizons_resolved": len((record.get("outcome") or {}).get("resolutions") or []),
        "ranking_model": ranking_model,
        "mean_brier": mean_scores.get("brier"),
        "mean_log": mean_scores.get("log"),
        "mean_crps": mean_scores.get("crps"),
        "has_sibyl": "sibyl" in record,
        "record_path": f"questions/{q.get('question_id')}.json",
    }


def _select_case_studies(
    summaries: list[dict[str, Any]], n_per_side: int
) -> dict[str, list[str]]:
    """Worst-N and best-N question_ids per score_family by mean Brier."""
    selection: dict[str, list[str]] = {"worst": [], "best": []}
    for family in ("spd", "binary"):
        ranked = [
            s
            for s in summaries
            if s.get("score_family") == family and s.get("mean_brier") is not None
        ]
        ranked.sort(key=lambda s: s["mean_brier"])
        best = ranked[:n_per_side]
        worst = ranked[-n_per_side:][::-1]
        selection["best"].extend([s["question_id"] for s in best])
        selection["worst"].extend([s["question_id"] for s in worst if s["question_id"] not in {b["question_id"] for b in best}])
    return selection


def _write_digest(
    out_dir: Path,
    summaries: list[dict[str, Any]],
    rollups: list[dict[str, Any]],
    weight_movement: list[dict[str, Any]],
    case_selection: dict[str, list[str]],
    *,
    months_back: int,
) -> None:
    lines: list[str] = [
        "# Scored-Forecast Analysis — Digest",
        "",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
        f"question-epoch window: last {months_back} months_",
        "",
        f"**{len(summaries)} scored questions** "
        f"({sum(1 for s in summaries if s['score_family'] == 'spd')} SPD, "
        f"{sum(1 for s in summaries if s['score_family'] == 'binary')} binary; "
        f"{sum(1 for s in summaries if s.get('has_sibyl'))} with Sibyl coverage).",
        "",
        "## Coverage by hazard/metric",
        "",
        "| hazard | metric | questions | mean ensemble Brier |",
        "|---|---|---|---|",
    ]
    by_hm: dict[tuple, list[dict[str, Any]]] = {}
    for s in summaries:
        by_hm.setdefault((s.get("hazard_code"), s.get("metric")), []).append(s)
    for (hz, metric), items in sorted(by_hm.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        briers = [s["mean_brier"] for s in items if s.get("mean_brier") is not None]
        mean_b = f"{sum(briers) / len(briers):.4f}" if briers else "—"
        lines.append(f"| {hz} | {metric} | {len(items)} | {mean_b} |")

    lines += [
        "",
        "## Model comparison (Brier by score family — never blend the two)",
        "",
        "_skill = 1 − mean/climatology-mean within the same (hazard, metric, "
        "family, score_type) group, aggregated here across groups; positive = "
        "beat the base rate. `__ext_climatology` / `__ext_uniform` are the "
        "reference forecasters, not Pythia models._",
        "",
        "| family | model | score_type | n | mean | median | skill vs climatology |",
        "|---|---|---|---|---|---|---|",
    ]
    # The rollup rows are per (hazard, metric); the digest table aggregates
    # per (family, model). Skill is averaged over the groups that HAVE a
    # climatology reference, weighted by sample count.
    agg: dict[tuple, dict[str, float]] = {}
    for r in rollups:
        if r.get("score_type") not in ("brier",):
            continue
        key = (r["score_family"], r["model_name"])
        a = agg.setdefault(key, {"n": 0, "vsum": 0.0, "msum": 0.0,
                                 "skill_n": 0, "skill_sum": 0.0})
        n = int(r["n_samples"] or 0)
        a["n"] += n
        a["vsum"] += float(r["mean_value"] or 0) * n
        a["msum"] += float(r["median_value"] or 0) * n
        if r.get("skill_vs_climatology") is not None:
            a["skill_n"] += n
            a["skill_sum"] += float(r["skill_vs_climatology"]) * n
    for (family, model), a in sorted(agg.items()):
        if not a["n"]:
            continue
        skill = (
            f"{a['skill_sum'] / a['skill_n']:+.3f}" if a["skill_n"] else "—"
        )
        lines.append(
            f"| {family} | {model} | brier | {a['n']} "
            f"| {a['vsum'] / a['n']:.4f} | {a['msum'] / a['n']:.4f} | {skill} |"
        )

    def _qline(s: dict[str, Any]) -> str:
        return (
            f"- `{s['question_id']}` ({s['hazard_code']}/{s['metric']}, "
            f"rc_level={s.get('rc_level')}, trace_quality={s.get('avg_trace_quality')}) "
            f"mean Brier {s.get('mean_brier'):.4f} → {s['record_path']}"
        )

    by_id = {s["question_id"]: s for s in summaries}
    lines += ["", "## Worst-scored questions (case studies)", ""]
    lines += [_qline(by_id[qid]) for qid in case_selection.get("worst", []) if qid in by_id]
    lines += ["", "## Best-scored questions (case studies)", ""]
    lines += [_qline(by_id[qid]) for qid in case_selection.get("best", []) if qid in by_id]

    movers = [m for m in weight_movement if m.get("weight_delta") is not None]
    movers.sort(key=lambda m: abs(m["weight_delta"]), reverse=True)
    if movers:
        lines += ["", "## Biggest calibration-weight movements", ""]
        for m in movers[:10]:
            lines.append(
                f"- {m['hazard_code']}/{m['metric']} {m['model_name']}: "
                f"{m.get('previous_weight')} → {m.get('weight')} "
                f"(Δ {m['weight_delta']:+.3f})"
            )

    lines += [
        "",
        "## Files",
        "",
        "- `ANALYST_GUIDE.md` — how to read everything (schemas, semantics, caveats).",
        "- `questions_index.csv` — one row per scored question.",
        "- `questions/{id}.json` — full reasoning→outcome record.",
        "- `case_studies/` — the best/worst records above (with Sibyl trials).",
        "- `scores_flat.csv`, `forecast_vs_outcome.csv`, `rollups.csv`, "
        "`calibration_weights.csv`, `calibration_advice.md`, `eiv_scores.csv`.",
        "- `briefing/` — condensed chat-uploadable digest + case studies.",
    ]
    path = out_dir / "digest.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    size_guard(path, DIGEST_BUDGET_KB)


def _load_staged_record(staging: Path, qid: str) -> dict[str, Any] | None:
    """Re-read a per-question record from staging (case studies only).

    Records are streamed to disk during the build and never all retained in
    memory — this is the read-back path for the ≤2N case-study ids.
    """

    path = staging / "questions" / f"{qid}.json"
    try:
        with path.open(encoding="utf-8") as fh:
            loaded = json.load(fh)
        return loaded if isinstance(loaded, dict) else None
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not re-read staged record for %s: %s", qid, exc)
        return None


def _write_briefing(
    out_dir: Path,
    records_by_id: dict[str, dict[str, Any]],
    case_selection: dict[str, list[str]],
) -> None:
    briefing_dir = out_dir / "briefing"
    briefing_dir.mkdir(parents=True, exist_ok=True)

    # 01: guide essence + digest (digest is already compact — reuse it).
    digest_text = (out_dir / "digest.md").read_text(encoding="utf-8")
    intro = (
        "# Pythia Scored-Forecast Briefing\n\n"
        "This is the condensed, chat-uploadable slice of a larger analysis "
        "bundle. Scores: lower is better; SPD Brier ranges 0–2, binary Brier "
        "0–1 — the two families must never be averaged together. A missing "
        "resolution horizon means 'unknown', not zero. Each case study below "
        "pairs the models' own reasoning with what actually happened; look "
        "for systematic reasoning failures (bad priors, over/under-reaction "
        "to signals, ignored dampeners), not one-off misses.\n\n---\n\n"
    )
    p1 = briefing_dir / "01_run_briefing.md"
    p1.write_text(intro + digest_text, encoding="utf-8")
    size_guard(p1, BRIEFING_BUDGET_KB)

    # 02: condensed case studies.
    lines = ["# Case Studies — Reasoning vs Outcome", ""]
    for side in ("worst", "best"):
        lines.append(f"## {side.capitalize()}-scored")
        lines.append("")
        for qid in case_selection.get(side, []):
            record = records_by_id.get(qid)
            if not record:
                continue
            q = record["question"]
            lines.append(f"### {qid}")
            lines.append("")
            lines.append(f"**Question:** {_truncate(q.get('wording'), 400)}")
            rc = record.get("regime_change") or {}
            lines.append(
                f"**RC:** level={rc.get('level')} score={rc.get('score')} "
                f"direction={rc.get('direction')}"
            )
            outcome = record.get("outcome") or {}
            res_bits = [
                f"h{r.get('horizon_m')}={r.get('value')} (bucket {r.get('realized_bucket')})"
                for r in outcome.get("resolutions") or []
            ]
            lines.append(f"**Outcome:** {', '.join(res_bits) or 'none'}")
            score_bits = {}
            for s in record.get("scores") or []:
                if s.get("score_type") == "brier" and s.get("model_name") in RANKING_MODEL_PREFERENCE:
                    score_bits.setdefault(s["model_name"], []).append(float(s.get("value") or 0))
            for model, vals in score_bits.items():
                lines.append(f"**{model} mean Brier:** {sum(vals) / len(vals):.4f}")
            lines.append("")
            for m in record.get("members") or []:
                trace = m.get("reasoning_trace") or {}
                prior = trace.get("prior") or {}
                tq = (m.get("trace_quality") or {}).get("trace_quality_score")
                lines.append(f"- **{m.get('model_name')}** (trace_quality={tq}):")
                if prior.get("rationale"):
                    lines.append(f"  - prior: {_truncate(prior.get('rationale'), 250)}")
                for u in (trace.get("updates") or [])[:3]:
                    lines.append(
                        f"  - update: {_truncate(u.get('signal'), 150)} "
                        f"({u.get('direction')}/{u.get('magnitude')})"
                    )
                if m.get("human_explanation"):
                    lines.append(f"  - said: {_truncate(m.get('human_explanation'), 300)}")
            lines.append("")
    p2 = briefing_dir / "02_case_studies.md"
    p2.write_text("\n".join(lines), encoding="utf-8")
    size_guard(p2, BRIEFING_BUDGET_KB)


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------


def build_bundle(
    db: str,
    out_dir: Path,
    *,
    months_back: int = 12,
    n_case_studies: int = 10,
    include_test: bool = False,
    include_sibyl_trials: str = "case-studies",
    keep_staging: bool = False,
) -> Path | None:
    con = open_db(db)
    db_path = resolve_db_path(db)
    try:
        if not table_exists(con, "scores") or not table_exists(con, "questions"):
            LOGGER.warning("scores/questions tables missing — nothing to bundle")
            return None

        cutoff = None
        if months_back and months_back > 0:
            now = datetime.now(timezone.utc)
            month = now.month - months_back
            year = now.year
            while month <= 0:
                month += 12
                year -= 1
            cutoff = f"{year:04d}-{month:02d}-01"

        window_clause = ""
        params: list[Any] = []
        if cutoff:
            window_clause = " AND (q.window_start_date IS NULL OR q.window_start_date >= ?)"
            params.append(cutoff)

        questions = rows_as_dicts(
            con,
            "SELECT DISTINCT q.question_id, q.hs_run_id, q.iso3, q.hazard_code, "
            "q.metric, q.target_month, q.window_start_date, q.window_end_date, "
            "q.wording, q.status, q.track, q.pythia_metadata_json "
            "FROM questions q JOIN scores s ON s.question_id = q.question_id "
            "WHERE 1=1" + _test_clause(con, "questions", "q", include_test) + window_clause
            + " ORDER BY q.question_id",
            params,
        )
        if not questions:
            LOGGER.warning("No scored questions in window — nothing to bundle")
            return None
        LOGGER.info("Bundling %d scored questions", len(questions))

        label = datetime.now(timezone.utc).strftime("%Y-%m")
        staging = out_dir / f"scored_forecast_analysis__{label}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        qids = [str(q["question_id"]) for q in questions]

        # Per-question records: genuinely streamed — each record is written
        # to staging and RELEASED. Records inline untruncated grounding
        # packs, full prompts and member responses (100s of KB each), so
        # retaining all of them (as an earlier version did via a
        # records_by_id dict) OOMs at production scale; only the ≤2N case
        # studies are re-read from staging after selection.
        summaries: list[dict[str, Any]] = []
        include_all_trials = include_sibyl_trials == "all"
        for q in questions:
            qid = str(q["question_id"])
            try:
                record = build_question_record(
                    con,
                    q,
                    include_test=include_test,
                    include_sibyl_trials=include_all_trials,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to build record for %s: %s", qid, exc)
                continue
            write_json(staging / "questions" / f"{qid}.json", record)
            summaries.append(_question_summary(record))
            del record

        write_csv(
            staging / "questions_index.csv",
            [
                "question_id", "iso3", "hazard_code", "metric", "score_family",
                "track", "target_month", "tier", "triage_score", "rc_level",
                "rc_score", "n_members", "avg_trace_quality",
                "n_horizons_resolved", "ranking_model", "mean_brier", "mean_log",
                "mean_crps", "has_sibyl", "record_path",
            ],
            summaries,
        )

        _emit_scores_flat(con, staging, qids)
        _emit_forecast_vs_outcome(con, staging, qids)
        rollups = _emit_rollups(con, staging, qids)
        weight_movement = _emit_calibration(con, staging)

        case_selection = _select_case_studies(summaries, n_case_studies)
        case_dir = staging / "case_studies"
        case_records: dict[str, dict[str, Any]] = {}
        for qid in case_selection.get("worst", []) + case_selection.get("best", []):
            record = _load_staged_record(staging, qid)
            if not record:
                continue
            if include_sibyl_trials in ("case-studies", "all") and "sibyl" in record:
                if "trials" not in record["sibyl"]:
                    sib = rows_as_dicts(
                        con,
                        "SELECT trials_json FROM sibyl_forecasts WHERE question_id = ? "
                        "ORDER BY created_at DESC LIMIT 1",
                        [qid],
                    )
                    if sib:
                        record["sibyl"]["trials"] = safe_json_loads(sib[0].get("trials_json"))
            write_json(case_dir / f"{qid}.json", record)
            case_records[qid] = record

        _write_digest(
            staging, summaries, rollups, weight_movement, case_selection,
            months_back=months_back,
        )
        _write_briefing(staging, case_records, case_selection)

        guide = build_analyst_guide(
            {"n_questions": len(summaries), "months_back": months_back}
        )
        guide += "\n\n" + build_question_record_schema_md()
        (staging / "ANALYST_GUIDE.md").write_text(guide, encoding="utf-8")

        table_counts = {}
        for t in ("questions", "scores", "resolutions", "forecasts_raw",
                  "forecasts_ensemble", "calibration_weights", "calibration_advice"):
            if table_exists(con, t):
                from scripts.ai_bundle.common import row_count

                table_counts[t] = row_count(con, t)
        write_manifest(
            staging,
            bundle_kind="scored_forecast_analysis",
            db_path=db_path,
            table_counts=table_counts,
            extra={
                "n_scored_questions": len(summaries),
                "months_back": months_back,
                "include_test": include_test,
                "case_studies": case_selection,
            },
        )

        zip_path = out_dir / f"scored_forecast_analysis__{label}.zip"
        write_bundle_zip(staging, zip_path)
        if not keep_staging:
            shutil.rmtree(staging, ignore_errors=True)
        zip_mb = zip_path.stat().st_size / 1e6
        LOGGER.info("Bundle written: %s (%.1f MB)", zip_path, zip_mb)
        if zip_mb > _ZIP_WARN_MB:
            # The digest/briefing guards cover the chat-uploadable files;
            # the ZIP itself is what can blow past Actions artifact limits.
            LOGGER.warning(
                "Bundle zip is %.0f MB (warn threshold %d MB) — consider a "
                "shorter --months-back or fewer case studies", zip_mb, _ZIP_WARN_MB,
            )
        return zip_path
    finally:
        try:
            con.close()
        except Exception:  # noqa: BLE001
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="DuckDB URL or path")
    parser.add_argument("--out-dir", default="ai_bundle", help="Output directory")
    parser.add_argument("--months-back", type=int, default=12,
                        help="Question-epoch window (window_start_date cutoff)")
    parser.add_argument("--n-case-studies", type=int, default=10,
                        help="Best/worst count per score family")
    parser.add_argument("--include-test", action="store_true",
                        help="Include is_test rows (default: excluded)")
    parser.add_argument("--include-sibyl-trials",
                        choices=["case-studies", "all", "none"],
                        default="case-studies",
                        help="Where to inline full Sibyl trial traces")
    parser.add_argument("--keep-staging", action="store_true",
                        help="Keep the unzipped staging directory")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[ai_bundle] %(message)s")
    zip_path = build_bundle(
        args.db,
        Path(args.out_dir),
        months_back=args.months_back,
        n_case_studies=args.n_case_studies,
        include_test=args.include_test,
        include_sibyl_trials=args.include_sibyl_trials,
        keep_staging=args.keep_staging,
    )
    if zip_path is None:
        # Nothing to bundle is a soft outcome, not a failure — the workflow
        # step is continue-on-error anyway, but exit 0 keeps logs green.
        print("[ai_bundle] no bundle produced (no scored questions)")
        return 0
    print(f"[ai_bundle] BUNDLE_PATH={zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
