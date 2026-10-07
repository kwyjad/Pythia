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

from scripts.ai_bundle import error_attribution as _err
from scripts.ai_bundle import provenance as _prov
from scripts.ai_bundle.common import (
    column_exists,
    latest_run_clause,
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
    RC_PROMOTED_TIER,
    triage_view,
)
from scripts.ai_bundle.guides import build_analyst_guide, build_question_record_schema_md
from pythia.tools.scoring_class import has_scoring_class, scored_only_sql

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


# ---------------------------------------------------------------------------
# Scoring class and resolution reading (ACE/PA, Oct 2026)
# ---------------------------------------------------------------------------


def _resolution_extra_cols(con, alias: str = "r") -> str:
    """``scoring_class``, ``scoring_class_reason`` and the resolution's date,
    NULL on a DB whose resolutions table predates them."""
    p = f"{alias}." if alias else ""
    sc = (f"{p}scoring_class" if column_exists(con, "resolutions", "scoring_class")
          else "CAST(NULL AS VARCHAR)")
    scr = (f"{p}scoring_class_reason" if column_exists(con, "resolutions", "scoring_class_reason")
           else "CAST(NULL AS VARCHAR)")
    ca = (f"CAST({p}created_at AS DATE)" if column_exists(con, "resolutions", "created_at")
          else "CAST(NULL AS DATE)")
    return f"{sc} AS scoring_class, {scr} AS scoring_class_reason, {ca} AS resolved_on"


def _resolution_reading(hazard: Any, metric: Any, observed_month: Any, resolved_on: Any) -> str | None:
    """The reading (first, or d60/d90, d180/d270) the resolution was taken at
    (``compute_resolutions.reading_label``); None when it cannot be said."""
    if not observed_month or resolved_on is None:
        return None
    try:
        from datetime import date as _date

        from pythia.tools.compute_resolutions import reading_label  # noqa: PLC0415

        as_of = resolved_on if isinstance(resolved_on, _date) else _date.fromisoformat(str(resolved_on)[:10])
        return reading_label(str(hazard or ""), str(metric or ""), str(observed_month)[:7], as_of)
    except Exception:  # noqa: BLE001 - a label never costs a row
        return None


def _is_indicative(row: Mapping[str, Any]) -> bool:
    return str(row.get("scoring_class") or "").lower() == "indicative"


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


#: A resolved value within this fraction of a bucket boundary is flagged
#: ``bucket_edge``: it would change bucket under a revision of a few percent
#: (ACLED revises monthly counts for weeks), so a forecast scored "wrong"
#: against it may have been one data revision away from "right".
BUCKET_EDGE_TOLERANCE = 0.05


def _bucket_edge(con, metric: str, value: float | None) -> tuple[bool | None, float | None]:
    """(is the value within 5% of its nearest bucket boundary, that boundary).

    ``nearest_boundary`` is the closest FINITE interior boundary (the open-ended
    top bucket's ``inf`` is not a boundary a value can sit beside: 5% of
    infinity is infinity, which is how 84 of 160 rows once read as edge
    cases). A value of 0 is never an edge case — it is its own bucket, and no
    revision of a few percent moves a zero. Binary rows carry neither.
    """
    if value is None or _score_family(metric) == "binary":
        return None, None
    v = float(value)
    interior = sorted(
        float(b) for b in (_thresholds_for_metric(con, metric) or [])
        if b is not None and 0 < float(b) < float("inf")
    )
    if not interior:
        return None, None
    nearest = min(interior, key=lambda b: abs(v - b))
    if v == 0:
        return False, nearest
    return (abs(v - nearest) <= BUCKET_EDGE_TOLERANCE * nearest), nearest


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


def _shown_prior_vector(shown: Mapping[str, Any] | None) -> tuple[list[float] | None, str | None]:
    """The month-1 distribution the prompt told members to start from.

    The level-and-volatility vector where the prompt showed one, else the
    anchor ``forecast_deviation`` reconstructs. (None, reason) when neither.
    """
    if not isinstance(shown, Mapping):
        return None, "base_rate_shown absent"
    lv = shown.get("level_volatility") or {}
    if lv.get("shown"):
        vec = (lv.get("spd_by_horizon") or {}).get("1")
        if isinstance(vec, list) and vec:
            return [float(x) for x in vec], "level_volatility_h1"
    anchor = shown.get("anchor") or {}
    probs = anchor.get("probs")
    if anchor.get("available") and isinstance(probs, list) and probs:
        if all(isinstance(x, (int, float)) for x in probs):
            return [float(x) for x in probs], f"anchor:{anchor.get('source')}"
    return None, "no base-rate distribution recorded for this question"


def rescore_trace_prior(record: dict[str, Any]) -> None:
    """Score each member's stated prior against the base rate the prompt SHOWED.

    ``trace_validation`` checks the prior against a base-rate summary the
    bundle does not have, so it returned a constant 0.7 for every member and
    40% of ``trace_quality_score`` was that constant. This replaces the prior
    component with the same distance rule (modal bucket equal 1.0, one apart
    0.7, further 0.3) measured against ``base_rate_shown``, and recomputes the
    composite with the original weights. Where nothing was recorded the prior
    is marked uncompared and the composite is the other two components
    re-weighted, never a guessed constant.
    """
    vec, source = _shown_prior_vector(record.get("base_rate_shown"))
    for m in record.get("members") or []:
        tq = m.get("trace_quality")
        if not isinstance(tq, dict) or not tq.get("has_trace"):
            continue
        delta = float((tq.get("delta_arithmetic") or {}).get("score") or 0.0)
        mag = float((tq.get("magnitude_consistency") or {}).get("score") or 0.0)
        prior = ((m.get("reasoning_trace") or {}).get("prior") or {}).get("spd")
        if vec is None or not isinstance(prior, list) or len(prior) != len(vec):
            reason = source if vec is None else "prior.spd missing or a different length from the shown vector"
            tq["prior_quality"] = {"score": None, "compared": False, "detail": reason}
            tq["trace_quality_score"] = round((0.4 * delta + 0.2 * mag) / 0.6, 4)
            tq["trace_quality_basis"] = "delta_and_magnitude_only"
            continue
        try:
            model_mode = max(range(len(prior)), key=lambda i: float(prior[i]))
        except (TypeError, ValueError):
            tq["prior_quality"] = {"score": 0.0, "compared": True, "detail": "prior.spd not numeric"}
            tq["trace_quality_score"] = round(0.4 * delta + 0.2 * mag, 4)
            tq["trace_quality_basis"] = "prior_vs_base_rate_shown"
            continue
        shown_mode = max(range(len(vec)), key=lambda i: vec[i])
        distance = abs(model_mode - shown_mode)
        score = 1.0 if distance == 0 else 0.7 if distance == 1 else 0.3
        tq["prior_quality"] = {
            "score": score, "compared": True, "shown_source": source,
            "model_mode": model_mode, "shown_mode": shown_mode, "distance": distance,
        }
        tq["trace_quality_score"] = round(0.4 * score + 0.4 * delta + 0.2 * mag, 4)
        tq["trace_quality_basis"] = "prior_vs_base_rate_shown"


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
            tier, triage_score = triage_view(t)
            record["triage"] = {
                "tier": tier,
                "triage_score": triage_score,
                "triage_skipped": tier == RC_PROMOTED_TIER,
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
            + ", " + _resolution_extra_cols(con, "")
            + " FROM resolutions WHERE question_id = ? ORDER BY horizon_m",
            [qid],
        )
        for r in resolution_rows:
            r["realized_bucket"] = _realized_bucket(con, metric, r.get("value"))
            r["resolution_reading"] = _resolution_reading(
                hz, metric, r.get("observed_month"), r.pop("resolved_on", None))
            if r.get("scoring_class_reason") is None:
                r.pop("scoring_class_reason", None)
        resolved_horizons = {int(r["horizon_m"]) for r in resolution_rows if r.get("horizon_m") is not None}
        record["outcome"] = {
            "resolutions": resolution_rows,
            "unresolved_horizons": sorted(set(range(1, 7)) - resolved_horizons),
        }

    if table_exists(con, "scores"):
        # One run per question: the latest. Earlier runs of a rerun question
        # stay in scores_flat.csv (flagged is_latest_run = false) but never
        # weigh on the record's scores or on any mean built from them.
        record["scores"] = rows_as_dicts(
            con,
            "SELECT s.horizon_m, s.model_name, s.score_type, s.value FROM scores s "
            "WHERE s.question_id = ?" + latest_run_clause(con, "s")
            + " ORDER BY s.model_name, s.score_type, s.horizon_m",
            [qid],
        )
        if column_exists(con, "scores", "run_id"):
            record["score_runs"] = [
                r["run_id"] for r in rows_as_dicts(
                    con,
                    "SELECT DISTINCT run_id FROM scores WHERE question_id = ? "
                    "AND run_id IS NOT NULL ORDER BY run_id",
                    [qid],
                )
            ]

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
    has_res = table_exists(con, "resolutions")
    has_source_desc = has_res and column_exists(con, "resolutions", "source_desc")
    has_run_id = column_exists(con, "scores", "run_id")
    res_cols = (
        "r.value AS resolved_value, r.observed_month, r.source_snapshot_ym, "
        + _resolution_extra_cols(con, "r")
        if has_res else
        "NULL AS resolved_value, NULL AS observed_month, NULL AS source_snapshot_ym, "
        "NULL AS scoring_class, NULL AS scoring_class_reason, NULL AS resolved_on"
    )
    rows = rows_as_dicts(
        con,
        "SELECT s.question_id, q.iso3, q.hazard_code, UPPER(q.metric) AS metric, "
        "CASE WHEN UPPER(q.metric) = 'EVENT_OCCURRENCE' THEN 'binary' ELSE 'spd' END "
        f"AS score_family, s.horizon_m, s.model_name, s.score_type, s.value, {res_cols}"
        + (", r.source_desc" if has_source_desc else ", NULL AS source_desc")
        + (", s.run_id" if has_run_id else ", NULL AS run_id")
        + (
            ", (s.run_id IS NULL OR s.run_id = (SELECT MAX(_lr.run_id) FROM scores _lr "
            "WHERE _lr.question_id = s.question_id AND _lr.run_id IS NOT NULL)) AS is_latest_run"
            if has_run_id else ", TRUE AS is_latest_run"
        )
        + " FROM scores s JOIN questions q ON q.question_id = s.question_id "
        + ("LEFT JOIN resolutions r ON r.question_id = s.question_id "
           "AND r.horizon_m = s.horizon_m " if has_res else "")
        + "WHERE s.question_id IN (SELECT UNNEST(?::VARCHAR[]))"
        " ORDER BY s.question_id, s.model_name, s.score_type, s.horizon_m",
        [qids],
    )
    for r in rows:
        r["resolution_reading"] = _resolution_reading(
            r.get("hazard_code"), r.get("metric"), r.get("observed_month"), r.get("resolved_on"))
    write_csv(
        out_dir / "scores_flat.csv",
        [
            "question_id", "iso3", "hazard_code", "metric", "score_family",
            "horizon_m", "model_name", "score_type", "value", "resolved_value",
            "observed_month", "source_snapshot_ym", "source_desc", "run_id",
            "is_latest_run", "scoring_class", "resolution_reading",
        ],
        rows,
    )


INDICATIVE_COLUMNS = [
    "question_id", "iso3", "hazard_code", "metric", "horizon_m", "observed_month", "value",
    "scoring_class_reason", "resolution_reading", "model_name", "score_type", "score_value",
]


def _emit_indicative_questions(con, out_dir: Path, qids: list[str]) -> int:
    """indicative_questions.csv: every indicative ACE/PA resolution of the
    bundled questions, one row per score of the latest run (or one row with
    empty score columns when none), with the reason it is not marked."""
    rows: list[dict[str, Any]] = []
    if qids and table_exists(con, "resolutions") and has_scoring_class(con):
        has_scores = table_exists(con, "scores")
        score_join = (
            "LEFT JOIN scores s ON s.question_id = r.question_id AND s.horizon_m = r.horizon_m"
            + latest_run_clause(con, "s") + " "
            if has_scores else ""
        )
        score_cols = ("s.model_name, s.score_type, s.value AS score_value"
                      if has_scores else
                      "NULL AS model_name, NULL AS score_type, NULL AS score_value")
        rows = rows_as_dicts(
            con,
            "SELECT r.question_id, q.iso3, q.hazard_code, UPPER(q.metric) AS metric, r.horizon_m, "
            "r.observed_month, r.value, " + _resolution_extra_cols(con, "r") + ", "
            f"{score_cols} FROM resolutions r "
            "JOIN questions q ON q.question_id = r.question_id "
            f"{score_join}"
            "WHERE r.scoring_class = 'indicative' "
            "AND r.question_id IN (SELECT UNNEST(?::VARCHAR[])) "
            "ORDER BY r.question_id, r.horizon_m, s.model_name, s.score_type"
            if has_scores else
            "SELECT r.question_id, q.iso3, q.hazard_code, UPPER(q.metric) AS metric, r.horizon_m, "
            "r.observed_month, r.value, " + _resolution_extra_cols(con, "r") + ", "
            f"{score_cols} FROM resolutions r "
            "JOIN questions q ON q.question_id = r.question_id "
            "WHERE r.scoring_class = 'indicative' "
            "AND r.question_id IN (SELECT UNNEST(?::VARCHAR[])) "
            "ORDER BY r.question_id, r.horizon_m",
            [qids],
        )
        for r in rows:
            r["resolution_reading"] = _resolution_reading(
                r.get("hazard_code"), r.get("metric"), r.get("observed_month"), r.get("resolved_on"))
    write_csv(out_dir / "indicative_questions.csv", INDICATIVE_COLUMNS, rows)
    return len(rows)


def _emit_forecast_vs_outcome(con, out_dir: Path, qids: list[str], ctx: Any = None) -> list[dict[str, Any]]:
    if not table_exists(con, "forecasts_ensemble") or not table_exists(con, "resolutions"):
        return []
    track_sql = "q.track" if column_exists(con, "questions", "track") else "NULL"
    rows = rows_as_dicts(
        con,
        f"SELECT fe.question_id, q.iso3, q.hazard_code, UPPER(q.metric) AS metric, {track_sql} AS track, "
        "fe.model_name, r.horizon_m, r.value AS resolved_value, "
        + ("r.observed_month" if column_exists(con, "resolutions", "observed_month")
           else "CAST(NULL AS VARCHAR) AS observed_month") + ", "
        + _resolution_extra_cols(con, "r") + ", "
        "fe.bucket_index, fe.probability, fe.ev_value "
        "FROM forecasts_ensemble fe "
        "JOIN questions q ON q.question_id = fe.question_id "
        "JOIN resolutions r ON r.question_id = fe.question_id "
        "AND r.horizon_m = fe.month_index "
        "WHERE fe.question_id IN (SELECT UNNEST(?::VARCHAR[])) "
        # One run per question: without it a rerun question's bucket rows
        # from different runs overwrote each other in the collapse below.
        + latest_run_clause(con, "fe", "forecasts_ensemble")
        + " ORDER BY fe.question_id, fe.model_name, r.horizon_m, fe.bucket_index",
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
                "track": r.get("track"),
                "model_name": r["model_name"],
                "horizon_m": r["horizon_m"],
                "resolved_value": r["resolved_value"],
                "ev_value": r.get("ev_value"),
                "scoring_class": r.get("scoring_class"),
                "resolution_reading": _resolution_reading(
                    r["hazard_code"], r["metric"], r.get("observed_month"), r.get("resolved_on")),
                "_probs": {},
            },
        )
        g["_probs"][int(r["bucket_index"] or 0)] = float(r["probability"] or 0.0)

    # The reference forecasters beside Pythia's aggregates, with the exact
    # vectors score_baselines scored (baseline_scored_forecasts), so "did we
    # beat climatology / persistence here" is readable row by row.
    refs = _prov.reference_vectors(con, qids)
    if refs:
        res_rows = rows_as_dicts(
            con,
            "SELECT r.question_id, r.horizon_m, r.value, r.observed_month, q.iso3, q.hazard_code, "
            + _resolution_extra_cols(con, "r") + ", "
            f"UPPER(q.metric) AS metric, {track_sql} AS track FROM resolutions r "
            "JOIN questions q ON q.question_id = r.question_id "
            "WHERE r.question_id IN (SELECT UNNEST(?::VARCHAR[]))",
            [qids],
        )
        resolved = {(str(r["question_id"]), int(r["horizon_m"])): r for r in res_rows}
        for (qid, model, h), vec in refs.items():
            res = resolved.get((qid, h))
            if res is None:
                continue
            grouped[(qid, model, h)] = {
                "question_id": qid,
                "iso3": res["iso3"],
                "hazard_code": res["hazard_code"],
                "metric": res["metric"],
                "track": res.get("track"),
                "model_name": model,
                "horizon_m": h,
                "resolved_value": res["value"],
                "ev_value": None,
                "scoring_class": res.get("scoring_class"),
                "resolution_reading": _resolution_reading(
                    res["hazard_code"], res["metric"], res.get("observed_month"), res.get("resolved_on")),
                "_probs": {i + 1: p for i, p in enumerate(vec)},
            }

    out_rows: list[dict[str, Any]] = []
    for g in grouped.values():
        probs = g.pop("_probs")
        g["probs"] = json.dumps(
            [round(probs.get(i, 0.0), 6) for i in range(1, max(probs) + 1)]
        ) if probs else None
        realized = _realized_bucket(con, g["metric"], g["resolved_value"])
        modal = max(probs, key=probs.get) if probs else None
        g["realized_bucket"] = realized
        g["p_realized_bucket"] = probs.get(realized) if realized is not None else None
        g["modal_bucket"] = modal
        g["p_modal_bucket"] = probs.get(modal) if modal is not None else None
        edge, boundary = _bucket_edge(con, g["metric"], g["resolved_value"])
        g["bucket_edge"] = edge
        g["nearest_boundary"] = boundary
        g["input_partial_month"] = (
            (ctx.qmeta.get(str(g["question_id"])) or {}).get("input_partial_month")
            if ctx is not None else None
        )
        out_rows.append(g)
    out_rows.sort(key=lambda r: (r["question_id"], r["model_name"], r["horizon_m"]))
    write_csv(
        out_dir / "forecast_vs_outcome.csv",
        [
            "question_id", "iso3", "hazard_code", "metric", "track", "model_name",
            "horizon_m", "resolved_value", "realized_bucket", "p_realized_bucket",
            "modal_bucket", "p_modal_bucket", "ev_value", "bucket_edge",
            "nearest_boundary", "input_partial_month", "probs",
            "scoring_class", "resolution_reading",
        ],
        out_rows,
    )
    return out_rows


def _attach_skill(rows: list[dict[str, Any]], samples: list[dict[str, Any]]) -> None:
    """Attach PAIRED skill-vs-climatology columns to rollup rows in place.

    skill = 1 − (model mean / climatology mean), both taken over the SAME
    (question_id, horizon_m) pairs — the ones where the model and
    ``__ext_climatology`` both have a score of this score_type — within one
    (hazard, metric, score_family, track, score_type) group. Never across
    score types, families or tracks. Dividing a model's mean over its own
    questions by climatology's mean over every question in the hazard and
    metric (both tracks) compared two different sets of questions and
    reported Track 2 DR/EVENT_OCCURRENCE at +0.80 where the paired figure was
    about +0.74. Where no climatology score pairs with the model, skill stays
    empty.

    ``n_questions`` is the PAIRED question count wherever the group has a
    climatology reference at all, else the model's own count;
    ``n_questions_scored`` is always the model's own count.
    """
    clim: dict[tuple, float] = {}
    clim_groups: set[tuple] = set()
    for sm in samples:
        if sm["model_name"] == "__ext_climatology" and sm["value"] is not None:
            clim[(sm["score_type"], sm["question_id"], sm["horizon_m"])] = float(sm["value"])
            clim_groups.add(_group_key(sm) + (sm["score_type"],))
    paired: dict[tuple, list[tuple[float, float, str]]] = {}
    for sm in samples:
        if sm["value"] is None:
            continue
        c = clim.get((sm["score_type"], sm["question_id"], sm["horizon_m"]))
        if c is None:
            continue
        key = _row_key(sm)
        paired.setdefault(key, []).append((float(sm["value"]), c, sm["question_id"]))
    for r in rows:
        key = _row_key(r)
        pairs = paired.get(key) or []
        has_clim = (_group_key(r) + (r.get("score_type"),)) in clim_groups
        r["n_paired"] = len(pairs)
        if pairs:
            pm = sum(p[0] for p in pairs) / len(pairs)
            cm = sum(p[1] for p in pairs) / len(pairs)
            r["paired_model_mean"] = round(pm, 6)
            r["climatology_mean"] = round(cm, 6)
            r["skill_vs_climatology"] = round(1.0 - pm / cm, 4) if cm > 0 else None
        else:
            r["paired_model_mean"] = None
            r["climatology_mean"] = None
            r["skill_vs_climatology"] = None
        r["n_questions"] = len({p[2] for p in pairs}) if has_clim else r["n_questions_scored"]


#: rollups.csv is split on these per-question columns as well as on hazard,
#: metric, family and track, so one row never pools two prompt versions, two
#: advice arms, two lineups or a partial-month input with a complete one.
ROLLUP_SPLIT_KEYS = _err.ROLLUP_SPLIT_KEYS


def _group_key(sm: Mapping[str, Any]) -> tuple:
    """The question-level part of a rollup key (shared with climatology)."""
    return (sm.get("hazard_code"), sm.get("metric"), sm.get("score_family"), sm.get("track"),
            sm.get("horizon_m")) + tuple(sm.get(k) for k in ROLLUP_SPLIT_KEYS)


def _row_key(sm: Mapping[str, Any]) -> tuple:
    """A rollup row's key: the group, the forecaster and its correction, the score."""
    return _group_key(sm) + (sm.get("correction"), sm.get("model_name"), sm.get("score_type"))


def _rollup_samples(con, qids: list[str], ctx: Any = None) -> list[dict[str, Any]]:
    """One row per score (latest run per question), with the question's track
    and, when an error-attribution context is given, its split columns."""
    track_sql = "q.track" if column_exists(con, "questions", "track") else "NULL"
    rows = rows_as_dicts(
        con,
        "SELECT q.hazard_code, UPPER(q.metric) AS metric, "
        "CASE WHEN UPPER(q.metric) = 'EVENT_OCCURRENCE' THEN 'binary' ELSE 'spd' END "
        f"AS score_family, {track_sql} AS track, s.model_name, s.score_type, "
        "s.question_id, s.horizon_m, s.value, "
        + (f"NOT ({scored_only_sql('s')}) AS indicative "
           if has_scoring_class(con) else "FALSE AS indicative ")
        + "FROM scores s JOIN questions q ON q.question_id = s.question_id "
        "WHERE s.question_id IN (SELECT UNNEST(?::VARCHAR[])) "
        + latest_run_clause(con, "s"),
        [qids],
    )
    for r in rows:
        if ctx is not None:
            r.update(ctx.sample_attrs(str(r["question_id"]), str(r["model_name"])))
        else:
            r.update({k: None for k in ROLLUP_SPLIT_KEYS})
            r["correction"] = None
    # prior_anchor_v1 and _v2 are one recalibration group: report each
    # wording on its own AND pooled (a relabelled copy per sample).
    return _err.with_pooled_block_versions(rows)


def _cost_per_question(costs: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Mean forecast-phase cost per question, per member model.

    An aggregate row (ensemble_mean_v2, track2_flash, ...) carries the mean
    total cost of the questions it covered; its own name logs no calls.
    """
    per_model: dict[str, list[float]] = {}
    totals: list[float] = []
    for by_model in costs.values():
        for model, cost in by_model.items():
            if model == "__total__":
                totals.append(float(cost))
            else:
                per_model.setdefault(model, []).append(float(cost))
    out = {m: round(sum(v) / len(v), 6) for m, v in per_model.items() if v}
    if totals:
        out["__total__"] = round(sum(totals) / len(totals), 6)
    return out


def _emit_rollups(
    con, out_dir: Path, qids: list[str],
    costs: Mapping[str, Mapping[str, float]] | None = None,
    ctx: Any = None,
) -> list[dict[str, Any]]:
    import statistics

    all_samples = _rollup_samples(con, qids, ctx)
    # An indicative ACE/PA month (pythia/tools/scoring_class.py) is a
    # selected sample: never in a mean or a skill figure, and each row says
    # how many questions it left out rather than dropping them silently.
    samples = [sm for sm in all_samples if not sm.get("indicative")]
    excluded: dict[tuple, set] = {}
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for sm in all_samples:
        groups.setdefault(_row_key(sm), [])
        if sm.get("indicative"):
            excluded.setdefault(_row_key(sm), set()).add(sm["question_id"])
        else:
            groups[_row_key(sm)].append(sm)
    first_of = {}
    for sm in all_samples:
        first_of.setdefault(_row_key(sm), sm)
    rows: list[dict[str, Any]] = []
    for key, sms in groups.items():
        first = sms[0] if sms else first_of[key]
        vals = [float(x["value"]) for x in sms if x["value"] is not None]
        rows.append({
            "hazard_code": first["hazard_code"], "metric": first["metric"],
            "score_family": first["score_family"], "track": first["track"],
            "horizon_m": first.get("horizon_m"),
            **{k: first.get(k) for k in ROLLUP_SPLIT_KEYS},
            "correction": first.get("correction"),
            "block_version_pooled": bool(first.get("block_version_pooled")),
            "model_name": first["model_name"], "score_type": first["score_type"],
            "n_samples": len(vals),
            "n_questions_scored": len({x["question_id"] for x in sms}),
            "mean_value": (sum(vals) / len(vals)) if vals else None,
            "median_value": statistics.median(vals) if vals else None,
            "n_indicative_excluded": len(excluded.get(key, ())),
        })
    rows.sort(key=lambda r: (str(r["score_family"]), str(r["hazard_code"]), str(r["metric"]),
                             str(r["track"]), int(r.get("horizon_m") or 0),
                             tuple(str(r.get(k)) for k in ROLLUP_SPLIT_KEYS),
                             str(r["score_type"]),
                             r["mean_value"] if r["mean_value"] is not None else 0.0))
    _attach_skill(rows, samples)
    per_q = _cost_per_question(costs or {})
    for r in rows:
        name = str(r.get("model_name") or "")
        if name.startswith("__ext_"):
            r["cost_per_question_usd"] = None
        elif name in per_q:
            r["cost_per_question_usd"] = per_q[name]
        else:
            r["cost_per_question_usd"] = per_q.get("__total__")
    write_csv(
        out_dir / "rollups.csv",
        [
            "hazard_code", "metric", "score_family", "track", "horizon_m", *ROLLUP_SPLIT_KEYS,
            "correction", "block_version_pooled", "model_name", "score_type",
            "n_samples", "n_questions", "n_questions_scored", "mean_value", "median_value",
            "n_paired", "paired_model_mean", "climatology_mean", "skill_vs_climatology",
            "cost_per_question_usd", "n_indicative_excluded",
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
        "n_runs": len(record.get("score_runs") or []),
        "is_rerun": len(record.get("score_runs") or []) > 1,
        "latest_run_id": (record.get("score_runs") or [None])[-1],
        "lineup_id": (record.get("lineup") or {}).get("lineup_id"),
        "resolution_series": record.get("resolution_series"),
        "spd_prompt_missing": record.get("spd_prompt") is None,
        "enso_observation_date": ((record.get("inject_status") or {}).get("enso") or {}).get("observation_date"),
        "gdacs_history_months": ((record.get("inject_status") or {}).get("gdacs_history") or {}).get("total_months"),
        "crisiswatch_edition_age_months": ((record.get("inject_status") or {}).get("crisiswatch") or {}).get("edition_age_months"),
        "baserate_source": ((record.get("inject_status") or {}).get("base_rate") or {}).get("source"),
        "cost_usd": (record.get("cost_usd") or {}).get("__total__"),
        "input_partial_month": record.get("input_partial_month"),
        **{k: (record.get("forecast_versions") or {}).get(k)
           for k in ("base_rate_block_version", "rc_guidance", "advice_arm", "recalibration_mode")},
        "record_path": f"questions/{q.get('question_id')}.json",
    }


def _attach_provenance(
    con, record: dict[str, Any], q: Mapping[str, Any], cost: Mapping[str, float]
) -> None:
    """Add what the forecast was made WITH to a scored question's record.

    Every helper degrades to a stated reason, so a failure here costs a
    field, never the record.
    """
    run_id = record.get("forecast_run_id")
    qid = str(q.get("question_id"))
    try:
        record["inject_status"] = _prov.inject_status(con, q, run_id)
    except Exception as exc:  # noqa: BLE001
        record["inject_status"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        record["lineup"] = _prov.lineup(con, run_id, qid)
    except Exception as exc:  # noqa: BLE001
        record["lineup"] = {"lineup_id": None, "reason": f"{type(exc).__name__}: {exc}"}
    record["resolution_series"] = _prov.resolution_series(q.get("hazard_code"), q.get("metric"))
    if record.get("spd_prompt") is None:
        record["spd_prompt_missing_reason"] = _prov.spd_prompt_missing_reason(con, qid, run_id)
    record["cost_usd"] = dict(cost)


def _attach_error_fields(
    con, record: dict[str, Any], q: Mapping[str, Any], ctx: Any, ctx_error: str | None
) -> dict[str, Any]:
    """Add input_partial_month, the forecast's prompt versions and
    base_rate_shown to a record; return the metadata inject_health reads.

    Degrades field by field: a failure states its reason in the record.
    """
    qid = str(q.get("question_id"))
    meta = (ctx.qmeta.get(qid) if ctx is not None else None) or dict(q)
    record["input_partial_month"] = meta.get("input_partial_month")
    record["input_partial_month_basis"] = meta.get(
        "input_partial_month_basis", ctx_error or "error-attribution context unavailable"
    )
    record["forecast_versions"] = {
        k: meta.get(k) for k in (
            "lineup_id", "base_rate_block_version", "rc_guidance", "advice_arm",
            "recalibration_mode", "forecast_date",
        )
    }
    if ctx is None:
        record["base_rate_shown"] = {"available": False, "reason": ctx_error or "context unavailable"}
        return meta
    try:
        record["base_rate_shown"] = _err.base_rate_shown(con, meta, record)
    except Exception as exc:  # noqa: BLE001
        record["base_rate_shown"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    return meta


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


def _lineups_seen(staging: Path, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """{lineup_id: {members, n_questions}} across the bundle's records."""
    out: dict[str, Any] = {}
    for summ in summaries:
        lid = summ.get("lineup_id")
        if not lid:
            continue
        entry = out.get(lid)
        if entry is None:
            rec = _load_staged_record(staging, str(summ.get("question_id"))) or {}
            entry = out[lid] = {
                "members": (rec.get("lineup") or {}).get("members") or [],
                "effort_source": (rec.get("lineup") or {}).get("effort_source"),
                "n_questions": 0,
            }
        entry["n_questions"] += 1
    return out


#: The aggregate that stands for each track in the sharpness table: the first
#: of these a question carries.
_PRIMARY_AGGREGATES = ("ensemble_bayesmc_v2", "ensemble_mean_v2", "track2_flash")


_SHARPNESS_REFERENCES = (
    "__ext_climatology",
    "__ext_persistence",
    "__ext_level_volatility",
    "__ext_level_transition",
    "__ext_conflictology12",
    "__ext_ref_pool",
)


def _sharpness_lines(fvo_rows: list[dict[str, Any]]) -> list[str]:
    """Digest table: how concentrated the primary aggregate was, per (hazard, metric, track).

    Mean max bucket probability says how sharp the forecasts were; mean
    probability on the realised bucket says whether the sharpness landed. The
    two move together only when the forecast is both sharp and right. The
    reference forecasters are reported beside it over the SAME (question,
    horizon) pairs, so a reader can see whether the aggregate was blunter
    than a reference that scored better.
    """
    by_q: dict[tuple, dict[str, dict[str, Any]]] = {}
    for r in fvo_rows:
        if r.get("metric") == "EVENT_OCCURRENCE" or _is_indicative(r):
            continue
        if r.get("model_name") not in _PRIMARY_AGGREGATES + _SHARPNESS_REFERENCES:
            continue
        by_q.setdefault((r["question_id"], r["horizon_m"]), {})[r["model_name"]] = r
    groups: dict[tuple, dict[str, list[dict[str, Any]]]] = {}
    for per_model in by_q.values():
        row = next((per_model[m] for m in _PRIMARY_AGGREGATES if m in per_model), None)
        if row is None or row.get("p_modal_bucket") is None:
            continue
        g = groups.setdefault((row.get("hazard_code"), row.get("metric"), row.get("track")), {})
        g.setdefault("primary", []).append(row)
        for ref in _SHARPNESS_REFERENCES:
            if ref in per_model and per_model[ref].get("p_modal_bucket") is not None:
                g.setdefault(ref, []).append(per_model[ref])
    if not groups:
        return []
    lines = [
        "",
        "## Sharpness of the primary aggregate (SPD)",
        "",
        "_Per (hazard, metric, track), over resolved (question, horizon) pairs of "
        "the latest run: the mean of the largest bucket probability, and the mean "
        "probability on the bucket that happened. Primary aggregate = "
        "ensemble_bayesmc_v2, else ensemble_mean_v2, else track2_flash; the "
        "reference rows cover the same pairs._",
        "",
        "| hazard | metric | track | forecaster | n | mean max bucket prob | mean prob on realised bucket |",
        "|---|---|---|---|---|---|---|",
    ]
    for (hz, metric, track), per in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        for who in ("primary",) + _SHARPNESS_REFERENCES:
            rs = per.get(who)
            if not rs:
                continue
            mx = sum(float(r["p_modal_bucket"]) for r in rs) / len(rs)
            real = [float(r["p_realized_bucket"] or 0.0) for r in rs if r.get("realized_bucket") is not None]
            rl = f"{sum(real) / len(real):.3f}" if real else "—"
            lines.append(
                f"| {hz} | {metric} | {'—' if track is None else f'T{track}'} | {who} "
                f"| {len(rs)} | {mx:.3f} | {rl} |"
            )
    return lines


def _emit_resolution_sources(con, out_dir: Path, qids: list[str]) -> list[dict[str, Any]]:
    """Per (hazard, metric) of the bundled questions: resolutions drawn from a
    source against zero-defaults. A group whose zero-defaults exceed half its
    resolutions FAILS here and is named in the digest: on the 5 October 2026
    release 64 of 96 ACE/PA resolutions were zero-defaults, 60 of them months
    IDMC had not reported yet, and every score and learned table below them
    rested on that inference."""
    try:
        from pythia.tools.compute_resolutions import (
            ZERO_DEFAULT_SHARE_LIMIT as limit,
            mostly_zero_default_groups,
        )
    except Exception:  # noqa: BLE001
        limit = 0.5

        def mostly_zero_default_groups(c):
            return [
                g for g, (a, z) in sorted(c.items())
                if not g.endswith("/EVENT_OCCURRENCE") and a + z and z / (a + z) > limit
            ]
    rows: list[dict[str, Any]] = []
    from pythia.tools._db_utils import column_exists

    if not qids or not (table_exists(con, "resolutions") and table_exists(con, "questions")) \
            or not column_exists(con, "resolutions", "source_desc"):
        write_csv(out_dir / "resolution_sources.csv",
                  ["hazard_code", "metric", "sourced", "zero_default", "zero_share", "verdict"], rows)
        return rows
    placeholders = ",".join("?" for _ in qids)
    found = rows_as_dicts(
        con,
        f"""
        SELECT upper(q.hazard_code) AS hazard_code, upper(q.metric) AS metric,
               COUNT(*) FILTER (WHERE COALESCE(r.source_desc, '') <> 'zero_default') AS sourced,
               COUNT(*) FILTER (WHERE r.source_desc = 'zero_default') AS zero_default
        FROM resolutions r JOIN questions q ON q.question_id = r.question_id
        WHERE r.question_id IN ({placeholders})
        GROUP BY 1, 2 ORDER BY 1, 2
        """,
        list(qids),
    )
    counts = {f"{r['hazard_code']}/{r['metric']}": (int(r["sourced"]), int(r["zero_default"])) for r in found}
    flagged = set(mostly_zero_default_groups(counts))
    for r in found:
        total = int(r["sourced"]) + int(r["zero_default"])
        key = f"{r['hazard_code']}/{r['metric']}"
        rows.append({
            **r,
            "zero_share": round(int(r["zero_default"]) / total, 3) if total else None,
            "verdict": "FAIL" if key in flagged else "PASS",
        })
    write_csv(out_dir / "resolution_sources.csv",
              ["hazard_code", "metric", "sourced", "zero_default", "zero_share", "verdict"], rows)
    return rows


def _resolution_source_lines(rows: list[dict[str, Any]]) -> list[str]:
    failed = [r for r in rows if r.get("verdict") == "FAIL"]
    if not failed:
        return []
    lines = [
        "## FAIL: groups resolved mostly by zero-defaults",
        "",
        "Their outcomes are mostly the resolver's inference, not a source's figure; read "
        "every score and calibration row for them with that in mind.",
        "",
        "| hazard | metric | sourced | zero-default | share |",
        "|---|---|---:|---:|---:|",
    ]
    for r in failed:
        lines.append(
            f"| {r['hazard_code']} | {r['metric']} | {r['sourced']} | {r['zero_default']} | "
            f"{r['zero_share']:.0%} |"
        )
    return lines + [""]


def _write_digest(
    out_dir: Path,
    summaries: list[dict[str, Any]],
    rollups: list[dict[str, Any]],
    weight_movement: list[dict[str, Any]],
    case_selection: dict[str, list[str]],
    *,
    months_back: int,
    fvo_rows: list[dict[str, Any]] | None = None,
    error_parts: Mapping[str, Any] | None = None,
    resolution_sources: list[dict[str, Any]] | None = None,
) -> None:
    error_parts = error_parts or {}
    lines: list[str] = [
        "# Scored-Forecast Analysis — Digest",
        "",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
        f"question-epoch window: last {months_back} months_",
        "",
        *_resolution_source_lines(resolution_sources or []),
        # The first table is generated from headline.json and nothing else,
        # so a report quoting the headline and this digest cannot disagree.
        *_err.headline_digest_lines(error_parts.get("headline") or {}),
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
        "## Model comparison (Brier and RPS by score family — never blend the two)",
        "",
        "_skill = 1 − (model mean / climatology mean) over PAIRED (question, "
        "horizon) scores only — the ones both the model and `__ext_climatology` "
        "scored — pooled across (hazard, metric) groups within a track; positive "
        "= beat the base rate. `__ext_climatology` / `__ext_uniform` / "
        "`__ext_persistence` / `__ext_level_volatility` / `__ext_level_transition` "
        "are the reference "
        "forecasters, not Pythia models. "
        "One run per question (the latest); RPS is SPD-only. Track 1 and Track 2 "
        "are different questions and are never pooled._",
        "",
        "| family | track | model | n | mean Brier | median Brier | Brier skill vs climatology "
        "| mean RPS | RPS skill vs climatology |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    # The rollup rows are per (hazard, metric, track); the digest table
    # aggregates per (family, track, model) and per score type. Skill pools
    # the PAIRED sums, never a ratio of unpaired means. RPS (stored as
    # score_type 'crps') sits beside Brier for SPD metrics: Brier ignores
    # bucket ORDER, so a forecast one bucket off and one five buckets off
    # score the same, and RPS is the score that tells them apart.
    agg: dict[tuple, dict[str, float]] = {}
    for r in rollups:
        if r.get("score_type") not in ("brier", "crps"):
            continue
        if r.get("block_version_pooled"):
            continue  # a pooled prior_anchor_v1+v2 copy; its samples are counted already
        key = (r["score_family"], r.get("track"), r["model_name"], r["score_type"])
        a = agg.setdefault(key, {"n": 0, "vsum": 0.0, "msum": 0.0,
                                 "pn": 0, "psum": 0.0, "csum": 0.0})
        n = int(r["n_samples"] or 0)
        a["n"] += n
        a["vsum"] += float(r["mean_value"] or 0) * n
        a["msum"] += float(r["median_value"] or 0) * n
        pn = int(r.get("n_paired") or 0)
        if pn and r.get("paired_model_mean") is not None and r.get("climatology_mean") is not None:
            a["pn"] += pn
            a["psum"] += float(r["paired_model_mean"]) * pn
            a["csum"] += float(r["climatology_mean"]) * pn

    def _skill(a: dict[str, float] | None) -> str:
        if not a or not a["pn"] or a["csum"] <= 0:
            return "—"
        return f"{1.0 - a['psum'] / a['csum']:+.3f}"

    def _tr(t: Any) -> str:
        return "—" if t is None else f"T{t}"

    for family, track, model in sorted({(k[0], k[1], k[2]) for k in agg},
                                       key=lambda x: (str(x[0]), str(x[1]), str(x[2]))):
        b = agg.get((family, track, model, "brier"))
        c = agg.get((family, track, model, "crps"))
        if not b or not b["n"]:
            continue
        rps = f"{c['vsum'] / c['n']:.4f}" if c and c["n"] else "—"
        lines.append(
            f"| {family} | {_tr(track)} | {model} | {b['n']} "
            f"| {b['vsum'] / b['n']:.4f} | {b['msum'] / b['n']:.4f} | {_skill(b)} "
            f"| {rps} | {_skill(c) if c else '—'} |"
        )

    n_ind = len({(r["question_id"], r["horizon_m"]) for r in (fvo_rows or []) if _is_indicative(r)})
    if n_ind:
        lines += [
            "",
            f"_{n_ind} indicative ACE/PA (question, horizon) resolution(s) are left out of the "
            "table above and of every skill figure: IDMC does not report those countries "
            "regularly, so their resolved months are a selected sample. They are listed in "
            "`indicative_questions.csv`._",
        ]
    lines += _sharpness_lines(fvo_rows or [])
    lines += _error_digest_lines(error_parts)

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
        "- Error attribution: `headline.json`, `trace_stages.csv` (+ summary), "
        "`update_value.csv` (+ summary), `rc_outcomes.csv` (+ summary), "
        "`unasked_outcomes.csv`, `experiments.csv`, `skill_history.csv`, "
        "`tail_outcomes.csv`, `binary_reliability.csv`, `inject_health.csv`.",
        "- `briefing/` — condensed chat-uploadable digest + case studies.",
    ]
    path = out_dir / "digest.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    size_guard(path, DIGEST_BUDGET_KB)


def _error_digest_lines(parts: Mapping[str, Any]) -> list[str]:
    """Digest sections for the error-attribution files: where error came from,
    which adjustments helped, the history, inject health, unasked outcomes."""
    lines: list[str] = []
    ts = [r for r in parts.get("trace_summary") or [] if r.get("score_type") == "rps"]
    if ts:
        lines += [
            "", "## Where the error came from (`trace_stages_summary.csv`, RPS)", "",
            "_Mean RPS of the base rate shown, the member's declared prior and its final "
            "SPD; prior − shown is the cost of the starting point, final − prior the cost "
            "of the adjustments (negative = better). 90% intervals resample questions._", "",
            "| hazard | metric | track | block | RC guidance | partial input | n q | shown | prior | final "
            "| prior − shown [90%] | final − prior [90%] |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]

        def ci(m: Any, lo: Any, hi: Any) -> str:
            if m is None:
                return "—"
            return f"{m:+.3f}" + (f" [{lo:+.3f}, {hi:+.3f}]" if lo is not None else " [—]")

        for r in ts:
            lines.append(
                f"| {r['hazard_code']} | {r['metric']} | T{r['track']} | {r['base_rate_block_version']} "
                f"| {r['rc_guidance']} | {r['input_partial_month']} | {r['n_questions']} "
                f"| {r['mean_shown'] if r['mean_shown'] is not None else '—'} | {r['mean_prior']} "
                f"| {r['mean_final']} | {ci(r['prior_minus_shown'], r['prior_minus_shown_ci90_low'], r['prior_minus_shown_ci90_high'])} "
                f"| {ci(r['final_minus_prior'], r['final_minus_prior_ci90_low'], r['final_minus_prior_ci90_high'])} |"
            )
    us = sorted(parts.get("update_summary") or [], key=lambda r: -int(r.get("n_updates") or 0))[:15]
    if us:
        lines += [
            "", "## Which adjustments helped (`update_value_summary.csv`, CLAIMED attribution)", "",
            "| signal class | hazard | metric | updates | toward outcome | mean ΔRPS [90%] | verdict |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in us:
            ci90 = (f" [{r['delta_rps_ci90_low']:+.3f}, {r['delta_rps_ci90_high']:+.3f}]"
                    if r.get("delta_rps_ci90_low") is not None else "")
            lines.append(
                f"| {r['signal_class']} | {r['hazard_code']} | {r['metric']} | {r['n_updates']} "
                f"| {r['share_toward_outcome']} | {r['mean_delta_rps']:+.3f}{ci90} | {r['verdict']} |"
            )
    lines += list(parts.get("history") or [])
    lines += _err.inject_digest_lines(parts.get("inject") or [])
    unasked = parts.get("unasked") or []
    if unasked:
        from collections import Counter as _Counter

        by = _Counter((r["month"], r["hazard_code"], r["trigger"]) for r in unasked)
        not_assessed = sum(1 for r in unasked if r.get("triage_tier") == "not assessed")
        lines += [
            "", "## Large outcomes with no question (`unasked_outcomes.csv`)", "",
            f"_{len(unasked)} cells across the horizon scanner's country list had a large "
            f"outcome and no question; {not_assessed} of them were not assessed by HS at all._", "",
            "| month | hazard | trigger | cells |", "|---|---|---|---|",
        ]
        for (month, hz, trig), n in sorted(by.items()):
            lines.append(f"| {month} | {hz} | {trig} | {n} |")
    return lines


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
        costs = _prov.question_costs(con, qids)
        # One read of what every error-attribution section shares. A failure
        # here stubs those files with the reason and costs nothing else.
        err_ctx = None
        err_ctx_error = None
        try:
            err_ctx = _err.build_context(con, qids, include_test)
        except Exception as exc:  # noqa: BLE001
            err_ctx_error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("error-attribution context unavailable: %s", err_ctx_error)
        inject_rows: list[dict[str, Any]] = []
        inject_error: str | None = None
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
            _attach_provenance(con, record, q, costs.get(qid) or {})
            meta = _attach_error_fields(con, record, q, err_ctx, err_ctx_error)
            try:
                rescore_trace_prior(record)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("trace prior rescoring failed for %s: %s", qid, exc)
            try:
                inject_rows.extend(
                    _err.inject_health_rows(qid, meta, record.get("inject_status") or {})
                )
            except Exception as exc:  # noqa: BLE001
                inject_error = f"{type(exc).__name__}: {exc}"
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
                "mean_crps", "has_sibyl", "n_runs", "is_rerun", "latest_run_id",
                "lineup_id", "resolution_series", "spd_prompt_missing",
                "enso_observation_date", "gdacs_history_months",
                "crisiswatch_edition_age_months", "baserate_source",
                "cost_usd", "input_partial_month", "base_rate_block_version",
                "rc_guidance", "advice_arm", "recalibration_mode", "record_path",
            ],
            summaries,
        )
        if inject_error:
            _err.write_stub(staging / "inject_health.csv", inject_error)
        else:
            write_csv(staging / "inject_health.csv", _err.INJECT_HEALTH_COLUMNS, inject_rows)

        _emit_scores_flat(con, staging, qids)
        try:
            n_indicative_rows = _emit_indicative_questions(con, staging, qids)
            indicative_error = None
        except Exception as exc:  # noqa: BLE001 - the table never costs the bundle
            n_indicative_rows = 0
            indicative_error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("indicative_questions.csv failed: %s", indicative_error)
            write_csv(staging / "indicative_questions.csv", ["stub_reason"],
                      [{"stub_reason": indicative_error}])
        resolution_sources = _emit_resolution_sources(con, staging, qids)
        fvo_rows = _emit_forecast_vs_outcome(con, staging, qids, err_ctx)
        rollups = _emit_rollups(con, staging, qids, costs, err_ctx)
        try:
            from scripts.ai_bundle import experiments as _exp

            _exp.emit_advice_experiment(con, staging, qids)
            _exp.emit_recalibration_effect(con, staging, qids)
        except Exception as exc:  # noqa: BLE001 - an experiment table never costs the bundle
            LOGGER.warning("experiment rollups skipped: %s", exc)
        err_sections, err_digest = _err.emit_all(err_ctx, staging, ctx_error=err_ctx_error)
        err_sections.files["inject_health.csv"] = (
            {"status": "stub", "rows": 0, "reason": inject_error} if inject_error
            else {"status": "ok", "rows": len(inject_rows)}
        )
        err_digest["inject"] = inject_rows
        weight_movement = _emit_calibration(con, staging)
        calibration_state = _prov.calibration_status(con)
        write_csv(
            staging / "calibration_status.csv",
            ["hazard_code", "metric", "n_questions_with_member_scores", "floor",
             "has_weights", "status"],
            calibration_state,
        )

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
            months_back=months_back, fvo_rows=fvo_rows, error_parts=err_digest,
            resolution_sources=resolution_sources,
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
                "resolved_questions": _prov.resolution_counts(con, qids),
                "calibration_status": calibration_state,
                "resolution_sources": resolution_sources,
                "groups_mostly_zero_defaults": [
                    f"{r['hazard_code']}/{r['metric']}" for r in resolution_sources
                    if r.get("verdict") == "FAIL"
                ],
                "lineups": _lineups_seen(staging, summaries),
                "indicative_questions": {
                    "file": "indicative_questions.csv",
                    "rows": n_indicative_rows,
                    "error": indicative_error,
                    "note": (
                        "ACE/PA months whose resolution is indicative (IDMC does not report "
                        "the country regularly): resolved and scored, but left out of every "
                        "skill figure and calibration step"
                    ),
                },
                "error_attribution": {
                    "files": err_sections.files,
                    "context_error": err_ctx_error,
                    "context_problems": err_ctx.problems if err_ctx is not None else [],
                },
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
    try:
        zip_path = build_bundle(
            args.db,
            Path(args.out_dir),
            months_back=args.months_back,
            n_case_studies=args.n_case_studies,
            include_test=args.include_test,
            include_sibyl_trials=args.include_sibyl_trials,
            keep_staging=args.keep_staging,
        )
    except Exception as exc:  # noqa: BLE001 - the bundle never fails calibration
        LOGGER.exception("scored bundle failed")
        print(f"::warning title=Scored bundle failed::{type(exc).__name__}: {exc}")
        return 0
    if zip_path is None:
        # Nothing to bundle is a soft outcome, not a failure — the workflow
        # step is continue-on-error anyway, but exit 0 keeps logs green.
        print("[ai_bundle] no bundle produced (no scored questions)")
        return 0
    print(f"[ai_bundle] BUNDLE_PATH={zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
