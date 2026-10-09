# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl track routes: /v1/sibyl/*.

Serves the parallel deep-research track: run coverage summaries (including
the budget_capped flag), the per-question JS-divergence table (sortable
track-vs-track disagreement is the most decision-useful output), and full
question detail (K trial distributions, belief-state traces, evidence, and
the standard-track SPD for overlay).

All endpoints tolerate DBs that pre-date the sibyl tables (empty payloads,
never 500s).
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from pythia.api.core import (
    _bucket_labels,
    _con,
    _execute,
    _rows_from_cursor,
    _table_exists,
    _table_has_columns,
    _test_filter,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Single-sourced from sibyl.config (import-light: os only — safe for the
# memory-constrained API process; see test_api_lazy_pipeline_import.py).
from sibyl.config import STANDARD_MODEL_PREFERENCE as _STANDARD_MODEL_PREFERENCE


#: How Sibyl researched, per run (sibyl/measure.py); never a score.
PROCESS_MEASURES = (
    "share_resolver_done", "docs_per_trial", "share_ledger_dated_figure",
    "share_forecasts_at_floor", "mean_jsd_from_reference", "reference_weight",
    "reference_weight_source",
    # Research depth (Oct 2026, review Part 1).
    "median_docs_per_trial", "share_trials_under_doc_gate", "steps_per_trial",
    "tool_calls_per_trial", "share_docs_wikipedia", "n_submit_gate_unmet",
    # The resolving source's reading (Oct 2026, review Part 5).
    "share_nowcast_done", "n_resolver_live_ok", "n_resolver_live_failed",
)
#: The shadow arm (sibyl/shadow.py, Oct 2026): NULL on earlier runs.
SHADOW_FIELDS = (
    "shadow_status", "shadow_model", "n_shadow_trials", "n_shadow_series",
    "n_shadow_skipped", "shadow_cost_usd",
)


def _maybe_json(raw: Any) -> Any:
    if raw is None or raw == "":
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _selection_pass_col(con) -> str:
    """``selection_pass`` (floor | fill) when the column exists, else NULL."""
    if _table_has_columns(con, "sibyl_forecasts", ["selection_pass"]):
        return "selection_pass"
    return "CAST(NULL AS TEXT) AS selection_pass"


def _evidence_ok_col(con) -> str:
    """``evidence_ok`` when the column exists, else NULL (unknown).

    FALSE marks a forecast that rested on no evidence (the July 2026 run,
    whose searches all failed). Such rows are listed, so the page can show
    them with a "no evidence" badge, and left out of every summary figure
    computed here and of every comparison elsewhere.
    """
    if _table_has_columns(con, "sibyl_forecasts", ["evidence_ok"]):
        return "evidence_ok"
    return "CAST(NULL AS BOOLEAN) AS evidence_ok"


def _latest_sibyl_run_id(con, include_test: bool) -> Optional[str]:
    # Pre-Sibyl / partial-schema DBs may have sibyl_forecasts without
    # sibyl_runs (or neither); this module's contract is to never 500 there.
    if not _table_exists(con, "sibyl_runs"):
        return None
    rows = _execute(
        con,
        f"""
        SELECT sibyl_run_id FROM sibyl_runs
        WHERE 1=1{_test_filter(include_test)}
        ORDER BY created_at DESC
        LIMIT 1
        """,
    ).fetchall()
    return str(rows[0][0]) if rows else None


@router.get("/v1/sibyl/runs")
def sibyl_runs(include_test: bool = Query(False)):
    """List Sibyl runs, newest first (run navigation)."""
    con = _con()
    if not _table_exists(con, "sibyl_runs"):
        return {"rows": []}
    rows = _rows_from_cursor(
        _execute(
            con,
            f"""
            SELECT * FROM sibyl_runs
            WHERE 1=1{_test_filter(include_test)}
            ORDER BY created_at DESC
            """,
        )
    )
    for r in rows:
        r["config"] = _maybe_json(r.pop("config_json", None))
    return {"rows": rows}


@router.get("/v1/sibyl/summary")
def sibyl_summary(
    sibyl_run_id: Optional[str] = Query(None),
    include_test: bool = Query(False),
):
    """Run-level coverage: forecast/skipped counts, cost, budget_capped, time_capped."""
    con = _con()
    if not _table_exists(con, "sibyl_runs"):
        return {"run": None, "questions": []}

    run_id = sibyl_run_id or _latest_sibyl_run_id(con, include_test)
    if not run_id:
        return {"run": None, "questions": []}

    run_rows = _rows_from_cursor(
        _execute(con, "SELECT * FROM sibyl_runs WHERE sibyl_run_id = ?", [run_id])
    )
    if not run_rows:
        return {"run": None, "questions": []}
    run = run_rows[0]
    run["config"] = _maybe_json(run.pop("config_json", None))
    # Pre-Oct-2026 rows: the column is absent, and absence is "not capped".
    run.setdefault("time_capped", False)
    # Process measures (Oct 2026): NULL on earlier runs, absent on older DBs.
    for key in PROCESS_MEASURES:
        run.setdefault(key, None)

    questions: List[Dict[str, Any]] = []
    if _table_exists(con, "sibyl_forecasts"):
        questions = _rows_from_cursor(
            _execute(
                con,
                f"""
                SELECT question_id, iso3, hazard_code, metric, status,
                       skip_reason, k, cost_usd, opus_cost_usd, brave_cost_usd,
                       volatility_score, js_divergence_vs_standard,
                       js_divergence_inter_trial, {_selection_pass_col(con)},
                       {_evidence_ok_col(con)}
                FROM sibyl_forecasts
                WHERE sibyl_run_id = ?
                ORDER BY volatility_score DESC NULLS LAST, question_id
                """,
                [run_id],
            )
        )
    # A forecast stored ok but resting on no evidence is not counted as a
    # forecast in the run's figures.
    run["n_no_evidence"] = sum(
        1 for q in questions
        if q.get("status") == "ok" and q.get("evidence_ok") is False
    )
    for key in SHADOW_FIELDS:
        run.setdefault(key, None)
    return {"run": run, "questions": questions, "shadow": _shadow_comparison(con, include_test),
            "pack": _pack_comparison(con, include_test)}


def _pack_comparison(con, include_test: bool) -> Optional[Dict[str, Any]]:
    """The structured-data pack arms compared (Part 6; a finding, never a switch)."""
    try:
        from sibyl.pack import pack_comparison  # noqa: PLC0415 - lazy, API process

        return pack_comparison(con, include_test=include_test)
    except Exception:  # noqa: BLE001
        logger.debug("sibyl pack comparison failed", exc_info=True)
        return None


def _shadow_comparison(con, include_test: bool) -> Optional[Dict[str, Any]]:
    """Shadow arm minus Sibyl over every scored question (a finding, never a switch)."""
    try:
        from sibyl.shadow import shadow_comparison  # noqa: PLC0415 - lazy, API process

        return shadow_comparison(con, include_test=include_test)
    except Exception:  # noqa: BLE001
        logger.debug("sibyl shadow comparison failed", exc_info=True)
        return None


@router.get("/v1/sibyl/calibration")
def sibyl_calibration(as_of_month: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}$")):
    """Sibyl's own calibration record and the advice in force.

    One row per (hazard, metric) class plus the pooled row (``*``/``*``) for
    the newest generation on or before *as_of_month*: distinct scored
    questions, coverage and bias with 90% intervals, the advice text (empty
    when the class is gated, with ``gate`` saying why), and the pooled row's
    advice / no-advice arm comparison. A DB without the table answers
    ``has_advice_table: false``, never a 500.
    """
    con = _con()
    empty = {
        "has_advice_table": False, "as_of_month": None, "months": [],
        "min_questions": None, "rows": [], "arm_comparison": None,
        "failure_types": None,
    }
    if not _table_exists(con, "sibyl_calibration_advice"):
        return empty
    months = [
        str(r[0]) for r in _execute(
            con,
            "SELECT DISTINCT as_of_month FROM sibyl_calibration_advice ORDER BY 1 DESC",
        ).fetchall()
    ]
    chosen = next((m for m in months if as_of_month is None or m <= as_of_month), None)
    out = dict(empty, has_advice_table=True, months=months, as_of_month=chosen)
    try:
        from sibyl.config import ADVICE_MIN_QUESTIONS

        out["min_questions"] = ADVICE_MIN_QUESTIONS
    except Exception:  # noqa: BLE001
        pass
    if chosen is None:
        return out
    rows = _rows_from_cursor(
        _execute(
            con,
            """
            SELECT as_of_month, hazard_code, metric, scope, n_questions, advice,
                   findings_json, advice_version, created_at
            FROM sibyl_calibration_advice
            WHERE as_of_month = ?
            ORDER BY CASE WHEN hazard_code = '*' THEN 1 ELSE 0 END, hazard_code, metric
            """,
            [chosen],
        )
    )
    for r in rows:
        findings = _maybe_json(r.pop("findings_json", None)) or {}
        r["gate"] = findings.get("gate")
        r["diagnostics"] = findings.get("diagnostics") or {}
        r["calibrated_values"] = findings.get("calibrated_values") or {}
        r["perspective_bias"] = findings.get("perspective_bias") or {}
        r["paired_skill"] = findings.get("paired_skill") or {}
        if r["scope"] == "pooled":
            out["arm_comparison"] = findings.get("arm_comparison")
            # Failure types from the post-mortems (Oct 2026): per class and
            # pooled; absent on generations written before.
            out["failure_types"] = findings.get("failure_types")
    out["rows"] = rows
    return out


@router.get("/v1/sibyl/questions")
def sibyl_questions(
    sibyl_run_id: Optional[str] = Query(None),
    include_test: bool = Query(False),
):
    """Per-question rows for the sortable divergence table.

    Default order is track-vs-track JS divergence descending — the most
    decision-useful ranking (the frontend re-sorts client-side).
    """
    con = _con()
    if not _table_exists(con, "sibyl_forecasts"):
        return {"sibyl_run_id": None, "rows": []}

    run_id = sibyl_run_id or _latest_sibyl_run_id(con, include_test)
    if not run_id:
        return {"sibyl_run_id": None, "rows": []}

    rows = _rows_from_cursor(
        _execute(
            con,
            f"""
            SELECT sibyl_run_id, run_id, question_id, iso3, hazard_code,
                   metric, status, skip_reason, as_of, k, aggregation,
                   volatility_score, triage_score,
                   js_divergence_vs_standard, js_divergence_inter_trial,
                   cost_usd, opus_cost_usd, brave_cost_usd,
                   pooled_quantiles_json, {_selection_pass_col(con)},
                   {_evidence_ok_col(con)}
            FROM sibyl_forecasts
            WHERE sibyl_run_id = ?
            ORDER BY js_divergence_vs_standard DESC NULLS LAST, question_id
            """,
            [run_id],
        )
    )
    for r in rows:
        r["pooled_quantiles"] = _maybe_json(r.pop("pooled_quantiles_json", None))
    return {"sibyl_run_id": run_id, "rows": rows}


def _standard_spd_by_month(con, run_id: str, question_id: str) -> Dict[str, Any]:
    """Preferred standard-track aggregate for the overlay chart."""
    for model_name in _STANDARD_MODEL_PREFERENCE:
        rows = _execute(
            con,
            """
            SELECT month_index, bucket_index, probability
            FROM forecasts_ensemble
            WHERE run_id = ? AND question_id = ? AND model_name = ?
            ORDER BY month_index, bucket_index
            """,
            [run_id, question_id, model_name],
        ).fetchall()
        if rows:
            by_month: Dict[int, Dict[int, float]] = {}
            for mi, bi, p in rows:
                by_month.setdefault(int(mi), {})[int(bi)] = float(p or 0.0)
            return {
                "model_name": model_name,
                "by_month": {
                    str(mi): [v for _, v in sorted(buckets.items())]
                    for mi, buckets in sorted(by_month.items())
                },
            }
    return {"model_name": None, "by_month": {}}


@router.get("/v1/sibyl/question_detail")
def sibyl_question_detail(
    question_id: str = Query(...),
    sibyl_run_id: Optional[str] = Query(None),
    include_test: bool = Query(False),
):
    """Full interpretability payload for one Sibyl question.

    Includes the K trial belief-state traces and evidence lists, the pooled
    SPD, and the standard-track SPD for the overlay.
    """
    con = _con()
    if not _table_exists(con, "sibyl_forecasts"):
        raise HTTPException(status_code=404, detail="sibyl_forecasts table not found")

    run_id = sibyl_run_id or _latest_sibyl_run_id(con, include_test)
    rows = _rows_from_cursor(
        _execute(
            con,
            """
            SELECT * FROM sibyl_forecasts
            WHERE question_id = ? AND sibyl_run_id = ?
            LIMIT 1
            """,
            [question_id, run_id],
        )
    )
    if not rows:
        raise HTTPException(status_code=404, detail="sibyl forecast not found")
    rec = rows[0]
    for src_key, dst_key in (
        ("pooled_quantiles_json", "pooled_quantiles"),
        ("trials_json", "trials"),
        ("bucket_probs_json", "bucket_probs"),
        ("leakage_json", "leakage"),
        # Oct 2026: the reference, the raw pool and the published vectors by
        # month (month 1 and 6 overlay), and the extra-trial measures.
        ("reference_json", "reference"),
        ("raw_by_month_json", "raw_by_month"),
        ("final_by_month_json", "final_by_month"),
        ("trial_checks_json", "trial_checks"),
        # Part 5: the resolving source's reading shown to the trials.
        ("resolver_reading_json", "resolver_reading"),
        ("pack_json", "pack"),
    ):
        rec[dst_key] = _maybe_json(rec.pop(src_key, None))

    question_rows = _rows_from_cursor(
        _execute(
            con,
            """
            SELECT question_id, iso3, hazard_code, metric, wording,
                   window_start_date, target_month
            FROM questions WHERE question_id = ?
            LIMIT 1
            """,
            [question_id],
        )
    ) if _table_exists(con, "questions") else []
    question = question_rows[0] if question_rows else None

    metric = str(rec.get("metric") or (question or {}).get("metric") or "")
    standard = (
        _standard_spd_by_month(con, str(rec["run_id"]), question_id)
        if rec.get("run_id")
        else {"model_name": None, "by_month": {}}
    )
    return {
        "record": rec,
        "question": question,
        "bucket_labels": _bucket_labels(con, metric),
        "standard_spd": standard,
    }
