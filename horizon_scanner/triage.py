# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Triage LLM module — separate Gemini API calls for triage scoring.

This module is responsible for running the triage-only prompt against
Gemini 2.5 Flash, performing 2 passes for reliability, and returning merged
triage results per hazard.  It receives regime-change results as context
input (from the RC module that runs first).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

from forecaster.providers import (
    ModelSpec,
    call_chat_ms,
    estimate_cost_usd,
)
from horizon_scanner._utils import (
    coerce_list,
    coerce_score_or_none,
    merge_unique,
    parse_json_response,
    repair_json_response,
    resolve_hs_model,
    short_error,
    status_from_error,
)
from horizon_scanner.llm_logging import log_hs_llm_call
from horizon_scanner.prompts import build_triage_prompt

logger = logging.getLogger(__name__)

HS_TEMPERATURE = float(os.getenv("HS_TEMPERATURE", "0.0"))


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

async def _call_triage_model(
    prompt_text: str,
    *,
    run_id: str | None = None,
    fallback_specs: list[ModelSpec] | None = None,
) -> tuple[str, Dict[str, Any], str, ModelSpec]:
    """Call the LLM for triage scoring.

    Returns (text, usage, error, model_spec).
    """

    spec = ModelSpec(
        name="Gemini",
        provider="google",
        model_id=resolve_hs_model(),
        active=True,
        purpose="hs_triage",
    )
    start = time.time()
    try:
        text, usage, error = await call_chat_ms(
            spec,
            prompt_text,
            temperature=HS_TEMPERATURE,
            prompt_key="hs.triage.v3",
            prompt_version="1.0.0",
            component="HorizonScanner",
            run_id=run_id,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.time() - start) * 1000)
        return "", {"elapsed_ms": elapsed_ms}, f"provider call error: {exc}", spec

    usage = usage or {}
    usage.setdefault("elapsed_ms", int((time.time() - start) * 1000))
    if not error:
        usage["fallback_used"] = False
        usage["model_selected"] = f"{spec.provider}:{spec.model_id}"
        return text, usage, error, spec

    primary_error = short_error(error)
    fallback_specs = fallback_specs or []
    for fallback_spec in fallback_specs:
        fallback_start = time.time()
        try:
            fallback_text, fallback_usage, fallback_error = await call_chat_ms(
                fallback_spec,
                prompt_text,
                temperature=HS_TEMPERATURE,
                prompt_key="hs.triage.v3",
                prompt_version="1.0.0",
                component="HorizonScanner",
                run_id=run_id,
            )
        except Exception as exc:  # noqa: BLE001
            fallback_error = f"provider call error: {exc}"
            fallback_text = ""
            fallback_usage = {}
        fallback_usage = fallback_usage or {}
        fallback_usage.setdefault("elapsed_ms", int((time.time() - fallback_start) * 1000))
        fallback_usage["fallback_used"] = True
        fallback_usage["primary_error"] = primary_error
        fallback_usage["model_selected"] = f"{fallback_spec.provider}:{fallback_spec.model_id}"
        if not fallback_error:
            return fallback_text, fallback_usage, "", fallback_spec
    usage["fallback_used"] = True
    usage["primary_error"] = primary_error
    usage["model_selected"] = f"{spec.provider}:{spec.model_id}"
    return "", usage, error, spec


# ---------------------------------------------------------------------------
# Pass extraction
# ---------------------------------------------------------------------------

def _build_triage_pass_hazards(
    hazards_raw: Dict[str, Any],
    expected_hazards: list[str],
) -> Dict[str, Dict[str, Any]]:
    """Extract triage fields from a single pass response."""

    normalized: Dict[str, Any] = {}
    if isinstance(hazards_raw, dict):
        for hz_code, hdata in hazards_raw.items():
            key = (hz_code or "").upper().strip()
            if key:
                normalized[key] = hdata if isinstance(hdata, dict) else {}

    pass_hazards: Dict[str, Dict[str, Any]] = {}
    for hz_code in expected_hazards:
        hdata = normalized.get(hz_code) if isinstance(normalized.get(hz_code), dict) else None
        if not hdata:
            pass_hazards[hz_code] = {
                "score": None,
                "score_valid": False,
                "drivers": [],
                "regime_shifts": [],
                "data_quality": {"status": "missing_in_model_output"},
                "scenario_stub": "",
            }
            continue

        data_quality = hdata.get("data_quality") or {}
        if not isinstance(data_quality, dict):
            data_quality = {"raw": data_quality}

        score_value = coerce_score_or_none(hdata.get("triage_score"))
        pass_hazards[hz_code] = {
            "score": score_value,
            "score_valid": score_value is not None,
            "drivers": coerce_list(hdata.get("drivers")),
            "regime_shifts": coerce_list(hdata.get("regime_shifts")),
            "data_quality": data_quality,
            "scenario_stub": str(hdata.get("scenario_stub") or ""),
        }

    return pass_hazards


# ---------------------------------------------------------------------------
# 2-pass merge
# ---------------------------------------------------------------------------

def _merge_triage_passes(
    pass_one: Dict[str, Dict[str, Any]],
    pass_two: Dict[str, Dict[str, Any]],
    expected_hazards: list[str],
) -> Dict[str, Dict[str, Any]]:
    """Merge two triage passes: average scores, deduplicate qualitative data."""

    merged: Dict[str, Dict[str, Any]] = {}
    for hz_code in expected_hazards:
        p1 = pass_one.get(hz_code, {})
        p2 = pass_two.get(hz_code, {})
        score1 = p1.get("score")
        score2 = p2.get("score")
        score1_valid = bool(p1.get("score_valid"))
        score2_valid = bool(p2.get("score_valid"))

        if score1_valid and score2_valid:
            avg_score = (float(score1) + float(score2)) / 2
            combined_status = "ok"
        elif score1_valid:
            avg_score = float(score1)
            combined_status = "degraded"
        elif score2_valid:
            avg_score = float(score2)
            combined_status = "degraded"
        else:
            avg_score = 0.0
            combined_status = "error"

        merged[hz_code] = {
            "triage_score": avg_score,
            "drivers": merge_unique(p1.get("drivers", []), p2.get("drivers", [])),
            "regime_shifts": merge_unique(
                p1.get("regime_shifts", []), p2.get("regime_shifts", []),
            ),
            "data_quality": {
                "status": combined_status,
                "score_pass1": score1,
                "score_pass2": score2,
                "pass1": p1.get("data_quality", {}),
                "pass2": p2.get("data_quality", {}),
            },
            "scenario_stub": p1.get("scenario_stub") or p2.get("scenario_stub") or "",
        }

    return merged


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_triage_for_country(
    run_id: str,
    iso3: str,
    country_name: str,
    hazard_catalog: Dict[str, str],
    resolver_features: Dict[str, Any],
    evidence_pack: Dict[str, Any] | None,
    fallback_specs: list[ModelSpec],
    rc_results: Dict[str, Any] | None = None,
    expected_hazards: list[str] | None = None,
) -> Dict[str, Any]:
    """Run triage scoring for a single country (2 passes).

    *rc_results* is the output from :func:`run_rc_for_country` — when provided,
    its per-hazard regime-change assessments are injected into the prompt as
    context so the model can factor them into triage scores.

    Returns::

        {
            "hazards": {"ACE": {"triage_score": ..., "drivers": [...], ...}, ...},
            "status": "ok" | "degraded" | "failed",
            "pass_results": [...],
        }
    """

    iso3_up = (iso3 or "").upper()
    if expected_hazards is None:
        expected_hazards = sorted(hazard_catalog.keys())

    logger.info("Running triage scoring for %s (%s)", country_name, iso3_up)

    prompt = build_triage_prompt(
        country_name=country_name,
        iso3=iso3_up,
        hazard_catalog=hazard_catalog,
        resolver_features=resolver_features,
        model_info={},
        evidence_pack=evidence_pack,
        regime_change_results=rc_results,
    )

    pass_results: list[dict[str, Any]] = []
    for pass_idx in (1, 2):
        call_start = time.time()
        text, usage, error, model_spec = asyncio.run(
            _call_triage_model(prompt, run_id=run_id, fallback_specs=fallback_specs)
        )
        usage = usage or {}
        usage.setdefault("elapsed_ms", int((time.time() - call_start) * 1000))

        log_error_text: str | None = None
        parsed: Dict[str, Any] = {}
        hazards_dict: Dict[str, Any] = {}
        parse_ok = True
        json_repair_used = False
        repair_error: str | None = None
        repair_model_selected: str | None = None

        if error:
            log_error_text = str(error)
            parse_ok = False
        else:
            raw = (text or "").strip()
            if not raw:
                log_error_text = "empty response"
                parse_ok = False
            else:
                try:
                    parsed = parse_json_response(raw)
                    hazards_dict = parsed.get("hazards") or {}
                except Exception as exc:  # noqa: BLE001
                    debug_dir = Path("debug/hs_triage_raw")
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    raw_path = debug_dir / f"{run_id}__{iso3}__triage_pass{pass_idx}.txt"
                    raw_path.write_text(text or "", encoding="utf-8")
                    log_error_text = f"triage parse failed: {type(exc).__name__}: {exc}"
                    parse_ok = False
                    logger.error(
                        "Triage parse failed for %s pass %s: %s (raw saved to %s)",
                        iso3_up, pass_idx, exc, raw_path,
                    )
                    repair_obj, repair_usage, repair_error, repair_spec = asyncio.run(
                        repair_json_response(
                            raw,
                            run_id=run_id,
                            fallback_specs=fallback_specs,
                            prompt_key="hs.triage.json_repair",
                        )
                    )
                    if not repair_error:
                        parsed = repair_obj
                        hazards_dict = parsed.get("hazards") or {}
                        parse_ok = True
                        json_repair_used = True
                        log_error_text = None
                        if repair_spec:
                            repair_model_selected = f"{repair_spec.provider}:{repair_spec.model_id}"
                        if repair_usage:
                            usage.setdefault("repair_usage", repair_usage)
                    else:
                        repair_model_selected = (
                            f"{repair_spec.provider}:{repair_spec.model_id}" if repair_spec else None
                        )

        hazards_dict = hazards_dict if isinstance(hazards_dict, dict) else {}

        usage_for_log = dict(usage or {})
        if json_repair_used:
            usage_for_log["json_repair_used"] = True
            if repair_model_selected:
                usage_for_log["repair_model_selected"] = repair_model_selected
        if repair_error:
            usage_for_log["repair_error"] = short_error(repair_error)
        if (usage_for_log.get("cost_usd") in (None, 0, 0.0)) and (usage_for_log.get("total_tokens") or 0):
            model_id_for_cost = getattr(model_spec, "model_id", None) or resolve_hs_model()
            if model_id_for_cost:
                usage_for_log["cost_usd"] = estimate_cost_usd(str(model_id_for_cost), usage_for_log)

        log_hs_llm_call(
            hs_run_id=run_id,
            iso3=iso3_up,
            hazard_code=f"triage_pass_{pass_idx}",
            model_spec=model_spec,
            prompt_text=prompt,
            response_text=text or "",
            usage=usage_for_log,
            error_text=log_error_text,
        )
        logger.info(
            "Triage logged call: hs_run_id=%s iso3=%s hazard=triage_pass_%s",
            run_id, iso3_up, pass_idx,
        )

        pass_results.append({
            "text": text or "",
            "usage": usage_for_log,
            "hazards": hazards_dict,
            "error_text": log_error_text,
            "parse_ok": parse_ok,
            "fallback_used": bool(usage_for_log.get("fallback_used")),
            "primary_error": usage_for_log.get("primary_error"),
            "json_repair_used": json_repair_used,
            "repair_error": repair_error,
            "model_selected": usage_for_log.get("model_selected"),
        })

    # Merge passes
    pass_one_hz = _build_triage_pass_hazards(pass_results[0]["hazards"], expected_hazards)
    pass_two_hz = _build_triage_pass_hazards(pass_results[1]["hazards"], expected_hazards)
    merged_hazards = _merge_triage_passes(pass_one_hz, pass_two_hz, expected_hazards)

    # Determine overall status
    p1_valid = any(pass_one_hz[hz].get("score_valid") for hz in expected_hazards)
    p2_valid = any(pass_two_hz[hz].get("score_valid") for hz in expected_hazards)
    if p1_valid and p2_valid:
        status = "ok"
    elif p1_valid or p2_valid:
        status = "degraded"
    else:
        status = "failed"

    total_tokens = sum(r["usage"].get("total_tokens") or 0 for r in pass_results)
    total_cost = sum(r["usage"].get("cost_usd") or 0.0 for r in pass_results)
    logger.info(
        "Triage scoring completed for %s (%s): status=%s tokens=%s cost_usd=%.4f",
        country_name, iso3_up, status, total_tokens, total_cost,
    )

    return {
        "hazards": merged_hazards,
        "status": status,
        "pass_results": pass_results,
        "primary_model_id": resolve_hs_model(),
        "fallback_model_id": fallback_specs[0].model_id if fallback_specs else None,
    }
