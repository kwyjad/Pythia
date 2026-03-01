# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Regime-change LLM module — per-hazard Gemini API calls for RC assessment.

This module runs separate RC assessments for each hazard (ACE, FL, DR, TC, HW),
each with its own dedicated Google Search grounding call.  DI is silenced
(defaults returned, no LLM call).  Seasonal filtering skips out-of-season
hazards based on resolver/data/seasonal_hazards.csv.

Each active hazard goes through:
  1. Per-hazard grounding call (Gemini + Google Search)
  2. Per-hazard RC prompt (2-pass for reliability)
  3. 2-pass merge → merged RC dict

The public entry point ``run_rc_for_country`` preserves the same return shape
as before::

    {
        "hazards": {"ACE": {"likelihood": ..., ...}, ...},
        "status": "ok" | "degraded" | "failed",
        "pass_results": [...],
    }
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import time
from datetime import date as date_type
from pathlib import Path
from typing import Any, Dict, Set

from forecaster.providers import (
    ModelSpec,
    call_chat_ms,
    estimate_cost_usd,
)
from horizon_scanner._utils import (
    merge_unique,
    merge_unique_signals,
    parse_json_response,
    repair_json_response,
    resolve_hs_model,
    short_error,
    status_from_error,
)
from horizon_scanner.llm_logging import log_hs_llm_call
from horizon_scanner.rc_grounding_prompts import (
    build_grounding_prompt,
    get_recency_days,
)
from horizon_scanner.rc_prompts import build_rc_prompt
from horizon_scanner.regime_change import coerce_regime_change
from horizon_scanner.seasonal_filter import get_active_hazards

logger = logging.getLogger(__name__)

HS_TEMPERATURE = float(os.getenv("HS_TEMPERATURE", "0.0"))

# All RC-eligible hazard codes (DI excluded).
_RC_HAZARDS = {"ACE", "DR", "FL", "HW", "TC"}

# Default RC values for hazards that are seasonally inactive.
_RC_DEFAULTS: Dict[str, Any] = {
    "likelihood": 0.05,
    "direction": "unclear",
    "magnitude": 0.05,
    "window": "",
    "rationale_bullets": [],
    "trigger_signals": [],
    "valid": True,
    "status": "seasonal_skip",
}

# Default RC values for DI (silenced — no resolution source yet).
_DI_DEFAULTS: Dict[str, Any] = {
    "likelihood": 0.05,
    "direction": "unclear",
    "magnitude": 0.05,
    "window": "",
    "rationale_bullets": [],
    "trigger_signals": [],
    "valid": True,
    "status": "silenced",
}


# ---------------------------------------------------------------------------
# Model call (unchanged from previous version)
# ---------------------------------------------------------------------------

async def _call_rc_model(
    prompt_text: str,
    *,
    run_id: str | None = None,
    fallback_specs: list[ModelSpec] | None = None,
) -> tuple[str, Dict[str, Any], str, ModelSpec]:
    """Call the LLM for regime-change assessment.

    Returns (text, usage, error, model_spec).
    """

    spec = ModelSpec(
        name="Gemini",
        provider="google",
        model_id=resolve_hs_model(),
        active=True,
        purpose="hs_regime_change",
    )
    start = time.time()
    try:
        text, usage, error = await call_chat_ms(
            spec,
            prompt_text,
            temperature=HS_TEMPERATURE,
            prompt_key="hs.regime_change.v2",
            prompt_version="2.0.0",
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
                prompt_key="hs.regime_change.v2",
                prompt_version="2.0.0",
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
# Per-hazard grounding
# ---------------------------------------------------------------------------

def _render_grounding_markdown(pack: dict[str, Any]) -> str:
    """Render an evidence pack dict as markdown for injection into RC prompts."""
    structural = (pack.get("structural_context") or "(none)").strip()
    recent_signals = pack.get("recent_signals") or []
    sources = pack.get("sources") or []

    lines = ["# Evidence pack", ""]
    lines.append(f"Query: {pack.get('query', '')}")
    if pack.get("recency_days"):
        lines.append(f"Recency window: last {pack.get('recency_days')} days")
    lines.append(f"Grounded: {bool(pack.get('grounded'))}")
    lines.append("")
    lines.append("Structural context (background only):")
    lines.append(structural if structural else "(none)")
    lines.append("")
    lines.append("Recent signals (last window):")
    if recent_signals:
        for sig in recent_signals:
            lines.append(f"- {sig}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("Sources:")
    if sources:
        for src in sources:
            if isinstance(src, dict):
                title = src.get("title") or src.get("url") or "(untitled)"
                url = src.get("url") or ""
            else:
                title = getattr(src, "title", "") or getattr(src, "url", "(untitled)")
                url = getattr(src, "url", "")
            lines.append(f"- {title} — {url}")
    else:
        lines.append("- (none)")
    return "\n".join(lines)


def _run_grounding_for_hazard(
    hazard_code: str,
    country_name: str,
    iso3: str,
    *,
    run_id: str | None = None,
) -> dict[str, Any] | None:
    """Run a per-hazard Google Search grounding call via Gemini.

    Returns an evidence pack dict with a ``markdown`` key ready for the
    RC prompt, or *None* on failure.
    """
    from pythia.web_research.backends.gemini_grounding import fetch_via_gemini

    try:
        grounding_prompt = build_grounding_prompt(hazard_code, country_name, iso3)
        recency = get_recency_days(hazard_code)
        evidence_pack = fetch_via_gemini(
            query=f"{country_name} ({iso3}) {hazard_code} grounding",
            recency_days=recency,
            include_structural=True,
            timeout_sec=60,
            max_results=10,
            custom_prompt=grounding_prompt,
        )
        pack = evidence_pack.to_dict() if hasattr(evidence_pack, "to_dict") else dict(evidence_pack)
        pack["markdown"] = _render_grounding_markdown(pack)
        return pack
    except Exception as exc:  # noqa: BLE001
        logger.warning("RC grounding failed for %s %s: %s", iso3, hazard_code, exc)
        return None


# ---------------------------------------------------------------------------
# Single-hazard 2-pass merge
# ---------------------------------------------------------------------------

def _merge_single_hazard_passes(
    p1: Dict[str, Any],
    p2: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge two passes of a single hazard's RC assessment.

    Averages numeric fields, reconciles categorical fields.
    """
    rc1_valid = bool(p1.get("valid"))
    rc2_valid = bool(p2.get("valid"))
    rc1_likelihood = p1.get("likelihood")
    rc2_likelihood = p2.get("likelihood")
    rc1_magnitude = p1.get("magnitude")
    rc2_magnitude = p2.get("magnitude")
    rc1_direction = p1.get("direction") or "unclear"
    rc2_direction = p2.get("direction") or "unclear"
    rc1_window = p1.get("window") or ""
    rc2_window = p2.get("window") or ""

    if rc1_valid and rc2_valid:
        rc_likelihood = (float(rc1_likelihood) + float(rc2_likelihood)) / 2
        rc_magnitude = (float(rc1_magnitude) + float(rc2_magnitude)) / 2
        rc_valid = True
        rc_status = "ok"
        if rc1_direction == "unclear" or rc2_direction == "unclear":
            rc_direction = "unclear"
        elif rc1_direction == rc2_direction:
            rc_direction = rc1_direction
        else:
            rc_direction = "mixed"
        if rc1_window == rc2_window:
            rc_window = rc1_window
        elif rc1_window and not rc2_window:
            rc_window = rc1_window
        elif rc2_window and not rc1_window:
            rc_window = rc2_window
        else:
            rc_window = ""
    elif rc1_valid:
        rc_likelihood = float(rc1_likelihood)
        rc_magnitude = float(rc1_magnitude)
        rc_valid = True
        rc_status = "degraded"
        rc_direction = rc1_direction
        rc_window = rc1_window
    elif rc2_valid:
        rc_likelihood = float(rc2_likelihood)
        rc_magnitude = float(rc2_magnitude)
        rc_valid = True
        rc_status = "degraded"
        rc_direction = rc2_direction
        rc_window = rc2_window
    else:
        rc_likelihood = 0.0
        rc_magnitude = 0.0
        rc_valid = False
        rc_status = "error"
        rc_direction = "unclear"
        rc_window = ""

    rc_bullets = merge_unique(
        p1.get("rationale_bullets") or [],
        p2.get("rationale_bullets") or [],
    )
    rc_signals = merge_unique_signals(
        p1.get("trigger_signals") or [],
        p2.get("trigger_signals") or [],
    )

    return {
        "likelihood": rc_likelihood,
        "direction": rc_direction,
        "magnitude": rc_magnitude,
        "window": rc_window,
        "rationale_bullets": rc_bullets,
        "trigger_signals": rc_signals,
        "valid": rc_valid,
        "status": rc_status,
    }


# ---------------------------------------------------------------------------
# Single-hazard 2-pass RC assessment
# ---------------------------------------------------------------------------

def _run_rc_for_single_hazard(
    hazard_code: str,
    country_name: str,
    iso3: str,
    resolver_features: Dict[str, Any],
    evidence_pack: Dict[str, Any] | None,
    *,
    run_id: str,
    fallback_specs: list[ModelSpec],
) -> Dict[str, Any]:
    """Run a 2-pass RC assessment for a single hazard.

    Returns a merged RC dict: ``{likelihood, direction, magnitude, window,
    rationale_bullets, trigger_signals, valid, status}``.
    """

    prompt = build_rc_prompt(
        hazard_code,
        country_name=country_name,
        iso3=iso3,
        resolver_features=resolver_features,
        evidence_pack=evidence_pack,
    )

    pass_rcs: list[Dict[str, Any]] = []
    pass_results: list[dict[str, Any]] = []

    for pass_idx in (1, 2):
        call_start = time.time()
        text, usage, error, model_spec = asyncio.run(
            _call_rc_model(prompt, run_id=run_id, fallback_specs=fallback_specs)
        )
        usage = usage or {}
        usage.setdefault("elapsed_ms", int((time.time() - call_start) * 1000))

        log_error_text: str | None = None
        parsed: Dict[str, Any] = {}
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
                except Exception as exc:  # noqa: BLE001
                    debug_dir = Path("debug/hs_rc_raw")
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    raw_path = debug_dir / f"{run_id}__{iso3}__{hazard_code}__rc_pass{pass_idx}.txt"
                    raw_path.write_text(text or "", encoding="utf-8")
                    log_error_text = f"rc parse failed: {type(exc).__name__}: {exc}"
                    parse_ok = False
                    logger.error(
                        "RC parse failed for %s %s pass %s: %s (raw saved to %s)",
                        iso3, hazard_code, pass_idx, exc, raw_path,
                    )
                    repair_obj, repair_usage, repair_error, repair_spec = asyncio.run(
                        repair_json_response(
                            raw,
                            run_id=run_id,
                            fallback_specs=fallback_specs,
                            prompt_key="hs.regime_change.json_repair",
                        )
                    )
                    if not repair_error:
                        parsed = repair_obj
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

        # The per-hazard RC prompt returns a flat RC object (not nested {hazards: ...}).
        # Coerce it through the standard RC normalizer.
        rc_obj = coerce_regime_change(parsed if parse_ok else None)

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
            iso3=iso3,
            hazard_code=f"rc_{hazard_code}_pass_{pass_idx}",
            model_spec=model_spec,
            prompt_text=prompt,
            response_text=text or "",
            usage=usage_for_log,
            error_text=log_error_text,
        )
        logger.info(
            "RC logged call: hs_run_id=%s iso3=%s hazard=rc_%s_pass_%s",
            run_id, iso3, hazard_code, pass_idx,
        )

        pass_rcs.append(rc_obj)
        pass_results.append({
            "text": text or "",
            "usage": usage_for_log,
            "error_text": log_error_text,
            "parse_ok": parse_ok,
            "fallback_used": bool(usage_for_log.get("fallback_used")),
            "primary_error": usage_for_log.get("primary_error"),
            "json_repair_used": json_repair_used,
            "repair_error": repair_error,
            "model_selected": usage_for_log.get("model_selected"),
        })

    # Merge two passes for this hazard
    merged = _merge_single_hazard_passes(pass_rcs[0], pass_rcs[1])
    return merged


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_rc_for_country(
    run_id: str,
    iso3: str,
    country_name: str,
    hazard_catalog: Dict[str, str],
    resolver_features: Dict[str, Any],
    evidence_pack: Dict[str, Any] | None,
    fallback_specs: list[ModelSpec],
    expected_hazards: list[str] | None = None,
    run_date: date_type | None = None,
) -> Dict[str, Any]:
    """Run per-hazard regime-change assessment for a single country.

    For each active hazard:
      1. Per-hazard grounding call (Gemini + Google Search)
      2. Per-hazard RC prompt (2-pass for reliability)
      3. 2-pass merge

    DI is silenced (defaults returned).  Hazards out of season per
    ``seasonal_hazards.csv`` are skipped with defaults.

    Returns::

        {
            "hazards": {"ACE": {"likelihood": ..., ...}, ...},
            "status": "ok" | "degraded" | "failed",
            "pass_results": [...],
        }
    """

    iso3_up = (iso3 or "").upper()
    run_date = run_date or date_type.today()
    if expected_hazards is None:
        expected_hazards = sorted(hazard_catalog.keys())

    # Determine which hazards are seasonally active for this country.
    active: Set[str] = get_active_hazards(iso3_up, run_date)
    logger.info(
        "Running RC assessment for %s (%s): active=%s run_date=%s",
        country_name, iso3_up, sorted(active), run_date,
    )

    # Phase 1: Parallel grounding calls for active hazards.
    hazards_to_ground = [hz for hz in expected_hazards if hz in active and hz in _RC_HAZARDS]
    grounding_results: Dict[str, dict[str, Any] | None] = {}

    if hazards_to_ground:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(hazards_to_ground))) as executor:
            futures = {
                executor.submit(
                    _run_grounding_for_hazard,
                    hz,
                    country_name,
                    iso3_up,
                    run_id=run_id,
                ): hz
                for hz in hazards_to_ground
            }
            for future in concurrent.futures.as_completed(futures):
                hz = futures[future]
                try:
                    grounding_results[hz] = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("RC grounding future failed for %s %s: %s", iso3_up, hz, exc)
                    grounding_results[hz] = None

    # Phase 2: Sequential RC calls per active hazard (2-pass each).
    merged_hazards: Dict[str, Dict[str, Any]] = {}
    all_pass_results: list[dict[str, Any]] = []
    rc_start = time.time()

    for hz_code in expected_hazards:
        if hz_code == "DI":
            merged_hazards[hz_code] = dict(_DI_DEFAULTS)
            continue

        if hz_code not in active or hz_code not in _RC_HAZARDS:
            merged_hazards[hz_code] = dict(_RC_DEFAULTS)
            logger.info("RC skip for %s %s (seasonal or unknown)", iso3_up, hz_code)
            continue

        hazard_evidence = grounding_results.get(hz_code)
        hazard_rc = _run_rc_for_single_hazard(
            hz_code,
            country_name,
            iso3_up,
            resolver_features,
            hazard_evidence,
            run_id=run_id,
            fallback_specs=fallback_specs,
        )
        merged_hazards[hz_code] = hazard_rc

    # Determine overall status across active hazards.
    active_statuses = [
        merged_hazards[hz].get("status", "error")
        for hz in expected_hazards
        if hz in active and hz in _RC_HAZARDS
    ]
    if not active_statuses:
        status = "ok"
    elif all(s == "error" for s in active_statuses):
        status = "failed"
    elif all(s == "ok" for s in active_statuses):
        status = "ok"
    else:
        status = "degraded"

    total_elapsed = int((time.time() - rc_start) * 1000)
    logger.info(
        "RC assessment completed for %s (%s): status=%s active=%s elapsed_ms=%s",
        country_name, iso3_up, status, sorted(active), total_elapsed,
    )

    return {
        "hazards": merged_hazards,
        "status": status,
        "pass_results": all_pass_results,
        "primary_model_id": resolve_hs_model(),
        "fallback_model_id": fallback_specs[0].model_id if fallback_specs else None,
    }
