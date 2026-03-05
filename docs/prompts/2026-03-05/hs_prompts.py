# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
from typing import Any, Dict


def build_hs_triage_prompt(
    country_name: str,
    iso3: str,
    hazard_catalog: Dict[str, str],
    resolver_features: Dict[str, Any],
    model_info: Dict[str, Any] | None = None,
    evidence_pack: Dict[str, Any] | None = None,
) -> str:
    """Build the Horizon Scanner v2 triage prompt.

    The prompt is intentionally structured so the model returns a single JSON
    payload with tiers, triage scores, and optional qualitative context per
    hazard. No forecasts are requested at this stage.
    """

    model_info = model_info or {}

    evidence_text = "Evidence pack unavailable (web research disabled or failed)."
    if evidence_pack:
        evidence_text = evidence_pack.get("markdown") or ""

    schema_example: dict[str, Any] = {
        "country": iso3,
        "hazards": {
            "ACE": {
                "triage_score": 0.0,
                "tier": "quiet|priority",
                "regime_change": {
                    "likelihood": 0.0,
                    "direction": "up|down|mixed|unclear",
                    "magnitude": 0.0,
                    "window": (
                        "month_1|month_2|month_3|month_4|month_5|month_6|"
                        "month_1-2|month_3-4|month_5-6"
                    ),
                    "rationale_bullets": ["..."],
                    "trigger_signals": [
                        {
                            "signal": "...",
                            "timeframe_months": 2,
                            "evidence_refs": ["..."],
                        }
                    ],
                },
                "drivers": ["..."],
                "regime_shifts": ["..."],
                "data_quality": {
                    "resolution_source": "...",
                    "reliability": "low|medium|high",
                    "notes": "...",
                },
                "scenario_stub": "...",
            }
        },
    }
    schema_text = json.dumps(schema_example, indent=2)

    # --- PROMPT_EXCERPT: hs_triage_start ---
    return f"""You are a strategic humanitarian risk analyst.\nYou are assessing {country_name} ({iso3}) for the next 1-6 months.\n\nHazards and their meanings:\n{json.dumps(hazard_catalog, indent=2)}\n\nResolver features (noisy, imperfect data from ACLED/IDMC/EM-DAT; see notes in each entry):\n\n{json.dumps(resolver_features, indent=2)}\n\nEvidence Pack (prioritize recent signals; use structural context only as background):\n\n{evidence_text}\n\nIf evidence is thin, say so.\n\nModel/data notes JSON:\n{json.dumps(model_info, indent=2)}\n\nYour task is to triage, not to forecast exact numbers.\n\nOutput requirements (strict):\n\n- Return a single JSON object only. No prose. No markdown fences. No extra keys.\n- Provide a hazards entry for every hazard code in the catalog (ACE, DI, DR, FL, HW, TC).\n- Each hazard must include a numeric triage_score (0.0 to 1.0). Score-first: decide triage_score first, then assign tier (if provided).\n- A tier is optional for interpretability (quiet/priority) and may be ignored by downstream code.\n\nScoring rubric (stable, deterministic):\n\n- Recent signals (from evidence pack): how many credible, recent signals point to elevated risk in the next 1-6 months.\n- Structural drivers (background): persistent conflict, climate exposure, governance, vulnerability; use as a modifier, not a trigger.\n- Resolver base-rate confidence (ACLED/IDMC/EM-DAT): whether historical base-rates suggest elevated risk; downweight noisy or sparse metrics.\n- data_quality: lower scores if data is sparse, outdated, or inconsistent.\n- Stability rule: for borderline cases, pick conservative scores near thresholds rather than flipping tiers.\n\nFor each hazard (ACE/DI/DR/FL/HW/TC):\n\n* Assign a `triage_score` between 0.0 and 1.0 representing the risk of unusually high recorded impact in the next 1-6 months.\n* Optionally assign a triage `tier` in [\"quiet\",\"priority\"].\n* You are providing a coarse risk score.\n* Provide a `regime_change` object (see REGIME CHANGE CALIBRATION below).\n\nFor non-quiet hazards:\n\n* Provide 2–4 key `drivers` that push risk up or down, citing evidence pack signals.\n* Provide 0–3 plausible `regime_shifts` (tipping points) with likelihood and timeframe.\n* Provide `data_quality` notes about the resolution dataset (ACLED, EM-DAT, IDMC, DTM) and its biases.\n* Optionally provide a short `scenario_stub` (3–4 sentences) describing the situation, humanitarian needs, and operational constraints.\n\nFor quiet hazards:\n\n* Still include the `regime_change` object with low values as described below.\n\n=== REGIME CHANGE CALIBRATION (critical — read carefully) ===\n\nThe `regime_change` object captures the probability and magnitude of a DEPARTURE FROM the country's OWN HISTORICAL BASELINE for that specific hazard. It is NOT a measure of absolute risk level.\n\nKey distinction — triage_score vs regime_change:\n- `triage_score` captures the OVERALL risk level, including ongoing/chronic situations. A country with severe but steady conflict has a HIGH triage_score.\n- `regime_change` captures ONLY the probability and magnitude of a BREAK from the established pattern. That same country with steady conflict has LOW regime_change likelihood, because the pattern is continuing as expected.\n\nExamples of what IS a regime change:\n- A country at peace seeing a new armed conflict emerge (ACE likelihood high)\n- A drought-free region entering an unprecedented dry spell (DR likelihood high)\n- Conflict fatalities suddenly doubling or tripling versus the recent 12-month trend\n- A new displacement crisis in a previously stable country\n\nExamples of what is NOT a regime change:\n- Ongoing conflict continuing at its established level, even if severe\n- Seasonal flooding in a flood-prone country during the usual flood season\n- Cyclone risk during cyclone season in a cyclone-exposed country\n- Chronic displacement continuing at roughly the same rate\n- General structural vulnerability without specific new triggers\n\nCalibration anchors — use the resolver features:\n- Compare current signals against the historical base rates in the resolver features.\n- If recent signals are consistent with the historical pattern (even if the historical pattern involves periodic crises), likelihood should be LOW (< 0.15).\n- Only assign likelihood > 0.35 when you can cite SPECIFIC, CONCRETE evidence of emerging change beyond the base rate.\n- Magnitude should reflect how FAR from the historical baseline the change would be, not the absolute severity.\n\nExpected distribution (across a full run of ~120 countries × 6 hazards = ~720 assessments):\n- ~80% of assessments: likelihood <= 0.10 (base-rate normal, no regime change signal)\n- ~10% of assessments: likelihood 0.10-0.30 (watch — some signals but not compelling)\n- ~7% of assessments: likelihood 0.30-0.55 (emerging — specific evidence of potential change)\n- ~3% of assessments: likelihood >= 0.55 (strong signal — clear, concrete evidence of imminent break)\n\nDefault values for the majority of hazard-country pairs:\n- likelihood: 0.05 (most hazards in most countries are following their established pattern)\n- magnitude: 0.05\n- direction: \"unclear\"\n\nDo NOT assign likelihood > 0.10 unless you can point to a specific, recent signal in the evidence pack that suggests a departure from the base rate. Structural vulnerability alone is not sufficient.\n\n=== END REGIME CHANGE CALIBRATION ===\n\nReturn exactly one JSON object matching this schema:\n{schema_text}\n"""
    # --- PROMPT_EXCERPT: hs_triage_end ---


# ---------------------------------------------------------------------------
# Split prompts: regime change (runs first) and triage (runs second)
# ---------------------------------------------------------------------------


def build_regime_change_prompt(
    country_name: str,
    iso3: str,
    hazard_catalog: Dict[str, str],
    resolver_features: Dict[str, Any],
    model_info: Dict[str, Any] | None = None,
    evidence_pack: Dict[str, Any] | None = None,
) -> str:
    """Build a prompt for regime-change-only assessment.

    This prompt is focused exclusively on detecting departures from the
    country's own historical baseline for each hazard.  It runs *before*
    the triage prompt so its outputs can inform triage scoring.
    """

    model_info = model_info or {}

    evidence_text = "Evidence pack unavailable (web research disabled or failed)."
    if evidence_pack:
        evidence_text = evidence_pack.get("markdown") or ""

    schema_example: dict[str, Any] = {
        "country": iso3,
        "hazards": {
            "ACE": {
                "regime_change": {
                    "likelihood": 0.0,
                    "direction": "up|down|mixed|unclear",
                    "magnitude": 0.0,
                    "window": (
                        "month_1|month_2|month_3|month_4|month_5|month_6|"
                        "month_1-2|month_3-4|month_5-6"
                    ),
                    "rationale_bullets": ["..."],
                    "trigger_signals": [
                        {
                            "signal": "...",
                            "timeframe_months": 2,
                            "evidence_refs": ["..."],
                        }
                    ],
                },
            }
        },
    }
    schema_text = json.dumps(schema_example, indent=2)

    return f"""You are a strategic humanitarian regime-change analyst.
You are assessing {country_name} ({iso3}) for the next 1-6 months.

Your ONLY task is to assess whether each hazard is DEPARTING from the country's own historical baseline. You are NOT scoring overall risk — only the probability and magnitude of a BREAK from the established pattern.

Hazards and their meanings:
{json.dumps(hazard_catalog, indent=2)}

Resolver features (noisy, imperfect data from ACLED/IDMC/EM-DAT; see notes in each entry):

{json.dumps(resolver_features, indent=2)}

Evidence Pack (prioritize recent signals; use structural context only as background):

{evidence_text}

If evidence is thin, say so.

Model/data notes JSON:
{json.dumps(model_info, indent=2)}

Output requirements (strict):

- Return a single JSON object only. No prose. No markdown fences. No extra keys.
- Provide a hazards entry for every hazard code in the catalog (ACE, DI, DR, FL, HW, TC).
- Each hazard must include a `regime_change` object as described below.

=== REGIME CHANGE CALIBRATION (critical — read carefully) ===

The `regime_change` object captures the probability and magnitude of a DEPARTURE FROM the country's OWN HISTORICAL BASELINE for that specific hazard. It is NOT a measure of absolute risk level.

Examples of what IS a regime change:
- A country at peace seeing a new armed conflict emerge (ACE likelihood high)
- A drought-free region entering an unprecedented dry spell (DR likelihood high)
- Conflict fatalities suddenly doubling or tripling versus the recent 12-month trend
- A new displacement crisis in a previously stable country

Examples of what is NOT a regime change:
- Ongoing conflict continuing at its established level, even if severe
- Seasonal flooding in a flood-prone country during the usual flood season
- Cyclone risk during cyclone season in a cyclone-exposed country
- Chronic displacement continuing at roughly the same rate
- General structural vulnerability without specific new triggers

Calibration anchors — use the resolver features:
- Compare current signals against the historical base rates in the resolver features.
- If recent signals are consistent with the historical pattern (even if the historical pattern involves periodic crises), likelihood should be LOW (< 0.15).
- Only assign likelihood > 0.35 when you can cite SPECIFIC, CONCRETE evidence of emerging change beyond the base rate.
- Magnitude should reflect how FAR from the historical baseline the change would be, not the absolute severity.

Expected distribution (across a full run of ~120 countries x 6 hazards = ~720 assessments):
- ~80% of assessments: likelihood <= 0.10 (base-rate normal, no regime change signal)
- ~10% of assessments: likelihood 0.10-0.30 (watch — some signals but not compelling)
- ~7% of assessments: likelihood 0.30-0.55 (emerging — specific evidence of potential change)
- ~3% of assessments: likelihood >= 0.55 (strong signal — clear, concrete evidence of imminent break)

Default values for the majority of hazard-country pairs:
- likelihood: 0.05 (most hazards in most countries are following their established pattern)
- magnitude: 0.05
- direction: "unclear"

Do NOT assign likelihood > 0.10 unless you can point to a specific, recent signal in the evidence pack that suggests a departure from the base rate. Structural vulnerability alone is not sufficient.

For each hazard, provide:
- likelihood (0.0 to 1.0): probability of a regime change occurring in the next 1-6 months
- direction: "up", "down", "mixed", or "unclear"
- magnitude (0.0 to 1.0): how far from the historical baseline the change would be
- window: the expected timeframe (e.g. "month_1-2", "month_3-4")
- rationale_bullets: 1-4 short bullets explaining the assessment
- trigger_signals: 0-4 specific observable signals with timeframe and evidence refs

=== END REGIME CHANGE CALIBRATION ===

Return exactly one JSON object matching this schema:
{schema_text}
"""


def _format_rc_context(rc_results: Dict[str, Any]) -> str:
    """Format merged regime-change results as readable context for the triage prompt."""

    hazards = rc_results.get("hazards") or {}
    if not hazards:
        return "Regime change assessment: unavailable."

    lines = []
    for hz_code in sorted(hazards.keys()):
        rc = hazards[hz_code]
        if not isinstance(rc, dict):
            continue
        likelihood = rc.get("likelihood")
        magnitude = rc.get("magnitude")
        direction = rc.get("direction", "unclear")
        window = rc.get("window", "")
        bullets = rc.get("rationale_bullets") or []

        parts = [f"{hz_code}:"]
        parts.append(f"likelihood={likelihood}")
        parts.append(f"magnitude={magnitude}")
        parts.append(f"direction={direction}")
        if window:
            parts.append(f"window={window}")
        line = " ".join(parts)
        if bullets:
            bullet_text = "; ".join(str(b) for b in bullets[:3])
            line += f" [{bullet_text}]"
        lines.append(line)

    return "\n".join(lines)


def build_triage_prompt(
    country_name: str,
    iso3: str,
    hazard_catalog: Dict[str, str],
    resolver_features: Dict[str, Any],
    model_info: Dict[str, Any] | None = None,
    evidence_pack: Dict[str, Any] | None = None,
    regime_change_results: Dict[str, Any] | None = None,
) -> str:
    """Build a triage-only prompt (no regime change assessment).

    When *regime_change_results* is provided (output from the RC module), the
    model is given those assessments as context to inform triage scoring.
    """

    model_info = model_info or {}

    evidence_text = "Evidence pack unavailable (web research disabled or failed)."
    if evidence_pack:
        evidence_text = evidence_pack.get("markdown") or ""

    rc_context = ""
    if regime_change_results:
        rc_text = _format_rc_context(regime_change_results)
        rc_context = f"""
Regime Change Assessments (already computed — use to inform triage scoring):

The following regime change assessments have already been computed for this country.
Use them to inform your triage scoring — elevated RC likelihood/magnitude should push
triage_score higher when combined with other risk signals. However, triage_score also
captures ongoing/chronic situations that are NOT regime changes.

{rc_text}

"""

    schema_example: dict[str, Any] = {
        "country": iso3,
        "hazards": {
            "ACE": {
                "triage_score": 0.0,
                "tier": "quiet|priority",
                "drivers": ["..."],
                "regime_shifts": ["..."],
                "data_quality": {
                    "resolution_source": "...",
                    "reliability": "low|medium|high",
                    "notes": "...",
                },
                "scenario_stub": "...",
            }
        },
    }
    schema_text = json.dumps(schema_example, indent=2)

    return f"""You are a strategic humanitarian risk analyst.
You are assessing {country_name} ({iso3}) for the next 1-6 months.

Hazards and their meanings:
{json.dumps(hazard_catalog, indent=2)}

Resolver features (noisy, imperfect data from ACLED/IDMC/EM-DAT; see notes in each entry):

{json.dumps(resolver_features, indent=2)}

Evidence Pack (prioritize recent signals; use structural context only as background):

{evidence_text}

If evidence is thin, say so.

Model/data notes JSON:
{json.dumps(model_info, indent=2)}
{rc_context}
Your task is to triage, not to forecast exact numbers.

Output requirements (strict):

- Return a single JSON object only. No prose. No markdown fences. No extra keys.
- Provide a hazards entry for every hazard code in the catalog (ACE, DI, DR, FL, HW, TC).
- Each hazard must include a numeric triage_score (0.0 to 1.0). Score-first: decide triage_score first, then assign tier (if provided).
- A tier is optional for interpretability (quiet/priority) and may be ignored by downstream code.

Scoring rubric (stable, deterministic):

- Recent signals (from evidence pack): how many credible, recent signals point to elevated risk in the next 1-6 months.
- Structural drivers (background): persistent conflict, climate exposure, governance, vulnerability; use as a modifier, not a trigger.
- Resolver base-rate confidence (ACLED/IDMC/EM-DAT): whether historical base-rates suggest elevated risk; downweight noisy or sparse metrics.
- Regime change context: if a regime change assessment indicates elevated likelihood/magnitude for a hazard, factor that into the triage score.
- data_quality: lower scores if data is sparse, outdated, or inconsistent.
- Stability rule: for borderline cases, pick conservative scores near thresholds rather than flipping tiers.

For each hazard (ACE/DI/DR/FL/HW/TC):

* Assign a `triage_score` between 0.0 and 1.0 representing the risk of unusually high recorded impact in the next 1-6 months.
* Optionally assign a triage `tier` in ["quiet","priority"].
* You are providing a coarse risk score.

For non-quiet hazards:

* Provide 2-4 key `drivers` that push risk up or down, citing evidence pack signals.
* Provide 0-3 plausible `regime_shifts` (tipping points) with likelihood and timeframe.
* Provide `data_quality` notes about the resolution dataset (ACLED, EM-DAT, IDMC, DTM) and its biases.
* Optionally provide a short `scenario_stub` (3-4 sentences) describing the situation, humanitarian needs, and operational constraints.

For quiet hazards:

* Minimal output is fine — just the triage_score and tier.

Return exactly one JSON object matching this schema:
{schema_text}
"""
