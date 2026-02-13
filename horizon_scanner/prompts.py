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
                "tier": "quiet|watchlist|priority",
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

    return f"""You are a strategic humanitarian risk analyst.\nYou are assessing {country_name} ({iso3}) for the next 1-6 months.\n\nHazards and their meanings:\n{json.dumps(hazard_catalog, indent=2)}\n\nResolver features (noisy, imperfect data from ACLED/IDMC/EM-DAT; see notes in each entry):\n\n{json.dumps(resolver_features, indent=2)}\n\nEvidence Pack (prioritize recent signals; use structural context only as background):\n\n{evidence_text}\n\nIf evidence is thin, say so.\n\nModel/data notes JSON:\n{json.dumps(model_info, indent=2)}\n\nYour task is to triage, not to forecast exact numbers.\n\nOutput requirements (strict):\n\n- Return a single JSON object only. No prose. No markdown fences. No extra keys.\n- Provide a hazards entry for every hazard code in the catalog (ACE, DI, DR, FL, HW, TC).\n- Each hazard must include a numeric triage_score (0.0 to 1.0). Score-first: decide triage_score first, then assign tier (if provided).\n- A tier is optional for interpretability (quiet/watchlist/priority) and may be ignored by downstream code.\n\nScoring rubric (stable, deterministic):\n\n- Recent signals (from evidence pack): how many credible, recent signals point to elevated risk in the next 1-6 months.\n- Structural drivers (background): persistent conflict, climate exposure, governance, vulnerability; use as a modifier, not a trigger.\n- Resolver base-rate confidence (ACLED/IDMC/EM-DAT): whether historical base-rates suggest elevated risk; downweight noisy or sparse metrics.\n- data_quality: lower scores if data is sparse, outdated, or inconsistent.\n- Stability rule: for borderline cases, pick conservative scores near thresholds rather than flipping tiers.\n\nFor each hazard (ACE/DI/DR/FL/HW/TC):\n\n* Assign a `triage_score` between 0.0 and 1.0 representing the risk of unusually high recorded impact in the next 1-6 months.\n* Optionally assign a triage `tier` in [\"quiet\",\"watchlist\",\"priority\"].\n* You are providing a coarse risk score.\n* Provide a `regime_change` object (see REGIME CHANGE CALIBRATION below).\n\nFor non-quiet hazards:\n\n* Provide 2–4 key `drivers` that push risk up or down, citing evidence pack signals.\n* Provide 0–3 plausible `regime_shifts` (tipping points) with likelihood and timeframe.\n* Provide `data_quality` notes about the resolution dataset (ACLED, EM-DAT, IDMC, DTM) and its biases.\n* Optionally provide a short `scenario_stub` (3–4 sentences) describing the situation, humanitarian needs, and operational constraints.\n\nFor quiet hazards:\n\n* Still include the `regime_change` object with low values as described below.\n\n=== REGIME CHANGE CALIBRATION (critical — read carefully) ===\n\nThe `regime_change` object captures the probability and magnitude of a DEPARTURE FROM the country's OWN HISTORICAL BASELINE for that specific hazard. It is NOT a measure of absolute risk level.\n\nKey distinction — triage_score vs regime_change:\n- `triage_score` captures the OVERALL risk level, including ongoing/chronic situations. A country with severe but steady conflict has a HIGH triage_score.\n- `regime_change` captures ONLY the probability and magnitude of a BREAK from the established pattern. That same country with steady conflict has LOW regime_change likelihood, because the pattern is continuing as expected.\n\nExamples of what IS a regime change:\n- A country at peace seeing a new armed conflict emerge (ACE likelihood high)\n- A drought-free region entering an unprecedented dry spell (DR likelihood high)\n- Conflict fatalities suddenly doubling or tripling versus the recent 12-month trend\n- A new displacement crisis in a previously stable country\n\nExamples of what is NOT a regime change:\n- Ongoing conflict continuing at its established level, even if severe\n- Seasonal flooding in a flood-prone country during the usual flood season\n- Cyclone risk during cyclone season in a cyclone-exposed country\n- Chronic displacement continuing at roughly the same rate\n- General structural vulnerability without specific new triggers\n\nCalibration anchors — use the resolver features:\n- Compare current signals against the historical base rates in the resolver features.\n- If recent signals are consistent with the historical pattern (even if the historical pattern involves periodic crises), likelihood should be LOW (< 0.15).\n- Only assign likelihood > 0.35 when you can cite SPECIFIC, CONCRETE evidence of emerging change beyond the base rate.\n- Magnitude should reflect how FAR from the historical baseline the change would be, not the absolute severity.\n\nExpected distribution (across a full run of ~120 countries × 6 hazards = ~720 assessments):\n- ~80% of assessments: likelihood <= 0.10 (base-rate normal, no regime change signal)\n- ~10% of assessments: likelihood 0.10-0.30 (watch — some signals but not compelling)\n- ~7% of assessments: likelihood 0.30-0.55 (emerging — specific evidence of potential change)\n- ~3% of assessments: likelihood >= 0.55 (strong signal — clear, concrete evidence of imminent break)\n\nDefault values for the majority of hazard-country pairs:\n- likelihood: 0.05 (most hazards in most countries are following their established pattern)\n- magnitude: 0.05\n- direction: \"unclear\"\n\nDo NOT assign likelihood > 0.10 unless you can point to a specific, recent signal in the evidence pack that suggests a departure from the base rate. Structural vulnerability alone is not sufficient.\n\n=== END REGIME CHANGE CALIBRATION ===\n\nReturn exactly one JSON object matching this schema:\n{schema_text}\n"""
