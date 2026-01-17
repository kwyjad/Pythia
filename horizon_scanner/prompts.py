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

    return f"""You are a strategic humanitarian risk analyst.\nYou are assessing {country_name} ({iso3}) for the next 1-6 months.\n\nHazards and their meanings:\n{json.dumps(hazard_catalog, indent=2)}\n\nResolver features (noisy, imperfect data from ACLED/IDMC/EM-DAT; see notes in each entry):\n\n{json.dumps(resolver_features, indent=2)}\n\nEvidence Pack (prioritize recent signals; use structural context only as background):\n\n{evidence_text}\n\nIf evidence is thin, say so.\n\nModel/data notes JSON:\n{json.dumps(model_info, indent=2)}\n\nYour task is to triage, not to forecast exact numbers.\n\nOutput requirements (strict):\n\n- Return a single JSON object only. No prose. No markdown fences. No extra keys.\n- Provide a hazards entry for every hazard code in the catalog (ACE, DI, DR, FL, HW, TC).\n- Each hazard must include a numeric triage_score (0.0 to 1.0). Score-first: decide triage_score first, then assign tier (if provided).\n- A tier is optional for interpretability (quiet/watchlist/priority) and may be ignored by downstream code.\n\nScoring rubric (stable, deterministic):\n\n- Recent signals (from evidence pack): how many credible, recent signals point to elevated risk in the next 1-6 months.\n- Structural drivers (background): persistent conflict, climate exposure, governance, vulnerability; use as a modifier, not a trigger.\n- Resolver base-rate confidence (ACLED/IDMC/EM-DAT): whether historical base-rates suggest elevated risk; downweight noisy or sparse metrics.\n- data_quality: lower scores if data is sparse, outdated, or inconsistent.\n- Stability rule: for borderline cases, pick conservative scores near thresholds rather than flipping tiers.\n\nFor each hazard (ACE/DI/DR/FL/HW/TC):\n\n* Assign a `triage_score` between 0.0 and 1.0 representing the risk of unusually high recorded impact in the next 1-6 months.\n* Optionally assign a triage `tier` in [\"quiet\",\"watchlist\",\"priority\"].\n* You are providing a coarse risk score.\n* Provide a `regime_change` object describing out-of-pattern / base-rate break risk in the next 1-6 months.\n\nFor non-quiet hazards:\n\n* Provide 2–4 key `drivers` that push risk up or down, citing evidence pack signals.\n* Provide 0–3 plausible `regime_shifts` (tipping points) with likelihood and timeframe.\n* Provide `data_quality` notes about the resolution dataset (ACLED, EM-DAT, IDMC, DTM) and its biases.\n* Optionally provide a short `scenario_stub` (3–4 sentences) describing the situation, humanitarian needs, and operational constraints.\n\nFor quiet hazards:\n\n* Still include the `regime_change` object with low values (e.g., likelihood <= 0.2 and magnitude <= 0.2).\n\nReturn exactly one JSON object matching this schema:\n{{\n  \"country\": \"{iso3}\",\n  \"hazards\": {{\n    \"ACE\": {{\n      \"triage_score\": 0.0,\n      \"tier\": \"quiet|watchlist|priority\",\n      \"regime_change\": {{\n        \"likelihood\": 0.0,\n        \"direction\": \"up|down|mixed|unclear\",\n        \"magnitude\": 0.0,\n        \"window\": \"month_1|month_2|month_3|month_4|month_5|month_6|month_1-2|month_3-4|month_5-6\",\n        \"rationale_bullets\": [\"...\"],\n        \"trigger_signals\": [\n          {{\n            \"signal\": \"...\",\n            \"timeframe_months\": 2,\n            \"evidence_refs\": [\"...\"]\n          }}\n        ]\n      }},\n      \"drivers\": [\"...\"],\n      \"regime_shifts\": [\"...\"],\n      \"data_quality\": {{\n        \"resolution_source\": \"...\",\n        \"reliability\": \"low|medium|high\",\n        \"notes\": \"...\"\n      }},\n      \"scenario_stub\": \"...\"\n    }}\n  }}\n}}\n"""
