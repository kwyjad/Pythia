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
) -> str:
    """Build the Horizon Scanner v2 triage prompt.

    The prompt is intentionally structured so the model returns a single JSON
    payload with tiers, triage scores, and optional qualitative context per
    hazard. No forecasts are requested at this stage.
    """

    model_info = model_info or {}

    return f"""You are a strategic humanitarian risk analyst.\nYou are assessing {country_name} ({iso3}) for the next 6–12 months.\n\nHazards and their meanings:\n{json.dumps(hazard_catalog, indent=2)}\n\nResolver features (noisy, imperfect data from ACLED/IDMC/EM-DAT; see notes in each entry):\n\n{json.dumps(resolver_features, indent=2)}\n\nModel/data notes JSON:\n{json.dumps(model_info, indent=2)}\n\nYour task is to triage, not to forecast exact numbers.\n\nFor each hazard:\n\n* Assign a triage `tier` in [\"quiet\",\"watchlist\",\"priority\"].\n* Assign a `triage_score` between 0.0 and 1.0 representing the risk of unusually high recorded impact in the next 6–12 months.\n* Do NOT try to compute a full SPD; you are providing a coarse risk score.\n\nFor non-quiet hazards:\n\n* Provide 2–4 key `drivers` that push risk up or down.\n* Provide 0–3 plausible `regime_shifts` (tipping points) with likelihood and timeframe.\n* Provide `data_quality` notes about the resolution dataset (ACLED, EM-DAT, IDMC, DTM) and its biases.\n* Optionally provide a short `scenario_stub` (3–4 sentences) describing the situation, humanitarian needs, and operational constraints.\n\nReturn a single JSON object only (no code fences, no prose):\n{{\n  \"country\": \"{iso3}\",\n  \"hazards\": {{\n    \"<hazard_code>\": {{\n      \"tier\": \"quiet|watchlist|priority\",\n      \"triage_score\": 0.0,\n      \"drivers\": [\"...\"],\n      \"regime_shifts\": [\"...\"],\n      \"data_quality\": {{\n        \"resolution_source\": \"...\",\n        \"reliability\": \"low|medium|high\",\n        \"notes\": \"...\"\n      }},\n      \"scenario_stub\": \"...\"\n    }}\n  }}\n}}\n\nReturn a single JSON object only. Do not wrap in markdown code fences. Do not include any prose.\n"""
