# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ICG CrisisWatch "On the Horizon" monthly fetch.

Fetches the latest ICG "On the Horizon" feature (published as part of
the monthly CrisisWatch Global Overview). This highlights ~3 conflict
risks and ~1 resolution opportunity expected in the next 3–6 months.

Called ONCE per HS run (not per-country). The result is cached in-memory
for the duration of the run, and injected into RC prompts for countries
that ICG has flagged.

Usage::

    horizon_data = fetch_on_the_horizon()
    if horizon_data:
        flagged = get_horizon_countries(horizon_data)
        # flagged = {"SOM": "ICG 'On the Horizon' ...", "SDN": "..."}
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# In-memory cache for the current HS run.
_HORIZON_CACHE: dict[str, Any] | None = None


def fetch_on_the_horizon(
    year: int | None = None,
) -> dict[str, Any] | None:
    """Fetch the latest ICG "On the Horizon" via Gemini grounding search.

    Returns a dict with:
        - raw_text: the raw Gemini grounding response text
        - countries: list of {name, iso3, risk_type, description}
        - source_url: URL of the CrisisWatch page if found

    Returns None on failure or if no data is found.
    """
    global _HORIZON_CACHE
    if _HORIZON_CACHE is not None:
        return _HORIZON_CACHE

    if year is None:
        year = datetime.now().year

    try:
        from pythia.web_research.backends.gemini_grounding import fetch_via_gemini
    except ImportError:
        log.debug("Gemini grounding unavailable — skipping On the Horizon fetch.")
        return None

    query = f'site:crisisgroup.org "on the horizon" {year}'
    custom_prompt = f"""\
You are a research assistant. Search for the latest ICG CrisisWatch \
"On the Horizon" section for {year}.

"On the Horizon" is a monthly feature by the International Crisis Group \
that highlights ~3 CONFLICT RISKS and ~1 RESOLUTION OPPORTUNITY expected \
to emerge or escalate in the next 3-6 months.

Find the most recent "On the Horizon" and extract:
1. Each country or situation flagged
2. Whether it is flagged as "Conflict Risk" or "Resolution Opportunity"
3. A brief (1-2 sentence) summary of why it was flagged

Return a JSON object with this structure:
{{
  "month": "Month Year",
  "conflict_risks": [
    {{"country": "...", "summary": "..."}}
  ],
  "resolution_opportunities": [
    {{"country": "...", "summary": "..."}}
  ]
}}

If you cannot find any "On the Horizon" content, return: {{"month": "unknown", "conflict_risks": [], "resolution_opportunities": []}}
"""

    try:
        evidence = fetch_via_gemini(
            query=query,
            recency_days=45,
            include_structural=False,
            timeout_sec=30,
            max_results=5,
            custom_prompt=custom_prompt,
        )

        if hasattr(evidence, "to_dict"):
            pack = evidence.to_dict()
        else:
            pack = dict(evidence)

        raw_text = pack.get("markdown") or pack.get("raw_text") or ""

        # Try to parse the JSON from the response
        countries = _parse_horizon_response(raw_text)

        result = {
            "raw_text": raw_text,
            "countries": countries,
        }

        _HORIZON_CACHE = result
        return result

    except Exception as exc:
        log.warning("On the Horizon fetch failed: %s", exc)
        return None


def _parse_horizon_response(text: str) -> list[dict[str, Any]]:
    """Parse the Gemini response to extract country-level flags."""
    countries: list[dict[str, Any]] = []

    # Try to find JSON in the response
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            for entry in data.get("conflict_risks", []):
                countries.append({
                    "name": entry.get("country", ""),
                    "risk_type": "Conflict Risk",
                    "description": entry.get("summary", ""),
                })
            for entry in data.get("resolution_opportunities", []):
                countries.append({
                    "name": entry.get("country", ""),
                    "risk_type": "Resolution Opportunity",
                    "description": entry.get("summary", ""),
                })
        except json.JSONDecodeError:
            pass

    return countries


def get_horizon_countries(
    horizon_data: dict[str, Any] | None,
) -> dict[str, str]:
    """Extract {country_name: formatted_note} for countries in "On the Horizon".

    Returns a dict mapping country names (uppercased) to a formatted string
    suitable for injection into RC prompts. The caller should map country
    names to ISO3 codes using the existing country utilities.
    """
    if not horizon_data:
        return {}

    result: dict[str, str] = {}
    for entry in horizon_data.get("countries", []):
        name = (entry.get("name") or "").strip()
        if not name:
            continue
        risk_type = entry.get("risk_type", "Conflict Risk")
        description = entry.get("description", "")

        note = (
            f'ICG "ON THE HORIZON" FLAG: International Crisis Group has '
            f"flagged {name} as a {risk_type} in their latest monthly "
            f'"On the Horizon" assessment. This is a strong expert signal — '
            f"ICG is very selective about what they flag here.\n"
            f"Context: {description}"
        )
        result[name.upper()] = note

    return result


def clear_cache() -> None:
    """Clear the in-memory cache (for testing or new HS run)."""
    global _HORIZON_CACHE
    _HORIZON_CACHE = None
