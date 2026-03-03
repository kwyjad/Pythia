# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Query generation for prediction market search.

Generates search queries from Fred question metadata using either an LLM
(primary) or keyword-based rules (fallback).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pythia.prediction_markets.config import get_query_generation_config

logger = logging.getLogger(__name__)

# Hazard code → human-readable labels for keyword fallback
_HAZARD_LABELS: dict[str, str] = {
    "AC": "armed conflict",
    "ACE": "armed conflict",
    "DR": "drought",
    "FL": "flood",
    "TC": "tropical cyclone",
    "HW": "heatwave",
    "DI": "displacement",
    "CU": "civil unrest",
    "EC": "economic crisis",
    "PHE": "public health emergency",
}

# Hazard-specific keyword expansions for fallback
_HAZARD_EXPANSIONS: dict[str, list[str]] = {
    "AC": ["war conflict", "military", "ceasefire peace"],
    "ACE": ["war conflict", "military", "ceasefire peace"],
    "DR": ["drought famine", "food crisis", "food insecurity"],
    "FL": ["flood disaster", "flooding"],
    "TC": ["hurricane typhoon cyclone", "extreme weather"],
    "HW": ["heatwave extreme heat", "extreme weather"],
    "DI": ["displacement refugees", "migration crisis"],
    "CU": ["protests unrest", "political crisis"],
    "EC": ["economic collapse", "financial crisis currency"],
    "PHE": ["epidemic outbreak", "health emergency disease"],
}


def _build_llm_prompt(
    question_text: str,
    country_name: str,
    iso3: str,
    hazard_code: str,
    hazard_name: str,
    forecast_start: str,
    forecast_end: str,
) -> str:
    return (
        "You are helping find prediction market questions relevant to a humanitarian forecast.\n\n"
        f"Fred question: \"{question_text}\"\n"
        f"Country: {country_name} ({iso3})\n"
        f"Hazard type: {hazard_code} ({hazard_name})\n"
        f"Forecast period: {forecast_start} to {forecast_end}\n\n"
        "Generate 3-5 short search queries (2-6 words each) that would find related prediction "
        "market questions on platforms like Metaculus, Polymarket, and Manifold.\n\n"
        "Focus on:\n"
        "- Direct event questions (e.g., \"Iran Israel war 2026\")\n"
        "- Leadership/stability questions (e.g., \"Khamenei supreme leader\")\n"
        "- Causal driver questions (e.g., \"Iran nuclear weapon\", \"US sanctions Iran\")\n"
        "- Consequence/outcome questions (e.g., \"Iran ceasefire\", \"Iran humanitarian crisis\")\n"
        "- For natural hazards: climate/weather questions (e.g., \"El Nino 2026\", \"drought East Africa\")\n\n"
        "Return ONLY a JSON array of query strings. No explanations."
    )


def _parse_llm_response(text: str) -> list[str] | None:
    """Parse LLM response to extract query list."""
    text = text.strip()
    # Try to find JSON array in the response
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [str(q).strip() for q in parsed if str(q).strip()]
        except (json.JSONDecodeError, TypeError):
            pass
    return None


async def generate_queries_llm(
    question_text: str,
    country_name: str,
    iso3: str,
    hazard_code: str,
    hazard_name: str,
    forecast_start: str,
    forecast_end: str,
    run_id: str | None = None,
) -> list[str] | None:
    """Generate search queries using an LLM call.

    Returns None if the LLM call fails (caller should use keyword fallback).
    """
    cfg = get_query_generation_config()
    model_id = cfg.get("model", "gemini-3-flash-preview")
    max_queries = int(cfg.get("max_queries", 5))

    prompt = _build_llm_prompt(
        question_text, country_name, iso3, hazard_code, hazard_name,
        forecast_start, forecast_end,
    )

    try:
        from forecaster.providers import ModelSpec, call_chat_ms

        spec = ModelSpec(
            name="PM-QueryGen",
            provider="google",
            model_id=model_id,
            active=True,
        )
        text, usage, error = await call_chat_ms(
            spec,
            prompt,
            temperature=0.3,
            prompt_key="prediction_markets.query_gen",
            prompt_version="1.0.0",
            component="prediction_markets",
            run_id=run_id,
        )
        if error:
            logger.warning("Query generation LLM error: %s", error)
            return None
        queries = _parse_llm_response(text)
        if queries:
            return queries[:max_queries]
        logger.warning("Query generation LLM returned unparseable response")
        return None
    except Exception as exc:
        logger.warning("Query generation LLM call failed: %s", exc)
        return None


def generate_queries_keyword(
    country_name: str,
    iso3: str,
    hazard_code: str,
    forecast_end: str,
) -> list[str]:
    """Generate search queries using keyword-based rules (no LLM).

    Always returns at least one query.
    """
    hz = hazard_code.upper()
    label = _HAZARD_LABELS.get(hz, hazard_code.lower())
    year = forecast_end[:4] if forecast_end else ""

    queries = [f"{country_name} {label}"]

    expansions = _HAZARD_EXPANSIONS.get(hz, [])
    for exp in expansions:
        queries.append(f"{country_name} {exp}")

    # Add year if available
    if year:
        queries = [f"{q} {year}" for q in queries]

    return queries


async def generate_queries(
    question_text: str,
    country_name: str,
    iso3: str,
    hazard_code: str,
    hazard_name: str,
    forecast_start: str,
    forecast_end: str,
    run_id: str | None = None,
) -> list[str]:
    """Generate search queries, trying LLM first with keyword fallback.

    Always returns at least one query.
    """
    # Try LLM-based generation
    queries = await generate_queries_llm(
        question_text, country_name, iso3, hazard_code, hazard_name,
        forecast_start, forecast_end, run_id=run_id,
    )
    if queries:
        return queries

    # Fallback to keyword-based
    logger.info("Using keyword fallback for query generation")
    return generate_queries_keyword(country_name, iso3, hazard_code, forecast_end)
