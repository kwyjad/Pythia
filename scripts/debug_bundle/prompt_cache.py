# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Did prompt caching actually happen, per phase, provider and model.

The flags were on in the workflow for a month while the batched path had a
hit rate of zero by construction, and in September only Anthropic reported
a single cache read across the whole cycle. Nobody noticed, because the
only evidence either way lived in ``usage_json`` and nothing read it.

The rule this table exists to enforce: a cache flag that is on proves
nothing — only a non-zero ``cache_read_input_tokens`` does. So the flag
states sit in the same rows as the token counts, and a reader can see at a
glance which of the two disagrees with the other.

Dollars saved is computed from the price table's cached rate against the
standard rate for the same model, on the tokens that were actually read
from cache. Where a model has no price entry the saving is blank rather
than zero: unknown and nothing are different answers.
"""

from __future__ import annotations

import os
from typing import Any

CACHE_FLAGS = (
    "PYTHIA_PROMPT_V3_ORDER",
    "PYTHIA_PROMPT_CACHE_ENABLED",
    "PYTHIA_BATCH_PROMPT_CACHE",
    "PYTHIA_BATCH_CACHE_TTL",
    "PYTHIA_ANTHROPIC_CACHE_MIN_CHARS",
)

FIELDNAMES = [
    "phase", "provider", "model_id", "n_calls", "n_batched_calls",
    "prompt_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
    "cached_tokens", "cache_hit_rate_pct", "estimated_saved_usd",
    "prompt_v3_order", "prompt_cache_enabled", "batch_prompt_cache",
    "note",
]


def _price_detail(model_id: str) -> dict[str, float] | None:
    try:
        from forecaster.providers import resolve_price_detail  # noqa: PLC0415

        return resolve_price_detail(model_id)
    except Exception:
        return None


def collect(con, *, predicate: str | None, params: list[Any]) -> list[dict[str, Any]]:
    """Per (phase, provider, model) cache rows. Never raises."""

    flags = {name: os.getenv(name, "<unset>") for name in CACHE_FLAGS}
    if not predicate:
        return []
    try:
        cur = con.execute(
            f"""
            SELECT
                COALESCE(phase,'') AS phase,
                COALESCE(provider,'') AS provider,
                COALESCE(model_id,'') AS model_id,
                COUNT(*) AS n_calls,
                SUM(CASE WHEN json_extract_string(usage_json,'$.service_tier') = 'batch'
                         THEN 1 ELSE 0 END) AS n_batched,
                SUM(COALESCE(TRY_CAST(json_extract_string(usage_json,'$.prompt_tokens') AS BIGINT),0))
                    AS prompt_tokens,
                SUM(COALESCE(TRY_CAST(json_extract_string(usage_json,'$.cache_read_input_tokens') AS BIGINT),0))
                    AS cache_read,
                SUM(COALESCE(TRY_CAST(json_extract_string(usage_json,'$.cache_creation_input_tokens') AS BIGINT),0))
                    AS cache_creation,
                SUM(COALESCE(TRY_CAST(json_extract_string(usage_json,'$.cached_tokens') AS BIGINT),0))
                    AS cached_tokens
            FROM llm_calls
            WHERE {predicate}
            GROUP BY 1,2,3
            ORDER BY prompt_tokens DESC
            """,
            params,
        )
    except Exception:
        return []
    cols = [d[0] for d in cur.description]
    raw = [dict(zip(cols, r)) for r in cur.fetchall()]

    rows: list[dict[str, Any]] = []
    for r in raw:
        prompt_tokens = int(r["prompt_tokens"] or 0)
        cache_read = int(r["cache_read"] or 0)
        cached = int(r["cached_tokens"] or 0)
        hit_tokens = cache_read + cached
        model_id = str(r["model_id"])
        prices = _price_detail(model_id)
        saved: Any = ""
        if prices:
            delta_per_token = (prices["input"] - prices["cached_input"]) / 1_000_000.0
            saved = round(hit_tokens * delta_per_token, 6)
        note = ""
        if str(r["provider"]).lower() == "google" and int(r["n_batched"] or 0):
            # Not a defect: Gemini batch mode is not eligible for the
            # implicit cache, so zero reads there is the provider's design.
            note = "Gemini batch mode is not eligible for implicit caching — zero reads expected"
        elif hit_tokens == 0 and prompt_tokens:
            note = "no cache reads — a flag being on is not evidence the cache worked"
        rows.append(
            {
                "phase": r["phase"],
                "provider": r["provider"],
                "model_id": model_id,
                "n_calls": int(r["n_calls"] or 0),
                "n_batched_calls": int(r["n_batched"] or 0),
                "prompt_tokens": prompt_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": int(r["cache_creation"] or 0),
                "cached_tokens": cached,
                "cache_hit_rate_pct": (
                    round(100.0 * hit_tokens / prompt_tokens, 1) if prompt_tokens else 0.0
                ),
                "estimated_saved_usd": saved,
                "prompt_v3_order": flags["PYTHIA_PROMPT_V3_ORDER"],
                "prompt_cache_enabled": flags["PYTHIA_PROMPT_CACHE_ENABLED"],
                "batch_prompt_cache": flags["PYTHIA_BATCH_PROMPT_CACHE"],
                "note": note,
            }
        )
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The two numbers the executive summary prints."""

    prompt_tokens = sum(int(r["prompt_tokens"]) for r in rows)
    hit = sum(int(r["cache_read_input_tokens"]) + int(r["cached_tokens"]) for r in rows)
    saved = sum(float(r["estimated_saved_usd"] or 0.0) for r in rows)
    by_provider: dict[str, dict[str, int]] = {}
    for r in rows:
        entry = by_provider.setdefault(str(r["provider"]), {"prompt_tokens": 0, "hit_tokens": 0})
        entry["prompt_tokens"] += int(r["prompt_tokens"])
        entry["hit_tokens"] += int(r["cache_read_input_tokens"]) + int(r["cached_tokens"])
    return {
        "prompt_tokens": prompt_tokens,
        "cache_hit_tokens": hit,
        "cache_hit_rate_pct": round(100.0 * hit / prompt_tokens, 1) if prompt_tokens else 0.0,
        "estimated_saved_usd": round(saved, 4),
        "by_provider": by_provider,
        "flags": {name: os.getenv(name, "<unset>") for name in CACHE_FLAGS},
    }
