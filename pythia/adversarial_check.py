# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Adversarial evidence check for RC Level 1+ cases.

Runs 2-3 targeted web searches for **counter-evidence** — reasons a
predicted regime change might NOT materialise — then synthesises results
via a single LLM call.  Output is stored in ``hs_adversarial_checks``
and formatted for later injection into the SPD prompt.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict

from forecaster.providers import ModelSpec, call_chat_ms
from horizon_scanner._utils import parse_json_response, resolve_hs_model
from horizon_scanner.regime_change import compute_level, compute_score, coerce_regime_change
from pythia.web_research import fetch_evidence_pack

logger = logging.getLogger(__name__)

_ADVERSARIAL_CHECK_TIMEOUT_SEC = 60
_MAX_COUNTER_EVIDENCE = 6
_MAX_HISTORICAL_ANALOGS = 3
_MAX_STABILIZING_FACTORS = 4

# ---------------------------------------------------------------------------
# Adversarial query builder
# ---------------------------------------------------------------------------

_ADVERSARIAL_VOCAB_UP: dict[str, list[str]] = {
    "ACE": [
        "{country} peace talks ceasefire diplomatic resolution {year}",
        "{country} conflict de-escalation stabilization {months}",
    ],
    "FL": [
        "{country} flood forecast revised improved {year}",
        "{country} flood preparedness infrastructure improved",
    ],
    "DR": [
        "{country} drought forecast revised rainfall improvement {year}",
        "{country} drought preparedness water infrastructure improved",
    ],
    "TC": [
        "{country} tropical cyclone forecast revised lower risk {year}",
        "{country} cyclone preparedness early warning system improved",
    ],
    "HW": [
        "{country} heatwave forecast revised cooler outlook {year}",
        "{country} heat preparedness infrastructure cooling improved",
    ],
}

_ADVERSARIAL_VOCAB_DOWN: dict[str, list[str]] = {
    "ACE": [
        "{country} conflict resumption violence renewed {year}",
        "{country} ceasefire violation peace process collapse {months}",
    ],
    "FL": [
        "{country} flood risk warning escalation {year}",
    ],
    "DR": [
        "{country} drought risk warning worsening {year}",
    ],
    "TC": [
        "{country} tropical cyclone risk warning escalation {year}",
    ],
    "HW": [
        "{country} heatwave risk warning escalation {year}",
    ],
}


def _build_adversarial_queries(
    country_name: str,
    iso3: str,
    hazard_code: str,
    rc_result: dict,
) -> list[str]:
    """Build 2-3 adversarial search queries based on hazard and RC direction.

    Parameters
    ----------
    country_name : str
        Human-readable country name.
    iso3 : str
        ISO-3166 alpha-3 code.
    hazard_code : str
        Hazard code (ACE, FL, DR, TC, HW).
    rc_result : dict
        Coerced regime-change dict with ``direction``, ``trigger_signals``, etc.

    Returns
    -------
    list[str]
        2-3 search query strings.
    """
    direction = (rc_result.get("direction") or "unclear").lower()
    country = country_name or iso3
    year = str(datetime.now().year)
    now = datetime.now()
    months = f"{now.strftime('%B')} {year}"

    hz = hazard_code.upper()
    if direction in ("up", "mixed", "unclear"):
        templates = _ADVERSARIAL_VOCAB_UP.get(hz, _ADVERSARIAL_VOCAB_UP.get("ACE", []))
    else:
        templates = _ADVERSARIAL_VOCAB_DOWN.get(hz, _ADVERSARIAL_VOCAB_DOWN.get("ACE", []))

    queries = [
        t.format(country=country, year=year, months=months)
        for t in templates
    ]

    # Add a trigger-specific adversarial query if trigger signals are available
    trigger_signals = rc_result.get("trigger_signals") or []
    if trigger_signals:
        first_trigger = trigger_signals[0]
        signal_text = first_trigger if isinstance(first_trigger, str) else first_trigger.get("signal", "")
        if signal_text:
            # Extract key terms and frame adversarially
            queries.append(
                f"{country} {signal_text} routine not escalation {year}"
            )

    # Truncate all queries to 10 words max — web search engines perform
    # poorly with long natural-language queries.
    queries = [" ".join(q.split()[:10]) for q in queries]

    return queries[:3]


# ---------------------------------------------------------------------------
# LLM synthesis
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT_TEMPLATE = """\
You are a devil's advocate analyst reviewing a regime change assessment.

The RC assessment for {country_name} ({iso3}) — {hazard_code} predicts:
- Direction: {direction} (regime change {direction_label})
- Likelihood: {likelihood}
- Magnitude: {magnitude}
- Window: {window}
- Key triggers: {trigger_bullets}

Your task: Based on the evidence below, identify reasons this regime change
might NOT materialize as predicted. Be specific and cite sources.

Search evidence:
{evidence_text}

Respond in JSON only (no commentary):
{{
  "counter_evidence": [
    {{"claim": "...", "source": "...", "relevance": "...", "strength": "strong|moderate|weak"}}
  ],
  "historical_analogs": [
    {{"analog": "...", "outcome": "...", "relevance": "..."}}
  ],
  "stabilizing_factors": ["...", "..."],
  "net_assessment": "strong_counter|moderate|weak_counter|inconclusive",
  "summary": "One sentence on counter-evidence strength"
}}

Rules:
- Only include counter-evidence that is specific and sourced
- "strong" counter-evidence directly contradicts a named trigger signal
- "moderate" provides context that weakens the RC hypothesis
- "weak" is tangential or speculative
- If you find no meaningful counter-evidence, set net_assessment to "inconclusive"\
  and say so in the summary — do not fabricate counter-evidence
- Historical analogs should be from the same country or closely comparable contexts
- Maximum {max_counter} counter_evidence items, {max_analogs} analogs, {max_factors} stabilizing_factors
"""

_DIRECTION_LABELS = {
    "up": "escalation predicted",
    "down": "de-escalation predicted",
    "mixed": "mixed signals",
    "unclear": "direction unclear",
}


async def _synthesize_counter_evidence(
    country_name: str,
    iso3: str,
    hazard_code: str,
    rc_result: dict,
    evidence_text: str,
    run_id: str,
) -> dict[str, Any]:
    """Call the LLM to synthesise counter-evidence into structured output.

    Parameters
    ----------
    country_name, iso3, hazard_code : str
        Country and hazard identifiers.
    rc_result : dict
        Coerced RC assessment dict.
    evidence_text : str
        Aggregated evidence text from adversarial searches.
    run_id : str
        Current HS run ID.

    Returns
    -------
    dict
        Structured adversarial check result.
    """
    direction = rc_result.get("direction") or "unclear"
    trigger_signals = rc_result.get("trigger_signals") or []
    trigger_bullets = "\n".join(
        f"- {s}" if isinstance(s, str)
        else f"- {s.get('signal', 'unknown')}"
        for s in trigger_signals
    ) or "- (none identified)"

    prompt = _SYNTHESIS_PROMPT_TEMPLATE.format(
        country_name=country_name,
        iso3=iso3,
        hazard_code=hazard_code,
        direction=direction,
        direction_label=_DIRECTION_LABELS.get(direction, direction),
        likelihood=rc_result.get("likelihood", 0.0),
        magnitude=rc_result.get("magnitude", 0.0),
        window=rc_result.get("window", ""),
        trigger_bullets=trigger_bullets,
        evidence_text=evidence_text,
        max_counter=_MAX_COUNTER_EVIDENCE,
        max_analogs=_MAX_HISTORICAL_ANALOGS,
        max_factors=_MAX_STABILIZING_FACTORS,
    )

    model_id = resolve_hs_model()
    spec = ModelSpec(
        name="Gemini",
        provider="google",
        model_id=model_id,
        active=True,
        purpose="hs_adversarial_check",
    )

    text, usage, error = await call_chat_ms(
        spec,
        prompt,
        temperature=0.0,
        prompt_key="hs.adversarial_check",
        prompt_version="1.0.0",
        component="HorizonScanner",
        run_id=run_id,
    )

    if error or not text:
        raise ValueError(f"LLM synthesis failed: {error or 'empty response'}")

    result = parse_json_response(text)

    # Enforce maximums
    result["counter_evidence"] = (result.get("counter_evidence") or [])[:_MAX_COUNTER_EVIDENCE]
    result["historical_analogs"] = (result.get("historical_analogs") or [])[:_MAX_HISTORICAL_ANALOGS]
    result["stabilizing_factors"] = (result.get("stabilizing_factors") or [])[:_MAX_STABILIZING_FACTORS]
    result.setdefault("net_assessment", "inconclusive")
    result.setdefault("summary", "")
    result["model_id"] = model_id

    return result


# ---------------------------------------------------------------------------
# Evidence aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_evidence_text(packs: list[dict]) -> str:
    """Aggregate evidence from multiple packs into a single text block."""
    parts: list[str] = []
    for i, pack in enumerate(packs, 1):
        signals = pack.get("recent_signals") or []
        structural = pack.get("structural_context") or ""
        sources = pack.get("sources") or []

        section = f"--- Search {i} ---\n"
        if structural:
            section += f"Context: {structural}\n"
        if signals:
            section += "Signals:\n"
            for sig in signals:
                section += f"  - {sig}\n"
        if sources:
            section += "Sources:\n"
            for src in sources:
                if isinstance(src, dict):
                    title = src.get("title", "")
                    url = src.get("url", "")
                    section += f"  - {title} ({url})\n"
        if not signals and not structural and sources:
            # When signals/structural are empty but sources exist,
            # include source titles as minimal evidence context
            section += "Source titles (no structured signals extracted):\n"
            for src in sources:
                if isinstance(src, dict):
                    title = src.get("title", "")
                    if title:
                        section += f"  - {title}\n"
        parts.append(section)
    return "\n".join(parts) if parts else "(no search evidence available)"


def _aggregate_sources(packs: list[dict]) -> list[dict]:
    """Collect unique sources from all evidence packs."""
    seen_urls: set[str] = set()
    sources: list[dict] = []
    for pack in packs:
        for src in pack.get("sources") or []:
            if not isinstance(src, dict):
                continue
            url = src.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "title": src.get("title", ""),
                    "url": url,
                    "date": src.get("date", ""),
                })
    return sources


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_adversarial_check(
    iso3: str,
    country_name: str,
    hazard_code: str,
    rc_result: dict,
    run_id: str,
    evidence_pack: dict | None = None,
) -> dict | None:
    """Run adversarial evidence check for a single country-hazard pair.

    Only executes for RC Level >= 1.  Returns ``None`` for lower levels
    or on error.

    Parameters
    ----------
    iso3 : str
        ISO-3166 alpha-3 country code.
    country_name : str
        Human-readable country name.
    hazard_code : str
        Hazard code (ACE, FL, DR, TC, HW).
    rc_result : dict
        Per-hazard RC assessment dict (coerced).
    run_id : str
        Current HS run ID.
    evidence_pack : dict or None
        Optional existing country-level evidence pack (unused, reserved
        for future deduplication).

    Returns
    -------
    dict or None
        Structured adversarial check result, or ``None`` if skipped/failed.
    """
    rc = coerce_regime_change(rc_result)
    likelihood = float(rc.get("likelihood") or 0.0)
    magnitude = float(rc.get("magnitude") or 0.0)
    score = compute_score(likelihood, magnitude)
    level = compute_level(likelihood, magnitude, score)

    if level < 1:
        return None

    iso3_up = (iso3 or "").upper()
    hz = (hazard_code or "").upper()
    start = time.time()

    logger.info(
        "Running adversarial check for %s %s (RC L%d, score=%.3f)",
        iso3_up, hz, level, score,
    )

    # 1. Build adversarial queries
    queries = _build_adversarial_queries(country_name, iso3_up, hz, rc)

    # 2. Run web searches
    retriever_enabled = os.getenv("PYTHIA_RETRIEVER_ENABLED", "0") == "1"
    model_id = (
        (os.getenv("PYTHIA_RETRIEVER_MODEL_ID") or "").strip()
        if retriever_enabled
        else None
    )

    packs: list[dict] = []
    for query in queries:
        try:
            # Adversarial checks always use web search regardless of the
            # deprecated PYTHIA_WEB_RESEARCH_ENABLED flag (which controls
            # the old question-level pipeline).
            _original_web_research = os.environ.get("PYTHIA_WEB_RESEARCH_ENABLED")
            os.environ["PYTHIA_WEB_RESEARCH_ENABLED"] = "1"
            try:
                pack = dict(
                    fetch_evidence_pack(
                        query,
                        purpose="hs_adversarial_check",
                        run_id=run_id,
                        hs_run_id=run_id,
                        model_id=model_id or None,
                    ) or {}
                )
            finally:
                # Restore original value
                if _original_web_research is not None:
                    os.environ["PYTHIA_WEB_RESEARCH_ENABLED"] = _original_web_research
                else:
                    os.environ.pop("PYTHIA_WEB_RESEARCH_ENABLED", None)
            packs.append(pack)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Adversarial search failed for %s %s query=%r: %s",
                iso3_up, hz, query, exc,
            )

    # 3. Aggregate evidence
    evidence_text = _aggregate_evidence_text(packs)
    all_sources = _aggregate_sources(packs)
    any_grounded = any(bool(p.get("grounded")) for p in packs)

    # 4. If no usable evidence, return inconclusive
    has_evidence = any(
        (p.get("recent_signals") or p.get("structural_context") or p.get("sources"))
        for p in packs
    )
    if not has_evidence:
        logger.info(
            "Adversarial check for %s %s: no evidence found, returning inconclusive",
            iso3_up, hz,
        )
        return {
            "counter_evidence": [],
            "historical_analogs": [],
            "stabilizing_factors": [],
            "net_assessment": "inconclusive",
            "summary": "No adversarial evidence found in search results",
            "sources": all_sources,
            "grounded": any_grounded,
            "model_id": "",
        }

    # 5. LLM synthesis
    try:
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                _synthesize_counter_evidence(
                    country_name=country_name,
                    iso3=iso3_up,
                    hazard_code=hz,
                    rc_result=rc,
                    evidence_text=evidence_text,
                    run_id=run_id,
                )
            )
        finally:
            loop.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Adversarial synthesis failed for %s %s: %s", iso3_up, hz, exc,
        )
        return None

    # Attach aggregated sources and grounding flag
    result["sources"] = all_sources
    result["grounded"] = any_grounded

    elapsed = time.time() - start
    logger.info(
        "Adversarial check for %s %s completed in %.1fs: net_assessment=%s",
        iso3_up, hz, elapsed, result.get("net_assessment"),
    )

    return result


# ---------------------------------------------------------------------------
# SPD prompt formatter
# ---------------------------------------------------------------------------

def format_adversarial_check_for_spd(check: dict | None, rc_level: int = 0) -> str:
    """Format adversarial check result for injection into the SPD prompt.

    Parameters
    ----------
    check : dict or None
        Adversarial check result from ``run_adversarial_check``.
    rc_level : int
        RC level for display.

    Returns
    -------
    str
        Formatted markdown block, or empty string if check is None.
    """
    if not check:
        return ""

    lines: list[str] = []
    net = check.get("net_assessment", "inconclusive")
    summary = check.get("summary", "")

    lines.append(f"ADVERSARIAL EVIDENCE CHECK (RC Level {rc_level} — counter-evidence review):")
    lines.append(f"Net assessment: {net}")
    lines.append(f"Summary: {summary}")
    lines.append("")

    counter_evidence = check.get("counter_evidence") or []
    if counter_evidence:
        lines.append("Counter-evidence:")
        for item in counter_evidence:
            strength = item.get("strength", "unknown")
            claim = item.get("claim", "")
            source = item.get("source", "")
            relevance = item.get("relevance", "")
            lines.append(f"- [{strength}] {claim} (Source: {source})")
            if relevance:
                lines.append(f"  → Relevance: {relevance}")
        lines.append("")

    analogs = check.get("historical_analogs") or []
    if analogs:
        lines.append("Historical analogs:")
        for item in analogs:
            analog = item.get("analog", "")
            outcome = item.get("outcome", "")
            lines.append(f"- {analog} → Outcome: {outcome}")
        lines.append("")

    factors = check.get("stabilizing_factors") or []
    if factors:
        lines.append("Stabilizing factors:")
        for factor in factors:
            lines.append(f"- {factor}")
        lines.append("")

    lines.append(
        'INSTRUCTION: Weigh this counter-evidence against the RC assessment and '
        'hazard tail pack. If counter-evidence is "strong_counter", your posterior '
        'should be closer to the base rate than the RC would suggest. If '
        '"inconclusive", the RC assessment stands largely unchallenged.'
    )

    return "\n".join(lines)
