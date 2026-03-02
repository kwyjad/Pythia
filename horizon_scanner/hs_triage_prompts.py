# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Per-hazard Horizon Scanner triage prompt builders.

Triage is the SECOND step in the HS pipeline, after RC. It answers:
"What is the overall humanitarian risk level for this hazard in this
country over the next 1–6 months?"

This is fundamentally different from RC:
- RC asks: "Is the PATTERN changing?"
- Triage asks: "How BAD is it (or will it be), whether or not the
  pattern is changing?"

A country with severe ongoing conflict and stable trends should have:
- HIGH triage_score (the situation is bad)
- LOW RC likelihood (the pattern isn't changing)

A country at peace with emerging instability should have:
- MODERATE triage_score (not yet severe)
- HIGH RC likelihood (the pattern is breaking)

Pipeline:
    1. RC grounding → RC prompt → RC result (per hazard)
    2. Seasonal/ACLED filtering
    3. Triage grounding → Triage prompt → Triage result (per hazard)
       RC result is injected as structured context into the triage prompt.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Shared triage output schema
# ---------------------------------------------------------------------------

_TRIAGE_SCHEMA = {
    "triage_score": 0.0,
    "tier": "quiet|priority",
    "drivers": ["..."],
    "data_quality": {
        "resolution_source": "...",
        "reliability": "low|medium|high",
        "notes": "...",
    },
    "scenario_stub": "...",
    "confidence_note": "...",
}


# ---------------------------------------------------------------------------
# Shared triage preamble
# ---------------------------------------------------------------------------

_TRIAGE_PREAMBLE = """You are a humanitarian risk analyst assessing {country_name} ({iso3}).
Your task: estimate the OVERALL HUMANITARIAN RISK LEVEL for {hazard_name} \
({hazard_code}) over the next 1–6 months.

This is a triage assessment, not a precise forecast. You are producing a \
coarse risk score that determines how much analytical attention this \
hazard-country pair receives downstream.

CRITICAL DISTINCTION — triage_score vs regime_change:
- triage_score captures the OVERALL risk level, including ongoing/chronic \
situations. Severe but steady conflict = HIGH triage_score.
- regime_change (provided below as context from a prior assessment) captures \
ONLY whether the pattern is breaking. Severe but steady conflict = LOW RC.
- These two scores are PARTIALLY INDEPENDENT. Do not simply copy RC into \
triage_score. A high RC should nudge triage_score upward (emerging risk), \
but a low RC does NOT mean low triage_score (chronic crises continue).

REGIME CHANGE CONTEXT (from prior RC assessment):
{rc_context}
"""

_TRIAGE_SCORING_RUBRIC = """
SCORING RUBRIC:

triage_score (0.0 to 1.0) — overall risk of significant humanitarian \
impact from this hazard in the next 1–6 months:

  0.00–0.49  QUIET: Low or negligible risk. No active crisis, no strong
             emerging signals. Background structural exposure or seasonal
             patterns may exist, but no active concern warranting
             forecasting resources.
  0.50–1.00  PRIORITY: Significant active concern. Ongoing situation with
             moderate-to-severe impact, strong emerging signals, seasonal
             peak approaching in a highly exposed country, or active
             humanitarian crisis. Scores above 0.85 indicate the most
             severe, large-scale emergencies.

SCORING INPUTS (in priority order):
1. CURRENT SITUATION (from evidence pack): What is happening RIGHT NOW?
   Active crisis, ongoing response, current impact levels. This is the
   strongest driver of triage_score.
2. REGIME CHANGE (from RC context): Is the pattern changing? If RC
   likelihood is high, nudge triage_score upward to account for emerging
   risk that may not yet be visible in current impact data.
3. SEASONAL CONTEXT: Is the country entering a high-risk season? Seasonal
   peaks elevate risk even without current signals.
4. STRUCTURAL EXPOSURE (from resolver features): Historical frequency
   and severity of this hazard in this country. Use as a floor — a
   country with a high base rate should not score very low unless there
   is strong evidence of unusually benign conditions.
5. VULNERABILITY AND CAPACITY: Coping capacity, humanitarian access,
   government response capacity. These amplify or attenuate risk.

IMPORTANT CALIBRATION RULES:
- Score the NEXT 6 MONTHS, not the current moment alone.
- For chronic crises (steady conflict, protracted displacement), the
  triage_score should reflect the ongoing severity, not just whether
  things are getting worse.
- For seasonal hazards, score relative to the upcoming season's expected
  severity, informed by climate forecasts and historical patterns.
- If evidence is thin (few sources, no recent signals), state this in
  confidence_note and score conservatively (closer to the structural
  base rate rather than inflating based on absence of information).
"""

_TRIAGE_OUTPUT_INSTRUCTIONS = """
OUTPUT: Return a single JSON object only. No prose. No markdown fences.
Match this schema exactly:

{schema}

Field rules:
- triage_score (0.0–1.0): overall risk score per rubric above.
- tier: one of "quiet", "priority". Derived from triage_score:
  quiet < 0.50, priority >= 0.50. Include for
  interpretability but score-first — decide triage_score then assign tier.
- drivers: 2–5 key factors pushing risk up or down, with evidence references.
  For quiet hazards, 1–2 drivers explaining why risk is low is sufficient.
- data_quality: note the primary resolution data source (ACLED, EM-DAT,
  IDMC, etc.), its reliability for this country, and any known biases or
  gaps.
- scenario_stub: For priority only. 2–4 sentences describing the
  plausible scenario over the next 6 months, humanitarian needs, and
  operational constraints. Omit or set to "" for quiet hazards.
- confidence_note: one sentence on how confident you are and what data
  gaps limit your assessment.
"""


# ---------------------------------------------------------------------------
# ACE — Armed Conflict Events
# ---------------------------------------------------------------------------

def build_triage_prompt_ace(
    country_name: str,
    iso3: str,
    resolver_features: Dict[str, Any],
    rc_result: Optional[Dict[str, Any]] = None,
    evidence_pack: Optional[Dict[str, Any]] = None,
    acled_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Build triage prompt for Armed Conflict Events (ACE).

    ACE triage is about the OVERALL conflict risk level, not just whether
    conflict is changing. A country with 5,000 conflict fatalities in the
    last year and a stable trend has a high triage_score even if RC is low.

    Parameters
    ----------
    rc_result : dict from the prior RC assessment for this hazard-country
    acled_summary : dict with trailing fatalities, events, trend data
    """

    evidence_text = _format_evidence(evidence_pack)
    rc_text = _format_rc_context(rc_result)
    acled_text = _format_acled_summary(acled_summary)
    resolver_text = json.dumps(resolver_features, indent=2, default=str)
    schema_text = json.dumps(_TRIAGE_SCHEMA, indent=2)

    preamble = _TRIAGE_PREAMBLE.format(
        country_name=country_name,
        iso3=iso3,
        hazard_name="Armed Conflict Events",
        hazard_code="ACE",
        rc_context=rc_text,
    )

    return f"""{preamble}

=== ACE-SPECIFIC TRIAGE GUIDANCE ===

ACLED BASE RATE DATA:
{acled_text}

RESOLVER FEATURES (historical context):
{resolver_text}

EVIDENCE PACK (current situation — from triage-focused web research):
{evidence_text}

ACE TRIAGE SCORING ANCHORS:

Use ACLED trailing-12-month fatalities as the primary quantitative anchor:

  0 fatalities/year, no armed groups active:         → 0.02–0.10
  1–99 fatalities/year, localized violence:           → 0.10–0.25
  100–499 fatalities/year, low-intensity conflict:    → 0.25–0.45
  500–1,999 fatalities/year, active conflict:         → 0.45–0.65
  2,000–9,999 fatalities/year, major conflict:        → 0.65–0.85
  10,000+ fatalities/year, severe/catastrophic:       → 0.85–1.00

These are starting points. Adjust based on:
- TREND: If fatalities are trending up, shift score upward within the
  band. If trending down, shift downward.
- RC RESULT: If RC likelihood > 0.30, add 0.05–0.15 to account for
  emerging escalation risk not yet reflected in trailing data.
- GEOGRAPHIC SCOPE: Conflict affecting the whole country vs. confined
  to a remote border region has different humanitarian implications.
- CIVILIAN TARGETING: Conflicts where civilians are the primary targets
  (mass atrocities, ethnic cleansing) warrant higher scores than those
  focused on combatant-vs-combatant.
- HUMANITARIAN ACCESS: Active conflict restricting humanitarian access
  amplifies humanitarian impact and warrants a higher score.
- DISPLACEMENT: If conflict is generating significant displacement
  (captured in DI, but cross-reference here), the humanitarian impact
  is larger than fatalities alone suggest.

WHAT DRIVES ACE TRIAGE SCORE DOWN:
- Active peace process with tangible progress (not just talks).
- Effective ceasefire holding for 3+ months.
- Peacekeeping/stabilization forces deployed and effective.
- Post-conflict transition proceeding normally.

RESOLUTION DATA: ACLED is the primary source. Note ACLED coverage gaps
in data_quality — some countries have systematically lower reporting.
EM-DAT may capture large discrete events that ACLED misses.

=== END ACE-SPECIFIC GUIDANCE ===

{_TRIAGE_SCORING_RUBRIC}
{_TRIAGE_OUTPUT_INSTRUCTIONS.format(schema=schema_text)}"""


# ---------------------------------------------------------------------------
# DR — Drought
# ---------------------------------------------------------------------------

def build_triage_prompt_dr(
    country_name: str,
    iso3: str,
    resolver_features: Dict[str, Any],
    rc_result: Optional[Dict[str, Any]] = None,
    evidence_pack: Optional[Dict[str, Any]] = None,
    climate_data: Optional[Dict[str, Any]] = None,
    season_context: Optional[str] = None,
) -> str:
    """Build triage prompt for Drought (DR).

    DR triage assesses overall drought-related humanitarian risk. This
    includes ongoing drought conditions, food insecurity from prior
    droughts, and emerging drought risk from RC and seasonal forecasts.
    """

    evidence_text = _format_evidence(evidence_pack)
    rc_text = _format_rc_context(rc_result)
    climate_text = _format_climate_data(climate_data)
    resolver_text = json.dumps(resolver_features, indent=2, default=str)
    schema_text = json.dumps(_TRIAGE_SCHEMA, indent=2)
    season_block = f"\nCurrent season context: {season_context}\n" if season_context else ""

    preamble = _TRIAGE_PREAMBLE.format(
        country_name=country_name,
        iso3=iso3,
        hazard_name="Drought",
        hazard_code="DR",
        rc_context=rc_text,
    )

    return f"""{preamble}

=== DR-SPECIFIC TRIAGE GUIDANCE ===

CLIMATE DATA (structured observations and forecasts):
{climate_text}
{season_block}
RESOLVER FEATURES (historical context):
{resolver_text}

EVIDENCE PACK (current situation — from triage-focused web research):
{evidence_text}

DR TRIAGE SCORING ANCHORS:

Use the combination of current food security status (IPC/FEWS NET), climate
conditions, and seasonal forecast as primary anchors:

  No drought, normal season, no food insecurity:          → 0.02–0.10
  Below-average rainfall but no crisis yet:               → 0.10–0.25
  Drought developing, IPC Phase 2 widespread:             → 0.25–0.45
  Active drought, IPC Phase 3 in parts of country:        → 0.45–0.65
  Severe drought, IPC Phase 3–4 widespread, aid needed:   → 0.65–0.85
  Catastrophic drought, IPC Phase 4–5, famine risk:       → 0.85–1.00

Adjust based on:
- SEASONAL TIMING: If the country is about to enter its growing season
  and forecasts are poor, score should reflect the anticipated impact,
  not just the current state.
- CONSECUTIVE FAILURES: Multi-season drought compounds impact. A second
  consecutive poor season warrants a significantly higher score than
  a single below-average season.
- RC RESULT: If RC indicates drought conditions are breaking from the
  historical pattern (e.g., unprecedented rainfall deficit), add
  0.05–0.15 to reflect emerging risk.
- COPING CAPACITY: Countries with strong social safety nets and food
  reserves can absorb moderate drought. Countries where households are
  already stressed from prior shocks are more vulnerable.
- MARKET ACCESS: Drought in a country with good import capacity and
  market function has lower humanitarian impact than drought in a
  landlocked country with poor infrastructure.

WHAT DRIVES DR TRIAGE SCORE DOWN:
- Above-normal rainfall forecast for the upcoming season.
- Good prior harvest providing food reserves.
- Government drought preparedness programs active.
- La Niña/El Niño conditions favorable for this region.

RESOLUTION DATA: EM-DAT for large drought events. FEWS NET IPC phases
are the gold standard for food security status. Note in data_quality
if IPC/FEWS NET coverage is absent for this country.

=== END DR-SPECIFIC GUIDANCE ===

{_TRIAGE_SCORING_RUBRIC}
{_TRIAGE_OUTPUT_INSTRUCTIONS.format(schema=schema_text)}"""


# ---------------------------------------------------------------------------
# FL — Flood
# ---------------------------------------------------------------------------

def build_triage_prompt_fl(
    country_name: str,
    iso3: str,
    resolver_features: Dict[str, Any],
    rc_result: Optional[Dict[str, Any]] = None,
    evidence_pack: Optional[Dict[str, Any]] = None,
    climate_data: Optional[Dict[str, Any]] = None,
    season_context: Optional[str] = None,
) -> str:
    """Build triage prompt for Flood (FL)."""

    evidence_text = _format_evidence(evidence_pack)
    rc_text = _format_rc_context(rc_result)
    climate_text = _format_climate_data(climate_data)
    resolver_text = json.dumps(resolver_features, indent=2, default=str)
    schema_text = json.dumps(_TRIAGE_SCHEMA, indent=2)
    season_block = f"\nCurrent season context: {season_context}\n" if season_context else ""

    preamble = _TRIAGE_PREAMBLE.format(
        country_name=country_name,
        iso3=iso3,
        hazard_name="Flood",
        hazard_code="FL",
        rc_context=rc_text,
    )

    return f"""{preamble}

=== FL-SPECIFIC TRIAGE GUIDANCE ===

CLIMATE DATA (structured observations and forecasts):
{climate_text}
{season_block}
RESOLVER FEATURES (historical context):
{resolver_text}

EVIDENCE PACK (current situation — from triage-focused web research):
{evidence_text}

FL TRIAGE SCORING ANCHORS:

Use the combination of current flood activity, seasonal position, and
forward-looking climate signals:

  Dry season, no flood risk, out of season:               → 0.02–0.08
  Pre-season, normal forecast, standard preparedness:     → 0.08–0.20
  Rainy season active, no unusual flooding:               → 0.15–0.30
  Active flooding, moderate displacement (<50k):          → 0.30–0.50
  Major flooding, significant displacement (50k–500k):    → 0.50–0.70
  Severe/catastrophic flooding, mass displacement:        → 0.70–0.90
  Unprecedented flooding, national emergency:             → 0.90–1.00

Adjust based on:
- SEASONAL POSITION: Score should reflect where the country is in its
  flood season. Pre-season with above-normal rainfall forecast warrants
  a higher score than pre-season with normal forecast.
- GloFAS SIGNALS: If GloFAS shows elevated flood probability on major
  river systems, this is a concrete forward-looking signal.
- COMPOUND RISK: Flooding on top of conflict (limiting response capacity)
  or flooding destroying crops mid-growing-season (amplifying food
  insecurity) warrants a higher score.
- RC RESULT: If RC indicates flood conditions are departing from the
  historical pattern, add 0.05–0.15.
- URBAN vs. RURAL: Urban flooding affects more people per event.
  Riverine flooding of agricultural land has longer-term food security
  implications.
- INFRASTRUCTURE: Dam conditions, levee integrity, drainage systems.
  Known infrastructure weaknesses amplify risk.

WHAT DRIVES FL TRIAGE SCORE DOWN:
- Dry season with months before flood season onset.
- Below-normal or normal rainfall forecast for the upcoming season.
- Effective flood management infrastructure.
- Recent investment in flood preparedness.

RESOLUTION DATA: EM-DAT is the primary source for historical flood
events. GloFAS provides real-time and forecast flood signals. Note in
data_quality if the country has limited hydrological monitoring.

=== END FL-SPECIFIC GUIDANCE ===

{_TRIAGE_SCORING_RUBRIC}
{_TRIAGE_OUTPUT_INSTRUCTIONS.format(schema=schema_text)}"""


# ---------------------------------------------------------------------------
# HW — Heatwave
# ---------------------------------------------------------------------------

def build_triage_prompt_hw(
    country_name: str,
    iso3: str,
    resolver_features: Dict[str, Any],
    rc_result: Optional[Dict[str, Any]] = None,
    evidence_pack: Optional[Dict[str, Any]] = None,
    climate_data: Optional[Dict[str, Any]] = None,
    season_context: Optional[str] = None,
) -> str:
    """Build triage prompt for Heatwave (HW)."""

    evidence_text = _format_evidence(evidence_pack)
    rc_text = _format_rc_context(rc_result)
    climate_text = _format_climate_data(climate_data)
    resolver_text = json.dumps(resolver_features, indent=2, default=str)
    schema_text = json.dumps(_TRIAGE_SCHEMA, indent=2)
    season_block = f"\nCurrent season context: {season_context}\n" if season_context else ""

    preamble = _TRIAGE_PREAMBLE.format(
        country_name=country_name,
        iso3=iso3,
        hazard_name="Heatwave",
        hazard_code="HW",
        rc_context=rc_text,
    )

    return f"""{preamble}

=== HW-SPECIFIC TRIAGE GUIDANCE ===

CLIMATE DATA (structured observations and forecasts):
{climate_text}
{season_block}
RESOLVER FEATURES (historical context):
{resolver_text}

EVIDENCE PACK (current situation — from triage-focused web research):
{evidence_text}

HW TRIAGE SCORING ANCHORS:

Heatwave humanitarian impact depends on the interaction between temperature
anomaly and vulnerability context. Use both:

  Out of hot season, no heat risk:                        → 0.02–0.08
  Approaching hot season, normal forecast:                → 0.08–0.15
  Hot season active, temperatures within historical norms: → 0.10–0.25
  Active heatwave, temperatures above norms, some impact: → 0.25–0.45
  Severe heatwave, significant health/infrastructure impact: → 0.45–0.65
  Extreme/record heatwave, mass health emergency:         → 0.65–0.85
  Unprecedented heat, critical infrastructure collapse:   → 0.85–1.00

Adjust based on:
- VULNERABILITY AMPLIFIERS: These can shift the score upward significantly:
  * Large outdoor labor force (agriculture, construction)
  * Unreliable power grid (limiting cooling access)
  * Urban heat island effect in densely populated cities
  * Limited water supply for cooling and hydration
  * Elderly/very young population without climate-controlled shelter
  * Conflict or displacement limiting coping mechanisms
- SEASONAL FORECAST: If IRI or CPC forecasts above-normal temperatures
  for the country's hot season, this is a concrete forward-looking signal.
- DURATION: Multi-week sustained heat is much more dangerous than brief
  spikes. Warm overnight temperatures preventing nighttime recovery
  amplify health impacts significantly.
- RC RESULT: If RC indicates heat conditions are anomalous, add 0.05–0.10.
- COMPOUND HAZARDS: Heat + drought (water stress), heat + power outages,
  heat + conflict (limited response capacity) all amplify risk.

WHAT DRIVES HW TRIAGE SCORE DOWN:
- Country not in or approaching its hot season.
- Below-normal or normal temperature forecast.
- Strong climate adaptation infrastructure (cooling centers, grid
  reliability, public health heat action plans).
- Temperate or cool climate with low historical heatwave impact.

IMPORTANT NOTE ON HW DATA QUALITY:
Heatwave is the LEAST well-documented hazard in humanitarian databases.
EM-DAT significantly undercounts heatwave events and mortality, especially
in low-income countries. The absence of heatwave records in EM-DAT does
NOT mean the country does not face heatwave risk. Use climate data and
vulnerability context to assess risk independently of historical event
records. State this data gap clearly in data_quality.

=== END HW-SPECIFIC GUIDANCE ===

{_TRIAGE_SCORING_RUBRIC}
{_TRIAGE_OUTPUT_INSTRUCTIONS.format(schema=schema_text)}"""


# ---------------------------------------------------------------------------
# TC — Tropical Cyclone
# ---------------------------------------------------------------------------

def build_triage_prompt_tc(
    country_name: str,
    iso3: str,
    resolver_features: Dict[str, Any],
    rc_result: Optional[Dict[str, Any]] = None,
    evidence_pack: Optional[Dict[str, Any]] = None,
    climate_data: Optional[Dict[str, Any]] = None,
    season_context: Optional[str] = None,
) -> str:
    """Build triage prompt for Tropical Cyclone (TC)."""

    evidence_text = _format_evidence(evidence_pack)
    rc_text = _format_rc_context(rc_result)
    climate_text = _format_climate_data(climate_data)
    resolver_text = json.dumps(resolver_features, indent=2, default=str)
    schema_text = json.dumps(_TRIAGE_SCHEMA, indent=2)
    season_block = f"\nCurrent season context: {season_context}\n" if season_context else ""

    preamble = _TRIAGE_PREAMBLE.format(
        country_name=country_name,
        iso3=iso3,
        hazard_name="Tropical Cyclone",
        hazard_code="TC",
        rc_context=rc_text,
    )

    return f"""{preamble}

=== TC-SPECIFIC TRIAGE GUIDANCE ===

CLIMATE DATA (structured observations and forecasts):
{climate_text}
{season_block}
RESOLVER FEATURES (historical context):
{resolver_text}

EVIDENCE PACK (current situation — from triage-focused web research):
{evidence_text}

TC TRIAGE SCORING ANCHORS:

TC risk is highly seasonal, geographically constrained, and event-driven.
Scoring depends heavily on whether the country is in its TC season and
whether any active storms are threatening.

  Outside TC season, no active storms:                    → 0.02–0.05
  TC season approaching, normal seasonal outlook:         → 0.05–0.15
  TC season active, normal activity, no direct threat:    → 0.10–0.25
  TC season active, above-normal outlook, no direct threat: → 0.20–0.35
  Active storm in basin with possible track to country:   → 0.35–0.55
  Active storm on likely track to country:                → 0.55–0.75
  Active intense storm on direct track, landfall expected: → 0.75–0.95

Adjust based on:
- HISTORICAL LANDFALL FREQUENCY: Countries that get hit regularly (e.g.,
  Philippines, Bangladesh, Madagascar) have higher baseline scores during
  their season than countries that rarely experience direct hits.
- ENSO STATE: Strong ENSO signal modifying expected season activity
  should shift the score. La Niña → more Atlantic storms, fewer East
  Pacific. El Niño → opposite.
- COASTAL VULNERABILITY: Low-lying coastal areas, population density
  in coastal zones, quality of building construction, early warning
  system effectiveness.
- COMPOUND RISK: TC + flooding (almost always), TC + displacement,
  TC hitting an area already affected by prior TC or other disaster.
- RC RESULT: If RC indicates unusual TC risk (anomalous SSTs, unexpected
  storm track), add 0.05–0.15.

TC SCORING IS UNIQUELY TIME-SENSITIVE:
Unlike other hazards, TC triage scores can legitimately swing from 0.05
to 0.80 within days as a storm develops and its track becomes clearer.
Score based on the BEST INFORMATION AVAILABLE AT THE TIME OF ASSESSMENT.
If a storm is forming but track is uncertain, reflect that uncertainty
in a moderate score rather than committing to high or low.

WHAT DRIVES TC TRIAGE SCORE DOWN:
- Country is outside its TC season.
- Below-normal seasonal outlook for the relevant basin.
- No active storms in the basin.
- Country is geographically sheltered (e.g., inland, lee side of
  mountain range).

RESOLUTION DATA: EM-DAT for historical TC impact. IBTrACS for historical
TC tracks and frequency. Note if the country has limited historical TC
records in EM-DAT (small island states may have sparse data).

=== END TC-SPECIFIC GUIDANCE ===

{_TRIAGE_SCORING_RUBRIC}
{_TRIAGE_OUTPUT_INSTRUCTIONS.format(schema=schema_text)}"""


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

TRIAGE_PROMPT_BUILDERS = {
    "ACE": build_triage_prompt_ace,
    "DR": build_triage_prompt_dr,
    "FL": build_triage_prompt_fl,
    "HW": build_triage_prompt_hw,
    "TC": build_triage_prompt_tc,
}


def build_triage_prompt(
    hazard_code: str,
    country_name: str,
    iso3: str,
    resolver_features: Dict[str, Any],
    rc_result: Optional[Dict[str, Any]] = None,
    evidence_pack: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> str:
    """Dispatch to the appropriate per-hazard triage prompt builder.

    Parameters
    ----------
    hazard_code : one of ACE, DR, FL, HW, TC
    country_name : full country name
    iso3 : ISO 3166-1 alpha-3 code
    resolver_features : historical base rate data from resolver
    rc_result : dict from the prior RC assessment (injected as context)
    evidence_pack : dict from triage-focused grounding call
    **kwargs : passed to hazard-specific builder (e.g., acled_summary,
        climate_data, season_context)
    """
    builder = TRIAGE_PROMPT_BUILDERS.get(hazard_code)
    if builder is None:
        raise ValueError(f"No triage prompt builder for hazard code: {hazard_code}")
    return builder(
        country_name=country_name,
        iso3=iso3,
        resolver_features=resolver_features,
        rc_result=rc_result,
        evidence_pack=evidence_pack,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_evidence(evidence_pack: Optional[Dict[str, Any]]) -> str:
    if not evidence_pack:
        return "Evidence pack unavailable (web research disabled or failed)."
    md = evidence_pack.get("markdown") or ""
    if not md.strip():
        return "Evidence pack was empty — no recent signals found."
    return md


def _format_rc_context(rc_result: Optional[Dict[str, Any]]) -> str:
    """Format the RC result as structured context for the triage prompt."""
    if not rc_result:
        return (
            "RC assessment unavailable. Score triage based on evidence pack "
            "and resolver features only. Be slightly more conservative without "
            "RC context — you may be missing emerging signals."
        )

    lk = rc_result.get("likelihood", 0.0)
    mag = rc_result.get("magnitude", 0.0)
    direction = rc_result.get("direction", "unclear")
    window = rc_result.get("window", "unclear")
    bullets = rc_result.get("rationale_bullets", [])
    confidence = rc_result.get("confidence_note", "")

    lines = [
        f"  RC likelihood: {lk:.2f}",
        f"  RC magnitude:  {mag:.2f}",
        f"  RC direction:  {direction}",
        f"  RC window:     {window}",
    ]
    if bullets:
        lines.append("  RC rationale:")
        for b in bullets[:4]:
            lines.append(f"    - {b}")
    if confidence:
        lines.append(f"  RC confidence: {confidence}")

    # Interpretation guidance
    if lk <= 0.10:
        lines.append("  → Pattern is stable. Focus triage on current/chronic conditions.")
    elif lk <= 0.30:
        lines.append("  → Some emerging signals. Factor into triage but don't overweight.")
    elif lk <= 0.55:
        lines.append("  → Significant emerging risk. Triage should reflect this — add 0.05–0.15 to what current conditions alone would suggest.")
    else:
        lines.append("  → Strong regime change signal. Triage should substantially account for imminent pattern break.")

    return "\n".join(lines)


def _format_acled_summary(acled_summary: Optional[Dict[str, Any]]) -> str:
    if not acled_summary:
        return (
            "ACLED summary unavailable. Assess based on evidence pack and "
            "resolver features only."
        )

    lines = []
    f12 = acled_summary.get("fatalities_trailing_12m")
    f3 = acled_summary.get("fatalities_trailing_3m")
    e12 = acled_summary.get("events_trailing_12m")
    e3 = acled_summary.get("events_trailing_3m")
    trend = acled_summary.get("trend_direction")
    pct = acled_summary.get("trend_pct_change")

    if f12 is not None:
        lines.append(f"- Fatalities (trailing 12 months): {f12:,}")
    if f3 is not None:
        lines.append(f"- Fatalities (trailing 3 months): {f3:,}")
    if e12 is not None:
        lines.append(f"- Events (trailing 12 months): {e12:,}")
    if e3 is not None:
        lines.append(f"- Events (trailing 3 months): {e3:,}")
    if trend:
        trend_str = f"- Trend: {trend}"
        if pct is not None:
            trend_str += f" ({pct:+.0f}% change, last 3m vs prior 3m)"
        lines.append(trend_str)

    top_types = acled_summary.get("top_event_types")
    if top_types:
        types_str = ", ".join(f"{t} ({c})" for t, c in top_types[:4])
        lines.append(f"- Top event types (12m): {types_str}")

    return "\n".join(lines) if lines else "ACLED data present but empty."


def _format_climate_data(climate_data: Optional[Dict[str, Any]]) -> str:
    if not climate_data:
        return "Structured climate data unavailable."

    lines = []
    for key, val in climate_data.items():
        if val is not None:
            label = key.replace("_", " ").capitalize()
            lines.append(f"- {label}: {val}")

    return "\n".join(lines) if lines else "Climate data present but empty."
