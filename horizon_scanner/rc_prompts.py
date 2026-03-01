# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Per-hazard Regime Change (RC) prompt builders.

Each hazard gets a dedicated RC assessment with:
- Hazard-specific search queries (for the web research step)
- Hazard-specific resolver features (subset of full resolver data)
- Hazard-specific evidence pack (from targeted web search)
- Hazard-specific prompt with tailored calibration anchors
- Hazard-specific examples of what does/doesn't constitute RC

The RC call is the FIRST step in the HS pipeline. It answers a single
question: "Is this hazard breaking from its historical base rate in this
country?" It does NOT assess overall risk level (that's triage).

RC results feed downstream into:
1. Seasonal/ACLED filtering (to decide whether triage is needed)
2. Per-hazard triage prompts (as context)
3. Hazard tail pack generation (for RC Level 2+)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Shared RC schema and calibration preamble
# ---------------------------------------------------------------------------

_RC_SCHEMA = {
    "likelihood": 0.0,
    "direction": "up|down|mixed|unclear",
    "magnitude": 0.0,
    "window": "month_1|month_2|month_3|month_4|month_5|month_6|month_1-2|month_3-4|month_5-6",
    "rationale_bullets": ["..."],
    "trigger_signals": [
        {
            "signal": "...",
            "timeframe_months": 2,
            "evidence_refs": ["..."],
        }
    ],
    "confidence_note": "...",
}

_RC_CALIBRATION_PREAMBLE = """You are assessing REGIME CHANGE for {country_name} ({iso3}) — specifically \
for the hazard: {hazard_name} ({hazard_code}).

Regime change means a DEPARTURE FROM the country's OWN HISTORICAL BASELINE \
for this specific hazard. It is NOT a measure of absolute risk level. \
A country with severe ongoing crisis but stable trends has LOW regime change. \
A country at peace seeing new armed conflict emerge has HIGH regime change.

Your task: estimate the probability and potential magnitude of a break from \
the established pattern in the next 1–6 months.

CRITICAL RULES:
- Default to likelihood 0.05 and magnitude 0.05 unless evidence says otherwise.
- Do NOT assign likelihood > 0.10 without citing a SPECIFIC, RECENT signal \
from the evidence pack that represents a departure from the base rate.
- Do NOT confuse ongoing chronic conditions with regime change.
- Do NOT confuse seasonal patterns with regime change.
- Structural vulnerability alone (poverty, weak governance, climate exposure) \
is NOT regime change — it is background context.

Expected distribution (across ~120 countries):
- ~80%: likelihood <= 0.10 (base-rate normal)
- ~10%: likelihood 0.10–0.30 (watch — some signals but not compelling)
- ~7%: likelihood 0.30–0.55 (emerging — specific evidence of potential change)
- ~3%: likelihood >= 0.55 (strong — clear, concrete evidence of imminent break)
"""

_RC_OUTPUT_INSTRUCTIONS = """
OUTPUT: Return a single JSON object only. No prose. No markdown fences.
Match this schema exactly:

{schema}

Rules for each field:
- likelihood (0.0–1.0): probability of a base-rate break in next 1–6 months.
- magnitude (0.0–1.0): how FAR from historical baseline the change would be, NOT absolute severity.
- direction: "up" (worsening), "down" (improving), "mixed", or "unclear".
- window: which part of the forecast window the break is most likely in.
- rationale_bullets: 2–4 concise bullets. Each bullet must cite evidence. \
If likelihood <= 0.10, one bullet explaining why the base rate holds is sufficient.
- trigger_signals: 0–3 specific, observable signals that would confirm or \
deny the regime change. These should be concrete and monitorable.
- confidence_note: one sentence on how confident you are in this assessment \
and what data gaps limit your confidence.
"""


# ---------------------------------------------------------------------------
# Suggested search queries per hazard (for web research step)
# ---------------------------------------------------------------------------

def get_rc_search_queries(
    hazard_code: str,
    country_name: str,
    iso3: str,
    year: int = 2026,
) -> List[str]:
    """Return 2–4 targeted search queries for RC evidence gathering.

    These are used by the web research step BEFORE the RC prompt is built.
    They should surface recent signals of change, not background information.
    """

    queries = {
        "ACE": [
            f"{country_name} armed conflict escalation {year}",
            f"{country_name} political violence military tensions {year}",
            f"{country_name} ceasefire peace talks breakdown {year}",
            f"{country_name} armed groups militia clashes {year}",
        ],
        "DR": [
            f"{country_name} drought rainfall deficit {year}",
            f"{country_name} crop failure food prices {year}",
            f"{country_name} water shortage dry spell {year}",
        ],
        "FL": [
            f"{country_name} flooding river levels {year}",
            f"{country_name} flood damage displacement {year}",
            f"{country_name} heavy rainfall dam overflow {year}",
        ],
        "HW": [
            f"{country_name} heatwave temperature record {year}",
            f"{country_name} extreme heat health emergency {year}",
        ],
        "TC": [
            f"{country_name} tropical cyclone hurricane typhoon {year}",
            f"{country_name} cyclone season outlook {year}",
        ],
    }
    return queries.get(hazard_code, [f"{country_name} {hazard_code} {year}"])


# ---------------------------------------------------------------------------
# ACE — Armed Conflict Events
# ---------------------------------------------------------------------------

def build_rc_prompt_ace(
    country_name: str,
    iso3: str,
    resolver_features: Dict[str, Any],
    evidence_pack: Optional[Dict[str, Any]] = None,
    acled_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Build RC prompt for Armed Conflict Events (ACE).

    This is the most critical RC assessment. The system's core value proposition
    is early warning for conflict escalation, and missing an emerging conflict
    is the most consequential failure mode.

    Parameters
    ----------
    acled_summary : optional dict with keys like:
        - fatalities_trailing_12m: total fatalities in last 12 months
        - fatalities_trailing_3m: total fatalities in last 3 months
        - events_trailing_12m: total events in last 12 months
        - events_trailing_3m: total events in last 3 months
        - trend_direction: "increasing" | "decreasing" | "stable"
        - trend_pct_change: float (3m vs prior 3m)
        - top_event_types: list of (event_type, count) tuples
        - recent_spikes: list of (week, event_count) for anomalous weeks
    """

    evidence_text = _format_evidence(evidence_pack)
    acled_text = _format_acled_summary(acled_summary)
    resolver_text = json.dumps(resolver_features, indent=2, default=str)
    schema_text = json.dumps(_RC_SCHEMA, indent=2)

    preamble = _RC_CALIBRATION_PREAMBLE.format(
        country_name=country_name,
        iso3=iso3,
        hazard_name="Armed Conflict Events",
        hazard_code="ACE",
    )

    return f"""{preamble}

=== ACE-SPECIFIC GUIDANCE ===

You are looking for signals that conflict patterns are CHANGING — not simply
that conflict exists. The question is: "Will the next 6 months look
meaningfully different from the last 12 months?"

ACLED BASE RATE DATA:
{acled_text}

RESOLVER FEATURES (historical context):
{resolver_text}

EVIDENCE PACK (from recent web search — prioritize recent signals):
{evidence_text}

WHAT CONSTITUTES ACE REGIME CHANGE (likelihood > 0.10):
- New armed group emerging or entering the country
- Breakdown of ceasefire, peace agreement, or political settlement
- Coup, attempted coup, or military takeover
- Election-related violence escalating beyond historical norms for that country
- Conflict spreading to a previously unaffected region within the country
- Foreign military intervention or withdrawal changing the conflict dynamic
- Significant shift in external support to armed actors (arms, funding, fighters)
- Fatality trends showing a clear, sustained break from the trailing 12-month average
  (not just a single bad week, but a pattern over 4+ weeks)
- Mass atrocities or targeting of civilians at a scale beyond the recent norm

WHAT IS NOT ACE REGIME CHANGE (keep likelihood <= 0.10):
- Ongoing conflict continuing at roughly the same level, even if severe
- Seasonal variations in conflict intensity (e.g., dry-season offensives)
  that are consistent with the country's historical pattern
- Protests or political unrest that has not crossed into armed violence
- Rhetoric or threats without corresponding operational indicators
- A single violent incident, unless it represents a fundamentally new pattern
- General instability or governance weakness without new triggers
- Media attention increasing on an existing conflict (attention ≠ escalation)

ANCHORING AGAINST ACLED DATA:
- If ACLED shows stable or declining trends over the last 3 months vs the
  prior 3 months, AND the evidence pack shows no new escalation triggers,
  likelihood should be <= 0.08.
- If ACLED shows a 30%+ increase in fatalities over the last 3 months vs
  prior 3 months, AND the evidence pack corroborates with specific events,
  likelihood should be 0.15–0.35 depending on whether the increase looks
  sustained vs. a single spike.
- If ACLED shows a 100%+ increase AND multiple evidence pack signals point
  to a structural shift (new actor, broken ceasefire, territorial change),
  likelihood should be 0.35–0.60.
- If the country has had zero or near-zero conflict fatalities for the past
  year, the default likelihood is 0.03. Only raise it if there are CONCRETE
  new triggers — not structural vulnerability or neighboring-country spillover
  risk alone.

CONFLICT EARLY WARNING SIGNALS TO LOOK FOR:
These are the signals that historically precede conflict escalation. Give
extra weight to any of these appearing in the evidence pack:
1. Political triggers: disputed elections, constitutional crises, leadership
   succession failures, collapse of power-sharing arrangements.
2. Security triggers: military mobilization, weapons proliferation reports,
   formation of new armed groups or militias, defections from security forces.
3. Economic triggers: sudden collapse in state revenue (e.g., oil price for
   oil-dependent states), hyperinflation, mass unemployment among youth.
4. Social triggers: inter-communal violence escalating, hate speech campaigns,
   mass displacement creating tensions, land/resource disputes intensifying.
5. External triggers: withdrawal of peacekeeping forces, change in foreign
   patron's policy, sanctions or arms embargoes being lifted/imposed,
   regional conflict spillover with concrete cross-border incidents.
6. Information triggers: media blackouts, internet shutdowns, expulsion of
   journalists or humanitarian observers.

MAGNITUDE CALIBRATION:
- 0.1–0.3: Conflict intensifies moderately (e.g., 50–100% above baseline
  fatalities, spread to 1–2 new sub-regions).
- 0.3–0.6: Major escalation (e.g., 2–5x baseline fatalities, new front
  opens, significant territorial changes, mass civilian displacement).
- 0.6–1.0: Transformative break (e.g., full-scale war onset in previously
  peaceful country, genocide/mass atrocity risk, state collapse).

=== END ACE-SPECIFIC GUIDANCE ===

{_RC_OUTPUT_INSTRUCTIONS.format(schema=schema_text)}"""


# ---------------------------------------------------------------------------
# DR — Drought
# ---------------------------------------------------------------------------

def build_rc_prompt_dr(
    country_name: str,
    iso3: str,
    resolver_features: Dict[str, Any],
    evidence_pack: Optional[Dict[str, Any]] = None,
    climate_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Build RC prompt for Drought (DR).

    Parameters
    ----------
    climate_data : optional dict with keys like:
        - chirps_percentile_90d: rainfall percentile vs 1981-2020 for last 90 days
        - iri_forecast_precip: IRI tercile forecast for next 1-3 months
        - enso_state: current ENSO phase and forecast
        - ndvi_anomaly: vegetation anomaly vs long-term mean
        - season_context: description of current growing season status
        - fewsnet_phase: current IPC phase if available
    """

    evidence_text = _format_evidence(evidence_pack)
    climate_text = _format_climate_data(climate_data)
    resolver_text = json.dumps(resolver_features, indent=2, default=str)
    schema_text = json.dumps(_RC_SCHEMA, indent=2)

    preamble = _RC_CALIBRATION_PREAMBLE.format(
        country_name=country_name,
        iso3=iso3,
        hazard_name="Drought",
        hazard_code="DR",
    )

    return f"""{preamble}

=== DR-SPECIFIC GUIDANCE ===

Drought is slow-onset and often predictable months in advance from climate
data. The key question is: "Are rainfall and vegetation patterns departing
from what is normal for THIS country at THIS time of year?"

CLIMATE DATA (structured observations and forecasts):
{climate_text}

RESOLVER FEATURES (historical context):
{resolver_text}

EVIDENCE PACK (from recent web search):
{evidence_text}

WHAT CONSTITUTES DR REGIME CHANGE (likelihood > 0.10):
- Cumulative rainfall for the current season significantly below normal
  (below 20th percentile of historical distribution for this point in the
  season), AND this is not a normal dry-year fluctuation for this country.
- IRI or other seasonal forecasts showing high probability (>50%) of
  below-normal rainfall for the remainder of the growing season.
- ENSO phase shift that historically correlates with drought in this region
  (e.g., El Niño onset for East Africa, La Niña for southern Africa).
- NDVI/vegetation indices showing anomalous decline vs. seasonal norms.
- Consecutive failed seasons (e.g., a second poor rainy season in a row),
  representing a compounding departure from the historical frequency.
- Food prices spiking above seasonal norms, indicating market stress from
  production shortfalls.
- Official drought declarations or emergency food security classifications
  (IPC Phase 3+) in regions that are not chronically food insecure.

WHAT IS NOT DR REGIME CHANGE (keep likelihood <= 0.10):
- A normal dry season in a country with pronounced wet/dry seasonality.
- Moderate below-average rainfall that is within the historical range of
  variability for this country (e.g., 30th–40th percentile).
- Chronic food insecurity continuing at its established level.
- Drought risk during the dry season in a country that always has a dry
  season — the regime change would be if the subsequent WET season fails.
- General climate vulnerability without specific current-season signals.

SEASONAL CONTEXT IS CRITICAL:
- If the country is currently in its dry season, drought RC should be
  assessed relative to the UPCOMING rainy season forecast, not current
  conditions (which are expected to be dry).
- If the country is mid-growing-season, current rainfall deficits are
  more immediately relevant.
- Multi-season drought (consecutive failed seasons) is a much stronger
  signal than a single below-average season.

MAGNITUDE CALIBRATION:
- 0.1–0.3: Below-average season likely, moderate crop losses, food prices
  elevated but manageable. Roughly a 1-in-5 to 1-in-10 year drought.
- 0.3–0.6: Severe drought developing, major crop failures across key
  agricultural zones, livestock losses, water rationing. Roughly a
  1-in-20 to 1-in-50 year event.
- 0.6–1.0: Exceptional/unprecedented drought, multi-season failure,
  potential famine conditions, mass displacement from agricultural areas.

=== END DR-SPECIFIC GUIDANCE ===

{_RC_OUTPUT_INSTRUCTIONS.format(schema=schema_text)}"""


# ---------------------------------------------------------------------------
# FL — Flood
# ---------------------------------------------------------------------------

def build_rc_prompt_fl(
    country_name: str,
    iso3: str,
    resolver_features: Dict[str, Any],
    evidence_pack: Optional[Dict[str, Any]] = None,
    climate_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Build RC prompt for Flood (FL).

    Parameters
    ----------
    climate_data : optional dict with keys like:
        - glofas_alerts: GloFAS flood alerts for major river basins
        - chirps_percentile_90d: rainfall percentile vs 1981-2020
        - iri_forecast_precip: IRI tercile forecast for next 1-3 months
        - enso_state: current ENSO phase and forecast
        - season_context: current rainy season status
    """

    evidence_text = _format_evidence(evidence_pack)
    climate_text = _format_climate_data(climate_data)
    resolver_text = json.dumps(resolver_features, indent=2, default=str)
    schema_text = json.dumps(_RC_SCHEMA, indent=2)

    preamble = _RC_CALIBRATION_PREAMBLE.format(
        country_name=country_name,
        iso3=iso3,
        hazard_name="Flood",
        hazard_code="FL",
    )

    return f"""{preamble}

=== FL-SPECIFIC GUIDANCE ===

Flood regime change means flooding at a scale or timing that departs from
the country's historical pattern. Many countries experience annual flooding
— that is the BASELINE, not a regime change.

CLIMATE DATA (structured observations and forecasts):
{climate_text}

RESOLVER FEATURES (historical context):
{resolver_text}

EVIDENCE PACK (from recent web search):
{evidence_text}

WHAT CONSTITUTES FL REGIME CHANGE (likelihood > 0.10):
- GloFAS showing elevated flood probability (>50% of exceeding 5-year
  return period) on major river systems, especially outside the typical
  peak flood months.
- Seasonal rainfall forecast indicating well-above-normal precipitation
  during the country's rainy season (upper tercile with >50% probability).
- ENSO phase that historically amplifies flooding in this region (e.g.,
  La Niña for parts of Southeast Asia and South America, El Niño for
  East Africa).
- Upstream developments that change flood dynamics: dam construction or
  failure risks, upstream deforestation, land use changes.
- Current-season rainfall already well above normal (>80th percentile)
  with more rain forecast.
- Back-to-back flood events within a season (soil saturation compounds
  subsequent flood impact).

WHAT IS NOT FL REGIME CHANGE (keep likelihood <= 0.10):
- Expected seasonal flooding during the country's normal rainy season,
  at roughly historical scale.
- A flood-prone country entering its rainy season — this is the baseline.
- General climate change increasing flood frequency — this is structural,
  not a 6-month regime change signal.
- River levels rising during the normal monsoon/rainy season onset.
- Localized flash flooding from individual storms (unless at anomalous
  frequency or scale).

SEASONALITY IS CRITICAL:
- A flood in the middle of the dry season is a much stronger RC signal
  than a flood during the peak rainy season.
- The question is always relative to "what is normal for this month in
  this country."

MAGNITUDE CALIBRATION:
- 0.1–0.3: Moderate flooding above seasonal norms, some infrastructure
  damage, localized displacement. A 1-in-10 to 1-in-20 year event.
- 0.3–0.6: Major flooding, significant displacement (>50,000), damage
  to critical infrastructure, crop destruction. A 1-in-20 to 1-in-50
  year event.
- 0.6–1.0: Catastrophic flooding, unprecedented scale, national-level
  emergency, massive displacement. A 1-in-100+ year event or entirely
  novel pattern.

=== END FL-SPECIFIC GUIDANCE ===

{_RC_OUTPUT_INSTRUCTIONS.format(schema=schema_text)}"""


# ---------------------------------------------------------------------------
# HW — Heatwave
# ---------------------------------------------------------------------------

def build_rc_prompt_hw(
    country_name: str,
    iso3: str,
    resolver_features: Dict[str, Any],
    evidence_pack: Optional[Dict[str, Any]] = None,
    climate_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Build RC prompt for Heatwave (HW).

    Parameters
    ----------
    climate_data : optional dict with keys like:
        - iri_forecast_temp: IRI seasonal temperature forecast
        - recent_temp_anomaly: recent temperature anomaly vs climatology
        - season_context: current season and typical peak heat months
    """

    evidence_text = _format_evidence(evidence_pack)
    climate_text = _format_climate_data(climate_data)
    resolver_text = json.dumps(resolver_features, indent=2, default=str)
    schema_text = json.dumps(_RC_SCHEMA, indent=2)

    preamble = _RC_CALIBRATION_PREAMBLE.format(
        country_name=country_name,
        iso3=iso3,
        hazard_name="Heatwave",
        hazard_code="HW",
    )

    return f"""{preamble}

=== HW-SPECIFIC GUIDANCE ===

Heatwave regime change means extreme heat events at a scale, duration, or
timing that departs from the country's historical pattern. Countries in hot
climates experience high temperatures annually — that is the baseline.

CLIMATE DATA (structured observations and forecasts):
{climate_text}

RESOLVER FEATURES (historical context):
{resolver_text}

EVIDENCE PACK (from recent web search):
{evidence_text}

WHAT CONSTITUTES HW REGIME CHANGE (likelihood > 0.10):
- Seasonal temperature forecasts showing well-above-normal temperatures
  (upper tercile with high probability) during the country's peak heat
  months.
- Record-breaking temperatures already observed in the current season,
  suggesting the heat pattern is anomalous.
- Heat events occurring outside the normal hot season window.
- Compound factors amplifying humanitarian impact: heatwave coinciding
  with power grid instability, water shortages, or harvest season.
- Multi-week sustained heat (not just a few hot days) at unprecedented
  levels.

WHAT IS NOT HW REGIME CHANGE (keep likelihood <= 0.10):
- Hot temperatures during the country's normal hot season, at roughly
  historical levels.
- A hot country being hot — the regime change is about DEPARTURE from
  that country's own norm.
- General climate warming trends — this is structural background.
- Brief (1-3 day) hot spells that are within historical variability.

HUMANITARIAN IMPACT CONTEXT:
- Heatwave humanitarian impact depends heavily on vulnerability: urban
  heat islands, outdoor labor exposure, elderly populations, power grid
  reliability, water availability.
- The same temperature may cause humanitarian crisis in one context and
  be routine in another. Focus on whether the TEMPERATURE PATTERN is
  anomalous for this country, not on absolute temperature levels.

MAGNITUDE CALIBRATION:
- 0.1–0.3: Temperatures moderately above seasonal norms (1–3°C above
  average), some health impacts, increased energy demand.
- 0.3–0.6: Severe heat event, temperatures well above norms (3–5°C),
  significant health impacts, power grid stress, agricultural damage.
- 0.6–1.0: Record-shattering heat, unprecedented temperatures, mass
  health emergency, critical infrastructure failure.

=== END HW-SPECIFIC GUIDANCE ===

{_RC_OUTPUT_INSTRUCTIONS.format(schema=schema_text)}"""


# ---------------------------------------------------------------------------
# TC — Tropical Cyclone
# ---------------------------------------------------------------------------

def build_rc_prompt_tc(
    country_name: str,
    iso3: str,
    resolver_features: Dict[str, Any],
    evidence_pack: Optional[Dict[str, Any]] = None,
    climate_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Build RC prompt for Tropical Cyclone (TC).

    Parameters
    ----------
    climate_data : optional dict with keys like:
        - enso_state: current ENSO phase and forecast
        - seasonal_outlook: NOAA/regional TC seasonal forecast
        - basin: which TC basin(s) affect this country
        - season_status: whether currently in/out of TC season
        - active_storms: any currently active storms in the relevant basin
    """

    evidence_text = _format_evidence(evidence_pack)
    climate_text = _format_climate_data(climate_data)
    resolver_text = json.dumps(resolver_features, indent=2, default=str)
    schema_text = json.dumps(_RC_SCHEMA, indent=2)

    preamble = _RC_CALIBRATION_PREAMBLE.format(
        country_name=country_name,
        iso3=iso3,
        hazard_name="Tropical Cyclone",
        hazard_code="TC",
    )

    return f"""{preamble}

=== TC-SPECIFIC GUIDANCE ===

Tropical cyclone regime change means TC activity that departs from the
country's historical exposure pattern. Many countries experience cyclones
annually during their TC season — that is the baseline.

TC risk is highly seasonal and geographically constrained. ENSO phase is
one of the strongest predictors of seasonal TC activity.

CLIMATE DATA (structured observations and forecasts):
{climate_text}

RESOLVER FEATURES (historical context):
{resolver_text}

EVIDENCE PACK (from recent web search):
{evidence_text}

WHAT CONSTITUTES TC REGIME CHANGE (likelihood > 0.10):
- Seasonal outlook from NOAA, Tropical Storm Risk, or regional agencies
  indicating well-above-normal TC activity for the relevant basin.
- ENSO phase shift that historically amplifies cyclone risk in this
  region (e.g., La Niña increasing Atlantic hurricane activity, El Niño
  increasing Eastern Pacific activity).
- Anomalously warm sea surface temperatures in the relevant basin,
  suggesting enhanced cyclogenesis potential.
- Active storm currently threatening or approaching the country outside
  the historical peak window.
- A country that rarely experiences direct TC landfall having a storm
  track that targets it.

WHAT IS NOT TC REGIME CHANGE (keep likelihood <= 0.10):
- Cyclone season approaching in a cyclone-exposed country — that is the
  baseline, even if forecasts call for an "active" season.
- A normal-activity TC season in a TC-prone country.
- General awareness that "it's hurricane season."
- TC activity in the broader basin that does not specifically threaten
  this country.

OUT-OF-SEASON CONTEXT:
- If the country is currently outside its TC season AND no active storm
  is threatening, likelihood should be <= 0.03 regardless of other
  factors. TC risk outside the season window is negligible.
- If the country is entering or within its TC season, assess based on
  seasonal outlook and current basin conditions.

MAGNITUDE CALIBRATION:
- 0.1–0.3: Above-normal TC season expected, elevated probability of
  landfall, but within range of historical variability.
- 0.3–0.6: Significantly above-normal season AND specific factors
  suggest this country may be disproportionately affected (SST patterns,
  steering flow anomalies).
- 0.6–1.0: Active intense storm on a direct track to the country, or
  extreme seasonal conditions suggesting record-breaking activity.
  (Note: this level is rare in an RC assessment made weeks/months ahead;
  it is more typical of a near-real-time assessment with an active storm.)

=== END TC-SPECIFIC GUIDANCE ===

{_RC_OUTPUT_INSTRUCTIONS.format(schema=schema_text)}"""


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

RC_PROMPT_BUILDERS = {
    "ACE": build_rc_prompt_ace,
    "DR": build_rc_prompt_dr,
    "FL": build_rc_prompt_fl,
    "HW": build_rc_prompt_hw,
    "TC": build_rc_prompt_tc,
}


def build_rc_prompt(
    hazard_code: str,
    country_name: str,
    iso3: str,
    resolver_features: Dict[str, Any],
    evidence_pack: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> str:
    """Dispatch to the appropriate per-hazard RC prompt builder."""
    builder = RC_PROMPT_BUILDERS.get(hazard_code)
    if builder is None:
        raise ValueError(f"No RC prompt builder for hazard code: {hazard_code}")
    return builder(
        country_name=country_name,
        iso3=iso3,
        resolver_features=resolver_features,
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


def _format_acled_summary(acled_summary: Optional[Dict[str, Any]]) -> str:
    if not acled_summary:
        return (
            "ACLED summary unavailable. Assess based on evidence pack and "
            "resolver features only. Note: absence of ACLED data may indicate "
            "either no conflict or poor data coverage — do not assume either."
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
        lines.append(f"- Political violence events (trailing 12 months): {e12:,}")
    if e3 is not None:
        lines.append(f"- Political violence events (trailing 3 months): {e3:,}")
    if trend:
        trend_str = f"- Trend: {trend}"
        if pct is not None:
            trend_str += f" ({pct:+.0f}% change, last 3m vs prior 3m)"
        lines.append(trend_str)

    top_types = acled_summary.get("top_event_types")
    if top_types:
        types_str = ", ".join(f"{t} ({c})" for t, c in top_types[:4])
        lines.append(f"- Top event types (12m): {types_str}")

    spikes = acled_summary.get("recent_spikes")
    if spikes:
        spike_str = ", ".join(f"week of {w}: {c} events" for w, c in spikes[:3])
        lines.append(f"- Recent anomalous weeks: {spike_str}")

    if not lines:
        return "ACLED summary provided but contained no usable data."

    return "\n".join(lines)


def _format_climate_data(climate_data: Optional[Dict[str, Any]]) -> str:
    if not climate_data:
        return (
            "Structured climate data unavailable. Assess based on evidence "
            "pack and resolver features only."
        )

    lines = []
    for key, val in climate_data.items():
        if val is not None:
            # Convert key from snake_case to readable label
            label = key.replace("_", " ").capitalize()
            lines.append(f"- {label}: {val}")

    if not lines:
        return "Climate data provided but contained no usable values."

    return "\n".join(lines)
