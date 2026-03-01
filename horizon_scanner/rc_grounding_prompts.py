# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Per-hazard Google Grounding query builders for RC evidence packs.

These replace the single generic country-level grounding query with
hazard-specific queries that surface the right signals for each hazard's
RC assessment. Each query is sent to Gemini with Google Search grounding
enabled, returning structured JSON with recent signals and sources.

Architecture:
    1. For each (country, hazard) pair that survives seasonal filtering,
       call fetch_via_gemini() with the hazard-specific grounding prompt.
    2. The returned evidence pack feeds directly into the corresponding
       per-hazard RC prompt (from rc_prompts.py).
    3. Each hazard gets its own grounding call because the search terms,
       signal types, source priorities, and recency windows are fundamentally
       different across hazards.

Cost: ~300 grounding calls per run (after seasonal filtering), up from
~120. Each call is lightweight (Gemini Flash, ~768 output tokens).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Recency windows per hazard (days)
# ---------------------------------------------------------------------------
# These control how far back the grounding search should look for signals.
# Conflict and displacement change fast; drought and TC seasons are slower.

RECENCY_DAYS = {
    "ACE": 90,   # Conflict signals are fast-moving
    "DR": 120,   # Drought is slow-onset, need longer window
    "FL": 90,    # Floods can be sudden but seasonal context matters
    "HW": 90,    # Heatwaves are seasonal, 90 days captures onset
    "TC": 120,   # TC season outlooks issued months ahead
}


# ---------------------------------------------------------------------------
# Output schema (shared across all hazards)
# ---------------------------------------------------------------------------
# This is the JSON shape Gemini must return. It matches what
# parse_gemini_grounding_response() and _extract_json_blob() expect.

_OUTPUT_SCHEMA_BLOCK = """\
Return strictly JSON with this shape:
{
  "recent_signals": ["<=8 bullets, see format below"],
  "structural_context": "max 6 lines of background context",
  "data_gaps": "what key information you could NOT find",
  "notes": "optional"
}

Signal bullet format (one per bullet):
  CATEGORY | TIMEFRAME | DIRECTION | signal text

Where:
  CATEGORY = one of: TRIGGER, DAMPENER, BASELINE
  TIMEFRAME = approximate: month_1, month_1-2, month_2-3, month_3-6, recent
  DIRECTION = UP (worsening), DOWN (improving), STABLE, UNCLEAR

Example bullet:
  TRIGGER | month_1-2 | UP | Rainfall for Oct-Nov was 35% below the 1991-2020 average across the southern agricultural belt, per CHIRPS satellite data.

Rules:
- Prioritize RECENT, SPECIFIC, QUANTIFIED signals over vague assessments.
- Every signal bullet must reference a source or data point.
- Do not include URLs in the JSON text (they come from grounding metadata).
- If you find no recent signals for this hazard, return an empty recent_signals array and explain in data_gaps.
- structural_context should be BRIEF background only — not a substitute for recent signals.
- data_gaps should honestly state what you could not find. This is critical for downstream confidence assessment.\
"""


# ---------------------------------------------------------------------------
# ACE — Armed Conflict Events
# ---------------------------------------------------------------------------

def build_grounding_prompt_ace(
    country_name: str,
    iso3: str,
    recency_days: int = RECENCY_DAYS["ACE"],
    acled_context: Optional[str] = None,
) -> str:
    """Build Gemini grounding prompt for ACE evidence gathering.

    This is the highest-priority grounding call. Conflict escalation is
    the system's most critical detection target, and web search is the
    primary source of early warning signals that precede statistical
    detection in ACLED data (which has ~1 week latency).

    Parameters
    ----------
    acled_context : optional brief text summarizing recent ACLED trends,
        e.g. "Fatalities trending up 40% over last 3 months vs prior 3 months."
        This helps Gemini focus its search on confirming/disconfirming
        the statistical signal.
    """

    acled_block = ""
    if acled_context:
        acled_block = (
            f"\n\nACLED statistical context (use to focus your search):\n"
            f"{acled_context}\n"
            f"Search for evidence that EXPLAINS or CONTRADICTS this trend.\n"
        )

    return f"""\
You are a conflict early warning research assistant using Google Search grounding.
Search for recent evidence of CHANGING conflict dynamics in {country_name} ({iso3})
over the last {recency_days} days.

You are NOT looking for background information on existing conflicts. You are
looking for signals that the PATTERN is changing — escalation, de-escalation,
new actors, broken ceasefires, political triggers.
{acled_block}
SEARCH PRIORITIES (in order):
1. Active hostilities: new offensives, territorial changes, sieges, major
   clashes in areas that were previously calm. Search for recent battle
   reports, military operations, frontline changes.
2. Political triggers: election disputes, constitutional crises, coup
   signals, collapse of power-sharing, leadership succession problems,
   failed negotiations. Search for recent political developments that
   could trigger or prevent violence.
3. Security force changes: military mobilization, defections, new armed
   group formation, weapons proliferation, mercenary deployment, foreign
   military intervention or withdrawal.
4. Ceasefire/peace dynamics: ceasefire violations, peace talk breakdowns,
   new agreements, mediation efforts. Search for recent updates on any
   active peace processes.
5. Social/communal triggers: inter-ethnic or inter-communal violence
   escalation, targeted attacks on civilians, hate speech campaigns,
   vigilante actions, land/resource disputes turning violent.
6. External factors: sanctions changes, arms embargo enforcement/violations,
   cross-border incidents, regional conflict spillover with concrete events,
   changes in foreign military support.

SIGNAL QUALITY GUIDANCE:
- A specific event ("Three clashes in Darfur's North region on Jan 15
  killed 23") is much more useful than a vague assessment ("tensions
  remain high").
- Quantified changes are better than qualitative ones: "fatalities doubled"
  vs "violence increased."
- Official statements of intent (military mobilization orders, formal
  ceasefire withdrawal) are strong signals.
- Humanitarian access restrictions, journalist expulsions, and internet
  shutdowns are important secondary indicators.
- Distinguish between RHETORIC (threats, warnings) and OPERATIONAL
  INDICATORS (troop movements, weapons transfers, actual attacks).

DAMPENER SIGNALS ARE EQUALLY IMPORTANT:
- Ceasefire agreements holding, peace talks progressing, elections
  proceeding normally, security sector reforms, DDR programs,
  de-escalation by external actors.
- If the country has active conflict, search specifically for both
  escalation AND de-escalation signals.

Focus on authoritative sources: ACLED, Crisis Group, UN OCHA, Armed
Conflict Location & Event Data, UNHCR, local conflict monitoring
organizations, major wire services (Reuters, AP, AFP).

{_OUTPUT_SCHEMA_BLOCK}"""


# ---------------------------------------------------------------------------
# DR — Drought
# ---------------------------------------------------------------------------

def build_grounding_prompt_dr(
    country_name: str,
    iso3: str,
    recency_days: int = RECENCY_DAYS["DR"],
    season_context: Optional[str] = None,
) -> str:
    """Build Gemini grounding prompt for DR evidence gathering.

    Drought is where the gap between web search and structured data is
    largest. Web search will NOT reliably surface rainfall percentiles or
    NDVI anomalies — those come from the structured data feeds (CHIRPS,
    IRI, etc.). The grounding call's job is to find the IMPACT signals
    and FORECAST signals that structured data misses: crop failure
    reports, food price spikes, government declarations, seasonal
    forecast interpretations, FEWS NET assessments.

    Parameters
    ----------
    season_context : optional text like "Currently mid-way through the
        October-December short rains season" to help Gemini frame its
        search correctly.
    """

    season_block = ""
    if season_context:
        season_block = (
            f"\n\nSeason context: {season_context}\n"
            f"Focus your search on conditions during THIS season.\n"
        )

    return f"""\
You are a food security and drought research assistant using Google Search grounding.
Search for recent evidence of DROUGHT CONDITIONS or EMERGING DROUGHT RISK in
{country_name} ({iso3}) over the last {recency_days} days.

You are looking for signals that drought conditions are DEVELOPING or WORSENING
beyond the country's normal seasonal patterns — or that previously concerning
conditions are IMPROVING.
{season_block}
SEARCH PRIORITIES (in order):
1. Rainfall and climate anomalies: reports of below-normal rainfall,
   delayed rainy season onset, early cessation of rains, consecutive
   dry spells. Search for FEWS NET, WMO, or national meteorological
   agency reports on current-season rainfall performance.
2. Crop and livestock impacts: crop failure reports, poor harvest
   forecasts, livestock distress or die-offs, pasture degradation.
   Search for FAO, FEWS NET, or government agricultural assessments.
3. Food security classifications: IPC/CH phase classifications,
   FEWS NET food security outlooks, WFP food security monitoring
   bulletins. Search for the most recent IPC or FEWS NET assessment
   for this country.
4. Food prices: staple food price spikes above seasonal norms,
   market disruptions, import dependency stress. Search for WFP
   market monitoring or VAM data.
5. Water stress: reservoir levels, water rationing, urban water
   supply disruptions, groundwater depletion reports.
6. Seasonal forecasts: IRI, ECMWF, or regional climate outlook
   forum predictions for upcoming seasons. La Niña/El Niño
   implications for this specific region.
7. Government/humanitarian response: drought emergency declarations,
   humanitarian appeals, food aid distributions, school feeding
   program expansions.

SIGNAL QUALITY GUIDANCE:
- A FEWS NET or IPC assessment is the gold standard for food security
  signals. If one exists, it should be the first bullet.
- Quantified crop losses ("maize harvest expected 40% below average")
  are much more useful than "poor harvest expected."
- Distinguish between current-season impacts and forecast/outlook
  signals — both matter but differently.
- Multi-season drought (consecutive poor seasons) is a much stronger
  signal than a single below-average season.

Focus on authoritative sources: FEWS NET, FAO GIEWS, WFP VAM, IPC,
IRI, national meteorological services, OCHA situation reports.

{_OUTPUT_SCHEMA_BLOCK}"""


# ---------------------------------------------------------------------------
# FL — Flood
# ---------------------------------------------------------------------------

def build_grounding_prompt_fl(
    country_name: str,
    iso3: str,
    recency_days: int = RECENCY_DAYS["FL"],
    season_context: Optional[str] = None,
) -> str:
    """Build Gemini grounding prompt for FL evidence gathering.

    Like drought, much of the quantitative flood signal comes from
    structured data (GloFAS, CHIRPS). The grounding call surfaces
    IMPACT signals (displacement, damage, humanitarian response) and
    CONTEXT signals (dam conditions, upstream developments, seasonal
    forecasts) that structured data doesn't capture.

    Parameters
    ----------
    season_context : optional text like "Currently entering the June-
        September monsoon season" to help Gemini focus its search.
    """

    season_block = ""
    if season_context:
        season_block = (
            f"\n\nSeason context: {season_context}\n"
            f"Frame your search around conditions for THIS season.\n"
        )

    return f"""\
You are a flood risk research assistant using Google Search grounding.
Search for recent evidence of FLOOD EVENTS, FLOOD RISK, or CHANGING FLOOD
DYNAMICS in {country_name} ({iso3}) over the last {recency_days} days.

You are looking for signals that flooding is occurring or likely to occur
at a scale BEYOND the country's normal seasonal pattern — or that an
expected flood season is tracking milder than usual.
{season_block}
SEARCH PRIORITIES (in order):
1. Active flooding: current flood events, river levels exceeding danger
   marks, flash floods, urban inundation. Search for OCHA flash updates,
   government disaster reports, FloodList, ReliefWeb.
2. River and dam conditions: river gauge levels on major systems, dam
   release alerts, reservoir levels approaching capacity, upstream
   rainfall reports. Search for national hydrological service bulletins
   or GloFAS reports.
3. Displacement and humanitarian impact: flood-related displacement
   numbers, shelter needs, crop/infrastructure damage assessments,
   humanitarian response activations (CERF, flash appeals).
4. Seasonal and climate context: above-normal rainfall forecasts for
   the upcoming season, La Niña/El Niño implications for rainfall
   in this region, regional climate outlook forum predictions.
5. Compounding factors: soil saturation from prior flooding, deforestation
   changing runoff patterns, rapid urbanization in flood plains, dam
   construction or infrastructure changes.
6. Cyclone-flood linkage: if tropical cyclones are expected or active
   in the region, search for their rainfall/flood implications.

SIGNAL QUALITY GUIDANCE:
- Specific flood events with quantified impact ("flooding in Province X
  displaced 50,000 people") are much more useful than "flood risk exists."
- River levels relative to danger thresholds are very useful if available.
- Distinguish between localized flash flooding (common, usually not RC)
  and widespread riverine flooding (potentially RC if above historical norms).
- Back-to-back flood events within a season compound impact and represent
  a stronger signal than a single event.

Focus on authoritative sources: OCHA, IFRC, FloodList, GloFAS bulletins,
national disaster management agencies, ReliefWeb, WMO.

{_OUTPUT_SCHEMA_BLOCK}"""


# ---------------------------------------------------------------------------
# HW — Heatwave
# ---------------------------------------------------------------------------

def build_grounding_prompt_hw(
    country_name: str,
    iso3: str,
    recency_days: int = RECENCY_DAYS["HW"],
    season_context: Optional[str] = None,
) -> str:
    """Build Gemini grounding prompt for HW evidence gathering.

    Heatwave data is sparser than conflict or drought in humanitarian
    reporting. The grounding call needs to look beyond humanitarian
    sources into meteorological reports, health system alerts, and
    energy/infrastructure reporting.

    Parameters
    ----------
    season_context : optional text like "Approaching the April-June
        pre-monsoon hot season" to help Gemini focus its search.
    """

    season_block = ""
    if season_context:
        season_block = (
            f"\n\nSeason context: {season_context}\n"
            f"Focus your search on conditions for THIS season.\n"
        )

    return f"""\
You are a climate and health research assistant using Google Search grounding.
Search for recent evidence of EXTREME HEAT EVENTS or EMERGING HEATWAVE RISK
in {country_name} ({iso3}) over the last {recency_days} days.

You are looking for signals that heat conditions are ANOMALOUS for this
country at this time of year — not simply that it is hot in a hot country.
{season_block}
SEARCH PRIORITIES (in order):
1. Temperature records: new temperature records set, temperatures
   exceeding historical norms for this time of year, sustained heat
   beyond typical duration. Search for national meteorological service
   reports, WMO statements, weather agency bulletins.
2. Health impacts: heat-related mortality or morbidity reports, hospital
   admissions for heat stress, government health advisories, school or
   workplace closures due to heat. These often appear in local media
   before humanitarian reporting.
3. Infrastructure stress: power grid failures from cooling demand,
   water supply stress from heat-driven demand, transport disruptions
   (rail buckling, road damage).
4. Seasonal temperature forecasts: IRI or CPC seasonal temperature
   outlooks showing above-normal temperatures for upcoming months,
   El Niño/La Niña implications for temperature in this region.
5. Agricultural/livelihood impacts: heat stress on crops during
   critical growth stages, livestock heat stress, outdoor worker
   impacts.
6. Compound hazards: heatwave coinciding with drought (amplifying
   water stress), heatwave coinciding with conflict (limiting
   cooling access), heatwave coinciding with power shortages.

SIGNAL QUALITY GUIDANCE:
- Temperature anomalies relative to the country's OWN historical norms
  are more relevant than absolute temperatures. +3°C above the monthly
  average in Canada is a stronger signal than 45°C in Kuwait (which may
  be normal for Kuwait in summer).
- Duration matters: a 2-day hot spell is less significant than a 2-week
  sustained heat event.
- Wet-bulb temperature extremes (combining heat and humidity) are
  particularly dangerous — note if mentioned.
- Night-time temperatures failing to drop (warm nights) amplify health
  impacts and are a sign of anomalous heat patterns.

Focus on authoritative sources: WMO, national meteorological services,
WHO heat-health warnings, IRI, CPC, ECMWF, Copernicus Climate Change
Service, ReliefWeb.

{_OUTPUT_SCHEMA_BLOCK}"""


# ---------------------------------------------------------------------------
# TC — Tropical Cyclone
# ---------------------------------------------------------------------------

def build_grounding_prompt_tc(
    country_name: str,
    iso3: str,
    recency_days: int = RECENCY_DAYS["TC"],
    season_context: Optional[str] = None,
) -> str:
    """Build Gemini grounding prompt for TC evidence gathering.

    TC evidence gathering is uniquely bimodal:
    - Pre-season / early season: looking for seasonal outlooks, SST
      anomalies, ENSO forecasts that predict above/below normal activity.
    - Active season with threatening storm: looking for track forecasts,
      intensity forecasts, impact projections for THIS specific storm.

    Parameters
    ----------
    season_context : optional text like "Currently in the Atlantic
        hurricane season (Jun-Nov), La Niña conditions present" or
        "Outside TC season for this basin."
    """

    season_block = ""
    if season_context:
        season_block = (
            f"\n\nSeason context: {season_context}\n"
            f"This determines whether to focus on seasonal outlooks or active storms.\n"
        )

    return f"""\
You are a tropical cyclone research assistant using Google Search grounding.
Search for recent evidence of TROPICAL CYCLONE RISK or ACTIVE STORMS
affecting {country_name} ({iso3}) over the last {recency_days} days.

You are looking for signals that TC activity for this country is DEPARTING
from historical norms — either an active storm threatening the country, or
seasonal conditions suggesting above-normal TC activity in the relevant basin.
{season_block}
SEARCH PRIORITIES (in order):
1. Active storms: any tropical cyclone, hurricane, or typhoon currently
   active or recently active in the basin affecting this country. Search
   for JTWC, NHC, or regional RSMC (e.g., TCWC New Delhi, JMA Tokyo,
   Météo-France La Réunion) advisories and track forecasts.
2. Seasonal outlooks: NOAA/CPC hurricane season outlook, Tropical Storm
   Risk (TSR/UCL) seasonal forecasts, regional meteorological agency
   seasonal TC predictions. Are they calling for above/below/near-normal
   activity?
3. ENSO state and forecast: current El Niño/La Niña status and forecast
   trajectory. ENSO is the single strongest seasonal predictor of TC
   activity in most basins.
4. Sea surface temperatures: SST anomalies in the relevant basin's main
   development region. Above-normal SSTs fuel TC activity.
5. Recent TC impacts: damage assessments, displacement, humanitarian
   response from any recent TC landfall in or near this country.
6. Pre-positioning and preparedness: government/humanitarian pre-positioning
   activities, evacuation plans activated — these can signal that agencies
   expect elevated risk.

BASIN IDENTIFICATION:
Determine which TC basin(s) affect {country_name} and focus search there:
- Atlantic (North Atlantic, Caribbean, Gulf of Mexico)
- Eastern Pacific
- Western Pacific (includes Philippines, Japan, Taiwan, Vietnam, etc.)
- North Indian Ocean (Bay of Bengal, Arabian Sea)
- South-West Indian Ocean (Madagascar, Mozambique, etc.)
- Australian region / South Pacific

SIGNAL QUALITY GUIDANCE:
- An active named storm with a forecast track toward the country is a
  very strong signal — include category, track, and timing.
- Seasonal outlooks need to be basin-specific. "Above-normal Atlantic
  season" is only relevant to Atlantic-basin countries.
- ENSO phase transitions (e.g., El Niño developing) are important
  multi-month signals.
- Historical landfall frequency for this specific country matters:
  a direct hit on a country that rarely gets hit is a stronger RC
  signal than one more storm hitting the Philippines.

Focus on authoritative sources: NOAA NHC, JTWC, regional RSMCs, WMO,
Tropical Storm Risk (UCL), OCHA, IFRC, national meteorological services.

{_OUTPUT_SCHEMA_BLOCK}"""


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

GROUNDING_PROMPT_BUILDERS = {
    "ACE": build_grounding_prompt_ace,
    "DR": build_grounding_prompt_dr,
    "FL": build_grounding_prompt_fl,
    "HW": build_grounding_prompt_hw,
    "TC": build_grounding_prompt_tc,
}


def build_grounding_prompt(
    hazard_code: str,
    country_name: str,
    iso3: str,
    **kwargs,
) -> str:
    """Dispatch to the appropriate per-hazard grounding prompt builder.

    Parameters
    ----------
    hazard_code : one of ACE, DR, FL, HW, TC
    country_name : full country name
    iso3 : ISO 3166-1 alpha-3 code
    **kwargs : passed to the hazard-specific builder (e.g., acled_context,
        season_context)
    """
    builder = GROUNDING_PROMPT_BUILDERS.get(hazard_code)
    if builder is None:
        raise ValueError(f"No grounding prompt builder for hazard code: {hazard_code}")
    return builder(country_name=country_name, iso3=iso3, **kwargs)


def get_recency_days(hazard_code: str) -> int:
    """Return the recommended recency window for a hazard's grounding call."""
    return RECENCY_DAYS.get(hazard_code, 120)
