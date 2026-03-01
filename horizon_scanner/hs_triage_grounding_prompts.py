# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Per-hazard Google Grounding query builders for TRIAGE evidence packs.

These are SEPARATE from the RC grounding prompts. The distinction:

  RC grounding asks:   "What is CHANGING?"
  Triage grounding asks: "What is the CURRENT SITUATION?"

RC grounding hunts for novelty — signals that the pattern is breaking.
Triage grounding hunts for the operational picture — what's happening now,
how severe is it, what's the humanitarian response status, what does the
upcoming season look like.

If you feed RC-flavored evidence into triage, the triage model will
overweight emerging signals and underweight chronic/ongoing conditions.
If you feed triage-flavored evidence into RC, the RC model will be biased
toward the status quo and miss emerging breaks. Separate calls solve this.

Pipeline:
    1. RC grounding   → RC prompt   → RC result
    2. Triage grounding → Triage prompt (with RC result injected) → Triage result

Cost: ~300 additional grounding calls per run (on top of ~300 RC grounding
calls), for ~600 total. Each is a Gemini Flash call at ~768 output tokens.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Recency windows per hazard (days) — triage looks further back than RC
# ---------------------------------------------------------------------------
# Triage needs to capture the full operational picture, including ongoing
# crises that may have been developing for months.

RECENCY_DAYS = {
    "ACE": 120,  # Ongoing conflict context needs longer window
    "DR": 180,   # Drought impacts unfold over seasons
    "FL": 120,   # Flood season + aftermath
    "HW": 120,   # Seasonal pattern + any ongoing heat events
    "TC": 120,   # Full season context
}


# ---------------------------------------------------------------------------
# Output schema (shared, same as RC grounding for parser compatibility)
# ---------------------------------------------------------------------------

_OUTPUT_SCHEMA_BLOCK = """\
Return strictly JSON with this shape:
{
  "recent_signals": ["<=8 bullets, see format below"],
  "structural_context": "max 8 lines of background context",
  "data_gaps": "what key information you could NOT find",
  "notes": "optional"
}

Signal bullet format (one per bullet):
  CATEGORY | TIMEFRAME | DIRECTION | signal text

Where:
  CATEGORY = one of: SITUATION, RESPONSE, FORECAST, VULNERABILITY
  TIMEFRAME = approximate: current, month_1, month_1-2, month_3-6, ongoing
  DIRECTION = WORSENING, IMPROVING, STABLE, UNCLEAR

Example bullet:
  SITUATION | current | STABLE | OCHA reports 2.4M people affected by ongoing flooding in southern provinces, with 340,000 displaced since August. Humanitarian response covering approximately 60% of those in need.

Rules:
- Prioritize CURRENT OPERATIONAL PICTURE over background information.
- Include humanitarian RESPONSE status where available — not just the
  hazard, but how well the response is covering needs.
- Include QUANTIFIED impacts: people affected, displaced, food insecure,
  mortality figures.
- Provide FORWARD-LOOKING context: seasonal forecasts, planned response
  scale-up or scale-down, upcoming risk windows.
- If you find no active situation for this hazard, say so clearly in
  data_gaps and provide background structural context instead.
- Do not include URLs in the JSON text (they come from grounding metadata).
- structural_context should cover: governance capacity, infrastructure,
  historical frequency/severity, humanitarian access constraints.\
"""


# ---------------------------------------------------------------------------
# ACE — Armed Conflict Events (Triage)
# ---------------------------------------------------------------------------

def build_triage_grounding_prompt_ace(
    country_name: str,
    iso3: str,
    recency_days: int = RECENCY_DAYS["ACE"],
    rc_summary: Optional[str] = None,
) -> str:
    """Build triage grounding prompt for ACE.

    Unlike RC grounding (which hunts for escalation/de-escalation signals),
    triage grounding builds the full operational picture: who is fighting,
    where, how many are affected, what the humanitarian response looks like.

    Parameters
    ----------
    rc_summary : optional one-line summary of the RC result to help focus
        the search. E.g., "RC assessed LOW likelihood of change — conflict
        is ongoing and stable" or "RC assessed MODERATE likelihood of
        escalation based on ceasefire breakdown."
    """

    rc_block = ""
    if rc_summary:
        rc_block = (
            f"\n\nPrior RC assessment summary: {rc_summary}\n"
            f"Use this to calibrate your search — if RC flagged something "
            f"specific, look for supporting or contradicting evidence.\n"
        )

    return f"""\
You are a humanitarian situation analyst using Google Search grounding.
Build a CURRENT OPERATIONAL PICTURE of armed conflict and political violence
in {country_name} ({iso3}) over the last {recency_days} days.

You are NOT primarily looking for change signals (that was done in a prior
step). You are building a comprehensive picture of the CURRENT STATE of
conflict in this country for humanitarian risk assessment.
{rc_block}
SEARCH PRIORITIES (in order):
1. Conflict overview: Who are the active parties? Where are the main
   areas of fighting? What is the current trajectory — active combat,
   stalemate, low-intensity? Search for OCHA situation reports, Crisis
   Group briefings, ACLED analyses, UN Secretary-General reports.
2. Humanitarian impact: How many people are affected by the conflict?
   Displacement numbers, civilian casualties, humanitarian access
   constraints, protection concerns. Search for OCHA, UNHCR, ICRC
   updates.
3. Humanitarian response: What is the scale of the response? Is there
   a Humanitarian Response Plan? What percentage of the appeal is
   funded? Are humanitarian organizations able to access affected
   populations? Search for Financial Tracking Service, OCHA dashboards.
4. Peace/political process: Is there an active peace process? What
   stage is it at? Is there a political transition underway? Elections
   upcoming? Search for mediation updates, UN political mission reports.
5. Regional context: Cross-border dimensions — refugee flows, arms
   flows, regional diplomatic efforts, peacekeeping mandates.
6. Civilian protection: reports of violations of international
   humanitarian law, targeting of civilians, sexual violence, child
   recruitment, attacks on schools/hospitals.

FOCUS ON QUANTIFIED IMPACTS:
- "3.2M people displaced" is much more useful than "significant displacement."
- "Humanitarian access restricted in 4 of 10 provinces" is better than
  "access challenges reported."
- "HRP 35% funded as of February 2026" gives concrete response context.

Focus on authoritative sources: OCHA (ReliefWeb, situation reports),
UNHCR, ICRC, Crisis Group, Security Council reports, ACLED analysis
pieces, humanitarian dashboards.

{_OUTPUT_SCHEMA_BLOCK}"""


# ---------------------------------------------------------------------------
# DR — Drought (Triage)
# ---------------------------------------------------------------------------

def build_triage_grounding_prompt_dr(
    country_name: str,
    iso3: str,
    recency_days: int = RECENCY_DAYS["DR"],
    rc_summary: Optional[str] = None,
    season_context: Optional[str] = None,
) -> str:
    """Build triage grounding prompt for DR.

    Triage grounding for drought focuses on the FOOD SECURITY PICTURE
    and HUMANITARIAN RESPONSE, not just rainfall anomalies (which the
    structured data feeds and RC grounding handle).
    """

    rc_block = ""
    if rc_summary:
        rc_block = (
            f"\n\nPrior RC assessment summary: {rc_summary}\n"
            f"Use this to calibrate your search.\n"
        )

    season_block = ""
    if season_context:
        season_block = (
            f"\n\nSeason context: {season_context}\n"
            f"Frame your search around the current/upcoming season.\n"
        )

    return f"""\
You are a food security and drought situation analyst using Google Search grounding.
Build a CURRENT OPERATIONAL PICTURE of drought and food security conditions
in {country_name} ({iso3}) over the last {recency_days} days.

You are building a comprehensive picture of the CURRENT STATE for
humanitarian risk assessment — ongoing food insecurity, drought impacts
already felt, humanitarian response coverage, and the seasonal outlook.
{rc_block}{season_block}
SEARCH PRIORITIES (in order):
1. Food security status: What is the current IPC/CH classification?
   How many people are food insecure (IPC Phase 3+)? Is the situation
   improving or deteriorating? Search for FEWS NET food security
   outlook, IPC analyses, WFP situation reports.
2. Agricultural conditions: How is the current growing season performing?
   Crop production forecasts, harvest estimates, livestock conditions.
   Search for FAO GIEWS crop prospects, government agricultural reports.
3. Humanitarian response: Scale of food assistance — how many people
   receiving food aid? What modalities (in-kind, cash, vouchers)?
   Is the response adequately funded? Any pipeline breaks? Search
   for WFP operations updates, HRP funding status, CERF allocations.
4. Market conditions: Staple food prices — are they above seasonal
   norms? Supply chain disruptions? Import dependency and terms of
   trade. Search for WFP market monitoring, VAM bulletins.
5. Nutrition: Acute malnutrition rates, nutrition surveys, SAM/MAM
   admission trends, nutrition response coverage. Search for
   UNICEF nutrition dashboards, Nutrition Cluster updates.
6. Water and sanitation: Drinking water availability, water point
   functionality, pastoralist water access. These are often early
   drought impact indicators.
7. Coping mechanisms: Are households employing negative coping
   strategies (selling productive assets, reducing meals, pulling
   children from school)? Are coping capacities being exhausted?

FOCUS ON QUANTIFIED IMPACTS:
- "4.3M people in IPC Phase 3+, of which 1.2M in Phase 4" gives a
  precise severity picture.
- "Maize prices 40% above 5-year average" quantifies market stress.
- "WFP reaching 1.8M of 3.5M targeted" shows response gap.

Focus on authoritative sources: FEWS NET, WFP (VAM, situation reports),
FAO GIEWS, IPC, OCHA, Nutrition Cluster, UNICEF.

{_OUTPUT_SCHEMA_BLOCK}"""


# ---------------------------------------------------------------------------
# FL — Flood (Triage)
# ---------------------------------------------------------------------------

def build_triage_grounding_prompt_fl(
    country_name: str,
    iso3: str,
    recency_days: int = RECENCY_DAYS["FL"],
    rc_summary: Optional[str] = None,
    season_context: Optional[str] = None,
) -> str:
    """Build triage grounding prompt for FL."""

    rc_block = ""
    if rc_summary:
        rc_block = (
            f"\n\nPrior RC assessment summary: {rc_summary}\n"
            f"Use this to calibrate your search.\n"
        )

    season_block = ""
    if season_context:
        season_block = (
            f"\n\nSeason context: {season_context}\n"
            f"Frame your search around the current/upcoming season.\n"
        )

    return f"""\
You are a disaster and flood situation analyst using Google Search grounding.
Build a CURRENT OPERATIONAL PICTURE of flood conditions and flood risk in
{country_name} ({iso3}) over the last {recency_days} days.

You are building a comprehensive picture of current flood impact,
ongoing response, and forward-looking seasonal risk.
{rc_block}{season_block}
SEARCH PRIORITIES (in order):
1. Active flood situation: Are there active floods? Where? How many
   people affected and displaced? What infrastructure is damaged?
   Search for OCHA flash updates, situation reports, national disaster
   management agency reports, FloodList.
2. Humanitarian impact and needs: Displacement numbers, shelter needs,
   WASH concerns (contaminated water post-flood), crop/livelihood
   damage, disease outbreaks (cholera, malaria spikes post-flood).
   Search for OCHA, IFRC, IOM DTM displacement tracking.
3. Humanitarian response: Flood response operations, search and rescue,
   relief distributions, emergency shelter, CERF allocations, flash
   appeals. Search for OCHA, IFRC, cluster updates.
4. Seasonal context: Where is the country in its rainy/monsoon season?
   Is it just beginning, at peak, or winding down? What is the
   rainfall forecast for the remainder of the season? Search for
   national meteorological service seasonal outlooks.
5. Infrastructure and preparedness: Dam conditions, levee status,
   flood early warning system coverage, government preparedness
   measures, pre-positioned stocks.
6. Recovery from prior events: If the country experienced flooding
   earlier in the season, what is the recovery status? Are communities
   still displaced? Has infrastructure been repaired? Accumulated
   damage from multiple events compounds vulnerability.

FOCUS ON QUANTIFIED IMPACTS:
- "Flooding has affected 1.8M people across 12 provinces, with 230,000
  displaced" gives precise scale.
- "45,000 hectares of cropland inundated during the main growing season"
  quantifies agricultural impact.

Focus on authoritative sources: OCHA (flash updates, sitreps), IFRC,
FloodList, IOM DTM, national disaster management agencies, WFP (food
security implications), WASH Cluster.

{_OUTPUT_SCHEMA_BLOCK}"""


# ---------------------------------------------------------------------------
# HW — Heatwave (Triage)
# ---------------------------------------------------------------------------

def build_triage_grounding_prompt_hw(
    country_name: str,
    iso3: str,
    recency_days: int = RECENCY_DAYS["HW"],
    rc_summary: Optional[str] = None,
    season_context: Optional[str] = None,
) -> str:
    """Build triage grounding prompt for HW.

    Heatwave humanitarian data is sparse. The grounding call needs to
    look broadly — meteorological reports, health system data, energy
    sector reports, agricultural impact — because there is rarely a
    single "heatwave situation report" from OCHA.
    """

    rc_block = ""
    if rc_summary:
        rc_block = (
            f"\n\nPrior RC assessment summary: {rc_summary}\n"
            f"Use this to calibrate your search.\n"
        )

    season_block = ""
    if season_context:
        season_block = (
            f"\n\nSeason context: {season_context}\n"
            f"Frame your search around the current/upcoming season.\n"
        )

    return f"""\
You are a climate and health situation analyst using Google Search grounding.
Build a CURRENT OPERATIONAL PICTURE of heat conditions and heatwave risk
in {country_name} ({iso3}) over the last {recency_days} days.

Heatwave is the LEAST well-documented hazard in humanitarian reporting.
You may need to look beyond OCHA and ReliefWeb into meteorological services,
health system reports, energy sector news, and local media.
{rc_block}{season_block}
SEARCH PRIORITIES (in order):
1. Current heat conditions: Have there been recent heatwave events?
   What were the temperatures relative to historical norms? Duration?
   Geographic extent? Search for national meteorological service
   reports, WMO statements, Copernicus Climate Change Service monthly
   bulletins.
2. Health impacts: Heat-related mortality and morbidity. Hospital
   admissions, official death tolls, WHO or Ministry of Health
   advisories. Search for WHO, health ministry reports, local media
   reporting on heat deaths. NOTE: heat mortality is vastly
   underreported — absence of reports does NOT mean absence of impact.
3. Infrastructure impacts: Power grid stress or failures, water supply
   strain, transport disruptions. These affect millions even when not
   directly killing people. Search for energy sector news, government
   advisories.
4. Seasonal forecast: What are IRI/CPC/ECMWF seasonal temperature
   forecasts for the upcoming months? Is the forecast above normal?
   Search for seasonal climate outlooks relevant to this region.
5. Vulnerable populations: Are there specific populations at elevated
   risk — outdoor workers, displaced populations in tents/informal
   shelters, elderly in urban heat islands, areas with unreliable
   power/water? Search for IOM camp condition reports, urban
   vulnerability assessments.
6. Government/humanitarian response: Heat action plans activated,
   cooling centers opened, work hour restrictions, water distribution.
   These indicate the authorities are taking the heat seriously.

IMPORTANT — DATA SCARCITY GUIDANCE:
If you find very little information about heatwave conditions in this
country, this does NOT mean there is no risk. It likely means the hazard
is underreported. State clearly in data_gaps what you could not find,
and provide whatever structural/seasonal context is available.

Focus on authoritative sources: WMO, national meteorological services,
WHO, Copernicus C3S, IRI, health ministries, OCHA (when available),
energy sector reporting.

{_OUTPUT_SCHEMA_BLOCK}"""


# ---------------------------------------------------------------------------
# TC — Tropical Cyclone (Triage)
# ---------------------------------------------------------------------------

def build_triage_grounding_prompt_tc(
    country_name: str,
    iso3: str,
    recency_days: int = RECENCY_DAYS["TC"],
    rc_summary: Optional[str] = None,
    season_context: Optional[str] = None,
) -> str:
    """Build triage grounding prompt for TC."""

    rc_block = ""
    if rc_summary:
        rc_block = (
            f"\n\nPrior RC assessment summary: {rc_summary}\n"
            f"Use this to calibrate your search.\n"
        )

    season_block = ""
    if season_context:
        season_block = (
            f"\n\nSeason context: {season_context}\n"
            f"Frame your search around the current season status.\n"
        )

    return f"""\
You are a tropical cyclone situation analyst using Google Search grounding.
Build a CURRENT OPERATIONAL PICTURE of tropical cyclone conditions and risk
for {country_name} ({iso3}) over the last {recency_days} days.

You are building the full TC risk picture: active storms, recent impacts
still being recovered from, seasonal outlook, and preparedness status.
{rc_block}{season_block}
SEARCH PRIORITIES (in order):
1. Active storms: Any tropical cyclones currently active or recently
   active in the basin affecting this country. Current position,
   intensity, forecast track, expected landfall. Search for JTWC,
   NHC, or regional RSMC advisories.
2. Recent TC impacts: If a cyclone recently hit or passed near the
   country, what was the humanitarian impact? Displacement, damage,
   casualties, ongoing relief operations. Search for OCHA flash
   updates, IFRC emergency appeals, national disaster reports.
3. Ongoing recovery: If the country was hit by a TC earlier in the
   season, what is the recovery status? Are people still displaced?
   Has infrastructure been repaired? Lingering vulnerability to
   the next storm. Search for OCHA situation reports, recovery
   updates, shelter cluster reports.
4. Seasonal outlook: What is the seasonal TC forecast for the relevant
   basin? Above/below/near normal? How much of the season remains?
   Search for NOAA/CPC seasonal outlook, TSR forecasts, WMO
   statements.
5. Preparedness: Government pre-positioning, evacuation plan status,
   early warning system functionality, humanitarian contingency
   planning. If agencies are pre-positioning, it signals elevated
   concern. Search for government disaster preparedness announcements,
   OCHA contingency plans.
6. Compound vulnerability: Is the country already dealing with other
   crises (conflict, flooding, drought) that would amplify TC impact?
   Limited response capacity from prior disasters?

FOCUS ON QUANTIFIED IMPACTS:
- "Cyclone X made landfall as a Category 3 with sustained winds of
  185 km/h, affecting 500,000 people" gives precise impact context.
- "Recovery from Cyclone Y (3 months ago) is only 40% complete, with
  45,000 people still in temporary shelters" shows lingering vulnerability.

Focus on authoritative sources: NOAA NHC, JTWC, regional RSMCs, OCHA,
IFRC, national disaster management agencies, WMO, TSR.

{_OUTPUT_SCHEMA_BLOCK}"""


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

TRIAGE_GROUNDING_BUILDERS = {
    "ACE": build_triage_grounding_prompt_ace,
    "DR": build_triage_grounding_prompt_dr,
    "FL": build_triage_grounding_prompt_fl,
    "HW": build_triage_grounding_prompt_hw,
    "TC": build_triage_grounding_prompt_tc,
}


def build_triage_grounding_prompt(
    hazard_code: str,
    country_name: str,
    iso3: str,
    **kwargs,
) -> str:
    """Dispatch to the appropriate per-hazard triage grounding prompt builder.

    Parameters
    ----------
    hazard_code : one of ACE, DR, FL, HW, TC
    country_name : full country name
    iso3 : ISO 3166-1 alpha-3 code
    **kwargs : passed to hazard-specific builder (e.g., rc_summary,
        season_context)
    """
    builder = TRIAGE_GROUNDING_BUILDERS.get(hazard_code)
    if builder is None:
        raise ValueError(
            f"No triage grounding prompt builder for hazard code: {hazard_code}"
        )
    return builder(country_name=country_name, iso3=iso3, **kwargs)


def get_recency_days(hazard_code: str) -> int:
    """Return the recommended recency window for triage grounding."""
    return RECENCY_DAYS.get(hazard_code, 120)
