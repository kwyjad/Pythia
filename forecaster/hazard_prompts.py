# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""
hazard_prompts.py — Hazard-specific reasoning guidance for SPD forecasting.

Each hazard type has fundamentally different generating processes. This module
provides tailored reasoning instructions that replace the generic "how to think
about this forecast" guidance in the SPD prompt.

Usage in build_spd_prompt_v2():
    from .hazard_prompts import get_hazard_reasoning_block
    hazard_block = get_hazard_reasoning_block(hazard_code, metric)
    # Insert into prompt where generic instructions currently sit
"""

from __future__ import annotations


def get_hazard_reasoning_block(hazard_code: str, metric: str) -> str:
    """Return hazard-specific reasoning guidance for the SPD prompt.

    Parameters
    ----------
    hazard_code : str
        Pythia hazard code (ACE, FL, DR, TC, HW, DI, etc.)
    metric : str
        Forecast metric (PA, FATALITIES)

    Returns
    -------
    str
        Multi-paragraph reasoning guidance tailored to the hazard/metric,
        or a generic fallback if the hazard is unrecognized.
    """
    hz = (hazard_code or "").upper().strip()
    m = (metric or "").upper().strip()

    if hz == "ACE" and m == "FATALITIES":
        return _ACE_FATALITIES
    if hz == "ACE" and m == "PA":
        return _ACE_PA
    if hz == "DI":
        return _DI
    if hz == "FL":
        return _FL
    if hz == "DR" and m == "PHASE3PLUS_IN_NEED":
        return _DR_PHASE3
    if hz == "DR":
        return _DR
    if hz == "TC":
        return _TC
    if hz == "HW":
        return _HW

    # Fallback for unrecognized hazard codes
    return _GENERIC


# ---------------------------------------------------------------------------
# Armed Conflict — Fatalities (ACE / FATALITIES)
# ---------------------------------------------------------------------------

_ACE_FATALITIES = """\
HAZARD-SPECIFIC REASONING GUIDANCE: ARMED CONFLICT — FATALITIES

This question asks about battle-related fatalities as recorded by ACLED. Conflict \
fatalities are driven by actor decisions, not natural processes. Keep the following \
in mind as you work through your Bayesian update:

Key reasoning principles for conflict fatalities:
- Conflict fatalities are HEAVY-TAILED and LUMPY. A single major battle or offensive \
can produce more fatalities in one month than the previous six months combined. Your \
SPD must reflect this: even in relatively calm periods, bucket 4/5 should rarely be \
zero unless there is a ceasefire or peace agreement with strong compliance.
- Base rates from ACLED history are your strongest prior anchor. If the Resolver \
history shows a country averaging 10–20 fatalities/month, your prior should be \
centred on bucket 2 (<25) with meaningful mass in bucket 1 (<5) and bucket 3 (25–100). \
Deviations from this base rate require specific evidence.
- VIEWS predicted fatalities and conflictforecast.org risk scores (if provided in \
structured data) are quantitative forecasts from ML models trained on conflict data. \
Use them as additional base-rate anchors: if VIEWS predicts elevated fatalities at \
lead 1–3, this is a signal to shift mass rightward. Weight VIEWS for trend direction \
and conflictforecast.org for escalation signals.
- Escalation dynamics matter more than levels. A country with 50 fatalities/month that \
is stable is different from one with 20/month that is rapidly escalating. Look for: \
new offensives announced, breakdown of ceasefires, new armed groups entering, external \
military intervention, or election-related violence cycles.
- De-escalation signals include: active peace talks with international mediation, \
ceasefires with monitoring mechanisms, seasonal patterns in fighting (e.g. rainy season \
in some regions reduces mobility), or exhaustion of combatant capacity.
- Month-to-month variation: conflict fatalities can shift rapidly. Your SPD should \
allow for meaningful month-to-month differences when there are time-specific signals \
(e.g. a planned offensive in month 2, elections in month 4). Do not default to \
identical SPDs across all 6 months unless the situation is genuinely static.
- ICG CrisisWatch flags (if present): "Deteriorated" or "Situation of Concern" flags \
are expert-curated signals. Treat them as moderate-strength evidence for rightward shift.

Common calibration errors to avoid:
- Anchoring too heavily on the most recent month rather than the 6–12 month trend.
- Treating "no major change" as evidence for bucket 1 when the base rate is bucket 2–3.
- Underweighting tail risk (bucket 5, >=500) in active conflict zones. Even in moderate \
conflicts, there is always some probability of a major escalation event.
- Overreacting to a single dramatic event when it is an outlier against a stable base rate.\
"""

# ---------------------------------------------------------------------------
# Armed Conflict — People Affected / Displacement (ACE / PA)
# ---------------------------------------------------------------------------

_ACE_PA = """\
HAZARD-SPECIFIC REASONING GUIDANCE: ARMED CONFLICT — DISPLACEMENT (PA)

This question asks about internally displaced people (IDPs) due to armed conflict, \
as recorded by IDMC with IOM DTM as fallback. Conflict displacement has different \
dynamics from conflict fatalities:

Key reasoning principles for conflict displacement:
- Displacement is STOCK-LIKE with FLOW SHOCKS. Monthly displacement figures reflect \
new displacements, not total IDP population. A major offensive can displace hundreds \
of thousands in a single month, then flows drop sharply — but the displaced people \
remain displaced. Your SPD should capture the probability of flow shocks.
- IDMC history may be SHORT or SPARSE. Unlike ACLED fatalities (which have strong \
monthly coverage), IDMC displacement data may have gaps or only cover recent years. \
If the history summary shows few months of data, widen your prior — you have less \
information to anchor on, so your SPD should be more spread across buckets.
- Displacement often LAGS fatalities. A battle in month 1 produces displacement in \
months 1–2 as civilians flee. If conflict fatalities are escalating, displacement \
typically follows with a 0–2 month lag. Factor this timing into month-specific SPDs.
- Scale depends on POPULATION DENSITY and GEOGRAPHY. The same intensity of conflict \
produces vastly different displacement in densely populated vs. sparsely populated \
areas, and in areas with accessible escape routes vs. besieged areas.
- Cross-border displacement is NOT counted here. This metric covers internal \
displacement only. If large populations are fleeing across borders rather than \
displacing internally, the PA figure may be lower than the conflict intensity suggests.
- Seasonal patterns: in some regions, displacement is seasonal (e.g. dry season \
offensives in South Sudan, lean season displacement in the Sahel). Check whether the \
forecast window overlaps with known seasonal patterns.

Common calibration errors to avoid:
- Confusing IDP stock (total displaced, often millions) with monthly new displacement \
flows (often tens of thousands). Your buckets measure monthly flows.
- Assuming displacement is proportional to fatalities — it is correlated but the \
relationship is nonlinear and context-dependent.
- Ignoring the possibility of mass displacement events (bucket 5, >=500k) in countries \
with large urban populations near conflict zones.\
"""

# ---------------------------------------------------------------------------
# Displacement Inflow (DI)
# ---------------------------------------------------------------------------

_DI = """\
HAZARD-SPECIFIC REASONING GUIDANCE: DISPLACEMENT INFLOW (DI)

This question asks about people entering a country due to events in NEIGHBOURING \
states — not internal displacement. There is NO Resolver base rate for DI.

Key reasoning principles for displacement inflow:
- You must CONSTRUCT your own prior. Use UNHCR/IOM flux estimates, known historical \
inflow episodes from neighbouring countries, and reference cases from similar \
situations. Typical monthly inflows during non-crisis periods are often in bucket 1 \
(<10k). During active neighbour-country crises, flows can surge to bucket 3–5 \
within weeks.
- The DRIVER is EXTERNAL. Focus your reasoning on what is happening in neighbouring \
countries, not domestic conditions. Key questions: Is there active conflict, famine, \
or state collapse in a neighbouring country? Are borders open or restricted? What \
is the geographic and ethnic/linguistic relationship between border communities?
- Inflows are EPISODIC and BINARY-ISH. Either a crisis in a neighbouring country is \
producing significant outflows toward this country, or it is not. The SPD should \
reflect this bimodal quality: heavy mass in bucket 1 during calm periods, with a \
clear probability mass in higher buckets if a neighbour-country crisis is active or \
plausible.
- Neighbouring country conflict/hazard signals: if the structured data includes \
information about armed conflict, drought, or food insecurity in neighbouring \
countries, treat these as the primary update signals. Check ACAPS risk radar and \
ReliefWeb reports for neighbour-country conditions.
- Border dynamics: government border policies, UNHCR registration capacity, and \
geographic barriers (rivers, mountains) all affect inflow volumes. A crisis in a \
neighbouring country with an open, flat border will produce larger inflows than one \
with restrictive border policies or geographic barriers.
- Month-to-month variation can be extreme. An inflow surge triggered by a new \
offensive in a neighbour can go from bucket 1 to bucket 4 in a single month.

Common calibration errors to avoid:
- Confusing internal displacement (ACE/PA) with cross-border inflows (DI). The \
metrics are measuring different things.
- Treating domestic conditions as the main driver — for DI, the generating process \
is external.
- Assuming steady-state flows when the situation is clearly deteriorating or \
improving in neighbouring countries.\
"""

# ---------------------------------------------------------------------------
# Flood (FL)
# ---------------------------------------------------------------------------

_FL = """\
HAZARD-SPECIFIC REASONING GUIDANCE: FLOOD (FL)

This question asks about people affected by flooding as recorded by IFRC Montandon.

Key reasoning principles for flood forecasting:
- Flooding is HIGHLY SEASONAL. Your prior should be strongly shaped by whether the \
forecast months overlap with the country's wet/rainy season. During dry season months, \
the prior should be heavily weighted toward bucket 1 (<10k). During peak rainy season, \
the prior should shift substantially rightward based on historical flood impacts.
- NMME seasonal outlook (if provided) is a key signal. Above-normal precipitation \
anomalies (positive σ) during rainy season months are a moderate-to-strong signal for \
rightward shift. Below-normal anomalies are a signal for leftward shift. The magnitude \
matters: +0.5σ is a modest signal, +1.5σ is a strong signal.
- ENSO and IOD phases affect flood risk regionally. La Niña typically increases flood \
risk in Southeast Asia, East Africa, and Australia. El Niño increases flood risk in \
Peru, Ecuador, and parts of East Africa. If these teleconnection patterns are relevant \
to this country, factor them in.
- Flood impacts are LUMPY and depend on specific events. A single major flood event \
can dominate the monthly PA figure. The difference between bucket 1 and bucket 3 may \
be a single cyclone making landfall or a river exceeding its flood stage.
- Urban vs. rural matters enormously for PA. Flooding in densely populated river \
basins or coastal cities produces far more people affected than equivalent rainfall \
in sparsely populated areas.
- IFRC Montandon data may be SPARSE for smaller events. The base rate from Resolver \
may undercount routine seasonal flooding if it only captures events large enough for \
IFRC reporting. Treat a sparse base rate as a lower bound, not the full picture.
- Month-to-month SPDs should vary with the seasonal cycle. Do not assign identical \
SPDs to a dry-season month and a peak-monsoon month.

Common calibration errors to avoid:
- Ignoring seasonality and assigning flat SPDs across all 6 months.
- Treating NMME anomalies as deterministic — they are probabilistic signals about \
conditions, not predictions of specific flood events.
- Underweighting tail risk during active monsoon/rainy seasons. Bucket 4–5 events \
do occur during extreme rainfall years.\
"""

# ---------------------------------------------------------------------------
# Drought (DR)
# ---------------------------------------------------------------------------

_DR = """\
HAZARD-SPECIFIC REASONING GUIDANCE: DROUGHT (DR)

This question asks about people affected by drought as recorded by IFRC Montandon.

Key reasoning principles for drought forecasting:
- Drought is SLOW-ONSET. Unlike floods or cyclones, drought impacts accumulate over \
months. A drought that begins in month 1 will typically produce increasing PA figures \
through months 2–4 as food security deteriorates, water sources dry up, and \
humanitarian needs grow. Your SPD should reflect this temporal pattern: if a drought \
is developing, later months should have more mass in higher buckets than earlier months.
- CUMULATIVE EFFECTS matter. Consecutive below-normal rainfall seasons compound \
drought impacts. If the country has already experienced one or more poor rainy \
seasons, the current forecast window starts from a weakened baseline — even \
near-normal rainfall may not prevent significant drought impacts.
- NMME seasonal outlook is especially important for drought. Below-normal \
precipitation anomalies (negative σ) are a direct driver. Temperature anomalies \
matter too: above-normal temperatures increase evapotranspiration and worsen \
drought conditions even with near-normal rainfall.
- IPC food insecurity phases (if provided in structured data) are one of the \
strongest signals for drought PA. IPC Phase 3+ (Crisis or worse) population \
estimates directly relate to how many people are drought-affected. If IPC data \
shows millions in Phase 3+, your SPD should reflect this.
- La Niña / El Niño: La Niña typically worsens drought in East Africa (Horn of \
Africa) and parts of Central America. El Niño can worsen drought in Southeast Asia, \
Southern Africa, and parts of South Asia. These are moderate-to-strong signals when \
they align with the forecast window.
- Drought PA figures can be VERY LARGE. Unlike conflict fatalities where bucket 5 \
(>=500k) is rare, drought can affect millions of people. In major drought events \
(Horn of Africa 2011, 2017, 2022), PA figures routinely exceed 500k per month. Do \
not treat bucket 5 as negligibly unlikely for drought-prone countries during dry seasons.
- Humanitarian response can REDUCE recorded PA. If a major drought response is \
underway, some affected populations may be reached by assistance and not appear in \
certain PA definitions. This is a modest downward signal.

Common calibration errors to avoid:
- Assigning flat SPDs across months for a developing drought — impacts should \
escalate over time.
- Ignoring compound effects from previous seasons.
- Underweighting IPC data when it is available — it is one of the most reliable \
forward-looking indicators for drought impacts.
- Treating drought as binary (either happening or not) rather than as a spectrum \
of severity.\
"""

# ---------------------------------------------------------------------------
# Drought — FEWS NET IPC Phase 3+ (DR / PHASE3PLUS_IN_NEED)
# ---------------------------------------------------------------------------

_DR_PHASE3 = """\
HAZARD-SPECIFIC REASONING GUIDANCE: DROUGHT — FEWS NET IPC PHASE 3+ (PHASE3PLUS_IN_NEED)

This question asks about the number of people in IPC Phase 3+ (Crisis or worse) food \
insecurity in {country}, as reported by FEWS NET's Current Situation assessment. \
Phase 3+ includes Crisis (Phase 3), Emergency (Phase 4), and Famine (Phase 5) populations.

Bucket interpretation for IPC Phase 3+ population:
- Bucket 1 (<100k): Minimal food insecurity — few or no populations in Crisis or worse. \
Typical for stable, food-secure countries or those with very small populations monitored \
by FEWS NET.
- Bucket 2 (100k–1M): Moderate crisis — sub-national pockets of food insecurity. Some \
districts or livelihood zones in IPC Phase 3, but national totals remain below 1 million. \
Common in countries with localized drought stress or seasonal lean-season peaks.
- Bucket 3 (1M–5M): Significant national-level food insecurity. Multiple regions in \
Crisis or Emergency phase. This is the range for countries with widespread but non-extreme \
drought or conflict-driven food insecurity (e.g. parts of the Sahel, East Africa in a \
moderate year).
- Bucket 4 (5M–15M): Severe food emergency. Large portions of the country in Phase 3+, \
often with significant Phase 4 (Emergency) populations. This is the range for major food \
crises (e.g. Ethiopia, DRC, or Nigeria in a bad year).
- Bucket 5 (>=15M): Catastrophic — Sudan/Ethiopia-scale famine risk. Phase 3+ populations \
exceed 15 million, implying a large fraction of the country's population is in Crisis or \
worse. Only a handful of the largest food crises in FEWS NET history have reached this \
level. Requires convergence of severe drought, conflict displacement, and economic collapse.

Key reasoning principles for IPC Phase 3+ forecasting:
- Phase 3+ populations are PERSISTENT and SLOW-CHANGING. Unlike conflict fatalities or \
flood PA, IPC Phase 3+ figures do not spike and drop within a single month. Food \
insecurity builds over weeks and months as rainfall deficits accumulate, food stocks \
deplete, and prices rise. Your SPD should reflect gradual trends rather than sharp \
month-to-month swings. Adjacent months should generally be similar unless there is a \
specific seasonal transition (e.g. harvest vs. lean season).
- FEWS NET ANALYSIS CYCLES MATTER. FEWS NET does not publish new IPC analyses for every \
country every month. Some countries are analysed quarterly, others twice a year. Between \
analysis cycles, the Phase 3+ figure effectively stays constant. If the most recent FEWS \
NET analysis is 2+ months old, the near-term forecast months will likely reflect that \
same figure unless a new analysis is imminent. Check the structured data for the most \
recent analysis date.
- SEASONAL DRIVERS are critical. Lean seasons (the months before harvest when food stocks \
are lowest) produce predictable spikes in Phase 3+ populations. The NMME seasonal outlook \
is especially relevant: below-normal precipitation during the growing season is a strong \
signal for higher Phase 3+ figures at the subsequent lean season. Above-normal rainfall \
during the growing season is a signal for improved food security at harvest.
- CONFLICT is a DOMINANT DRIVER in many FEWS NET-monitored countries. In countries like \
Sudan, South Sudan, DRC, Somalia, and northern Nigeria, conflict-driven displacement and \
market disruption are the primary cause of food insecurity, not drought alone. If conflict \
is escalating, Phase 3+ populations are likely to rise regardless of rainfall conditions.
- RESOLUTION SOURCE DISTINCTION: FEWS NET publishes both a "Current Situation" analysis \
(what is happening now) and a "Most Likely" projection (what will happen over the next \
4–8 months). This question resolves against the Current Situation analysis. However, the \
Most Likely projection (if provided in structured data as phase3plus_projection) is a \
strong forward-looking signal — treat it as moderate-to-strong evidence for the direction \
of change.
- CUMULATIVE EFFECTS matter. Consecutive poor rainy seasons compound food insecurity. If \
a country has already experienced one or more below-average seasons, even near-normal \
rainfall may not be sufficient to return Phase 3+ populations to low levels. Recovery \
from acute food insecurity takes months, not weeks.
- IPC data is the STRONGEST ANCHOR for this metric. When FEWS NET Current Situation or \
Most Likely projection data is available in the structured data, it should receive heavy \
weight in your Bayesian update. IPC data directly measures what this question asks about.

Common calibration errors to avoid:
- Assigning volatile month-to-month SPDs as if Phase 3+ populations fluctuate like \
conflict fatalities. Phase 3+ changes gradually — your SPDs across adjacent months \
should typically be similar or show a smooth trend.
- Ignoring the lean season / harvest season cycle. Phase 3+ peaks during lean seasons \
and drops after harvests. Your SPDs should reflect this seasonal pattern.
- Underweighting FEWS NET projection data when it is available. The Most Likely projection \
is produced by food security analysts and is one of the most reliable forward-looking \
indicators for this metric.
- Treating bucket 5 (>=15M) as negligible for large, crisis-affected countries. Sudan in \
2024–2025 and Ethiopia in severe drought years have exceeded 15M Phase 3+.
- Ignoring conflict signals in countries where conflict, not drought, is the primary \
food insecurity driver.\
"""

# ---------------------------------------------------------------------------
# Tropical Cyclone (TC)
# ---------------------------------------------------------------------------

_TC = """\
HAZARD-SPECIFIC REASONING GUIDANCE: TROPICAL CYCLONE (TC)

This question asks about people affected by tropical cyclones as recorded by IFRC \
Montandon.

Key reasoning principles for tropical cyclone forecasting:
- Cyclone impacts are EPISODIC and BINARY. In any given month, either a significant \
cyclone makes landfall near populated areas, or it does not. Your SPD should reflect \
this: during cyclone season months, there should be substantial mass in bucket 1 \
(no major landfall) alongside meaningful probability in higher buckets (a major \
landfall occurs). Outside cyclone season, the SPD should be very heavily weighted \
toward bucket 1.
- SEASONALITY is the dominant factor. Every cyclone basin has a well-defined season:
  - Atlantic/Caribbean: June–November (peak Aug–Oct)
  - Western Pacific/Philippines: May–December (peak Jul–Nov)
  - Bay of Bengal/South Asia: April–June and October–December
  - Southwest Indian Ocean/Madagascar: November–April
  - South Pacific/Fiji: November–April
  If the forecast window is entirely outside the relevant cyclone season, bucket 1 \
should receive 90%+ probability for those months.
- FREQUENCY vs. IMPACT is a two-stage process. Think about cyclone risk as: \
(probability of a cyclone forming and tracking toward the country) × (impact if it \
makes landfall). Seasonal forecasts affect the first factor. Country vulnerability, \
population exposure, and cyclone intensity affect the second.
- NMME and seasonal cyclone outlooks (if available) provide signals about basin-wide \
activity levels. Above-normal SSTs in the relevant basin increase cyclone frequency. \
La Niña typically increases Atlantic hurricane activity but suppresses Eastern Pacific. \
El Niño does the reverse.
- A single major cyclone can produce EXTREME PA. Cyclone Idai (Mozambique 2019) \
affected 1.85 million people. During active cyclone seasons in exposed countries, \
bucket 5 (>=500k) is a realistic possibility, not a tail event.
- Track uncertainty is high at monthly horizons. You cannot predict whether a specific \
cyclone will hit a specific country months in advance. Your SPD should reflect this \
fundamental uncertainty rather than trying to predict specific events.

Common calibration errors to avoid:
- Assigning the same SPD to cyclone-season and off-season months.
- Treating cyclone risk as zero outside peak season — early/late season storms do occur.
- Underweighting the impact of a single major cyclone (the distribution of cyclone \
impacts is extremely heavy-tailed).
- Ignoring the two-stage nature of cyclone risk (formation probability × landfall impact).\
"""

# ---------------------------------------------------------------------------
# Heatwave (HW)
# ---------------------------------------------------------------------------

_HW = """\
HAZARD-SPECIFIC REASONING GUIDANCE: HEATWAVE (HW)

This question asks about people affected by heatwaves as recorded by IFRC Montandon.

Key reasoning principles for heatwave forecasting:
- Heatwave impacts are SEASONAL and PREDICTABLE in timing. Unlike cyclones or \
conflict, the timing of heatwave risk is highly predictable: it occurs during the \
hot season for each country. Your SPD should have the strongest rightward shift \
during known hot-season months and be heavily weighted toward bucket 1 during \
cool-season months.
- NMME temperature anomalies are a direct driver. Above-normal temperature forecasts \
(positive σ) during the hot season are a moderate-to-strong signal for higher PA. \
The combination of above-normal temperatures AND above-normal humidity (or below-normal \
precipitation limiting cooling) is the strongest signal.
- Climate trend matters. Global warming means heatwave base rates are shifting upward \
over time. Historical averages from 10+ years ago may underestimate current risk. If \
the Resolver history is from an earlier period, consider adjusting upward slightly.
- IFRC Montandon data for heatwaves may be VERY SPARSE. Many countries have minimal \
recorded heatwave PA in the Montandon database because heatwave impacts are often \
undercounted (excess mortality may not be attributed to heat, and displacement from \
heat is rare). If the base rate shows mostly zeros, this may be a data limitation \
rather than evidence of no risk.
- Population vulnerability: heatwave impacts are disproportionately concentrated in \
countries with large outdoor labor forces, limited air conditioning, urban heat \
islands, and vulnerable populations (elderly, children). South Asia, the Middle East, \
and parts of sub-Saharan Africa are highest risk.
- Unlike other hazards, heatwave PA rarely reaches bucket 5 (>=500k) in IFRC \
Montandon data, because the metric captures people affected enough to need \
humanitarian assistance, not total heat-exposed population. Bucket 3–4 is a more \
realistic ceiling for most countries in most years, though extreme events in highly \
exposed countries (India, Pakistan) can exceed this.

Common calibration errors to avoid:
- Ignoring seasonality — assigning heatwave risk to winter months.
- Overweighting heatwave risk in countries where IFRC Montandon has no historical \
record, unless there is strong evidence of a novel extreme event.
- Confusing "people exposed to high temperatures" (which can be hundreds of millions) \
with "people affected as humanitarian agencies would record" (which is much smaller).
- Treating heatwave risk as static across months when the hot season clearly peaks \
in specific months.\
"""

# ---------------------------------------------------------------------------
# Generic fallback
# ---------------------------------------------------------------------------

_GENERIC = """\
HAZARD-SPECIFIC REASONING GUIDANCE

No hazard-specific reasoning guidance is available for this hazard type. Apply \
general Bayesian principles:
- Anchor on the Resolver base rate if available.
- Identify the 3–6 most decision-relevant update signals.
- Consider seasonality if the hazard has seasonal patterns.
- Think about whether the generating process is episodic (single events dominate) \
or cumulative (impacts build over time).
- Ensure your SPD reflects genuine uncertainty — avoid overconfident narrow \
distributions unless the evidence is very strong.\
"""
