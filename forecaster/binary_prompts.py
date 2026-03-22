# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Binary event prompt builder for EVENT_OCCURRENCE questions.

Binary questions produce a single probability per month (not a 5-bucket SPD).
They ask: "Will GDACS report a significant event (Orange/Red alert)?"
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Any

LOG = logging.getLogger(__name__)


def build_binary_event_prompt(
    *,
    question: dict,
    base_rate: dict,
    current_alerts: list[dict],
    structured_data: dict,
    hs_triage_entry: dict | None = None,
    today: str,
    gdacs_event_history: dict | None = None,
) -> str:
    """Build the full prompt for a binary event forecast.

    Parameters
    ----------
    question : dict
        Question metadata (iso3, hazard_code, metric, wording, etc.)
    base_rate : dict
        Output of build_binary_base_rate() — historical event rates.
    current_alerts : list[dict]
        Recent GDACS alerts for the country/hazard.
    structured_data : dict
        Structured data bundle (NMME, ACAPS, ReliefWeb, etc.)
    hs_triage_entry : dict | None
        Horizon Scanner triage entry for context.
    today : str
        Today's date as ISO string.
    gdacs_event_history : dict | None
        GDACS event occurrence history for seasonal frequency context.

    Returns
    -------
    str
        Complete prompt for the LLM.
    """
    iso3 = question.get("iso3", "???")
    hazard_code = (question.get("hazard_code") or "").upper()
    country = question.get("country_name") or iso3

    hazard_names = {"DR": "drought", "FL": "flooding", "TC": "tropical cyclone"}
    hazard_name = hazard_names.get(hazard_code, hazard_code)

    # Derive forecast months from question window
    window_start = question.get("window_start_date")
    forecast_months = _derive_forecast_months(window_start)

    sections = []

    # Section 1: Role and task
    sections.append(_section_role_and_task(country, hazard_name))

    # Section 2: Base rate
    sections.append(_section_base_rate(country, hazard_name, base_rate))

    # Section 3: Current situation
    sections.append(_section_current_situation(
        current_alerts, structured_data, hs_triage_entry, country, hazard_code
    ))

    # Section 3b: GDACS event history (seasonal frequency context)
    if gdacs_event_history:
        try:
            from forecaster.prompts import _format_gdacs_event_history_for_prompt
            cal_months = []
            for fm in forecast_months:
                try:
                    cal_months.append(int(fm.split("-")[1]))
                except (IndexError, ValueError):
                    pass
            gdacs_block = _format_gdacs_event_history_for_prompt(
                gdacs_event_history, cal_months
            )
            if gdacs_block:
                sections.append(gdacs_block)
        except Exception:
            pass

    # Section 4: Hazard-specific reasoning
    sections.append(get_binary_hazard_reasoning_block(hazard_code))

    # Section 5: Output instructions
    sections.append(_section_output_instructions(forecast_months))

    return "\n\n".join(s for s in sections if s)


def _derive_forecast_months(window_start) -> list[str]:
    """Derive 6 forecast month labels from window_start_date."""
    if isinstance(window_start, str):
        try:
            parts = window_start.split("-")
            y, m = int(parts[0]), int(parts[1])
        except (IndexError, ValueError):
            return [f"month_{i}" for i in range(1, 7)]
    elif isinstance(window_start, date):
        y, m = window_start.year, window_start.month
    else:
        return [f"month_{i}" for i in range(1, 7)]

    months = []
    for _ in range(6):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def _section_role_and_task(country: str, hazard_name: str) -> str:
    return f"""\
ROLE AND TASK

You are a careful probabilistic forecaster specializing in humanitarian \
event prediction. Your task is to estimate the probability that a \
significant {hazard_name} event will affect {country} during each of \
the next 6 months.

A "significant event" is defined as: GDACS reports an Orange or Red alert \
level {hazard_name} event with {country} in the affected countries list.
- Orange alert: "Potential need for international assistance"
- Red alert: "Likely need for international assistance"
- Green alerts (minor events) do NOT count.

You are being scored with the Brier score: (your_probability - outcome)^2
Lower is better. A well-calibrated forecaster assigns 10% to events that \
happen 10% of the time."""


def _section_base_rate(country: str, hazard_name: str, base_rate: dict) -> str:
    if not base_rate:
        return f"HISTORICAL BASE RATE: No historical data available for {country} / {hazard_name}."

    total_months = base_rate.get("total_months", 0)
    event_months = base_rate.get("event_months", 0)
    base_rate_pct = base_rate.get("base_rate_pct", 0.0)
    seasonal = base_rate.get("seasonal_pattern", {})
    recent_12m_events = base_rate.get("recent_12m_events", 0)
    recent_12m_rate = base_rate.get("recent_12m_rate", 0.0)
    trend = base_rate.get("trend", "unknown")

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    seasonal_lines = []
    for i, name in enumerate(month_names, 1):
        pct = seasonal.get(str(i), seasonal.get(i, 0.0))
        seasonal_lines.append(f"  {name}: {pct:.0f}%")

    seasonal_row1 = "  ".join(seasonal_lines[:6])
    seasonal_row2 = "  ".join(seasonal_lines[6:])

    return f"""\
HISTORICAL BASE RATE (GDACS, 2015\u2013present):
{country} has had significant {hazard_name} events in {event_months} of \
{total_months} months ({base_rate_pct:.1f}%).

Seasonal pattern (% of months with events by calendar month):
{seasonal_row1}
{seasonal_row2}

Recent 12 months: {recent_12m_events} events ({recent_12m_rate:.1f}%)
Trend: {trend}"""


def _section_current_situation(
    current_alerts: list[dict],
    structured_data: dict,
    hs_triage_entry: dict | None,
    country: str,
    hazard_code: str,
) -> str:
    parts = ["CURRENT SITUATION"]

    # Current GDACS alerts
    if current_alerts:
        parts.append(f"\nActive GDACS alerts for {country}:")
        for alert in current_alerts[:10]:
            level = alert.get("alertlevel", "?")
            event = alert.get("event_name", alert.get("event_id", "?"))
            ym = alert.get("ym", "?")
            parts.append(f"  - [{level}] {event} ({ym})")
    else:
        parts.append(f"\nNo active GDACS alerts for {country}.")

    # Structured data injection (reuse existing formatted data where possible)
    if structured_data:
        # NMME seasonal outlook
        nmme = structured_data.get("nmme_seasonal_outlook") or structured_data.get("nmme")
        if nmme:
            if isinstance(nmme, str):
                parts.append(f"\nNMME SEASONAL OUTLOOK:\n{nmme}")
            elif isinstance(nmme, dict):
                parts.append(f"\nNMME SEASONAL OUTLOOK:\n{json.dumps(nmme, indent=2)}")

        # ENSO
        enso = structured_data.get("enso") or structured_data.get("enso_context")
        if enso:
            if isinstance(enso, str):
                parts.append(f"\nENSO STATE:\n{enso}")

        # ACAPS INFORM severity
        inform = structured_data.get("inform_severity") or structured_data.get("acaps_inform_severity")
        if inform:
            if isinstance(inform, str):
                parts.append(f"\nINFORM SEVERITY:\n{inform}")

        # Risk radar
        risk_radar = structured_data.get("risk_radar") or structured_data.get("acaps_risk_radar")
        if risk_radar:
            if isinstance(risk_radar, str):
                parts.append(f"\nACAPS RISK RADAR:\n{risk_radar}")

        # Seasonal TC outlook (for TC)
        if hazard_code == "TC":
            tc_outlook = structured_data.get("seasonal_tc") or structured_data.get("seasonal_tc_context")
            if tc_outlook:
                if isinstance(tc_outlook, str):
                    parts.append(f"\nSEASONAL TC OUTLOOK:\n{tc_outlook}")

        # ReliefWeb reports
        reliefweb = structured_data.get("reliefweb") or structured_data.get("reliefweb_reports")
        if reliefweb:
            if isinstance(reliefweb, str):
                parts.append(f"\nRECENT RELIEFWEB REPORTS:\n{reliefweb}")
            elif isinstance(reliefweb, list):
                for rpt in reliefweb[:5]:
                    title = rpt.get("title", "")
                    if title:
                        parts.append(f"  - {title}")

    # HS triage context
    if hs_triage_entry:
        triage_score = hs_triage_entry.get("triage_score", "?")
        tier = hs_triage_entry.get("tier", "?")
        parts.append(f"\nHORIZON SCANNER TRIAGE: score={triage_score}, tier={tier}")

    return "\n".join(parts)


def _section_output_instructions(forecast_months: list[str]) -> str:
    months_str = ", ".join(f'"{m}"' for m in forecast_months)
    return f"""\
OUTPUT INSTRUCTIONS

For EACH of the 6 forecast months ({", ".join(forecast_months)}), provide:
1. Your prior probability (from base rate + seasonality alone)
2. Key evidence updates that shift the probability up or down
3. Your final posterior probability

Respond with a JSON object:
{{
  "months": {{
    "YYYY-MM": {{
      "prior": 0.XX,
      "evidence_updates": ["update 1", "update 2"],
      "posterior": 0.XX,
      "reasoning": "brief explanation"
    }}
  }}
}}

All probabilities must be between 0.01 and 0.99. Never assign exactly \
0 or 1 \u2014 there is always some residual uncertainty."""


# ---- Binary hazard reasoning blocks ----

def get_binary_hazard_reasoning_block(hazard_code: str) -> str:
    """Return hazard-specific reasoning guidance for binary event prediction."""
    hz = (hazard_code or "").upper().strip()
    if hz == "DR":
        return _BINARY_DR
    if hz == "FL":
        return _BINARY_FL
    if hz == "TC":
        return _BINARY_TC
    return _BINARY_GENERIC


_BINARY_DR = """\
HAZARD-SPECIFIC REASONING: DROUGHT (BINARY EVENT)

You are estimating the probability that GDACS reports an Orange or Red \
drought alert affecting this country in each month.

Key reasoning principles:
- Drought alerts in GDACS are based on precipitation deficit severity and \
spatial extent. They are triggered by sustained below-normal rainfall, not \
single dry months. An Orange/Red alert typically requires multi-month \
drought conditions.
- NMME precipitation anomalies are the strongest forward-looking signal. \
Negative anomalies (below-normal rainfall forecasts) increase drought alert \
probability, especially when combined with above-normal temperature forecasts.
- ENSO phase matters: La Ni\u00f1a increases drought risk in the Horn of Africa \
and Central America. El Ni\u00f1o increases drought risk in Southeast Asia, \
Southern Africa, and parts of South Asia.
- Drought alerts have PERSISTENCE: once a drought alert is issued, it tends \
to continue for several months. If there is a current Orange/Red alert, the \
probability of continued alerts in the next 1-3 months is substantially \
higher than the base rate.
- Seasonal patterns are strong. Drought alerts cluster in dry seasons and \
during/after failed rainy seasons. Weight your estimate heavily toward \
seasonality.
- IPC food insecurity data (if available) is a lagging but strong signal. \
Elevated IPC Phase 3+ populations indicate ongoing drought impacts that \
make continued GDACS alerts likely."""


_BINARY_FL = """\
HAZARD-SPECIFIC REASONING: FLOOD (BINARY EVENT)

You are estimating the probability that GDACS reports an Orange or Red \
flood alert affecting this country in each month.

Key reasoning principles:
- Flood alerts in GDACS are triggered by significant flooding events with \
potential humanitarian impact. They are highly seasonal \u2014 concentrated in \
the wet/monsoon season for each country.
- NMME precipitation anomalies are a key signal. Above-normal rainfall \
forecasts during the wet season increase flood alert probability. The \
magnitude matters: +0.5\u03c3 is a modest signal, +1.5\u03c3 is a strong signal.
- ENSO phase affects regional flood risk. La Ni\u00f1a typically increases flood \
risk in Southeast Asia, East Africa, and Australia. El Ni\u00f1o increases flood \
risk in Peru, Ecuador, and parts of East Africa.
- Unlike drought, flood alerts are EPISODIC. A flood event may trigger an \
alert for 1-2 weeks, then the alert lapses. The probability of a flood \
alert in any given month depends on whether a significant rainfall event \
occurs, which is inherently uncertain at monthly horizons.
- During off-season months, flood alert probability should be very low \
(close to but not exactly 0). During peak monsoon/rainy season, it can \
be substantially higher than the annual base rate.
- Recent flood events do NOT strongly predict the next month's floods \
(unlike drought persistence). Each month's risk is relatively independent \
once you account for seasonality."""


_BINARY_TC = """\
HAZARD-SPECIFIC REASONING: TROPICAL CYCLONE (BINARY EVENT)

You are estimating the probability that GDACS reports an Orange or Red \
tropical cyclone alert affecting this country in each month.

Key reasoning principles:
- Cyclone alerts are HIGHLY SEASONAL. Every cyclone basin has a well-defined \
season. Outside the season, assign probabilities very close to the minimum \
(0.01-0.03). During peak season, probabilities can be substantially higher.
- Basin seasons:
  - Atlantic/Caribbean: June\u2013November (peak Aug\u2013Oct)
  - Western Pacific/Philippines: May\u2013December (peak Jul\u2013Nov)
  - Bay of Bengal/South Asia: April\u2013June and October\u2013December
  - Southwest Indian Ocean/Madagascar: November\u2013April
  - South Pacific/Fiji: November\u2013April
- Seasonal cyclone outlooks (TSR, NOAA CPC, BoM) provide basin-level \
activity forecasts. Above-normal predicted activity increases the \
probability of an alert for countries in that basin.
- ENSO phase is a major driver: La Ni\u00f1a increases Atlantic hurricane \
activity but suppresses Eastern Pacific. El Ni\u00f1o does the reverse. IOD \
phase affects Indian Ocean cyclone tracks.
- SST anomalies in the relevant basin affect cyclone intensity and \
frequency. Warmer SSTs generally increase the probability of significant \
cyclone events.
- A cyclone alert in one month does NOT increase the probability for the \
next month (cyclones are discrete events). Each month's risk is primarily \
determined by seasonality and basin-level conditions."""


_BINARY_GENERIC = """\
HAZARD-SPECIFIC REASONING: BINARY EVENT

Apply general Bayesian principles:
- Anchor on the historical base rate for this country-hazard combination.
- Adjust for seasonality: when does this type of event typically occur?
- Consider current conditions and structured data signals.
- Think about persistence: is there an ongoing event that makes continuation likely?
- Ensure probabilities reflect genuine uncertainty."""


# ---- Base rate builder ----

def build_binary_base_rate(
    iso3: str,
    hazard_code: str,
    *,
    db_url: str | None = None,
    conn=None,
) -> dict:
    """Build base rate statistics for binary event prediction.

    Queries facts_resolved for metric='event_occurrence' rows matching
    the (iso3, hazard_code), aggregates by calendar month for seasonality,
    computes overall and recent event rates.

    Parameters
    ----------
    iso3 : str
        Country ISO3 code.
    hazard_code : str
        Hazard code (DR, FL, TC).
    db_url : str | None
        DuckDB URL. Ignored if conn is provided.
    conn : duckdb connection | None
        Existing DuckDB connection (preferred).

    Returns
    -------
    dict
        Base rate statistics including total_months, event_months,
        base_rate_pct, seasonal_pattern, recent_12m_events, recent_12m_rate,
        and trend.
    """
    close_conn = False
    if conn is None:
        try:
            import duckdb
            from resolver.db import duckdb_io
            db = db_url or duckdb_io.DEFAULT_DB_URL
            conn = duckdb_io.get_db(db)
            close_conn = True
        except Exception as exc:
            LOG.warning("Cannot open DB for base rate: %s", exc)
            return {}

    try:
        return _query_base_rate(conn, iso3, hazard_code)
    except Exception as exc:
        LOG.warning("Base rate query failed for %s/%s: %s", iso3, hazard_code, exc)
        return {}
    finally:
        if close_conn:
            try:
                from resolver.db import duckdb_io
                duckdb_io.close_db(conn)
            except Exception:
                pass


def _query_base_rate(conn, iso3: str, hazard_code: str) -> dict:
    """Query facts_resolved to compute binary base rate stats."""
    # Check if facts_resolved exists
    try:
        conn.execute("SELECT 1 FROM facts_resolved LIMIT 0")
    except Exception:
        return {}

    iso3_up = iso3.upper()
    hz_up = hazard_code.upper()

    # Get all event_occurrence rows for this country/hazard
    rows = conn.execute(
        """
        SELECT ym, value
        FROM facts_resolved
        WHERE upper(iso3) = ?
          AND upper(hazard_code) = ?
          AND lower(metric) = 'event_occurrence'
        ORDER BY ym
        """,
        [iso3_up, hz_up],
    ).fetchall()

    if not rows:
        return {
            "total_months": 0,
            "event_months": 0,
            "base_rate_pct": 0.0,
            "seasonal_pattern": {str(m): 0.0 for m in range(1, 13)},
            "recent_12m_events": 0,
            "recent_12m_rate": 0.0,
            "trend": "unknown",
        }

    total_months = len(rows)
    event_months = sum(1 for _, v in rows if v and float(v) >= 1)
    base_rate_pct = (event_months / total_months * 100) if total_months > 0 else 0.0

    # Seasonal pattern: count events per calendar month
    month_counts: dict[int, int] = {m: 0 for m in range(1, 13)}
    month_totals: dict[int, int] = {m: 0 for m in range(1, 13)}

    for ym, v in rows:
        try:
            parts = str(ym).split("-")
            cal_month = int(parts[1])
        except (IndexError, ValueError):
            continue
        month_totals[cal_month] = month_totals.get(cal_month, 0) + 1
        if v and float(v) >= 1:
            month_counts[cal_month] = month_counts.get(cal_month, 0) + 1

    seasonal_pattern = {}
    for m in range(1, 13):
        total = month_totals.get(m, 0)
        events = month_counts.get(m, 0)
        seasonal_pattern[str(m)] = (events / total * 100) if total > 0 else 0.0

    # Recent 12 months
    recent_rows = rows[-12:] if len(rows) >= 12 else rows
    recent_12m_events = sum(1 for _, v in recent_rows if v and float(v) >= 1)
    recent_12m_rate = (recent_12m_events / len(recent_rows) * 100) if recent_rows else 0.0

    # Trend: compare recent 12m rate to overall rate
    if total_months < 24:
        trend = "unknown"
    elif recent_12m_rate > base_rate_pct * 1.3:
        trend = "increasing"
    elif recent_12m_rate < base_rate_pct * 0.7:
        trend = "decreasing"
    else:
        trend = "stable"

    return {
        "total_months": total_months,
        "event_months": event_months,
        "base_rate_pct": base_rate_pct,
        "seasonal_pattern": seasonal_pattern,
        "recent_12m_events": recent_12m_events,
        "recent_12m_rate": recent_12m_rate,
        "trend": trend,
    }


def parse_binary_response(raw_text: str, expected_months: list[str] | None = None) -> dict[str, float]:
    """Parse binary forecast JSON response into {YYYY-MM: probability} dict.

    Handles markdown code fences, validates probabilities, clamps out-of-range.
    """
    # Strip markdown code fences
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the text
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                LOG.warning("Could not parse binary response JSON")
                return {}
        else:
            LOG.warning("No JSON found in binary response")
            return {}

    months_data = data.get("months", data)
    if not isinstance(months_data, dict):
        LOG.warning("Expected dict for months data, got %s", type(months_data))
        return {}

    result: dict[str, float] = {}
    for month_key, month_val in months_data.items():
        if isinstance(month_val, dict):
            prob = month_val.get("posterior", month_val.get("probability", month_val.get("prior")))
        elif isinstance(month_val, (int, float)):
            prob = month_val
        else:
            continue

        if prob is None:
            continue

        try:
            p = float(prob)
        except (TypeError, ValueError):
            continue

        # Clamp to [0.01, 0.99]
        p = max(0.01, min(0.99, p))
        result[month_key] = p

    # Fill missing months from expected_months with None (caller handles fallback)
    if expected_months:
        for m in expected_months:
            if m not in result:
                LOG.warning("Missing forecast for month %s", m)

    return result
