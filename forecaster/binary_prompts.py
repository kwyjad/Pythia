# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Binary event prompt builder for EVENT_OCCURRENCE questions.

Binary questions produce a single probability per month (not a bucketed SPD).
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
    rc_level: int | None = None,
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
    rc_level : int | None
        Horizon Scanner regime-change level. At >= 1 the prompt requires an
        ``rc_reconciliation`` field in the JSON.

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

    # GDACS event history block (seasonal frequency context) — shared by both
    # section orders below.
    gdacs_block = ""
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
            ) or ""
        except Exception:
            gdacs_block = ""

    from forecaster.prompts import _prompt_v3_order_enabled

    if _prompt_v3_order_enabled():
        # V3 (static-first) order: role/task (country-generic), hazard
        # reasoning, output instructions lead — identical across every
        # country of the same hazard — and the per-question data trails.
        sections = [
            _section_role_and_task_generic(hazard_name),
            get_binary_hazard_reasoning_block(hazard_code),
            _section_output_instructions(forecast_months),
            (
                "If the QUESTION DATA contains a REGIME CHANGE FLAG, add the "
                '"rc_reconciliation" field it asks for to the JSON object.'
            ),
            (
                "The QUESTION DATA follows below.\n\n"
                f"QUESTION: Will a significant {hazard_name} event "
                f"(GDACS Orange/Red alert) affect {country} ({iso3}) in each "
                "of the 6 forecast months?"
            ),
            _section_base_rate(country, hazard_name, base_rate),
            _section_current_situation(
                current_alerts, structured_data, hs_triage_entry, country, hazard_code
            ),
            gdacs_block,
            _section_rc_reconciliation(rc_level),
            (
                "END OF QUESTION DATA.\n"
                "Now apply the reasoning guidance above and produce ONLY the "
                "JSON object specified in OUTPUT INSTRUCTIONS."
            ),
        ]
        return "\n\n".join(s for s in sections if s)

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
    if gdacs_block:
        sections.append(gdacs_block)

    # Section 4: Hazard-specific reasoning
    sections.append(get_binary_hazard_reasoning_block(hazard_code))

    # Section 4b: regime-change reconciliation requirement (RC >= 1)
    sections.append(_section_rc_reconciliation(rc_level))

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


def _section_role_and_task_generic(hazard_name: str) -> str:
    """Country-generic role/task section for the V3 (static-first) order.

    Same text as _section_role_and_task with the country references
    generalized so the section is byte-identical across all countries of a
    hazard (the country is named in the QUESTION line of the dynamic tail).
    """
    return f"""\
ROLE AND TASK

You are a careful probabilistic forecaster specializing in humanitarian \
event prediction. Your task is to estimate the probability that a \
significant {hazard_name} event will affect the target country (named in \
the QUESTION DATA below) during each of the next 6 months.

A "significant event" is defined as: GDACS reports an Orange or Red alert \
level {hazard_name} event with the target country in the affected countries \
list.
- Orange alert: "Potential need for international assistance"
- Red alert: "Likely need for international assistance"
- Green alerts (minor events) do NOT count.

You are being scored with the Brier score: (your_probability - outcome)^2
Lower is better. A well-calibrated forecaster assigns 10% to events that \
happen 10% of the time."""


def _section_base_rate(country: str, hazard_name: str, base_rate: dict) -> str:
    if not base_rate:
        return f"HISTORICAL BASE RATE: No historical data available for {country} / {hazard_name}."
    if base_rate.get("history_available") is False:
        from forecaster.gdacs_history import unavailable_line

        return "HISTORICAL BASE RATE (GDACS): " + unavailable_line(
            hazard_name, country,
            base_rate.get("unavailable_reason") or "too few months of GDACS coverage",
        )

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

    coverage_start = base_rate.get("coverage_start")
    coverage_end = base_rate.get("coverage_end")
    if coverage_start and coverage_end:
        coverage_label = f"{coverage_start} to {coverage_end}"
    else:
        coverage_label = "coverage window unknown"

    return f"""\
HISTORICAL BASE RATE (GDACS, {coverage_label}):
{country} has had significant (Orange/Red) {hazard_name} alerts in {event_months} of \
the {total_months} calendar months GDACS covers ({base_rate_pct:.1f}%). A month \
with no alert naming {country} counts as a month without an event.

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

    # Recent significant GDACS event months (Orange/Red only — Green does
    # not count per the question definition). facts_resolved has no event
    # name for these monthly rows, so render level + month only.
    if current_alerts:
        parts.append(
            f"\nRecent significant GDACS event months for {country} (Orange/Red alerts):"
        )
        for alert in current_alerts[:10]:
            level = alert.get("alertlevel") or "?"
            ym = alert.get("ym", "?")
            parts.append(f"  - [{level}] {ym}")
    else:
        parts.append(f"\nNo recent Orange/Red GDACS event months for {country}.")

    # Structured data injection (reuse existing formatted data where possible)
    if structured_data:
        # NMME seasonal outlook
        nmme = structured_data.get("nmme_seasonal_outlook") or structured_data.get("nmme")
        if nmme:
            if isinstance(nmme, str):
                parts.append(f"\nNMME SEASONAL OUTLOOK:\n{nmme}")
            elif isinstance(nmme, dict):
                parts.append(f"\nNMME SEASONAL OUTLOOK:\n{json.dumps(nmme, indent=2)}")

        # ENSO — the header states the date the index was OBSERVED, so a
        # model can tell a current reading from an old one. August 2026's
        # drought prompts said "Current state: Neutral" through a strong El
        # Niño (the pre-September scraped phase) with nothing to date it.
        enso = structured_data.get("enso") or structured_data.get("enso_context")
        if enso:
            if isinstance(enso, str):
                parts.append(f"\n{_enso_header(enso)}\n{enso}")

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
                titles = [rpt.get("title", "") for rpt in reliefweb[:5]]
                titles = [t for t in titles if t]
                if titles:
                    # Header required: without it the bullets render visually
                    # nested under whatever section preceded (ENSO STATE in
                    # the July 2026 test run).
                    parts.append("\nRECENT RELIEFWEB REPORTS:")
                    parts.extend(f"  - {t}" for t in titles)

    # HS triage context
    if hs_triage_entry:
        triage_score = hs_triage_entry.get("triage_score", "?")
        tier = hs_triage_entry.get("tier", "?")
        parts.append(f"\nHORIZON SCANNER TRIAGE: score={triage_score}, tier={tier}")

    return "\n".join(parts)


_ENSO_OBSERVED_RE = re.compile(r"Observed (\d{4}-\d{2}(?:-\d{2})?)")


def _enso_header(enso_text: str) -> str:
    """``ENSO STATE (observed YYYY-MM-DD):`` from the ENSO block's own text.

    The block (``ENSOForecast.to_prompt_context``) ends its state line with
    "Observed <date>"; a block with no such date says so in the header
    rather than letting the reading pass as current.
    """
    m = _ENSO_OBSERVED_RE.search(enso_text or "")
    if m:
        return f"ENSO STATE (index observed {m.group(1)}):"
    return "ENSO STATE (observation date not stated; it may be stale):"


def _section_rc_reconciliation(rc_level: int | None) -> str:
    """Ask for an explicit reconciliation when HS flagged a regime change.

    At RC level >= 1 the Horizon Scanner has judged this country-hazard to be
    departing from its base rate. Five August 2026 drought questions carried
    that flag and were forecast at 1.7-15% beside a near-empty history; the
    model never had to say how it weighed the two.
    """
    if rc_level is None or int(rc_level) < 1:
        return ""
    return (
        f"REGIME CHANGE FLAG: the Horizon Scanner rates this hazard at regime-"
        f"change level {int(rc_level)} (a departure from the historical base "
        "rate is judged likely). Your JSON MUST include a top-level "
        '"rc_reconciliation" field: one or two sentences saying how you weighed '
        "this flag against the base rate, and why your probabilities move (or "
        "do not move) away from it."
    )


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

All probabilities must be between 0.001 and 0.999. Never assign exactly \
0 or 1 \u2014 there is always some residual uncertainty. Rare events with \
very low monthly base rates may warrant probabilities well below 0.01."""


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
- NMME rainfall probabilities are the strongest forward-looking signal. \
A chance of a dry month (driest third) well above the ordinary 1 in 3 increases \
drought alert probability, especially when combined with above-normal temperature forecasts.
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
- NMME rainfall probabilities are a key signal. A chance of a wet month \
(wettest third) above the ordinary 1 in 3 during the wet season increases \
flood alert probability; the further above 33%, the stronger the signal.
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
(0.001-0.01). During peak season, probabilities can be substantially higher.
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


def _query_base_rate(conn, iso3: str, hazard_code: str, today: date | None = None) -> dict:
    """Binary base-rate stats over the SOURCE's calendar window.

    Every calendar month GDACS covers for the hazard counts once; a month is
    an event month when any row for it is Orange/Red. The old version
    counted the country's rows over the country's own span, which printed
    "2026-05 to 2026-07 ... 0 of 9 months" — see ``forecaster.gdacs_history``.
    """
    from forecaster.gdacs_history import gdacs_calendar_series, seasonal_frequency

    try:
        conn.execute("SELECT 1 FROM facts_resolved LIMIT 0")
    except Exception:
        return {}
    series = gdacs_calendar_series(conn, iso3, hazard_code, today=today)
    months = series["months"]
    total_months = series["total_months"]
    event_months = series["event_months"]
    base_rate_pct = (event_months / total_months * 100) if total_months else 0.0
    seasonal = seasonal_frequency(months)
    seasonal_pattern = {str(m): float(seasonal[m]["frequency_pct"]) for m in range(1, 13)}

    recent = months[-12:]
    recent_12m_events = sum(1 for m in recent if m["occurred"])
    recent_12m_rate = (recent_12m_events / len(recent) * 100) if recent else 0.0
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
        # The source's calendar window — the header renders this, and the
        # denominator is exactly the months in it.
        "coverage_start": series["window_start"],
        "coverage_end": series["window_end"],
        "history_available": series["history_available"],
        "unavailable_reason": series["unavailable_reason"],
    }


def parse_rc_reconciliation(raw_text: str) -> str | None:
    """The top-level ``rc_reconciliation`` string, or None when absent.

    Tolerates code fences and prose around the JSON exactly as
    :func:`parse_binary_response` does. Truncated to 1,000 characters.
    """
    text = (raw_text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    data = None
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None
    if not isinstance(data, dict):
        return None
    value = data.get("rc_reconciliation")
    if isinstance(value, str) and value.strip():
        return value.strip()[:1000]
    return None


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

        # Clamp to [0.001, 0.999]. The floor must stay below plausible
        # rare-event monthly base rates: a 1% floor put a hard minimum
        # under the Brier score of well-calibrated low forecasts.
        clamped = max(0.001, min(0.999, p))
        if clamped != p:
            LOG.debug(
                "Clamped binary probability %s -> %s for month %s",
                p, clamped, month_key,
            )
        result[month_key] = clamped

    # Completeness is enforced per model: a response missing any expected
    # month is rejected outright rather than passed through partially —
    # partial forecasts previously flowed into aggregation and left silent
    # gaps in forecasts_ensemble that scoring later skipped.
    if expected_months:
        missing = [m for m in expected_months if m not in result]
        if missing:
            LOG.warning(
                "Binary response missing %d/%d expected months (%s); "
                "rejecting this model's forecast",
                len(missing), len(expected_months), ", ".join(missing),
            )
            return {}

    return result
