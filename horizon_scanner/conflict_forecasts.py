# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Load conflict forecasts from DuckDB for prompt injection.

Provides :func:`load_conflict_forecasts` which queries the
``conflict_forecasts`` table and returns a dict ready to pass as the
``conflict_forecasts`` kwarg to ACE RC / triage prompt builders.

Also provides :func:`format_conflict_forecasts_for_prompt` which renders
the forecast data as a text block suitable for prompt injection.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Any, Optional

log = logging.getLogger(__name__)

_STALENESS_DAYS = 45


def _db_url() -> str:
    """Resolve the Pythia DuckDB URL."""
    url = os.getenv("PYTHIA_DB_URL", "").strip()
    if url:
        return url
    try:
        from pythia.config import load as load_config
        cfg = load_config()
        url = str((cfg.get("app") or {}).get("db_url", "")).strip()
        if url:
            return url
    except Exception:
        pass
    from resolver.db.duckdb_io import DEFAULT_DB_URL
    return DEFAULT_DB_URL


def load_conflict_forecasts(
    iso3: str,
    db_url: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Load latest conflict forecasts for a country.

    Returns a dict suitable for the ``conflict_forecasts`` parameter
    accepted by the ACE RC and triage prompt builders, or *None* if no
    data is available.

    Keys returned::

        views_fatalities  – list of {lead_months, value} for predicted fatalities
        views_p_gte25     – list of {lead_months, value} for P(≥25 BRD)
        views_issue_date  – forecast issue date (str)
        views_model       – model version string
        views_stale       – True if data is >45 days old

        cf_risk_3m        – armed conflict risk 3-month value
        cf_risk_12m       – armed conflict risk 12-month value
        cf_intensity_3m   – violence intensity 3-month value
        cf_issue_date     – forecast issue date (str)
        cf_stale          – True if data is >45 days old

        cast_total        – list of {lead_months, value} for total predicted events
        cast_battles      – list of {lead_months, value} for predicted battle events
        cast_erv          – list of {lead_months, value} for predicted ERV events
        cast_vac          – list of {lead_months, value} for predicted VAC events
        cast_issue_date   – forecast issue date (str)
        cast_stale        – True if data is >45 days old
    """
    try:
        from resolver.db.duckdb_io import get_db
    except Exception:
        log.debug("DuckDB helpers unavailable — skipping conflict forecast load.")
        return None

    db_url = db_url or _db_url()

    try:
        con = get_db(db_url)
    except Exception:
        log.debug("Could not connect to DuckDB at %s", db_url)
        return None

    try:
        # Check the table exists.
        tables = [
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        ]
        if "conflict_forecasts" not in tables:
            return None

        # Fetch VIEWS forecasts (latest issue date).
        views_data = _load_views(con, iso3)

        # Fetch conflictforecast.org forecasts (latest issue date).
        cf_data = _load_conflictforecast_org(con, iso3)

        # Fetch ACLED CAST forecasts (latest issue date).
        cast_data = _load_acled_cast(con, iso3)

    except Exception as exc:
        log.warning("Failed to load conflict forecasts for %s: %s", iso3, exc)
        return None
    finally:
        pass  # Let the resolver connection cache manage lifecycle.

    if not views_data and not cf_data and not cast_data:
        return None

    result: dict[str, Any] = {}
    if views_data:
        result.update(views_data)
    if cf_data:
        result.update(cf_data)
    if cast_data:
        result.update(cast_data)

    return result


def _load_views(con, iso3: str) -> Optional[dict[str, Any]]:
    """Load VIEWS forecast data for a country."""
    row = con.execute(
        """
        SELECT MAX(forecast_issue_date)
        FROM conflict_forecasts
        WHERE source = 'VIEWS' AND iso3 = ?
        """,
        [iso3.upper()],
    ).fetchone()

    if not row or row[0] is None:
        return None
    latest_date = row[0]

    result = con.execute(
        """
        SELECT metric, lead_months, value, model_version
        FROM conflict_forecasts
        WHERE source = 'VIEWS' AND iso3 = ? AND forecast_issue_date = ?
        ORDER BY metric, lead_months
        """,
        [iso3.upper(), latest_date],
    ).fetchall()

    if not result:
        return None

    fatalities: list[dict[str, Any]] = []
    p_gte25: list[dict[str, Any]] = []
    model_version = ""

    for metric, lead, value, model in result:
        model_version = model or model_version
        entry = {"lead_months": lead, "value": float(value)}
        if metric == "views_predicted_fatalities":
            fatalities.append(entry)
        elif metric == "views_p_gte25_brd":
            p_gte25.append(entry)

    issue_date = latest_date if isinstance(latest_date, date) else date.fromisoformat(str(latest_date))
    stale = (date.today() - issue_date).days > _STALENESS_DAYS

    return {
        "views_fatalities": fatalities,
        "views_p_gte25": p_gte25,
        "views_issue_date": str(latest_date),
        "views_model": model_version,
        "views_stale": stale,
    }


def _load_conflictforecast_org(con, iso3: str) -> Optional[dict[str, Any]]:
    """Load conflictforecast.org data for a country."""
    row = con.execute(
        """
        SELECT MAX(forecast_issue_date)
        FROM conflict_forecasts
        WHERE source = 'conflictforecast_org' AND iso3 = ?
        """,
        [iso3.upper()],
    ).fetchone()

    if not row or row[0] is None:
        return None
    latest_date = row[0]

    result = con.execute(
        """
        SELECT metric, value
        FROM conflict_forecasts
        WHERE source = 'conflictforecast_org' AND iso3 = ? AND forecast_issue_date = ?
        """,
        [iso3.upper(), latest_date],
    ).fetchall()

    if not result:
        return None

    data: dict[str, float] = {}
    for metric, value in result:
        data[metric] = float(value)

    issue_date = latest_date if isinstance(latest_date, date) else date.fromisoformat(str(latest_date))
    stale = (date.today() - issue_date).days > _STALENESS_DAYS

    return {
        "cf_risk_3m": data.get("cf_armed_conflict_risk_3m"),
        "cf_risk_12m": data.get("cf_armed_conflict_risk_12m"),
        "cf_intensity_3m": data.get("cf_violence_intensity_3m"),
        "cf_issue_date": str(latest_date),
        "cf_stale": stale,
    }


def _load_acled_cast(con, iso3: str) -> Optional[dict[str, Any]]:
    """Load ACLED CAST forecast data for a country."""
    row = con.execute(
        """
        SELECT MAX(forecast_issue_date)
        FROM conflict_forecasts
        WHERE source = 'ACLED_CAST' AND iso3 = ?
        """,
        [iso3.upper()],
    ).fetchone()

    if not row or row[0] is None:
        return None
    latest_date = row[0]

    result = con.execute(
        """
        SELECT metric, lead_months, value
        FROM conflict_forecasts
        WHERE source = 'ACLED_CAST' AND iso3 = ? AND forecast_issue_date = ?
        ORDER BY metric, lead_months
        """,
        [iso3.upper(), latest_date],
    ).fetchall()

    if not result:
        return None

    total: list[dict[str, Any]] = []
    battles: list[dict[str, Any]] = []
    erv: list[dict[str, Any]] = []
    vac: list[dict[str, Any]] = []

    for metric, lead, value in result:
        entry = {"lead_months": lead, "value": float(value)}
        if metric == "cast_total_events":
            total.append(entry)
        elif metric == "cast_battles_events":
            battles.append(entry)
        elif metric == "cast_erv_events":
            erv.append(entry)
        elif metric == "cast_vac_events":
            vac.append(entry)

    issue_date = (
        latest_date
        if isinstance(latest_date, date)
        else date.fromisoformat(str(latest_date))
    )
    stale = (date.today() - issue_date).days > _STALENESS_DAYS

    return {
        "cast_total": total,
        "cast_battles": battles,
        "cast_erv": erv,
        "cast_vac": vac,
        "cast_issue_date": str(latest_date),
        "cast_stale": stale,
    }


def format_conflict_forecasts_for_prompt(
    forecasts: Optional[dict[str, Any]],
) -> str:
    """Format conflict forecast data as a text block for prompt injection.

    Returns an empty string if no data is available.
    """
    if not forecasts:
        return ""

    parts: list[str] = []
    parts.append("EXTERNAL CONFLICT FORECASTS:")

    # VIEWS section
    views_fat = forecasts.get("views_fatalities")
    views_p25 = forecasts.get("views_p_gte25")
    if views_fat or views_p25:
        stale_note = " [WARNING: DATA >45 DAYS OLD]" if forecasts.get("views_stale") else ""
        model = forecasts.get("views_model", "fatalities003")
        issue = forecasts.get("views_issue_date", "unknown")
        parts.append(f"\n### VIEWS Early Warning System ({model}){stale_note}")
        parts.append(f"Forecast issued: {issue}")
        parts.append("Source: Uppsala University / PRIO. ML ensemble trained on UCDP/ACLED data.")

        if views_fat:
            line = "Predicted fatalities (state-based armed conflict), next 1-6 months:"
            vals = " | ".join(
                f"M{e['lead_months']}: {e['value']:.1f}"
                for e in sorted(views_fat, key=lambda x: x["lead_months"])
            )
            parts.append(f"{line}\n  {vals}")

        if views_p25:
            line = "Probability of ≥25 battle-related deaths:"
            vals = " | ".join(
                f"M{e['lead_months']}: {e['value'] * 100:.1f}%"
                for e in sorted(views_p25, key=lambda x: x["lead_months"])
            )
            parts.append(f"{line}\n  {vals}")

    # conflictforecast.org section
    cf_risk_3m = forecasts.get("cf_risk_3m")
    cf_risk_12m = forecasts.get("cf_risk_12m")
    cf_intensity_3m = forecasts.get("cf_intensity_3m")
    has_cf = any(v is not None for v in (cf_risk_3m, cf_risk_12m, cf_intensity_3m))
    if has_cf:
        stale_note = " [WARNING: DATA >45 DAYS OLD]" if forecasts.get("cf_stale") else ""
        issue = forecasts.get("cf_issue_date", "unknown")
        parts.append(f"\n### conflictforecast.org (Mueller/Rauh){stale_note}")
        parts.append(f"Forecast issued: {issue}")
        parts.append("Source: FEA/BSE. NLP + ML model trained on news text + panel data.")
        parts.append("NOTE: This model is particularly useful for detecting escalation/de-escalation driven by news signals.")

        if cf_risk_3m is not None:
            parts.append(f"Armed conflict risk (3-month horizon): {cf_risk_3m:.3f}")
        if cf_risk_12m is not None:
            parts.append(f"Armed conflict risk (12-month horizon): {cf_risk_12m:.3f}")
        if cf_intensity_3m is not None:
            parts.append(f"Violence intensity outlook (3-month): {cf_intensity_3m:.3f}")

    # ACLED CAST section
    cast_total = forecasts.get("cast_total")
    cast_battles = forecasts.get("cast_battles")
    cast_erv = forecasts.get("cast_erv")
    cast_vac = forecasts.get("cast_vac")
    has_cast = any(v for v in (cast_total, cast_battles, cast_erv, cast_vac))
    if has_cast:
        stale_note = " [WARNING: DATA >45 DAYS OLD]" if forecasts.get("cast_stale") else ""
        issue = forecasts.get("cast_issue_date", "unknown")
        parts.append(f"\n### ACLED CAST (Conflict Alert System Tool){stale_note}")
        parts.append(f"Forecast issued: {issue}")
        parts.append("Source: ACLED. ML ensemble (random forest + XGBoost) trained on ACLED event data.")
        parts.append("NOTE: CAST forecasts event *counts* (not fatalities). Event-type breakdown is unique to CAST.")

        if cast_total:
            line = "Predicted total conflict events, next months:"
            vals = " | ".join(
                f"M{e['lead_months']}: {e['value']:.0f}"
                for e in sorted(cast_total, key=lambda x: x["lead_months"])
            )
            parts.append(f"{line}\n  {vals}")

        if cast_battles:
            line = "Predicted battle events:"
            vals = " | ".join(
                f"M{e['lead_months']}: {e['value']:.0f}"
                for e in sorted(cast_battles, key=lambda x: x["lead_months"])
            )
            parts.append(f"{line}\n  {vals}")

        if cast_erv:
            line = "Predicted explosions/remote violence events:"
            vals = " | ".join(
                f"M{e['lead_months']}: {e['value']:.0f}"
                for e in sorted(cast_erv, key=lambda x: x["lead_months"])
            )
            parts.append(f"{line}\n  {vals}")

        if cast_vac:
            line = "Predicted violence against civilians events:"
            vals = " | ".join(
                f"M{e['lead_months']}: {e['value']:.0f}"
                for e in sorted(cast_vac, key=lambda x: x["lead_months"])
            )
            parts.append(f"{line}\n  {vals}")

    if len(parts) <= 1:
        return ""

    parts.append(
        "\nThe conflict forecast data above comes from independent sources. "
        "VIEWS provides ML-based fatality predictions (treat as a statistical "
        "baseline — good at trends, weak at sudden onset). conflictforecast.org "
        "provides news-driven risk scores (better at detecting shifts and "
        "escalation signals). ACLED CAST provides event count predictions by "
        "type (battles, ERV, VAC) based on historical patterns. "
        "These are inputs to your assessment, not "
        "substitutes for it. Where they disagree, note the disagreement and "
        "reason about why."
    )

    return "\n".join(parts)


def format_conflict_forecasts_for_research(
    forecasts: Optional[dict[str, Any]],
) -> str:
    """Format conflict forecast data for the Forecaster research prompt.

    Uses a more structured tabular format suitable for the research brief.
    """
    if not forecasts:
        return ""

    parts: list[str] = ["QUANTITATIVE CONFLICT FORECASTS:"]

    # VIEWS table
    views_fat = forecasts.get("views_fatalities")
    views_p25 = forecasts.get("views_p_gte25")
    if views_fat or views_p25:
        model = forecasts.get("views_model", "fatalities003")
        issue = forecasts.get("views_issue_date", "unknown")
        stale_note = " [STALE — >45 days old]" if forecasts.get("views_stale") else ""
        parts.append(f"\n### VIEWS {model} ensemble (issued {issue}){stale_note}")

        header = "| Lead month | Predicted fatalities | P(≥25 BRD) |"
        sep = "|-----------|---------------------|-------------|"
        parts.append(header)
        parts.append(sep)

        # Build lookup dicts
        fat_by_lead = {e["lead_months"]: e["value"] for e in (views_fat or [])}
        p25_by_lead = {e["lead_months"]: e["value"] for e in (views_p25 or [])}
        all_leads = sorted(set(list(fat_by_lead.keys()) + list(p25_by_lead.keys())))

        for lead in all_leads:
            fat_val = f"{fat_by_lead[lead]:.1f}" if lead in fat_by_lead else "n/a"
            p25_val = f"{p25_by_lead[lead] * 100:.1f}%" if lead in p25_by_lead else "n/a"
            parts.append(f"| Month {lead}   | {fat_val:>19} | {p25_val:>11} |")

    # conflictforecast.org
    cf_risk_3m = forecasts.get("cf_risk_3m")
    cf_risk_12m = forecasts.get("cf_risk_12m")
    cf_intensity_3m = forecasts.get("cf_intensity_3m")
    has_cf = any(v is not None for v in (cf_risk_3m, cf_risk_12m, cf_intensity_3m))
    if has_cf:
        issue = forecasts.get("cf_issue_date", "unknown")
        stale_note = " [STALE]" if forecasts.get("cf_stale") else ""
        parts.append(f"\n### conflictforecast.org — news-based (issued {issue}){stale_note}")
        if cf_risk_3m is not None:
            parts.append(f"Armed conflict risk (3m): {cf_risk_3m:.3f}")
        if cf_risk_12m is not None:
            parts.append(f"Armed conflict risk (12m): {cf_risk_12m:.3f}")
        if cf_intensity_3m is not None:
            parts.append(f"Violence intensity outlook (3m): {cf_intensity_3m:.3f}")

    # ACLED CAST
    cast_total = forecasts.get("cast_total")
    cast_battles = forecasts.get("cast_battles")
    cast_erv = forecasts.get("cast_erv")
    cast_vac = forecasts.get("cast_vac")
    has_cast = any(v for v in (cast_total, cast_battles, cast_erv, cast_vac))
    if has_cast:
        issue = forecasts.get("cast_issue_date", "unknown")
        stale_note = " [STALE]" if forecasts.get("cast_stale") else ""
        parts.append(f"\n### ACLED CAST — event count forecasts (issued {issue}){stale_note}")

        header = "| Lead month | Total events | Battles | ERV | VAC |"
        sep = "|-----------|-------------|---------|-----|-----|"
        parts.append(header)
        parts.append(sep)

        total_by_lead = {e["lead_months"]: e["value"] for e in (cast_total or [])}
        battles_by_lead = {e["lead_months"]: e["value"] for e in (cast_battles or [])}
        erv_by_lead = {e["lead_months"]: e["value"] for e in (cast_erv or [])}
        vac_by_lead = {e["lead_months"]: e["value"] for e in (cast_vac or [])}
        all_leads = sorted(set(
            list(total_by_lead) + list(battles_by_lead)
            + list(erv_by_lead) + list(vac_by_lead)
        ))

        for lead in all_leads:
            t = f"{total_by_lead[lead]:.0f}" if lead in total_by_lead else "n/a"
            b = f"{battles_by_lead[lead]:.0f}" if lead in battles_by_lead else "n/a"
            e = f"{erv_by_lead[lead]:.0f}" if lead in erv_by_lead else "n/a"
            v = f"{vac_by_lead[lead]:.0f}" if lead in vac_by_lead else "n/a"
            parts.append(f"| Month {lead}   | {t:>11} | {b:>7} | {e:>3} | {v:>3} |")

    return "\n".join(parts) if len(parts) > 1 else ""
