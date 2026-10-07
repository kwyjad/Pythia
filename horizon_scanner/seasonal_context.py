# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Load NMME seasonal forecasts from DuckDB for prompt injection.

Provides :func:`load_seasonal_forecasts` which queries the
``seasonal_forecasts`` table and returns a dict ready to pass as the
``climate_data`` kwarg to RC / triage prompt builders.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

log = logging.getLogger(__name__)

# Hazard codes for which seasonal climate data is relevant.
CLIMATE_HAZARDS = {"DR", "FL", "TC"}

_VARIABLE_LABELS = {
    "tmp2m": "temperature",
    "prate": "precipitation",
}

# Units the ingest stores (resolver.ingestion.nmme.UNITS). Rows written
# before Oct 2026 carry no unit: their precipitation figure was raw mm/s
# rounded to zero and their category was read off sigma thresholds they were
# never in, so they are not printed at all.
_UNIT_SUFFIX = {"degC": " °C", "mm/day": " mm/day"}

# CPC's tercile probabilities (resolver.ingestion.nmme): the chance the month
# falls in the driest / wettest third of the model climatology. An ordinary
# month is one in three for each, in every climate, which is why the rainfall
# block reads these and not a category cut from the mm/day anomaly.
_PROB_BELOW = "prate_prob_below"
_PROB_ABOVE = "prate_prob_above"
_CLIMATOLOGY_SHARE = 1.0 / 3.0

RAINFALL_PROBABILITY_MISSING = (
    "unavailable: CPC published no tercile probability for this country in "
    "this issue (it publishes none over a dry-season or arid mask), so there "
    "is no chance of a dry or wet month to give"
)

_TERCILE_LABELS = {
    "above_normal": "above-normal",
    "below_normal": "below-normal",
    "near_normal": "near-normal",
}


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


def _format_outlook_line(
    variable: str,
    rows: list[dict],
    short_leads: tuple[int, ...] = (1, 2, 3),
) -> str:
    """Build a one-line outlook summary for a variable.

    Example: "Above-normal temperature anomaly (+1.20 °C) for leads 1-3"
    """
    label = _VARIABLE_LABELS.get(variable, variable)

    # Average the anomaly over the short-lead months.
    short = [r for r in rows if r["lead_months"] in short_leads]
    if not short:
        short = rows[:3]

    mean_anomaly = sum(r["anomaly_value"] for r in short) / len(short)

    # Use the majority tercile across those leads.
    tercile_counts: dict[str, int] = {}
    for r in short:
        tc = r.get("tercile_category", "near_normal")
        tercile_counts[tc] = tercile_counts.get(tc, 0) + 1
    majority_tercile = max(tercile_counts, key=tercile_counts.get)  # type: ignore[arg-type]

    tercile_label = _TERCILE_LABELS.get(majority_tercile, majority_tercile)
    sign = "+" if mean_anomaly >= 0 else ""
    unit = _UNIT_SUFFIX.get(str(short[0].get("units") or ""), "")
    lead_range = f"{short_leads[0]}-{short_leads[-1]}" if len(short_leads) > 1 else str(short_leads[0])

    return (
        f"{tercile_label.capitalize()} {label} anomaly "
        f"({sign}{mean_anomaly:.2f}{unit}) for leads {lead_range}"
    )


def _format_detail(variable: str, rows: list[dict]) -> str:
    """Per-lead detail string for a variable."""
    label = _VARIABLE_LABELS.get(variable, variable)
    parts = []
    for r in sorted(rows, key=lambda r: r["lead_months"]):
        sign = "+" if r["anomaly_value"] >= 0 else ""
        tercile = _TERCILE_LABELS.get(r.get("tercile_category", ""), "")
        unit = _UNIT_SUFFIX.get(str(r.get("units") or ""), "")
        parts.append(
            f"Lead {r['lead_months']}: {sign}{r['anomaly_value']:.2f}{unit} ({tercile})"
        )
    return f"{label.capitalize()}: " + "; ".join(parts)


def _pct(p: Optional[float]) -> str:
    return "unavailable" if p is None else f"{round(p * 100):d}%"


def format_rainfall_block(
    prate_rows: list[dict],
    below_rows: list[dict],
    above_rows: list[dict],
) -> tuple[str, str]:
    """The rainfall outlook line and its per-lead detail.

    Each lead shows the chance of a dry month (bottom third) and of a wet
    month (top third) beside the one in three an ordinary month carries, and
    the ensemble mean anomaly in mm/day with no category: a fixed +/-0.5
    mm/day cut reads a different thing in the Sahel dry season and the
    Central American wet season. A lead with no probability says so.
    """
    below = {int(r["lead_months"]): float(r["anomaly_value"]) for r in below_rows}
    above = {int(r["lead_months"]): float(r["anomaly_value"]) for r in above_rows}
    anomaly = {int(r["lead_months"]): float(r["anomaly_value"]) for r in prate_rows}
    leads = sorted(set(below) | set(above) | set(anomaly))
    climatology = _pct(_CLIMATOLOGY_SHARE)

    if not below and not above:
        outlook = f"Rainfall tercile probabilities {RAINFALL_PROBABILITY_MISSING}."
    else:
        short = [lead for lead in leads if lead in (1, 2, 3) and (lead in below or lead in above)]
        b = [below[lead] for lead in short if lead in below]
        a = [above[lead] for lead in short if lead in above]
        if len(short) > 1:
            span = f"leads {short[0]}-{short[-1]}"
        elif short:
            span = f"lead {short[0]}"
        else:
            span = "the first three leads (none carries a probability)"
        outlook = (
            f"Chance of a dry month (driest third of the model climatology) "
            f"{_pct(sum(b) / len(b) if b else None)} and of a wet month (wettest third) "
            f"{_pct(sum(a) / len(a) if a else None)}, averaged over {span}; "
            f"an ordinary month is {climatology} for each"
        )

    parts = []
    for lead in leads:
        if lead not in below and lead not in above:
            prob = f"tercile probabilities {RAINFALL_PROBABILITY_MISSING}"
        else:
            prob = f"dry {_pct(below.get(lead))}, wet {_pct(above.get(lead))} (ordinary {climatology} each)"
        if lead in anomaly:
            sign = "+" if anomaly[lead] >= 0 else ""
            prob += f"; ensemble mean anomaly {sign}{anomaly[lead]:.2f} mm/day"
        parts.append(f"Lead {lead}: {prob}")
    detail = "Rainfall: " + "; ".join(parts) if parts else ""
    return outlook, detail


def load_seasonal_forecasts(
    iso3: str,
    db_url: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Load latest NMME seasonal forecasts for a country.

    Returns a dict suitable for the ``climate_data`` parameter accepted
    by the per-hazard RC and triage prompt builders, or *None* if no
    data is available.

    Keys returned:
        nmme_temp_outlook   – one-line temperature summary (leads 1-3)
        nmme_precip_outlook – chance of a dry and of a wet month (leads 1-3)
                              beside the one in three an ordinary month carries
        nmme_temp_detail    – per-lead temperature breakdown
        nmme_precip_detail  – per lead: dry and wet chance, then the mm/day
                              anomaly with no category
        nmme_issue_date     – forecast issue date
    """
    try:
        from resolver.db.duckdb_io import get_db
    except Exception:
        log.debug("DuckDB helpers unavailable — skipping seasonal load.")
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
        if "seasonal_forecasts" not in tables:
            return None

        cols = {
            r[1] for r in con.execute("PRAGMA table_info('seasonal_forecasts')").fetchall()
        }
        if "units" not in cols:
            return None
        # Get the latest issue date for this country.
        row = con.execute(
            """
            SELECT MAX(forecast_issue_date)
            FROM seasonal_forecasts
            WHERE iso3 = ? AND units IS NOT NULL
            """,
            [iso3.upper()],
        ).fetchone()

        if not row or row[0] is None:
            return None
        latest_date = row[0]

        # Fetch all rows for this country and issue date.
        result = con.execute(
            """
            SELECT variable, lead_months, anomaly_value, tercile_category, units
            FROM seasonal_forecasts
            WHERE iso3 = ? AND forecast_issue_date = ?
              AND units IS NOT NULL AND anomaly_value IS NOT NULL
              AND variable IN ('tmp2m', 'prate', 'prate_prob_below', 'prate_prob_above')
            ORDER BY variable, lead_months
            """,
            [iso3.upper(), latest_date],
        ).fetchall()

        if not result:
            return None

    except Exception as exc:
        log.warning("Failed to load seasonal forecasts for %s: %s", iso3, exc)
        return None
    finally:
        pass  # Let the resolver connection cache manage lifecycle.

    # Group by variable.
    by_var: dict[str, list[dict]] = {}
    for var, lead, anomaly, tercile, units in result:
        by_var.setdefault(var, []).append(
            {
                "variable": var,
                "lead_months": lead,
                "anomaly_value": float(anomaly),
                "tercile_category": tercile or "near_normal",
                "units": units,
            }
        )

    climate_data: dict[str, Any] = {}

    if "tmp2m" in by_var:
        climate_data["nmme_temp_outlook"] = _format_outlook_line("tmp2m", by_var["tmp2m"])
        climate_data["nmme_temp_detail"] = _format_detail("tmp2m", by_var["tmp2m"])

    rain_vars = ("prate", _PROB_BELOW, _PROB_ABOVE)
    if any(v in by_var for v in rain_vars):
        outlook, detail = format_rainfall_block(
            by_var.get("prate", []), by_var.get(_PROB_BELOW, []), by_var.get(_PROB_ABOVE, []),
        )
        climate_data["nmme_precip_outlook"] = outlook
        if detail:
            climate_data["nmme_precip_detail"] = detail
    else:
        # Say so: a missing precipitation outlook printed nothing, and a
        # reader of the temperature line alone can take rain to be normal.
        climate_data["nmme_precip_outlook"] = "unavailable (no NMME precipitation forecast for this country)"
    if "tmp2m" not in by_var:
        climate_data["nmme_temp_outlook"] = "unavailable (no NMME temperature forecast for this country)"

    climate_data["nmme_issue_date"] = str(latest_date)

    return climate_data
