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
CLIMATE_HAZARDS = {"DR", "FL", "HW", "TC"}

_VARIABLE_LABELS = {
    "tmp2m": "temperature",
    "prate": "precipitation",
}

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

    Example: "Above-normal temperature anomaly (+1.2σ) for leads 1-3"
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
    lead_range = f"{short_leads[0]}-{short_leads[-1]}" if len(short_leads) > 1 else str(short_leads[0])

    return (
        f"{tercile_label.capitalize()} {label} anomaly "
        f"({sign}{mean_anomaly:.2f}σ) for leads {lead_range}"
    )


def _format_detail(variable: str, rows: list[dict]) -> str:
    """Per-lead detail string for a variable."""
    label = _VARIABLE_LABELS.get(variable, variable)
    parts = []
    for r in sorted(rows, key=lambda r: r["lead_months"]):
        sign = "+" if r["anomaly_value"] >= 0 else ""
        tercile = _TERCILE_LABELS.get(r.get("tercile_category", ""), "")
        parts.append(
            f"Lead {r['lead_months']}: {sign}{r['anomaly_value']:.2f}σ ({tercile})"
        )
    return f"{label.capitalize()}: " + "; ".join(parts)


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
        nmme_precip_outlook – one-line precipitation summary (leads 1-3)
        nmme_temp_detail    – per-lead temperature breakdown
        nmme_precip_detail  – per-lead precipitation breakdown
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

        # Get the latest issue date for this country.
        row = con.execute(
            """
            SELECT MAX(forecast_issue_date)
            FROM seasonal_forecasts
            WHERE iso3 = ?
            """,
            [iso3.upper()],
        ).fetchone()

        if not row or row[0] is None:
            return None
        latest_date = row[0]

        # Fetch all rows for this country and issue date.
        result = con.execute(
            """
            SELECT variable, lead_months, anomaly_value, tercile_category
            FROM seasonal_forecasts
            WHERE iso3 = ? AND forecast_issue_date = ?
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
    for var, lead, anomaly, tercile in result:
        by_var.setdefault(var, []).append(
            {
                "variable": var,
                "lead_months": lead,
                "anomaly_value": float(anomaly) if anomaly is not None else 0.0,
                "tercile_category": tercile or "near_normal",
            }
        )

    climate_data: dict[str, Any] = {}

    if "tmp2m" in by_var:
        climate_data["nmme_temp_outlook"] = _format_outlook_line("tmp2m", by_var["tmp2m"])
        climate_data["nmme_temp_detail"] = _format_detail("tmp2m", by_var["tmp2m"])

    if "prate" in by_var:
        climate_data["nmme_precip_outlook"] = _format_outlook_line("prate", by_var["prate"])
        climate_data["nmme_precip_detail"] = _format_detail("prate", by_var["prate"])

    climate_data["nmme_issue_date"] = str(latest_date)

    return climate_data if len(climate_data) > 1 else None
