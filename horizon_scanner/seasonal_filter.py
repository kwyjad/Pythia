# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Seasonal hazard filter for Horizon Scanner RC pipeline.

Reads resolver/data/seasonal_hazards.csv and determines which hazards
are active (in-season) for a given country based on the 6 forecast
months following the run date.

A hazard is considered active for RC if *any* of the 6 forecast months
has a base-rate value > 0 in the CSV.  ACE is always active (conflict
is not seasonal).  DI is never active for RC (silenced until a good
resolution source is found).
"""

from __future__ import annotations

import csv
import logging
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Dict, Set, Tuple

logger = logging.getLogger(__name__)

_CSV_PATH = Path(__file__).resolve().parent.parent / "resolver" / "data" / "seasonal_hazards.csv"

_MONTH_COLS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
               "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

_ALWAYS_ACTIVE = {"ACE"}
_NEVER_ACTIVE_RC = {"DI"}


@lru_cache(maxsize=1)
def _load_seasonal_data() -> Dict[Tuple[str, str], list[float]]:
    """Load seasonal_hazards.csv into {(ISO3, Hazard): [jan..dec]} dict.

    Values of ``x`` (meaning no data) are treated as 0.0.
    """
    data: Dict[Tuple[str, str], list[float]] = {}
    csv_path = _CSV_PATH
    if not csv_path.exists():
        logger.warning("seasonal_hazards.csv not found at %s", csv_path)
        return data

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iso3 = (row.get("ISO") or "").strip().upper()
            hazard = (row.get("Hazard") or "").strip().upper()
            if not iso3 or not hazard:
                continue
            monthly: list[float] = []
            for col in _MONTH_COLS:
                raw = (row.get(col) or "0").strip()
                if raw.lower() == "x":
                    monthly.append(0.0)
                else:
                    try:
                        monthly.append(float(raw))
                    except (ValueError, TypeError):
                        monthly.append(0.0)
            data[(iso3, hazard)] = monthly

    logger.info("Loaded seasonal hazard data: %d country-hazard pairs", len(data))
    return data


def _forecast_months(run_date: date) -> list[int]:
    """Return the 6 forecast month indices (0-based: 0=JAN .. 11=DEC).

    For a run in month M, the forecast covers months M+1 through M+6.
    Example: run_date in January (month 1) -> Feb(1), Mar(2), Apr(3),
    May(4), Jun(5), Jul(6).
    """
    base = run_date.month  # 1-based
    return [((base - 1 + offset) % 12) for offset in range(1, 7)]


def get_active_hazards(iso3: str, run_date: date) -> Set[str]:
    """Return the set of hazard codes that should get RC assessment.

    - ACE: always active
    - DI: never active (silenced)
    - FL, DR, TC, HW: active if any of the 6 forecast months has
      base-rate > 0 in seasonal_hazards.csv.  If a country/hazard pair
      is missing from the CSV, the hazard is conservatively treated as
      active.
    """
    iso3_up = (iso3 or "").upper()
    seasonal = _load_seasonal_data()
    fc_months = _forecast_months(run_date)

    active: Set[str] = set(_ALWAYS_ACTIVE)

    for hazard_code in ("FL", "DR", "TC", "HW"):
        key = (iso3_up, hazard_code)
        if key not in seasonal:
            active.add(hazard_code)
            continue
        monthly_rates = seasonal[key]
        if any(monthly_rates[m] > 0.0 for m in fc_months):
            active.add(hazard_code)

    return active
