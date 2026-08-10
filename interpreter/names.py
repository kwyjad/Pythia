# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Human names for the codes the report would otherwise print raw.

A reader with no forecasting background should never meet "NIC —
DR/EVENT_OCCURRENCE". Country names come from resolver/data/countries.csv
(the same list the rest of the system resolves against); hazards and metrics
are small fixed maps, because their codes are a closed set defined in
pythia/config.yaml.

Shared by the pack builder (so the model sees names, not codes), the
renderer and the PDF, so all three agree on what a thing is called.
"""

from __future__ import annotations

import csv
import functools
from pathlib import Path

# The two families the report splits its sections by. Drought, flood and
# tropical cyclone move with the weather; armed conflict does not, and the
# reasoning a reader needs for each is different enough to keep apart.
# Which aggregate speaks for a question, best first.
#
# ensemble_mean_v2 leads by owner decision (2026-08-10). BayesMC and the mean
# disagree sharply on thin anchors: on IDN_TC_PA_2026-09 BayesMC put the
# forecast at 14.1x its base rate where the mean put it at 2.25x, and it was
# the BayesMC figure that led the August report. Declared once and read by the
# pack builder, the prompt pack, the API and the printed map, because it
# previously lived in three copies plus an implicit fourth (the printed map
# had no preference at all and simply took the largest across every model,
# Sibyl included).
AGGREGATE_PREFERENCE = ("ensemble_mean_v2", "ensemble_bayesmc_v2", "track2_flash")

CLIMATE_HAZARDS = ("DR", "FL", "TC")
CONFLICT_HAZARDS = ("ACE", "ACO")

HAZARD_NAMES = {
    "ACE": "Armed conflict",
    "ACO": "Armed conflict",
    "DR": "Drought",
    "FL": "Flood",
    "TC": "Tropical cyclone",
    "HW": "Heatwave",
    "CU": "Civil unrest",
    "DI": "Displacement influx",
    "EC": "Economic crisis",
    "PHE": "Public health emergency",
}

# What the question actually asks for, in words. These are the subject of a
# sentence, so they read as nouns.
METRIC_NAMES = {
    "PA": "people affected",
    "FATALITIES": "deaths",
    "EVENT_OCCURRENCE": "a major disaster alert",
    "PHASE3PLUS_IN_NEED": "people in crisis-level hunger",
}

# Short forms for headings, where the hazard name already carries the sense.
METRIC_SHORT = {
    "PA": "people affected",
    "FATALITIES": "deaths",
    "EVENT_OCCURRENCE": "major alert",
    "PHASE3PLUS_IN_NEED": "crisis-level hunger",
}

_COUNTRIES_CSV = (
    Path(__file__).resolve().parents[1] / "resolver" / "data" / "countries.csv"
)


@functools.lru_cache(maxsize=1)
def _country_map() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        # utf-8-sig: the checked-in file carries a BOM, which would otherwise
        # land in the first header name and lose the country_name column.
        with _COUNTRIES_CSV.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                iso3 = (row.get("iso3") or "").strip().upper()
                name = (row.get("country_name") or "").strip()
                if iso3 and name:
                    out[iso3] = name
    except OSError:
        # A missing list degrades to printing the code, which is ugly but
        # true; it must never stop a report being written.
        return {}
    return out


def country_name(iso3: str | None) -> str:
    code = (iso3 or "").strip().upper()
    if not code:
        return "Unknown country"
    return _country_map().get(code, code)


def iso3_codes() -> frozenset[str]:
    """Every ISO3 code the country table knows.

    Used by the style check to tell a country code from an agency acronym:
    FAO and IOM are prose, SOM and NIC are codes a reader cannot decode.
    """
    return frozenset(_country_map())


def hazard_name(hazard_code: str | None) -> str:
    code = (hazard_code or "").strip().upper()
    return HAZARD_NAMES.get(code, code or "Unknown hazard")


def metric_name(metric: str | None, *, short: bool = False) -> str:
    code = (metric or "").strip().upper()
    table = METRIC_SHORT if short else METRIC_NAMES
    return table.get(code, code.replace("_", " ").lower() or "impact")


def hazard_family(hazard_code: str | None) -> str:
    """``climate`` | ``conflict`` | ``other`` — the report's section split."""
    code = (hazard_code or "").strip().upper()
    if code in CLIMATE_HAZARDS:
        return "climate"
    if code in CONFLICT_HAZARDS:
        return "conflict"
    return "other"


FAMILY_LABELS = {
    "climate": "Climate hazards",
    "conflict": "Conflict",
    "other": "Other hazards",
}


def describe_pair(
    iso3: str | None, hazard_code: str | None, metric: str | None
) -> str:
    """The heading form: 'Nicaragua, drought: major alert'."""
    return (
        f"{country_name(iso3)}, {hazard_name(hazard_code).lower()}: "
        f"{metric_name(metric, short=True)}"
    )
