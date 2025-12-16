# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Hazard mapping helpers for IDMC IDU rows."""
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

CANON = {
    "FL": ("Flood", "natural"),
    "DR": ("Drought", "natural"),
    "TC": ("Tropical Cyclone", "natural"),
    "HW": ("Heat Wave", "natural"),
    "ACO": ("Armed Conflict - Onset", "human-induced"),
    "ACE": ("Armed Conflict - Escalation", "human-induced"),
    "DI": (
        "Displacement Influx (cross-border from neighbouring country)",
        "human-induced",
    ),
    "CU": ("Civil Unrest", "human-induced"),
    "EC": ("Economic Crisis", "human-induced"),
    "PHE": ("Public Health Emergency", "epidemic"),
}


def _norm(value: Optional[str]) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _disaster_to_code(
    hazard_category: str,
    hazard_subcategory: str,
    hazard_type: str,
    hazard_subtype: str,
) -> Optional[str]:
    tokens = " ".join(
        [
            _norm(hazard_category),
            _norm(hazard_subcategory),
            _norm(hazard_type),
            _norm(hazard_subtype),
        ]
    )
    if any(keyword in tokens for keyword in ["flood", "flash flood", "river flood"]):
        return "FL"
    if "drought" in tokens:
        return "DR"
    if any(
        keyword in tokens
        for keyword in [
            "tropical cyclone",
            "cyclone",
            "hurricane",
            "typhoon",
            "storm",
        ]
    ):
        return "TC"
    if any(
        keyword in tokens
        for keyword in ["heat wave", "extreme heat", "heatwave", "heat"]
    ):
        return "HW"
    if any(
        keyword in tokens
        for keyword in ["epidemic", "outbreak", "disease", "pandemic"]
    ):
        return "PHE"
    if "economic" in tokens and "crisis" in tokens:
        return "EC"
    return None


def _conflict_to_code(
    violence_type: str,
    conflict_type: str,
    notes: str,
) -> Optional[str]:
    text = " ".join([_norm(violence_type), _norm(conflict_type), _norm(notes)])
    if any(
        keyword in text
        for keyword in [
            "riot",
            "riots",
            "protest",
            "protests",
            "civil unrest",
            "unrest",
        ]
    ):
        return "CU"
    return "ACE"


def map_row_to_hazard(row: pd.Series) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    displacement_type = _norm(row.get("displacement_type"))
    if displacement_type == "disaster":
        code = _disaster_to_code(
            row.get("hazard_category"),
            row.get("hazard_subcategory"),
            row.get("hazard_type"),
            row.get("hazard_subtype"),
        )
        if code:
            label, hazard_class = CANON[code]
            return code, label, hazard_class
        return None, None, "natural"

    if displacement_type == "conflict":
        code = _conflict_to_code(
            row.get("violence_type"),
            row.get("conflict_type"),
            row.get("notes") or row.get("event_details"),
        )
        if code and code in CANON:
            label, hazard_class = CANON[code]
            return code, label, hazard_class
        return None, None, "human-induced"

    if displacement_type == "development":
        return None, None, "human-induced"

    return None, None, None


def apply_hazard_mapping(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.assign(
            hazard_code=pd.Series(dtype="object"),
            hazard_label=pd.Series(dtype="object"),
            hazard_class=pd.Series(dtype="object"),
        )

    codes: list[Optional[str]] = []
    labels: list[Optional[str]] = []
    classes: list[Optional[str]] = []
    for _, row in frame.iterrows():
        code, label, hazard_class = map_row_to_hazard(row)
        codes.append(code)
        labels.append(label)
        classes.append(hazard_class)

    mapped = frame.copy()
    mapped["hazard_code"] = codes
    mapped["hazard_label"] = labels
    mapped["hazard_class"] = classes
    return mapped
