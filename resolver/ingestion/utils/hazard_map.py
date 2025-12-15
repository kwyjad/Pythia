# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Utilities for normalising hazard labels to Resolver's canonical vocabulary."""

from __future__ import annotations

from typing import Mapping, Optional

CLASSIF_TO_SHOCK: Mapping[str, str] = {
    "nat-cli-dro-dro": "drought",
    "nat-met-sto-tro": "tropical_cyclone",
    "nat-hyd-flo-riv": "flood",
    "nat-hyd-flo-fla": "flood",
}

_DEFAULT_MAP = {
    "eq": "earthquake",
    "earthquake": "earthquake",
    "fl": "flood",
    "flood": "flood",
    "tc": "tropical_cyclone",
    "cyclone": "tropical_cyclone",
    "storm": "tropical_cyclone",
    "hurricane": "tropical_cyclone",
    "volcano": "volcano",
    "vo": "volcano",
    "wf": "wildfire",
    "wildfire": "wildfire",
    "fire": "wildfire",
    "ls": "landslide",
    "landslide": "landslide",
    "tsunami": "tsunami",
    "dz": "drought",
    "drought": "drought",
    "conflict": "conflict",
}


def _normalise(token: str) -> str:
    return "".join(ch for ch in token.lower() if ch.isalnum())


def map_hazard(token: str | None, overrides: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Return the canonical hazard string for ``token`` if known."""

    if not token:
        return None
    normalised = _normalise(str(token))
    if not normalised:
        return None
    if overrides:
        for raw, canonical in overrides.items():
            if _normalise(str(raw)) == normalised:
                value = str(canonical).strip().lower()
                return value or None
    if normalised in _DEFAULT_MAP:
        return _DEFAULT_MAP[normalised]
    return None
