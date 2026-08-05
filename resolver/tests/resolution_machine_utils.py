# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Shared helpers for the resolution-machine test suites."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from resolver.resolution_machine.rulebook import RULEBOOK_PATH, Rulebook

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "resolution_machine"
IBTRACS_SAMPLE_CSV = FIXTURES / "ibtracs_sample.csv"
SYNTHETIC_COUNTRIES_GEOJSON = FIXTURES / "synthetic_countries.geojson"


def _deep_merge(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def make_rulebook(overrides: dict[str, Any] | None = None) -> Rulebook:
    """The real rulebook.yaml, optionally with test overrides deep-merged."""
    with open(RULEBOOK_PATH, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if overrides:
        data = _deep_merge(data, overrides)
    return Rulebook(data, RULEBOOK_PATH)


def seed_ibtracs(con, scope: str = "last3years") -> int:
    """Parse + store the fixture IBTrACS CSV into ``con``."""
    from resolver.resolution_machine.ibtracs import parse_ibtracs_csv, store_ibtracs

    frame = parse_ibtracs_csv(IBTRACS_SAMPLE_CSV)
    return store_ibtracs(
        con, frame, scope, "https://example.test/ibtracs.last3years.list.v04r01.csv"
    )


def silent_sweep_evidence(iso3: str, ym: str) -> dict[str, Any]:
    """A well-formed silent ReliefWeb sweep record (as the sweep returns)."""
    return {
        "iso3": iso3,
        "ym": ym,
        "hazard_key": "cyclone",
        "silent": True,
        "inconclusive": False,
        "total_hits": 0,
        "queries": [
            {
                "kind": "disaster_type",
                "url": "https://api.reliefweb.int/v2/reports",
                "payload": {"filter": {"conditions": ["…"]}},
                "hits": 0,
                "sample": [],
            },
            {
                "kind": "keywords",
                "url": "https://api.reliefweb.int/v2/reports",
                "payload": {"query": {"value": "cyclone OR hurricane"}},
                "hits": 0,
                "sample": [],
            },
        ],
        "retrieved_at": "2026-01-15T00:00:00+00:00",
        "error": None,
    }
