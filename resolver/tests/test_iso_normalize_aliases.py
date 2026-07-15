# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ACLED country-name forms must resolve to ISO3 (attribution regression).

countries.csv stores several countries under labels that differ from the
names the ACLED API returns ("Great Britain" vs "United Kingdom", "Guinea
Conakry" vs "Guinea", ...). The July 2026 run silently stored ZERO political
events for GIN/COG/BIH/KOR/TLS and others because the name-fallback path in
``resolve_iso3`` had no aliases for the ACLED forms.
"""

from __future__ import annotations

import pytest

from resolver.ingestion.utils.iso_normalize import resolve_iso3, to_iso3


ACLED_NAME_FORMS = [
    ("United Kingdom", "GBR"),
    ("Republic of Congo", "COG"),
    ("Republic of the Congo", "COG"),
    ("North Macedonia", "MKD"),
    ("Slovakia", "SVK"),
    ("South Korea", "KOR"),
    ("Bosnia and Herzegovina", "BIH"),
    ("Guinea", "GIN"),
    ("East Timor", "TLS"),
    ("Vatican City", "VAT"),
    ("Vatican (City)", "VAT"),
    ("Turks and Caicos Islands", "TCA"),
    ("Kosovo", "RKS"),
]


@pytest.mark.parametrize("name,expected", ACLED_NAME_FORMS)
def test_acled_name_form_resolves(name: str, expected: str) -> None:
    assert to_iso3(name) == expected


@pytest.mark.parametrize("name,expected", ACLED_NAME_FORMS)
def test_resolve_iso3_country_name_fallback(name: str, expected: str) -> None:
    iso3, reason = resolve_iso3({"country": name}, name_keys=("country",))
    assert iso3 == expected
    assert reason is None


def test_guinea_alias_does_not_shadow_neighbours() -> None:
    assert to_iso3("Guinea-Bissau") == "GNB"
    assert to_iso3("Equatorial Guinea") == "GNQ"
    assert to_iso3("Papua New Guinea") == "PNG"


def test_congo_alias_does_not_shadow_drc() -> None:
    assert to_iso3("Democratic Republic of Congo") == "COD"
    assert to_iso3("DR Congo") == "COD"
