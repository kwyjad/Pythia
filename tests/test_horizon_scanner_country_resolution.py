# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import logging
import sys
import types

import pytest

if "duckdb" not in sys.modules:  # pragma: no cover - dependency may be absent in CI smoke runs
    duckdb_stub = types.ModuleType("duckdb")

    class _DummyCatalogException(Exception):
        pass

    class _DummyConnection:
        def execute(self, *args, **kwargs):
            return self

        def fetchall(self):
            return []

        def fetchone(self):
            return None

        def close(self):
            return None

    duckdb_stub.CatalogException = _DummyCatalogException
    duckdb_stub.connect = lambda *args, **kwargs: _DummyConnection()
    duckdb_stub.__pythia_stub__ = True
    sys.modules["duckdb"] = duckdb_stub

from horizon_scanner.horizon_scanner import (
    _norm_country_key,
    _load_country_list,
    _load_country_registry,
    _resolve_country,
)


@pytest.mark.parametrize(
    "alias",
    [
        "Democratic Republic of Congo",
        "Democratic Republic of the Congo",
        "DR Congo",
        "drc",
    ],
)
def test_resolve_country_aliases_map_to_cod(alias):
    iso3_to_name, name_to_iso3 = _load_country_registry()

    name, iso3 = _resolve_country(alias, iso3_to_name, name_to_iso3)

    assert iso3 == "COD"
    assert name == iso3_to_name["COD"]


def test_resolve_country_accepts_registry_and_skips_unknown(caplog):
    iso3_to_name, name_to_iso3 = _load_country_registry()

    name, iso3 = _resolve_country("Germany", iso3_to_name, name_to_iso3)
    assert (name, iso3) == ("Germany", "DEU")

    caplog.set_level(logging.WARNING)
    resolved, skipped, requested = _load_country_list(["Germany", "FooBarLand"])

    assert resolved == [("Germany", "DEU")]
    assert len(skipped) == 1
    assert requested == ["Germany", "FooBarLand"]
    assert "FooBarLand" in caplog.text
    assert "skipped 1 unknown/invalid entries" in caplog.text


@pytest.mark.parametrize(
    "raw",
    [
        "Cote d'Ivoire",
        "Côte d’Ivoire",
        "Cote dIvoire",
        "cote   d ivoire",
    ],
)
def test_norm_country_key_handles_accents_and_spacing(raw):
    normalized = _norm_country_key(raw)
    assert normalized == "cote d ivoire"


def test_resolve_country_handles_cote_d_ivoire_variants():
    iso3_to_name, name_to_iso3 = _load_country_registry()

    for variant in ["Cote d'Ivoire", "Côte d’Ivoire", "Cote dIvoire"]:
        name, iso3 = _resolve_country(variant, iso3_to_name, name_to_iso3)
        assert iso3 == "CIV"
        assert name == iso3_to_name["CIV"]
