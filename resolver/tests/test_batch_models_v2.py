# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import pytest


@pytest.fixture()
def batch_models(monkeypatch):
    import importlib
    import sys
    from types import ModuleType

    monkeypatch.setitem(sys.modules, "duckdb", ModuleType("duckdb"))
    for module in [
        "resolver.api.batch_models",
        "resolver.query.selectors",
        "resolver.query",
        "resolver.db.duckdb_io",
        "resolver.db",
    ]:
        sys.modules.pop(module, None)

    return importlib.import_module("resolver.api.batch_models")


def test_import_batch_models_succeeds(batch_models):
    assert hasattr(batch_models, "ResolveQuery")


@pytest.mark.parametrize(
    "series_input,expected",
    [
        (None, "stock"),
        ("", "stock"),
        ("NEW", "new"),
        ("stock", "stock"),
        ("weird", "stock"),
    ],
)
def test_series_normalisation(series_input, expected, batch_models):
    query = batch_models.ResolveQuery(
        cutoff="2024-01-31",
        iso3="PHL",
        hazard_code="TC",
        series=series_input,
    )

    assert query.series == expected


@pytest.mark.parametrize(
    "backend_input,expected",
    [
        (None, None),
        ("csv", "files"),
        ("files", "files"),
        ("db", "db"),
        ("auto", "auto"),
    ],
)
def test_backend_normalisation(backend_input, expected, batch_models):
    query = batch_models.ResolveQuery(
        cutoff="2024-01-31",
        iso3="PHL",
        hazard_code="TC",
        backend=backend_input,
    )

    assert query.backend == expected


def test_backend_validation_error(batch_models):
    with pytest.raises(ValueError, match="backend must be one of files, db, or auto"):
        batch_models.ResolveQuery(
            cutoff="2024-01-31",
            iso3="PHL",
            hazard_code="TC",
            backend="nope",
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"iso3": None, "country": None}, "provide either country or iso3 for each query"),
        ({"hazard": None, "hazard_code": None}, "provide either hazard or hazard_code for each query"),
    ],
)
def test_identifier_requirements(kwargs, message, batch_models):
    base_kwargs = {
        "cutoff": "2024-01-31",
        "iso3": "PHL",
        "hazard_code": "TC",
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        batch_models.ResolveQuery(**base_kwargs)


def test_identifier_combination_passes(batch_models):
    query = batch_models.ResolveQuery(
        cutoff="2024-01-31",
        iso3="PHL",
        hazard_code="TC",
    )

    assert query.iso3 == "PHL"
    assert query.hazard_code == "TC"


def test_response_row_config_accepts_attribute_names(batch_models):
    row = batch_models.ResolveResponseRow(
        ok=True,
        iso3="PHL",
        hazard_code="TC",
        cutoff="2024-01-31",
        value=123,
        series_requested="new",
        series_returned="stock",
        series_semantics="stock",
    )

    assert row.iso3 == "PHL"
    assert row.series_requested == "new"
    assert row.series_returned == "stock"
