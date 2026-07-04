# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for user-input validation boundaries in the API layer.

The iso3 and forecaster_run_id query params are interpolated into (or
parameterized within) SQL, so malformed values must be rejected with 400
before they reach a query string.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi import HTTPException

from pythia.api.app import _run_filter_cte, _validate_iso3_param


class _StubCon:
    """Stands in for a DuckDB connection in _run_filter_cte tests."""


def test_validate_iso3_accepts_valid_codes():
    assert _validate_iso3_param("ETH") == "ETH"
    assert _validate_iso3_param("eth") == "ETH"
    assert _validate_iso3_param(" som ") == "SOM"


def test_validate_iso3_passes_through_empty():
    assert _validate_iso3_param(None) is None
    assert _validate_iso3_param("") is None
    assert _validate_iso3_param("   ") is None


@pytest.mark.parametrize(
    "bad",
    [
        "ET",  # too short
        "ETHH",  # too long
        "E1H",  # digit
        "ETH' OR '1'='1",  # injection attempt
        "'; DROP TABLE facts_resolved; --",
    ],
)
def test_validate_iso3_rejects_malformed(bad):
    with pytest.raises(HTTPException) as exc_info:
        _validate_iso3_param(bad)
    assert exc_info.value.status_code == 400


def test_run_filter_cte_rejects_malformed_run_id(monkeypatch):
    import pythia.api.app as app_mod

    monkeypatch.setattr(app_mod, "_table_has_columns", lambda con, table, cols: True)
    with pytest.raises(HTTPException) as exc_info:
        _run_filter_cte(_StubCon(), "fc_123' OR '1'='1")
    assert exc_info.value.status_code == 400


def test_run_filter_cte_accepts_normal_run_id(monkeypatch):
    import pythia.api.app as app_mod

    monkeypatch.setattr(app_mod, "_table_has_columns", lambda con, table, cols: True)
    cte, join = _run_filter_cte(_StubCon(), "fc_1774107846")
    assert "fc_1774107846" in cte
    assert "JOIN fc_run_filter" in join
