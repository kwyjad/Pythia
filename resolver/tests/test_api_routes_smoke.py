"""Smoke tests for FastAPI resolver endpoints."""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


@pytest.fixture()
def api_module(fast_exports):  # noqa: ANN001 - shared fixture from bootstrap
    """Reload the API module so fixtures and env overrides apply."""

    return importlib.reload(importlib.import_module("resolver.api.app"))


@pytest.fixture()
def api_client(api_module):  # noqa: ANN001
    return TestClient(api_module.app)


def test_resolve_endpoint_returns_row(api_client):
    """Ensure GET /resolve exists and returns a populated payload."""

    params = {
        "iso3": "PHL",
        "hazard_code": "TC",
        "cutoff": "2024-01-31",
        "series": "new",
        "backend": "files",
    }

    response = api_client.get("/resolve", params=params)
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["iso3"] == "PHL"
    assert payload["hazard_code"] == "TC"
    assert payload["value"] == 1500
    assert payload["series_returned"] == "new"
