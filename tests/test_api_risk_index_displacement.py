"""The PA risk index says where conflict displacement is not forecast (Oct 2026).

ACE/PA is asked only where IDMC reports regularly. A country's PA total
without it is lower by whatever displacement would have added, so the index
names the countries with a conflict-deaths question and no displacement
question, rather than let a missing question read as no displacement risk.
The fixture's USA has ACE/FATALITIES, FL/PA and TC/PA, and no ACE/PA.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import pythia.api.app as app_mod
from pythia.api.app import app
from tests.test_api_risk_index_smoke import api_env  # noqa: F401  (fixture)


def test_pa_index_names_countries_without_displacement(api_env) -> None:  # noqa: F811
    app_mod._READ_CON = None
    client = TestClient(app)
    payload = client.get(
        "/v1/risk_index", params={"metric": "PA", "target_month": "2026-01"}
    ).json()
    assert [c["iso3"] for c in payload["conflict_displacement_not_forecast"]] == ["USA"]
    assert "not a forecast of no displacement" in payload["conflict_displacement_note"]
    assert payload["rows"][0]["conflict_displacement_forecast"] is False


def test_other_metrics_carry_no_note(api_env) -> None:  # noqa: F811
    app_mod._READ_CON = None
    client = TestClient(app)
    payload = client.get(
        "/v1/risk_index", params={"metric": "FATALITIES", "target_month": "2026-01"}
    ).json()
    assert payload["conflict_displacement_not_forecast"] == []
    assert payload["conflict_displacement_note"] is None
