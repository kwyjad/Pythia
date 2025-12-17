# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for ACLED monthly client ISO3 fallback logic."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict

import pandas as pd
import pytest

from resolver.ingestion import acled_client as acled_client_module
from resolver.ingestion.acled_client import ACLEDClient


@pytest.fixture(autouse=True)
def _mock_acled_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "resolver.ingestion.acled_client.acled_auth.get_access_token",
        lambda: "test-token",
    )


def _patch_fetch(monkeypatch: pytest.MonkeyPatch, payloads: Dict[int, Dict[str, Any]]):
    calls: list[Dict[str, Any]] = []

    def _fake_fetch(self: ACLEDClient, params: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(dict(params))
        page = int(params.get("page", 1))
        return payloads.get(page, {"data": []})

    monkeypatch.setattr(ACLEDClient, "_fetch_page", _fake_fetch)
    return calls


def test_acled_iso3_fallback_and_monthly_aggregation(monkeypatch: pytest.MonkeyPatch):
    calls = _patch_fetch(
        monkeypatch,
        {
            1: {
                "data": [
                    {"event_date": "2023-01-15", "country": "Albania", "fatalities": "1"},
                    {"event_date": "2023-01-20", "country": "Albania", "fatalities": "3"},
                ]
            },
            2: {"data": []},
        },
    )

    client = ACLEDClient()
    frame = client.fetch_events(date(2023, 1, 1), date(2023, 1, 31))

    assert not frame.empty
    assert frame["iso3"].tolist() == ["ALB", "ALB"]
    assert calls, "_fetch_page should be invoked"
    assert calls[0]["fields"].split("|") == ["event_date", "iso3", "country", "fatalities"]

    monthly = client.monthly_fatalities(date(2023, 1, 1), date(2023, 1, 31))
    assert len(monthly) == 1
    row = monthly.iloc[0]
    assert row["iso3"] == "ALB"
    assert row["fatalities"] == 4
    assert row["month"] == pd.Timestamp("2023-01-01 00:00:00")


def test_acled_existing_iso3_preserved(monkeypatch: pytest.MonkeyPatch):
    _patch_fetch(
        monkeypatch,
        {
            1: {
                "data": [
                    {"event_date": "2023-02-05", "country": "Kenya", "iso3": "KEN", "fatalities": "2"},
                    {"event_date": "2023-02-06", "country": "Kenya", "iso3": "KEN", "fatalities": "1"},
                ]
            },
            2: {"data": []},
        },
    )

    client = ACLEDClient()
    frame = client.fetch_events(date(2023, 2, 1), date(2023, 2, 28))
    assert frame["iso3"].unique().tolist() == ["KEN"]

    monthly = client.monthly_fatalities(date(2023, 2, 1), date(2023, 2, 28))
    assert len(monthly) == 1
    row = monthly.iloc[0]
    assert row["iso3"] == "KEN"
    assert row["fatalities"] == 3


def test_prepare_dataframe_resolves_cod_alias():
    config = {
        "keys": {
            "iso3": ["iso3"],
            "country": ["country"],
            "date": ["event_date"],
            "event_type": ["event_type"],
            "fatalities": ["fatalities"],
            "notes": [],
        }
    }
    countries = pd.DataFrame(
        {"iso3": ["COD"], "country_name": ["Democratic Republic of the Congo"]}
    )
    records = [
        {
            "event_date": "2025-01-15",
            "country": "Democratic Republic of the Congo",
            "fatalities": 7,
        }
    ]

    frame = acled_client_module._prepare_dataframe(records, config, countries)

    assert not frame.empty
    assert frame.loc[0, "iso3"] == "COD"
