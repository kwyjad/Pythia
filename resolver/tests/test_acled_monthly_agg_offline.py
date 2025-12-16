# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import json
from pathlib import Path

import pandas as pd
import pytest

from resolver.ingestion.acled_client import ACLEDClient

pytestmark = pytest.mark.offline


def _load_fixture() -> pd.DataFrame:
    fixture_path = Path(__file__).resolve().parent / "../ingestion/tests/fixtures/acled_sample_events.json"
    with fixture_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    frame = pd.DataFrame(payload)
    frame["event_date"] = pd.to_datetime(frame["event_date"], utc=True).dt.tz_convert(None)
    frame["iso3"] = frame["iso3"].astype(str).str.upper()
    frame["country"] = frame["country"].astype(str)
    frame["fatalities"] = frame["fatalities"].astype(int)
    return frame


def test_monthly_fatalities_groups_correctly(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    events = _load_fixture()
    client = ACLEDClient()

    monkeypatch.setattr(client, "fetch_events", lambda *_, **__: events.copy())

    caplog.set_level("DEBUG", logger="resolver.ingestion.acled.client")
    result = client.monthly_fatalities("2024-01-01", "2024-02-29")

    assert list(result.columns) == ["iso3", "month", "fatalities", "source", "updated_at"]
    assert result["fatalities"].tolist() == [10, 3]
    assert result["iso3"].tolist() == ["ETH", "KEN"]
    assert result["month"].dt.strftime("%Y-%m-%d").tolist() == ["2024-02-01", "2024-01-01"]
    assert (result["fatalities"] >= 0).all()
    assert not result.isna().any().any()
    assert (result.sort_values(["iso3", "month"]).index == result.index).all()
    assert all(result["source"] == "ACLED")
    assert "Grouped to monthly fatalities" in " ".join(caplog.messages)
