import json
from datetime import date

import pandas as pd

from resolver.ingestion.idmc import client


def test_fetch_helix_last180_path_and_headers(monkeypatch):
    captured = {}

    def fake_http_get(url, *, headers=None, timeout=None, retries=None, verify=None):
        captured["url"] = url
        captured["headers"] = headers or {}
        captured["timeout"] = timeout
        captured["retries"] = retries
        payload = json.dumps(
            [
                {
                    "iso3": "USA",
                    "event_date": "2024-01-02",
                    "new_displacements": 10,
                }
            ]
        ).encode("utf-8")
        return 200, {}, payload, {"status": 200}

    monkeypatch.setenv("IDMC_USER_AGENT", "pytest-agent/1.0")
    monkeypatch.setattr(client, "HELIX_BASE", "https://helix-tools-api.idmcdb.org")
    monkeypatch.setattr(client, "http_get", fake_http_get)

    frame, diagnostics = client._fetch_helix_idus_last180("abc123")

    expected_url = (
        "https://helix-tools-api.idmcdb.org"
        "/external-api/idus/last-180-days/?client_id=abc123&format=json"
    )
    assert captured["url"] == expected_url
    assert captured["headers"]["Accept"] == "application/json"
    assert captured["headers"]["User-Agent"] == "pytest-agent/1.0"
    assert diagnostics["status"] == 200
    assert diagnostics["last_request_path"].startswith("/external-api/idus/last-180-days/")
    assert "abc123" not in diagnostics["last_request_path"]
    assert "format=json" in diagnostics["last_request_path"]
    assert diagnostics["http_status_counts"] == {"2xx": 1, "4xx": 0, "5xx": 0}
    assert not frame.empty


def test_normalise_helix_last180_monthly_filters_and_rolls_up():
    payload = pd.DataFrame(
        [
            {"iso3": "usa", "event_date": "2024-01-05", "value": 5},
            {"iso3": "USA", "event_date": "2024-01-20", "value": 7},
            {"iso3": "FRA", "event_date": "2024-02-10", "value": 3},
            {"iso3": "ESP", "event_date": "2024-02-11", "value": 4},
        ]
    )

    normalized = client._normalise_helix_last180_monthly(
        payload,
        window_start=date(2024, 1, 1),
        window_end=date(2024, 2, 29),
        countries=["USA", "FRA"],
    )

    assert list(normalized.columns) == list(client.FLOW_EXPORT_COLUMNS) + [client.HDX_PREAGG_COLUMN]
    assert set(normalized["iso3"]) == {"USA", "FRA"}

    usa_row = normalized.loc[normalized["iso3"] == "USA"].iloc[0]
    assert usa_row["value"] == 12
    assert usa_row["as_of_date"] == pd.Timestamp("2024-01-31")
    assert usa_row["metric"] == client.FLOW_METRIC
    assert usa_row["series_semantics"] == client.FLOW_SERIES_SEMANTICS
    assert usa_row["source"] == "idmc_idu"

    fra_row = normalized.loc[normalized["iso3"] == "FRA"].iloc[0]
    assert fra_row["as_of_date"] == pd.Timestamp("2024-02-29")
    assert fra_row["value"] == 3
    assert bool(fra_row[client.HDX_PREAGG_COLUMN]) is False
