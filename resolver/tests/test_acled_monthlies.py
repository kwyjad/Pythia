import datetime as dt

import json
from pathlib import Path

import pytest

from resolver.ingestion import acled_client


def _rows_to_map(rows):
    out = {}
    for row in rows:
        key = (row["iso3"], row["hazard_code"], row["metric"], row["as_of_date"])
        out[key] = row
    return out


def test_acled_monthly_aggregation(monkeypatch):
    monkeypatch.delenv("ACLED_PARSE_PARTICIPANTS", raising=False)
    records = [
        {
            "event_date": "2023-01-15",
            "event_type": "Battles",
            "country": "Kenya",
            "iso3": "KEN",
            "fatalities": "3",
            "notes": "",
        },
        {
            "event_date": "2023-01-20",
            "event_type": "Riots",
            "country": "Kenya",
            "iso3": "",
            "fatalities": "1",
            "notes": "Reports of protesters 200 strong in the streets",
        },
        {
            "event_date": "2023-01-22",
            "event_type": "Protests",
            "country": "Kenya",
            "iso3": "KEN",
            "fatalities": "0",
            "notes": "Approximately protesters totaling 1,500 joined",
        },
        {
            "event_date": "2023-02-10",
            "event_type": "Battles",
            "country": "Uganda",
            "iso3": "UGA",
            "fatalities": "30",
            "notes": "",
        },
        {
            "event_date": "2023-02-15",
            "event_type": "Battles",
            "country": "Uganda",
            "iso3": "UGA",
            "fatalities": "5",
            "notes": "",
        },
        {
            "event_date": "2023-02-18",
            "event_type": "Protests",
            "country": "Uganda",
            "iso3": "UGA",
            "fatalities": "0",
            "notes": "Participants numbered 200 downtown",
        },
    ]

    config = acled_client.load_config()
    config.setdefault("participants", {})["enabled"] = True
    countries, shocks = acled_client.load_registries()
    publication_date = "2023-03-01"
    ingested_at = "2023-03-02T00:00:00Z"

    rows = acled_client._build_rows(
        records,
        config,
        countries,
        shocks,
        "https://example.com/acled",
        publication_date,
        ingested_at,
    )
    assert rows, "expected rows from aggregation"

    rows_map = _rows_to_map(rows)

    kenya_conflict = rows_map[("KEN", "ACE", "fatalities_battle_month", "2023-01")]
    assert kenya_conflict["value"] == 3
    assert "Prev12m" in kenya_conflict["definition_text"]

    uganda_escalation = rows_map[("UGA", "ACE", "fatalities_battle_month", "2023-02")]
    assert uganda_escalation["value"] == 35
    assert "battle_fatalities=35" in uganda_escalation["method"]

    uganda_onset = rows_map[("UGA", "ACO", "fatalities_battle_month", "2023-02")]
    assert "onset_rule_v1" in uganda_onset["method"]
    assert uganda_onset["value"] == 35

    kenya_unrest = rows_map[("KEN", "CU", "events", "2023-01")]
    assert kenya_unrest["value"] == 2
    assert kenya_unrest["unit"] == "events"

    uganda_unrest = rows_map[("UGA", "CU", "events", "2023-02")]
    assert uganda_unrest["value"] == 1

    kenya_participants = rows_map[("KEN", "CU", "participants", "2023-01")]
    assert kenya_participants["value"] == 1700
    assert kenya_participants["unit"] == "persons"

    uganda_participants = rows_map[("UGA", "CU", "participants", "2023-02")]
    assert uganda_participants["value"] == 200
    assert uganda_participants["unit"] == "persons"


def test_fetch_events_raises_on_http_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESOLVER_START_ISO", "2024-01-01")
    monkeypatch.setenv("RESOLVER_END_ISO", "2024-01-31")
    diagnostics_root = tmp_path / "diagnostics" / "ingestion"
    monkeypatch.setattr(acled_client, "ACLED_DIAGNOSTICS", diagnostics_root / "acled")
    monkeypatch.setattr(acled_client, "ACLED_RUN_PATH", diagnostics_root / "acled_client" / "acled_client_run.json")
    monkeypatch.setattr(acled_client, "ACLED_HTTP_DIAG_PATH", diagnostics_root / "acled" / "http_diag.json")

    class StubResponse:
        status_code = 403
        headers: dict[str, str] = {}
        text = "forbidden"
        url = "https://acleddata.com/api/acled/read?_format=json"

        def json(self) -> dict[str, object]:  # pragma: no cover - defensive
            return {}

    class StubSession:
        def get(self, url, params, headers, timeout):  # noqa: D401 - test stub
            assert headers["Authorization"] == "Bearer TEST"
            return StubResponse()

    monkeypatch.setattr(acled_client.acled_auth, "get_access_token", lambda: "TEST")
    monkeypatch.setattr(acled_client.requests, "Session", lambda: StubSession())

    config = {
        "query": {},
    }

    with pytest.raises(RuntimeError) as excinfo:
        acled_client.fetch_events(config)
    assert "HTTP 403" in str(excinfo.value)


def test_fetch_events_zero_rows_creates_diagnostic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESOLVER_START_ISO", "2024-01-01")
    monkeypatch.setenv("RESOLVER_END_ISO", "2024-01-31")
    diagnostics_root = tmp_path / "diagnostics" / "ingestion"
    monkeypatch.setattr(acled_client, "ACLED_DIAGNOSTICS", diagnostics_root / "acled")
    monkeypatch.setattr(acled_client, "ACLED_RUN_PATH", diagnostics_root / "acled_client" / "acled_client_run.json")
    monkeypatch.setattr(acled_client, "ACLED_HTTP_DIAG_PATH", diagnostics_root / "acled" / "http_diag.json")

    payload = {"data": []}

    class StubResponse:
        status_code = 200
        headers: dict[str, str] = {}
        text = json.dumps(payload)
        url = "https://acleddata.com/api/acled/read?_format=json"

        def json(self) -> dict[str, object]:
            return payload

    class StubSession:
        def get(self, url, params, headers, timeout):
            assert headers["Authorization"] == "Bearer TEST"
            return StubResponse()

    monkeypatch.setattr(acled_client.acled_auth, "get_access_token", lambda: "TEST")
    monkeypatch.setattr(acled_client.requests, "Session", lambda: StubSession())

    config = {
        "query": {},
    }

    records, source_url, meta = acled_client.fetch_events(config)
    assert records == []
    assert meta["http_status"] == 200
    assert source_url.startswith("https://acleddata.com/api/acled/read")

    diag_path = diagnostics_root / "acled" / "zero_rows.json"
    assert diag_path.is_file(), "expected zero_rows.json diagnostic"
    diagnostic = json.loads(diag_path.read_text(encoding="utf-8"))
    assert diagnostic["reason"] == "empty dataframe"
    assert diagnostic["status"] == 200
