import json
from pathlib import Path

import pytest
import requests

from resolver.ingestion import emdat_client
from resolver.ingestion.emdat_client import EmdatClient


class _DummyResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, object]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _RecordingSession:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload
        self.sent: list[dict[str, object]] = []

    def post(self, url, data=None, headers=None, timeout=None):  # noqa: D401, ANN001
        decoded = json.loads(data.decode("utf-8")) if isinstance(data, (bytes, bytearray)) else data
        self.sent.append(decoded)
        return _DummyResponse(self._payload)


def _fixed_perf_counter(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> None:
    iterator = iter(values)
    monkeypatch.setattr(emdat_client.time, "perf_counter", lambda: next(iterator))


def test_emdat_probe_writes_metadata(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(emdat_client, "_current_timestamp", lambda: "2024-06-01T00:00:00Z")
    _fixed_perf_counter(monkeypatch, [100.0, 100.321])

    payload = {
        "data": {
            "api_version": "2024-05",
            "public_emdat": {
                "total_available": 0,
                "info": {
                    "timestamp": "2024-05-01T00:00:00Z",
                    "version": "dataset-v1",
                },
            },
        }
    }
    session = _RecordingSession(payload)
    client = EmdatClient(network=True, api_key="token", session=session)

    result = client.probe(from_year=2020, to_year=2021, iso=["PHL"])

    assert result["ok"] is True
    assert result["http_status"] == 200
    assert result["api_version"] == "2024-05"
    assert result["info"] == {
        "timestamp": "2024-05-01T00:00:00Z",
        "version": "dataset-v1",
    }
    assert result["total_available"] == 0
    assert result["recorded_at"] == "2024-06-01T00:00:00Z"
    assert result["filters"]["iso"] == ["PHL"]
    assert result["filters"]["from"] == 2020
    assert result["filters"]["to"] == 2021

    probe_path = Path("diagnostics/ingestion/emdat/probe.json")
    assert probe_path.exists()
    saved = json.loads(probe_path.read_text(encoding="utf-8"))
    assert saved["ok"] is True
    assert saved["http_status"] == 200
    assert saved["api_version"] == "2024-05"
    assert saved["info"]["version"] == "dataset-v1"
    assert pytest.approx(saved["elapsed_ms"], rel=0.01) == result["elapsed_ms"]


def test_emdat_probe_handles_timeout(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(emdat_client, "_current_timestamp", lambda: "2024-06-02T00:00:00Z")
    _fixed_perf_counter(monkeypatch, [200.0, 200.25])

    class _TimeoutSession:
        def post(self, *args, **kwargs):  # noqa: D401, ANN001
            raise requests.Timeout("connection timeout")

    client = EmdatClient(network=True, api_key="token", session=_TimeoutSession())

    result = client.probe(from_year=2019, to_year=2020)

    assert result["ok"] is False
    assert "timeout" in (result.get("error") or "").lower()
    assert result["http_status"] is None
    assert result["elapsed_ms"] is None
    assert result["total_available"] is None

    probe_path = Path("diagnostics/ingestion/emdat/probe.json")
    assert probe_path.exists()
    saved = json.loads(probe_path.read_text(encoding="utf-8"))
    assert saved["ok"] is False
    assert saved["http_status"] is None
    assert saved["total_available"] is None
    assert "timeout" in (saved.get("error") or "").lower()
