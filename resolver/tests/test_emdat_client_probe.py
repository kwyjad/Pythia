import json
from pathlib import Path

import pytest
import requests

from resolver.ingestion import emdat_client
from resolver.ingestion.emdat_client import (
    EmdatClient,
    _build_effective_params,
    probe_emdat,
)


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
    assert result["status"] == 200
    assert result["api_version"] == "2024-05"
    assert result["table_version"] == "dataset-v1"
    assert result["metadata_timestamp"] == "2024-05-01T00:00:00Z"
    assert result["total_available"] == 0
    assert result["recorded_at"] == "2024-06-01T00:00:00Z"
    assert result["requests"]["2xx"] == 1
    assert result["latency_ms"] == pytest.approx(321, rel=0.01)
    assert result["filters"]["iso"] == ["PHL"]
    assert result["filters"]["from"] == 2020
    assert result["filters"]["to"] == 2021

    probe_path = Path("diagnostics/ingestion/emdat/probe.json")
    assert probe_path.exists()
    saved = json.loads(probe_path.read_text(encoding="utf-8"))
    assert saved["ok"] is True
    assert saved["status"] == 200
    assert saved["api_version"] == "2024-05"
    assert saved["table_version"] == "dataset-v1"
    assert saved["requests"]["2xx"] == 1
    assert saved["latency_ms"] == pytest.approx(result["latency_ms"], rel=0.01)


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
    assert result["status"] == "error"
    assert result["latency_ms"] is None
    assert result["total_available"] is None

    probe_path = Path("diagnostics/ingestion/emdat/probe.json")
    assert probe_path.exists()
    saved = json.loads(probe_path.read_text(encoding="utf-8"))
    assert saved["ok"] is False
    assert saved["status"] == "error"
    assert saved["total_available"] is None
    assert "timeout" in (saved.get("error") or "").lower()


def test_probe_emdat_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ProbeResponse:
        status_code = 200

        def raise_for_status(self) -> None:  # noqa: D401
            return None

        def json(self) -> dict[str, object]:
            return {
                "data": {
                    "api_version": "2024-05",
                    "public_emdat": {"info": {"version": "v1", "timestamp": "2024-05-01"}},
                }
            }

    class _Session:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def post(self, url, json=None, headers=None, timeout=None):  # noqa: ANN001
            self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
            return _ProbeResponse()

    session = _Session()
    result = probe_emdat("token", "https://example.test/graphql", timeout_s=3, session=session)

    assert result["ok"] is True
    assert result["status"] == 200
    assert result["api_version"] == "2024-05"
    assert result["table_version"] == "v1"
    assert result["metadata_timestamp"] == "2024-05-01"
    assert result["requests"]["2xx"] == 1
    assert session.calls[0]["headers"]["Authorization"] == "token"
    assert session.calls[0]["json"]["query"]


def test_probe_emdat_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ProbeResponse:
        status_code = 401

        def raise_for_status(self) -> None:  # noqa: D401
            raise requests.HTTPError("unauthorized")

        def json(self) -> dict[str, object]:  # pragma: no cover - not reached
            return {}

    class _Session:
        def post(self, *args, **kwargs):  # noqa: ANN001
            return _ProbeResponse()

    result = probe_emdat("token", "https://example.test/graphql", timeout_s=3, session=_Session())

    assert result["ok"] is False
    assert result["status"] == 401
    assert "unauthorized" in (result.get("error") or "").lower()


def test_build_effective_params_iso_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(emdat_client.EMDAT_NETWORK_ENV, raising=False)
    cfg = {
        "include_hist": False,
        "default_from_year": 2020,
        "default_to_year": 2021,
        "iso": [],
    }

    params, meta = _build_effective_params(
        network_requested=False,
        api_key_present=False,
        cfg=cfg,
        source_mode="file",
        network_env=None,
        source_override=None,
    )

    assert "iso" not in params["filters"]
    assert meta["iso"] == []
    assert params["source_type"] == "file"
    assert params["network"] is False

    cfg_with_iso = {
        "include_hist": True,
        "default_from_year": 2019,
        "default_to_year": 2020,
        "iso": ["phl", " idn "],
    }

    params_with_iso, meta_with_iso = _build_effective_params(
        network_requested=True,
        api_key_present=True,
        cfg=cfg_with_iso,
        source_mode="api",
        network_env="1",
        source_override="api",
    )

    assert params_with_iso["filters"]["iso"] == ["IDN", "PHL"]
    assert meta_with_iso["iso"] == ["IDN", "PHL"]
    assert params_with_iso["filters"]["include_hist"] is True
    assert params_with_iso["iso_values"] == ["IDN", "PHL"]
    assert params_with_iso["source_override"] == "api"
