# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import csv
import json
from datetime import date
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
    assert result["requests"]["total"] == 1
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
    assert saved["requests"]["total"] == 1
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
    assert result["requests"]["total"] == 1
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


def test_emdat_main_records_diagnostics(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    staging_dir = Path("resolver/staging")
    staging_dir.mkdir(parents=True, exist_ok=True)
    csv_path = staging_dir / "emdat_pa.csv"

    monkeypatch.setenv("EMDAT_NETWORK", "1")
    monkeypatch.setenv("EMDAT_API_KEY", "token")
    monkeypatch.setenv("EMDAT_SOURCE", "api")
    monkeypatch.delenv("RESOLVER_SKIP_EMDAT", raising=False)

    monkeypatch.setattr(emdat_client, "OUT_DIR", staging_dir)
    monkeypatch.setattr(emdat_client, "OUT_PATH", csv_path)

    sample_payload = {
        "data": {
            "api_version": "2024-05",
            "public_emdat": {
                "info": {"timestamp": "2024-02-05T00:00:00Z"},
                "data": [
                    {
                        "disno": "2024-0001-PHL",
                        "classif_key": "nat-hyd-flo-riv",
                        "type": "Flood",
                        "subtype": "",
                        "iso": "PHL",
                        "country": "Philippines",
                        "start_year": 2024,
                        "start_month": 1,
                        "start_day": 1,
                        "end_year": 2024,
                        "end_month": 1,
                        "end_day": 2,
                        "total_affected": 125,
                        "entry_date": "2024-02-01",
                        "last_update": "2024-02-05",
                    }
                ],
            },
        }
    }

    def _fake_post_with_status(self, payload):  # noqa: ANN001
        self._ensure_online()
        return 200, sample_payload, 15.0

    monkeypatch.setattr(emdat_client.EmdatClient, "_post_with_status", _fake_post_with_status)

    def _fake_probe(api_key, base_url, timeout_s, session):  # noqa: ANN001
        return {
            "ok": True,
            "status": 200,
            "latency_ms": 20,
            "elapsed_ms": 20,
            "api_version": "2024-05",
            "table_version": "dataset-v1",
            "metadata_timestamp": "2024-02-05T00:00:00Z",
            "error": None,
            "requests": {"total": 1, "2xx": 1, "4xx": 0, "5xx": 0},
        }

    monkeypatch.setattr(emdat_client, "probe_emdat", _fake_probe)

    captured: dict[str, dict[str, int] | str | None] = {}
    original_finalize = emdat_client.diagnostics_emitter.finalize_run

    def _capture_finalize(context, status, **kwargs):  # noqa: ANN001
        captured["http"] = kwargs.get("http")
        captured["counts"] = kwargs.get("counts")
        return original_finalize(context, status, **kwargs)

    monkeypatch.setattr(emdat_client.diagnostics_emitter, "finalize_run", _capture_finalize)

    assert emdat_client.main([]) is True

    with csv_path.open(newline="", encoding="utf-8") as handle:
        data_rows = list(csv.DictReader(handle))
    assert len(data_rows) == 1
    assert data_rows[0]["metric"] == "total_affected"

    http_stats = captured.get("http") or {}
    count_stats = captured.get("counts") or {}
    assert http_stats.get("2xx", 0) >= 1
    assert http_stats.get("total", 0) >= http_stats.get("2xx", 0)
    assert count_stats.get("fetched") == 1
    assert count_stats.get("normalized") == 1
    assert count_stats.get("written") == 1


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


def test_default_year_bounds_uses_current_year(monkeypatch: pytest.MonkeyPatch) -> None:
    today = date(2024, 6, 1)
    monkeypatch.setattr(emdat_client, "date", type("_D", (), {"today": staticmethod(lambda: today)})())

    cfg = {"default_from_year": 2022, "default_to_year": None}
    from_year, to_year = emdat_client._default_year_bounds(cfg)
    assert from_year == 2022
    assert to_year == 2024

    cfg_missing_to = {"default_from_year": 2022}
    from_year_missing, to_year_missing = emdat_client._default_year_bounds(cfg_missing_to)
    assert from_year_missing == 2022
    assert to_year_missing == 2024
