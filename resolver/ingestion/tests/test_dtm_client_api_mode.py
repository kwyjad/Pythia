from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import pytest

from resolver.ingestion import dtm_client
import resolver.ingestion.dtm_auth as dtm_auth


@pytest.fixture
def sample_api_records() -> List[Dict[str, Any]]:
    return [
        {
            "iso3": "KEN",
            "admin1": "",
            "month": date(2024, 1, 1),
            "value": 120.0,
            "cause": "conflict",
            "measure": "flow",
            "source_id": "dtm_api_admin0",
            "as_of": "2024-02-01",
        }
    ]


@pytest.fixture
def sample_api_summary() -> Dict[str, Any]:
    return {
        "row_counts": {"admin0": 1, "admin1": 0, "admin2": 0, "total": 1},
        "http_counts": {"2xx": 1, "4xx": 0, "5xx": 0, "timeout": 0, "error": 0},
    }


def _fake_fetch_factory(records: List[Dict[str, Any]], summary: Dict[str, Any]):
    calls: List[Dict[str, Any]] = []

    def _fake_fetch(cfg: Dict[str, Any], **kwargs: Any):
        calls.append(kwargs)
        http_counts = kwargs.get("http_counts")
        if http_counts is not None:
            http_counts["2xx"] = http_counts.get("2xx", 0) + 1
        return records, summary

    return _fake_fetch, calls


def test_build_rows_api_mode_top_level(monkeypatch, sample_api_records, sample_api_summary):
    fake_fetch, calls = _fake_fetch_factory(sample_api_records, sample_api_summary)
    monkeypatch.setattr(dtm_client, "_fetch_api_data", fake_fetch)

    def fail_read(*_: Any, **__: Any):
        raise AssertionError("_read_source should not be called in API mode")

    monkeypatch.setattr(dtm_client, "_read_source", fail_read)
    diagnostics: Dict[str, Any] = {}
    rows = dtm_client.build_rows(
        {"api": {}, "sources": []},
        window_start="2024-01-01",
        window_end="2024-02-01",
        diagnostics=diagnostics,
        http_counts={"2xx": 0, "4xx": 0, "5xx": 0, "timeout": 0, "error": 0},
    )

    assert len(rows) == 1
    assert len(calls) == 1
    assert diagnostics["mode"] == "api"
    assert "top-level-api" in diagnostics["trigger"]
    assert diagnostics["row_counts"]["total"] == 1
    assert diagnostics["http_counts"] == sample_api_summary["http_counts"]


def test_build_rows_api_mode_from_source_entry(
    monkeypatch, sample_api_records, sample_api_summary
):
    fake_fetch, calls = _fake_fetch_factory(sample_api_records, sample_api_summary)
    monkeypatch.setattr(dtm_client, "_fetch_api_data", fake_fetch)
    diagnostics: Dict[str, Any] = {}
    rows = dtm_client.build_rows(
        {"sources": [{"type": "api"}]},
        diagnostics=diagnostics,
        http_counts={"2xx": 0, "4xx": 0, "5xx": 0, "timeout": 0, "error": 0},
    )

    assert len(rows) == 1
    assert len(calls) == 1
    assert diagnostics["mode"] == "api"
    assert "source-type-api" in diagnostics["trigger"]


def test_build_rows_file_mode(monkeypatch):
    def unexpected_fetch(*_: Any, **__: Any) -> None:
        raise AssertionError("_fetch_api_data should not run in file mode")

    monkeypatch.setattr(dtm_client, "_fetch_api_data", unexpected_fetch)

    def fake_read(entry, cfg, **_):
        return dtm_client.SourceResult(
            source_name="file",
            records=[
                {
                    "iso3": "UGA",
                    "admin1": "Central",
                    "month": date(2024, 1, 1),
                    "value": 10.0,
                    "cause": "conflict",
                    "measure": "flow",
                    "source_id": "file",
                    "as_of": "2024-02-01",
                }
            ],
        )

    monkeypatch.setattr(dtm_client, "_read_source", fake_read)
    diagnostics: Dict[str, Any] = {}
    rows = dtm_client.build_rows(
        {
            "sources": [{"type": "file", "id_or_path": "dummy.csv"}],
            "admin_agg": "country",
        },
        diagnostics=diagnostics,
    )

    assert len(rows) == 1
    assert diagnostics["mode"] == "file"


def test_read_source_api_type_delegates():
    result = dtm_client._read_source(
        {"type": "api", "name": "api-source"},
        {},
        no_date_filter=False,
        window_start=None,
        window_end=None,
    )
    assert result.status == "delegated"
    assert result.rows == 0


def test_main_writes_api_diagnostics(
    monkeypatch, sample_api_records, sample_api_summary
):
    monkeypatch.setenv("DTM_API_KEY", "test-key")
    monkeypatch.setattr(dtm_auth, "check_api_key_configured", lambda: True)
    monkeypatch.setattr(dtm_client, "load_config", lambda: {"enabled": True, "api": {}, "sources": []})
    monkeypatch.setattr(dtm_client, "write_rows", lambda rows: None)
    monkeypatch.setattr(dtm_client, "ensure_header_only", lambda: None)
    monkeypatch.setattr(dtm_client, "diagnostics_start_run", lambda *args, **kwargs: object())
    monkeypatch.setattr(dtm_client, "diagnostics_finalize_run", lambda *args, **kwargs: {})
    monkeypatch.setattr(dtm_client, "diagnostics_append_jsonl", lambda *args, **kwargs: None)

    def fake_fetch(cfg, **kwargs):
        http_counts = kwargs.get("http_counts")
        if http_counts is not None:
            http_counts["2xx"] = http_counts.get("2xx", 0) + 1
        results = kwargs.get("results")
        if results is not None:
            results.append(dtm_client.SourceResult(source_name="dtm_api", status="ok"))
        return sample_api_records, sample_api_summary

    monkeypatch.setattr(dtm_client, "_fetch_api_data", fake_fetch)

    captured: Dict[str, Any] = {}

    def fake_write_json(path, obj):
        captured[str(path)] = obj

    monkeypatch.setattr(dtm_client, "write_json", fake_write_json)

    exit_code = dtm_client.main([])
    assert exit_code == 0

    payload = captured[str(dtm_client.RUN_DETAILS_PATH)]
    assert payload["mode"] == "api"
    assert "top-level-api" in (payload.get("trigger") or "")
    assert payload["row_counts"]["total"] == 1
    assert payload["http_counts"] == sample_api_summary["http_counts"]
