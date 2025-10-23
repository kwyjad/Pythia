import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pytest

from resolver.ingestion import dtm_client


def _setup_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    out_path = tmp_path / "data" / "staging" / "dtm_displacement.csv"
    diagnostics_dir = tmp_path / "diagnostics" / "ingestion"

    monkeypatch.setattr(dtm_client, "OUT_PATH", out_path)
    monkeypatch.setattr(dtm_client, "OUT_DIR", out_path.parent)
    monkeypatch.setattr(dtm_client, "OUTPUT_PATH", out_path)
    monkeypatch.setattr(
        dtm_client,
        "META_PATH",
        out_path.with_suffix(out_path.suffix + ".meta.json"),
    )
    monkeypatch.setattr(dtm_client, "DEFAULT_OUTPUT", out_path)
    monkeypatch.setattr(dtm_client, "DIAGNOSTICS_DIR", diagnostics_dir)
    monkeypatch.setattr(
        dtm_client,
        "CONNECTORS_REPORT",
        diagnostics_dir / "connectors_report.jsonl",
    )
    monkeypatch.setattr(
        dtm_client,
        "CONFIG_ISSUES_PATH",
        diagnostics_dir / "dtm_config_issues.json",
    )
    monkeypatch.setattr(
        dtm_client,
        "RESOLVED_SOURCES_PATH",
        diagnostics_dir / "dtm_sources_resolved.json",
    )
    monkeypatch.setattr(
        dtm_client,
        "RUN_DETAILS_PATH",
        diagnostics_dir / "dtm_run.json",
    )
    return out_path


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: Dict[str, Any],
        *,
        headers: Optional[Dict[str, str]] = None,
        url: str,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        body = json.dumps(payload).encode("utf-8")
        self.content = body
        self.text = body.decode("utf-8")
        self.request = SimpleNamespace(url=url)

    def json(self) -> Dict[str, Any]:
        return self._payload


class FakeClient:
    def __init__(self, responses: Iterable[Dict[str, Any]]) -> None:
        self._responses: Iterator[Dict[str, Any]] = iter(responses)
        self.calls: List[Dict[str, Any]] = []

    def get(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> FakeResponse:
        params = params or {}
        headers = headers or {}
        self.calls.append({"url": url, "params": dict(params), "headers": dict(headers)})
        try:
            payload = next(self._responses)
        except StopIteration:  # pragma: no cover - defensive
            payload = {"status_code": 200, "payload": {"results": []}}
        status_code = int(payload.get("status_code", 200))
        data = payload.get("payload", {})
        resp_headers = payload.get("headers", {})
        page = params.get("page")
        req_url = f"{url}?page={page}" if page is not None else url
        return FakeResponse(status_code, data, headers=resp_headers, url=req_url)

    def close(self) -> None:  # pragma: no cover - no-op cleanup
        return None


def test_parse_args_defaults_to_records_mode() -> None:
    args = dtm_client.parse_args([])
    assert args.mode == "records"


def test_backfill_rejects_header_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESOLVER_BACKFILL", "1")
    with pytest.raises(SystemExit):
        dtm_client.main(["--mode", "header-only"])


def test_http_retry_and_pagination(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    out_path = _setup_paths(monkeypatch, tmp_path)

    responses = [
        {"status_code": 401, "payload": {"error": "bad key"}},
        {
            "status_code": 200,
            "payload": {
                "results": [
                    {"country": "KEN", "date": "2024-01-01", "value": 5},
                ]
            },
            "headers": {"X-RateLimit-Remaining": "9"},
        },
        {
            "status_code": 200,
            "payload": {
                "results": [
                    {"country": "KEN", "date": "2024-02-01", "value": 7},
                ]
            },
            "headers": {"X-RateLimit-Remaining": "8"},
        },
        {"status_code": 200, "payload": {"results": []}},
    ]

    created_clients: List[FakeClient] = []

    def fake_client_factory(*args, **kwargs):
        client = FakeClient(responses)
        created_clients.append(client)
        return client

    monkeypatch.setattr(dtm_client.httpx, "Client", fake_client_factory)

    monkeypatch.setenv("DTM_API_PRIMARY_KEY", "primary")
    monkeypatch.setenv("DTM_API_SECONDARY_KEY", "secondary")
    monkeypatch.delenv("RESOLVER_SKIP_DTM", raising=False)
    monkeypatch.delenv("DTM_STRICT", raising=False)
    monkeypatch.delenv("DTM_STRICT_EMPTY", raising=False)

    config = {
        "enabled": True,
        "base_url": "https://dtm.example.test/api",
        "sources": [
            {
                "type": "api",
                "id_or_path": "displacement",
                "date_column": "date",
                "country_column": "country",
                "value_column": "value",
                "measure": "flow",
                "page_size": 1,
            }
        ],
    }
    monkeypatch.setattr(dtm_client, "load_config", lambda: config)

    exit_code = dtm_client.main([])
    assert exit_code == 0

    client = created_clients[0]
    assert len(client.calls) >= 3
    first_headers = client.calls[0]["headers"]
    second_headers = client.calls[1]["headers"]
    assert first_headers.get("X-Api-Key") == "primary"
    assert second_headers.get("X-Api-Key") == "secondary"

    meta_payload = json.loads(dtm_client.META_PATH.read_text(encoding="utf-8"))
    assert meta_payload["row_count"] == 2
    assert meta_payload.get("auth_used") == "secondary"
    assert meta_payload.get("pages") == len(client.calls)
    histogram = meta_payload.get("status_histogram", {})
    assert histogram.get("401") == 1
    assert histogram.get("200") == len(client.calls) - 1
    assert meta_payload.get("request_url") or meta_payload.get("request_urls")

    csv_text = out_path.read_text(encoding="utf-8")
    assert csv_text.count("\n") >= 2  # header + at least two rows
