from datetime import date
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from resolver.ingestion.idmc import client, config
from resolver.ingestion.idmc.http import HttpRequestError


class _FakeResponse:
    def __init__(self, *, status_code: int, json_body: Dict[str, Any] | None = None, content: bytes = b"", headers: Dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self._json = json_body
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"status={self.status_code}")

    def json(self) -> Dict[str, Any]:
        return self._json or {}


@pytest.fixture
def cfg(tmp_path: Path) -> config.IdmcConfig:
    cfg_obj = config.load()
    cfg_obj.cache.dir = str(tmp_path / "cache")
    cfg_obj.cache.ttl_seconds = 0
    cfg_obj.api.countries = ["AFG", "PAK"]
    return cfg_obj


def test_idmc_hdx_prefers_configured_resource(monkeypatch: pytest.MonkeyPatch) -> None:
    good_id = "good-resource"
    small_id = "small-resource"

    def fake_get(url: str, *args: Any, **kwargs: Any) -> _FakeResponse:
        if "package_show" in url:
            payload = {
                "success": True,
                "result": {
                    "resources": [
                        {
                            "id": small_id,
                            "name": "idus_view_flat old",
                            "format": "CSV",
                            "url": "https://example.invalid/small.csv",
                            "last_modified": "2024-01-01T00:00:00",
                        },
                        {
                            "id": good_id,
                            "name": "idus_view_flat",
                            "format": "CSV",
                            "url": "https://example.invalid/good.csv",
                            "last_modified": "2024-02-01T00:00:00",
                        },
                    ]
                },
            }
            return _FakeResponse(status_code=200, json_body=payload)
        if url.endswith("small.csv"):
            body = b"iso3,value\nAFG,1\n"
            return _FakeResponse(status_code=200, content=body, headers={"Content-Length": str(len(body))})
        assert url.endswith("good.csv")
        rows = ["iso3,figure"] + ["AFG,10"] * 2500
        body = ("\n".join(rows) + "\n").encode("utf-8")
        return _FakeResponse(status_code=200, content=body, headers={"Content-Length": str(len(body))})

    monkeypatch.setenv("IDMC_HDX_RESOURCE_ID", good_id)
    monkeypatch.setattr(client.requests, "get", fake_get)

    frame, diag = client._hdx_fetch_latest_csv()

    assert not frame.empty
    assert diag.get("resource_id") == good_id
    assert diag.get("resource_bytes") and int(diag["resource_bytes"]) > 10_000
    assert "iso3" in frame.columns
    assert "figure" in frame.columns


def test_idmc_hdx_single_fetch_reused_across_chunks(monkeypatch: pytest.MonkeyPatch, cfg: config.IdmcConfig) -> None:
    call_count = 0

    fallback_frame = pd.DataFrame(
        {
            "iso3": ["AFG", "PAK"],
            "figure": [100, 200],
            "displacement_end_date": ["2024-01-31", "2024-02-29"],
        }
    )

    def fake_hdx_fetch() -> tuple[pd.DataFrame, Dict[str, Any]]:
        nonlocal call_count
        call_count += 1
        return fallback_frame, {
            "dataset": "preliminary-internal-displacement-updates",
            "resource_id": "cached",
            "resource_bytes": 15_000,
            "resource_url": "https://example.invalid/good.csv",
        }

    def fake_http_get(*args: Any, **kwargs: Any) -> None:
        raise HttpRequestError("boom", {"status": 502, "attempts": 1, "retries": 0})

    monkeypatch.setenv("IDMC_ALLOW_HDX_FALLBACK", "1")
    monkeypatch.setattr(client, "_hdx_fetch_latest_csv", fake_hdx_fetch)
    monkeypatch.setattr(client, "http_get", fake_http_get)

    frames, diagnostics = client.fetch(
        cfg,
        network_mode="live",
        window_start=date(2024, 1, 1),
        window_end=date(2024, 2, 29),
        chunk_by_month=True,
        allow_hdx_fallback=True,
    )

    assert call_count == 1
    monthly = frames.get("monthly_flow")
    assert monthly is not None
    assert isinstance(monthly, pd.DataFrame)
    assert int(monthly["figure"].sum()) == 300
    assert diagnostics.get("fallback_used") is True
