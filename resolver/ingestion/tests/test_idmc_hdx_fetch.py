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
            "source_tag": "idmc_idu",
        }

    def fake_http_get(*args: Any, **kwargs: Any) -> None:
        raise HttpRequestError("boom", {"status": 502, "attempts": 1, "retries": 0})

    monkeypatch.setenv("IDMC_ALLOW_HDX_FALLBACK", "1")
    monkeypatch.setattr(client, "_hdx_fetch_once", fake_hdx_fetch)
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


def test_idmc_hdx_selects_data_tab(monkeypatch: pytest.MonkeyPatch) -> None:
    resource_id = "sheet"
    gid_value = "987654321"

    def fake_get(url: str, *args: Any, **kwargs: Any) -> _FakeResponse:
        if "package_show" in url:
            payload = {
                "success": True,
                "result": {
                    "resources": [
                        {
                            "id": resource_id,
                            "name": "idus_view_flat (Google)",
                            "format": "CSV",
                            "url": "https://docs.google.com/spreadsheets/d/abc/pub?gid=0&single=true&output=csv",
                        }
                    ]
                },
            }
            return _FakeResponse(status_code=200, json_body=payload)
        assert "gid=" in url
        assert f"gid={gid_value}" in url
        body = b"iso3,figure,displacement_end_date\nAFG,3,2024-01-31\n"
        return _FakeResponse(status_code=200, content=body, headers={"Content-Length": str(len(body))})

    monkeypatch.setenv("IDMC_HDX_RESOURCE_ID", resource_id)
    monkeypatch.setenv("IDMC_HDX_GID", gid_value)
    monkeypatch.setattr(client.requests, "get", fake_get)

    frame, diag = client._hdx_fetch_latest_csv()

    assert not frame.empty
    assert diag.get("resource_gid_used") == gid_value
    assert "figure" in frame.columns
    assert "displacement_end_date" in frame.columns


def test_idmc_helix_fallback_used_on_hdx_empty(
    monkeypatch: pytest.MonkeyPatch, cfg: config.IdmcConfig
) -> None:
    def fake_http_get(*args: Any, **kwargs: Any) -> None:
        raise HttpRequestError("boom", {"status": 504, "attempts": 1, "retries": 0})

    def fake_hdx_once() -> tuple[None, Dict[str, Any]]:
        return None, {"zero_rows_reason": "hdx_empty_or_bad_header"}

    def fake_helix_fetch(**kwargs: Any) -> tuple[pd.DataFrame, Dict[str, Any]]:
        frame = pd.DataFrame(
            {
                "ISO3": ["AFG"],
                "New displacements": [7],
                "displacement_end_date": ["2024-01-31"],
            }
        )
        diag = {
            "rows": 1,
            "columns": list(frame.columns),
            "status_code": 200,
            "bytes": 128,
            "source": "helix",
            "source_tag": "idmc_gidd",
            "request_url": "https://helix.example/export?start_year=2023",
        }
        return frame, diag

    monkeypatch.setenv("IDMC_ALLOW_HDX_FALLBACK", "1")
    monkeypatch.setenv("IDMC_USE_HELIX_IF_IDU_UNREACHABLE", "1")
    monkeypatch.setenv("IDMC_HELIX_CLIENT_ID", "dummy")
    monkeypatch.setattr(client, "http_get", fake_http_get)
    monkeypatch.setattr(client, "_hdx_fetch_once", fake_hdx_once)
    monkeypatch.setattr(client, "_helix_fetch_csv", fake_helix_fetch)

    frames, diagnostics = client.fetch(
        cfg,
        network_mode="live",
        window_start=date(2024, 1, 1),
        window_end=date(2024, 1, 31),
        allow_hdx_fallback=True,
    )

    fallback = diagnostics.get("fallback") or {}
    assert fallback.get("type") == "helix"
    assert fallback.get("rows") == 1
    assert fallback.get("source_tag") == "idmc_gidd"
    assert "request_url" in fallback

    monthly = frames.get("monthly_flow")
    assert monthly is not None
    assert "idmc_source" in monthly.columns
    assert set(monthly["idmc_source"].unique()) == {"idmc_gidd"}


def test_idmc_value_alias_includes_figure() -> None:
    cfg_obj = config.load()
    aliases = cfg_obj.field_aliases.value_flow
    assert "figure" in aliases


def test_summary_zero_reason_codes(monkeypatch: pytest.MonkeyPatch, cfg: config.IdmcConfig) -> None:
    def fake_http_get(*args: Any, **kwargs: Any) -> None:
        raise HttpRequestError("boom", {"status": 502, "attempts": 1, "retries": 0})

    def fake_hdx_once() -> tuple[None, Dict[str, Any]]:
        return None, {"zero_rows_reason": "hdx_empty_or_bad_header"}

    def helix_success(**kwargs: Any) -> tuple[pd.DataFrame, Dict[str, Any]]:
        frame = pd.DataFrame(
            {
                "ISO3": ["PAK"],
                "New displacements": [11],
                "displacement_end_date": ["2024-02-29"],
            }
        )
        return frame, {
            "rows": 1,
            "columns": list(frame.columns),
            "status_code": 200,
            "bytes": 256,
            "source": "helix",
            "source_tag": "idmc_gidd",
            "request_url": "https://helix.example/export?start_year=2023",
        }

    def helix_fail(**kwargs: Any) -> tuple[pd.DataFrame, Dict[str, Any]]:
        return pd.DataFrame(), {
            "rows": 0,
            "columns": [],
            "zero_rows_reason": "helix_http_error",
            "source": "helix",
            "source_tag": "idmc_gidd",
            "request_url": "https://helix.example/export?start_year=2023",
        }

    monkeypatch.setenv("IDMC_ALLOW_HDX_FALLBACK", "1")
    monkeypatch.setenv("IDMC_USE_HELIX_IF_IDU_UNREACHABLE", "1")
    monkeypatch.setenv("IDMC_HELIX_CLIENT_ID", "dummy")
    monkeypatch.setattr(client, "http_get", fake_http_get)
    monkeypatch.setattr(client, "_hdx_fetch_once", fake_hdx_once)
    monkeypatch.setattr(client, "_helix_fetch_csv", helix_success)

    frames, diagnostics = client.fetch(
        cfg,
        network_mode="live",
        window_start=date(2024, 2, 1),
        window_end=date(2024, 2, 29),
        allow_hdx_fallback=True,
    )

    filters_block = diagnostics.get("filters") or {}
    assert filters_block.get("zero_rows_reason") in (None, "")
    fallback = diagnostics.get("fallback") or {}
    assert fallback.get("type") == "helix"
    assert fallback.get("rows") == 1

    monkeypatch.setattr(client, "_helix_fetch_csv", helix_fail)

    frames_empty, diagnostics_empty = client.fetch(
        cfg,
        network_mode="live",
        window_start=date(2024, 3, 1),
        window_end=date(2024, 3, 31),
        allow_hdx_fallback=True,
    )

    assert frames_empty.get("monthly_flow") is not None
    zero_reason = (diagnostics_empty.get("filters") or {}).get("zero_rows_reason")
    assert zero_reason == "helix_http_error"
    fallback_empty = diagnostics_empty.get("fallback") or {}
    assert fallback_empty.get("zero_rows_reason") == "helix_http_error"
