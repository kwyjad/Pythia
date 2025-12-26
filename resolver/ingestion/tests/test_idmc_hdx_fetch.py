# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from datetime import date
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from resolver.ingestion.idmc import client, config
from resolver.ingestion.idmc.export import (
    FLOW_EXPORT_COLUMNS,
    FLOW_METRIC,
    FLOW_SERIES_SEMANTICS,
)
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


def test_idmc_hdx_selects_large_resource(monkeypatch: pytest.MonkeyPatch) -> None:
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
        rows = ["iso3,figure"] + ["AFG,10"] * 12000
        body = ("\n".join(rows) + "\n").encode("utf-8")
        return _FakeResponse(status_code=200, content=body, headers={"Content-Length": str(len(body))})

    monkeypatch.setenv("IDMC_HDX_RESOURCE_ID", good_id)
    monkeypatch.setattr(client.requests, "get", fake_get)

    frame, diag = client._hdx_fetch_latest_csv()

    assert not frame.empty
    assert diag.get("resource_id") == good_id
    assert diag.get("resource_bytes") and int(diag["resource_bytes"]) >= 50_000
    assert "iso3" in frame.columns
    assert "figure" in frame.columns


def test_idmc_hdx_single_download_reused(monkeypatch: pytest.MonkeyPatch, cfg: config.IdmcConfig) -> None:
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
            "resource_bytes": 60_000,
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
    assert set(FLOW_EXPORT_COLUMNS).issubset(monthly.columns)
    assert int(monthly["value"].sum()) == 300
    assert set(monthly["source"].unique()).issubset({"idmc_idu", "idmc_hdx"})
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
        rows = ["iso3,figure,displacement_end_date"] + ["AFG,3,2024-01-31"] * 3000
        body = ("\n".join(rows) + "\n").encode("utf-8")
        return _FakeResponse(status_code=200, content=body, headers={"Content-Length": str(len(body))})

    monkeypatch.setenv("IDMC_HDX_RESOURCE_ID", resource_id)
    monkeypatch.setenv("IDMC_HDX_GID", gid_value)
    monkeypatch.setattr(client.requests, "get", fake_get)

    frame, diag = client._hdx_fetch_latest_csv()

    assert not frame.empty
    assert diag.get("resource_gid_used") == gid_value
    assert "figure" in frame.columns
    assert "displacement_end_date" in frame.columns


def test_idmc_helix_fallback(
    monkeypatch: pytest.MonkeyPatch, cfg: config.IdmcConfig
) -> None:
    def fake_http_get(*args: Any, **kwargs: Any) -> None:
        raise HttpRequestError("boom", {"status": 504, "attempts": 1, "retries": 0})

    def fake_hdx_once() -> tuple[None, Dict[str, Any]]:
        return None, {"zero_rows_reason": "hdx_empty_or_bad_header"}

    def fake_helix_chain(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, Dict[str, Any]]:
        frame = pd.DataFrame(
            {
                "iso3": ["AFG"],
                "as_of_date": [pd.Timestamp("2024-01-31")],
                "metric": FLOW_METRIC,
                "value": [7],
                "series_semantics": FLOW_SERIES_SEMANTICS,
                "source": "idmc_gidd",
                "ym": ["2024-01"],
                "record_id": ["AFG-new_displacements-2024-01-7-idmc_gidd"],
                client.HDX_PREAGG_COLUMN: False,
            }
        )
        diag = {
            "status": 200,
            "rows": 1,
            "raw_rows": 1,
            "helix_endpoint": "idus_last180",
        }
        return frame, diag

    monkeypatch.setenv("IDMC_ALLOW_HDX_FALLBACK", "1")
    monkeypatch.setenv("IDMC_HELIX_CLIENT_ID", "dummy")
    monkeypatch.setattr(client, "http_get", fake_http_get)
    monkeypatch.setattr(client, "_hdx_fetch_once", fake_hdx_once)
    monkeypatch.setattr(client, "_fetch_helix_chain", fake_helix_chain)

    frames, diagnostics = client.fetch(
        cfg,
        network_mode="live",
        window_start=date(2024, 1, 1),
        window_end=date(2024, 1, 31),
        allow_hdx_fallback=True,
    )

    assert diagnostics.get("helix_endpoint") == "idus_last180"
    helix_block = diagnostics.get("helix") or {}
    assert helix_block.get("helix_endpoint") == "idus_last180"
    assert helix_block.get("raw_rows") == 1

    monthly = frames.get("monthly_flow")
    assert monthly is not None
    assert set(FLOW_EXPORT_COLUMNS).issubset(monthly.columns)
    assert set(monthly["source"].unique()) == {"idmc_gidd"}


def test_idmc_value_alias_includes_figure() -> None:
    cfg_obj = config.load()
    aliases = cfg_obj.field_aliases.value_flow
    assert "figure" in aliases


def test_hdx_displacements_parses_small_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    csv_rows = [
        "CountryISO3,displacement_date,figure",
        "AFG,2024-01-05,5",
        "AFG,2024-01-20,7",
        "AFG,2024-02-10,9",
    ]
    payload = ("\n".join(csv_rows) + "\n").encode("utf-8")

    def fake_pick(*_args: Any, **_kwargs: Any) -> tuple[str, Dict[str, Any]]:
        return "https://example.invalid/tiny.csv", {}

    def fake_download(*_args: Any, **_kwargs: Any) -> tuple[bytes, Dict[str, Any]]:
        return payload, {"status_code": 200, "bytes": len(payload), "content_length": len(payload)}

    monkeypatch.setattr(client, "_hdx_pick_displacement_csv", fake_pick)
    monkeypatch.setattr(client, "_hdx_download_resource", fake_download)

    frame, diagnostics = client._fetch_hdx_displacements(
        package_id="pkg",
        base_url="https://example.invalid",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 2, 29),
        iso3_list=["AFG"],
    )

    assert diagnostics.get("rows") == 2
    assert diagnostics.get("used") is True
    assert diagnostics.get("resource_bytes") == len(payload)
    expected_columns = set(FLOW_EXPORT_COLUMNS) | {"__hdx_preaggregated__"}
    assert set(frame.columns) == expected_columns
    assert "figure" not in frame.columns
    values = dict(zip(frame["as_of_date"].astype(str), frame["value"].astype(int)))
    assert values == {"2024-01-31": 12, "2024-02-29": 9}
    assert set(frame["source"].unique()) == {"idmc_hdx"}


def test_helix_last180_fallback_respects_env_disable(
    monkeypatch: pytest.MonkeyPatch, cfg: config.IdmcConfig
) -> None:
    monkeypatch.setenv("IDMC_ALLOW_HDX_FALLBACK", "0")
    monkeypatch.setenv("IDMC_HELIX_CLIENT_ID", "dummy-client")

    monkeypatch.setattr(
        client,
        "_fetch_helix_idus_last180",
        lambda *_args, **_kwargs: (pd.DataFrame(), {"status": 502}),
    )

    called = False

    def forbidden_fetch(*_args: Any, **_kwargs: Any) -> tuple[pd.DataFrame, Dict[str, Any]]:
        nonlocal called
        called = True
        return pd.DataFrame(), {}

    monkeypatch.setattr(client, "_fetch_hdx_displacements", forbidden_fetch)
    monkeypatch.setattr(
        client,
        "fetch_idu_json",
        lambda *args, **kwargs: (pd.DataFrame(), {"mode": "cache", "http": {}}),
    )

    frames, diagnostics = client.fetch(
        cfg,
        network_mode="live",
        window_start=date(2024, 1, 1),
        window_end=date(2024, 1, 31),
        allow_hdx_fallback=False,
    )

    assert called is False
    assert frames.get("monthly_flow") is not None
    fallback_summary = diagnostics.get("fallback") or {}
    assert not fallback_summary


def test_summary_zero_reason_codes(monkeypatch: pytest.MonkeyPatch, cfg: config.IdmcConfig) -> None:
    def fake_http_get(*args: Any, **kwargs: Any) -> None:
        raise HttpRequestError("boom", {"status": 502, "attempts": 1, "retries": 0})

    def fake_hdx_once() -> tuple[None, Dict[str, Any]]:
        return None, {"zero_rows_reason": "hdx_empty_or_bad_header"}

    def helix_success(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, Dict[str, Any]]:
        frame = pd.DataFrame(
            {
                "iso3": ["PAK"],
                "as_of_date": [pd.Timestamp("2024-02-29")],
                "metric": FLOW_METRIC,
                "value": [11],
                "series_semantics": FLOW_SERIES_SEMANTICS,
                "source": "idmc_gidd",
                "ym": ["2024-02"],
                "record_id": ["PAK-new_displacements-2024-02-11-idmc_gidd"],
                client.HDX_PREAGG_COLUMN: False,
            }
        )
        return frame, {
            "status": 200,
            "rows": 1,
            "raw_rows": 1,
            "helix_endpoint": "idus_last180",
        }

    def helix_fail(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, Dict[str, Any]]:
        return pd.DataFrame(), {
            "status": 502,
            "rows": 0,
            "raw_rows": 0,
            "helix_endpoint": "idus_last180",
            "zero_rows_reason": "helix_http_error",
        }

    monkeypatch.setenv("IDMC_ALLOW_HDX_FALLBACK", "1")
    monkeypatch.setenv("IDMC_HELIX_CLIENT_ID", "dummy")
    monkeypatch.setattr(client, "http_get", fake_http_get)
    monkeypatch.setattr(client, "_hdx_fetch_once", fake_hdx_once)
    monkeypatch.setattr(client, "_fetch_helix_chain", helix_success)

    frames, diagnostics = client.fetch(
        cfg,
        network_mode="live",
        window_start=date(2024, 2, 1),
        window_end=date(2024, 2, 29),
        allow_hdx_fallback=True,
    )

    filters_block = diagnostics.get("filters") or {}
    assert filters_block.get("zero_rows_reason") in (None, "")
    helix_block = diagnostics.get("helix") or {}
    assert helix_block.get("helix_endpoint") == "idus_last180"
    assert helix_block.get("raw_rows") == 1

    monkeypatch.setattr(client, "_fetch_helix_chain", helix_fail)

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
    helix_block_empty = diagnostics_empty.get("helix") or {}
    assert helix_block_empty.get("zero_rows_reason") == "helix_http_error"
