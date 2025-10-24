"""High-level dtm_client tests covering CLI behaviours."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pytest

from resolver.ingestion import dtm_client


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DTM_API_KEY", raising=False)
    monkeypatch.delenv("DTM_API_PRIMARY_KEY", raising=False)
    monkeypatch.delenv("DTM_API_SECONDARY_KEY", raising=False)
    monkeypatch.delenv("RESOLVER_START_ISO", raising=False)
    monkeypatch.delenv("RESOLVER_END_ISO", raising=False)


@pytest.fixture()
def patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Dict[str, Path]:
    diagnostics_dir = tmp_path / "diagnostics" / "ingestion"
    out_path = tmp_path / "outputs" / "dtm.csv"
    mappings = {
        "OUT_PATH": out_path,
        "OUT_DIR": out_path.parent,
        "OUTPUT_PATH": out_path,
        "DEFAULT_OUTPUT": out_path,
        "META_PATH": out_path.with_suffix(out_path.suffix + ".meta.json"),
        "DIAGNOSTICS_DIR": diagnostics_dir,
        "RUN_DETAILS_PATH": diagnostics_dir / "dtm_run.json",
        "API_REQUEST_PATH": diagnostics_dir / "dtm_api_request.json",
        "API_SAMPLE_PATH": diagnostics_dir / "dtm_api_sample.json",
    }
    for name, value in mappings.items():
        monkeypatch.setattr(dtm_client, name, value)
    return mappings


def test_build_rows_requires_api_config() -> None:
    with pytest.raises(ValueError) as excinfo:
        dtm_client.build_rows({}, no_date_filter=False, window_start=None, window_end=None)
    assert "DTM is API-only" in str(excinfo.value)


def test_main_writes_outputs_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch, patch_paths: Dict[str, Path], tmp_path: Path
) -> None:
    monkeypatch.setenv("DTM_API_KEY", "primary")

    class DummyClient:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.rate_limit_delay = 0
            self.timeout = 0

        def get_countries(self, *_: Any, **__: Any) -> pd.DataFrame:
            return pd.DataFrame([{"CountryName": "Kenya", "ISO3": "KEN"}])

        def get_idp_admin0(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "CountryName": ["Kenya"],
                    "ReportingDate": ["2024-05-15"],
                    "TotalIDPs": [120],
                }
            )

        def get_idp_admin1(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

        def get_idp_admin2(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(dtm_client, "DTMApiClient", DummyClient)
    monkeypatch.setattr(dtm_client, "resolve_accept_names", lambda *_: ["Kenya"])
    monkeypatch.setattr(dtm_client, "load_config", lambda: {"enabled": True, "api": {"admin_levels": ["admin0"]}})
    monkeypatch.setattr(dtm_client, "diagnostics_start_run", lambda *_, **__: object())
    monkeypatch.setattr(dtm_client, "diagnostics_finalize_run", lambda *_, **__: {})
    monkeypatch.setattr(dtm_client, "diagnostics_append_jsonl", lambda *_, **__: None)
    monkeypatch.setattr(dtm_client, "resolve_ingestion_window", lambda: (None, None))

    rc = dtm_client.main([])
    assert rc == 0
    csv_lines = patch_paths["OUT_PATH"].read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) == 3
    run_payload = json.loads(patch_paths["RUN_DETAILS_PATH"].read_text(encoding="utf-8"))
    assert run_payload["rows"]["written"] == 2
    assert run_payload["rows"]["fetched"] == 1
    request = json.loads(patch_paths["API_REQUEST_PATH"].read_text(encoding="utf-8"))
    assert request["admin_levels"] == ["admin0"]


def test_main_strict_empty_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, patch_paths: Dict[str, Path], tmp_path: Path
) -> None:
    monkeypatch.setenv("DTM_API_KEY", "primary")

    class EmptyClient:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.rate_limit_delay = 0
            self.timeout = 0

        def get_countries(self, *_: Any, **__: Any) -> pd.DataFrame:
            return pd.DataFrame([{"CountryName": "Kenya", "ISO3": "KEN"}])

        def get_idp_admin0(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame(columns=["CountryName", "ReportingDate", "TotalIDPs"])

        def get_idp_admin1(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

        def get_idp_admin2(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(dtm_client, "DTMApiClient", EmptyClient)
    monkeypatch.setattr(dtm_client, "resolve_accept_names", lambda *_: ["Kenya"])
    monkeypatch.setattr(dtm_client, "load_config", lambda: {"enabled": True, "api": {"admin_levels": ["admin0"]}})
    monkeypatch.setattr(dtm_client, "diagnostics_start_run", lambda *_, **__: object())
    monkeypatch.setattr(dtm_client, "diagnostics_finalize_run", lambda *_, **__: {})
    monkeypatch.setattr(dtm_client, "diagnostics_append_jsonl", lambda *_, **__: None)
    monkeypatch.setattr(dtm_client, "resolve_ingestion_window", lambda: (None, None))

    rc = dtm_client.main(["--strict-empty"])
    assert rc == 3


def test_main_retries_with_secondary_key(
    monkeypatch: pytest.MonkeyPatch, patch_paths: Dict[str, Path], tmp_path: Path
) -> None:
    monkeypatch.setenv("DTM_API_PRIMARY_KEY", "primary")
    monkeypatch.setenv("DTM_API_SECONDARY_KEY", "secondary")

    attempt = {"count": 0}

    class FailingClient:
        def __init__(self, *_: Any, subscription_key: Optional[str] = None) -> None:
            self.subscription_key = subscription_key
            self.rate_limit_delay = 0
            self.timeout = 0

        def get_countries(self, *_: Any, **__: Any) -> pd.DataFrame:
            return pd.DataFrame([{"CountryName": "Kenya", "ISO3": "KEN"}])

        def get_idp_admin0(self, **_: Any) -> pd.DataFrame:
            attempt["count"] += 1
            if self.subscription_key == "primary":
                raise dtm_client.DTMUnauthorizedError(401, "unauthorized")
            return pd.DataFrame(
                {
                    "CountryName": ["Kenya"],
                    "ReportingDate": ["2024-06-01"],
                    "TotalIDPs": [25],
                }
            )

        def get_idp_admin1(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

        def get_idp_admin2(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(dtm_client, "DTMApiClient", FailingClient)
    monkeypatch.setattr(dtm_client, "resolve_accept_names", lambda *_: ["Kenya"])
    monkeypatch.setattr(dtm_client, "load_config", lambda: {"enabled": True, "api": {"admin_levels": ["admin0"]}})
    monkeypatch.setattr(dtm_client, "diagnostics_start_run", lambda *_, **__: object())
    monkeypatch.setattr(dtm_client, "diagnostics_finalize_run", lambda *_, **__: {})
    monkeypatch.setattr(dtm_client, "diagnostics_append_jsonl", lambda *_, **__: None)
    monkeypatch.setattr(dtm_client, "resolve_ingestion_window", lambda: (None, None))

    rc = dtm_client.main([])
    assert rc == 0
    assert attempt["count"] == 2
    run_payload = json.loads(patch_paths["RUN_DETAILS_PATH"].read_text(encoding="utf-8"))
    assert run_payload["http"]["retries"] == 1
