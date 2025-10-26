import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from resolver.ingestion import dtm_client as dtm


def _patch_diagnostics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "diagnostics"
    mapping = {
        "DIAGNOSTICS_DIR": root,
        "DTM_DIAGNOSTICS_DIR": root / "dtm",
        "DIAGNOSTICS_RAW_DIR": root / "raw",
        "DIAGNOSTICS_METRICS_DIR": root / "metrics",
        "DIAGNOSTICS_SAMPLES_DIR": root / "samples",
        "DIAGNOSTICS_LOG_DIR": root / "logs",
        "CONNECTORS_REPORT": root / "connectors_report.jsonl",
        "RUN_DETAILS_PATH": root / "dtm_run.json",
        "API_REQUEST_PATH": root / "dtm_api_request.json",
        "API_SAMPLE_PATH": root / "dtm_api_sample.json",
        "API_RESPONSE_SAMPLE_PATH": root / "dtm_api_response_sample.json",
        "DISCOVERY_SNAPSHOT_PATH": (root / "dtm" / "discovery_countries.csv"),
        "DISCOVERY_FAIL_PATH": (root / "dtm" / "discovery_fail.json"),
        "DTM_HTTP_LOG_PATH": (root / "dtm" / "dtm_http.ndjson"),
        "DISCOVERY_RAW_JSON_PATH": root / "raw" / "dtm_countries.json",
        "PER_COUNTRY_METRICS_PATH": root / "metrics" / "dtm_per_country.jsonl",
        "SAMPLE_ROWS_PATH": root / "samples" / "dtm_sample.csv",
        "DTM_CLIENT_LOG_PATH": root / "logs" / "dtm_client.log",
        "METRICS_SUMMARY_PATH": root / "metrics" / "metrics.json",
        "SAMPLE_ADMIN0_PATH": root / "samples" / "sample_admin0.csv",
    }
    for name, value in mapping.items():
        monkeypatch.setattr(dtm, name, value)
    dtm._reset_admin0_sample_counter()
    dtm._clear_discovery_error_log()


def test_timeout_then_operations_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_diagnostics(monkeypatch, tmp_path)
    monkeypatch.setenv("DTM_API_KEY", "test-key")

    def fake_get(path: str, key: str, **_: object) -> pd.DataFrame:
        if "country-list" in path:
            raise requests.exceptions.Timeout("simulated timeout")
        return pd.DataFrame([
            {"admin0Name": "Freedonia", "admin0Pcode": "FRE"},
            {"admin0Name": "Sylvania", "admin0Pcode": "SYL"},
        ])

    monkeypatch.setattr(dtm, "_get_country_list_via_http", fake_get)

    metrics = dtm._init_metrics_summary()
    result = dtm._perform_discovery(dtm.load_config(), metrics=metrics, api_key="test-key")

    assert result.stage_used == "operations"

    payload = json.loads(dtm.DISCOVERY_FAIL_PATH.read_text(encoding="utf-8"))
    assert payload["used_stage"] == "operations"
    stages = {stage["stage"]: stage.get("status") for stage in payload.get("stages", [])}
    assert stages.get("countries") in {"error", "empty"}


def test_both_discovery_fail_uses_static(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_diagnostics(monkeypatch, tmp_path)
    monkeypatch.setenv("DTM_API_KEY", "test-key")

    def always_timeout(path: str, key: str, **_: object) -> pd.DataFrame:
        raise requests.exceptions.Timeout(f"{path} timed out")

    monkeypatch.setattr(dtm, "_get_country_list_via_http", always_timeout)

    def fake_static(path: Path | None = None) -> pd.DataFrame:
        return pd.DataFrame([
            {"admin0Name": "Freedonia", "admin0Pcode": "FRE"},
            {"admin0Name": "Sylvania", "admin0Pcode": "SYL"},
        ])

    monkeypatch.setattr(dtm, "_load_static_iso3", fake_static)

    metrics = dtm._init_metrics_summary()
    result = dtm._perform_discovery(dtm.load_config(), metrics=metrics, api_key="test-key")

    assert result.stage_used == "static_iso3"

    metrics_payload = json.loads(dtm.METRICS_SUMMARY_PATH.read_text(encoding="utf-8"))
    assert metrics_payload["countries_attempted"] > 0

    sample_path = dtm.SAMPLE_ADMIN0_PATH
    assert sample_path.exists()
    sample = pd.read_csv(sample_path)
    cols = set(sample.columns)
    assert {"admin0Name", "admin0Pcode"}.issubset(cols)
    if "operation" in cols and not sample.empty:
        assert sample["operation"].notna().any()


def test_admin0_only_skips_subnational(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_diagnostics(monkeypatch, tmp_path)
    monkeypatch.setenv("DTM_API_KEY", "test-key")

    class StubClient:
        base_url = "https://example.test"

        def __init__(self, config: object, *, subscription_key: str | None = None) -> None:
            self.config = config
            self.subscription_key = subscription_key
            self.client = self

        def get_idp_admin0(
            self,
            *,
            country: str | None = None,
            from_date: str | None = None,
            to_date: str | None = None,
            http_counts: dict | None = None,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "CountryName": country or "Kenya",
                        "ReportingDate": "2023-01-01",
                        "TotalIDPs": 1200,
                        "Cause": "conflict",
                    }
                ]
            )

        def get_idp_admin1(self, **_: object) -> pd.DataFrame:  # pragma: no cover - should not run
            raise AssertionError("admin1 fetch should be skipped when admin0 only")

        def get_idp_admin2(self, **_: object) -> pd.DataFrame:  # pragma: no cover - should not run
            raise AssertionError("admin2 fetch should be skipped when admin0 only")

    def fake_discovery(
        cfg: object,
        *,
        metrics: dict | None = None,
        api_key: str | None = None,
    ) -> dtm.DiscoveryResult:
        frame = pd.DataFrame([
            {"admin0Name": "Kenya", "admin0Pcode": "KEN"},
        ])
        result = dtm.DiscoveryResult(
            countries=["Kenya"],
            frame=frame,
            stage_used="static_iso3",
            report={"stages": [], "errors": [], "attempts": {}, "latency_ms": {}, "used_stage": "static_iso3"},
        )
        if metrics is not None:
            metrics["countries_attempted"] = 1
            metrics["stage_used"] = "static_iso3"
            dtm._write_metrics_summary_file(metrics)
        return result

    monkeypatch.setattr(dtm, "DTMApiClient", StubClient)
    monkeypatch.setattr(dtm, "_perform_discovery", fake_discovery)

    cfg = dtm.load_config()
    cfg.setdefault("api", {})["admin_levels"] = ["admin0"]
    cfg.setdefault("output", {})["measure"] = "flow"

    http_counts: dict = {}
    records, summary = dtm._fetch_api_data(
        cfg,
        no_date_filter=True,
        window_start=None,
        window_end=None,
        http_counts=http_counts,
    )

    assert summary["row_counts"]["admin0"] > 0
    assert summary["row_counts"].get("admin1", 0) == 0
    assert summary["row_counts"].get("admin2", 0) == 0
    assert records, "Expected normalized records for admin0 run"


def test_soft_skip_no_country_match(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_diagnostics(monkeypatch, tmp_path)
    monkeypatch.setenv("DTM_API_KEY", "test-key")

    class MockSDKClient:
        def get_idp_admin0_data(self, **kwargs: object) -> pd.DataFrame:
            raise ValueError("No Country found matching your query.")

    def fake_init(self, config: dict, *, subscription_key: str | None = None) -> None:
        self.config = config
        self.client = MockSDKClient()
        self.rate_limit_delay = 0.0
        self.timeout = 60
        self._http_counts = {}

    monkeypatch.setattr(dtm.DTMApiClient, "__init__", fake_init, raising=False)

    def fake_discovery(
        cfg: dict,
        *,
        metrics: dict | None = None,
        api_key: str | None = None,
    ) -> dtm.DiscoveryResult:
        frame = pd.DataFrame([
            {"admin0Name": "Atlantis", "admin0Pcode": "ATL"},
        ])
        result = dtm.DiscoveryResult(
            countries=["Atlantis"],
            frame=frame,
            stage_used="static_iso3",
            report={"stages": [], "errors": [], "attempts": {}, "latency_ms": {}, "used_stage": "static_iso3"},
        )
        return result

    monkeypatch.setattr(dtm, "_perform_discovery", fake_discovery)

    cfg = dtm.load_config()
    cfg.setdefault("api", {})["admin_levels"] = ["admin0"]

    http_counts: dict = {}
    records, summary = dtm._fetch_api_data(
        cfg,
        no_date_filter=True,
        window_start=None,
        window_end=None,
        http_counts=http_counts,
    )

    assert records == []
    assert summary["rows"]["fetched"] == 0
    assert summary["http_counts"].get("skip_no_match") == 1
    assert summary.get("extras", {}).get("zero_rows_reason") == "no_country_match"

    metrics_payload = json.loads(dtm.METRICS_SUMMARY_PATH.read_text(encoding="utf-8"))
    assert metrics_payload["countries_skipped_no_match"] == 1
    assert metrics_payload["rows_fetched"] == 0
    assert metrics_payload["countries_failed_other"] == 0


def test_static_iso3_accepts_commas_in_names() -> None:
    csv_path = Path("resolver/ingestion/static/iso3_master.csv")
    df = pd.read_csv(csv_path, dtype=str, engine="python")
    assert {"admin0Pcode", "admin0Name"}.issubset(df.columns)
    assert (df["admin0Name"].str.contains(",")).any(), "Expect at least one comma-in-name row"
