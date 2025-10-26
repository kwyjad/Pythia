import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest

from resolver.ingestion import dtm_client


@pytest.fixture(autouse=True)
def _redirect_dtm_paths(tmp_path, monkeypatch):
    diagnostics_root = tmp_path / "diagnostics" / "ingestion"
    dtm_dir = diagnostics_root / "dtm"
    replacements = {
        "DIAGNOSTICS_ROOT": diagnostics_root,
        "DTM_DIAGNOSTICS_DIR": dtm_dir,
        "DTM_RAW_DIR": dtm_dir / "raw",
        "DTM_METRICS_DIR": dtm_dir / "metrics",
        "DTM_SAMPLES_DIR": dtm_dir / "samples",
        "DTM_LOG_DIR": dtm_dir / "logs",
        "CONNECTORS_REPORT": diagnostics_root / "connectors_report.jsonl",
        "RUN_DETAILS_PATH": dtm_dir / "dtm_run.json",
        "API_REQUEST_PATH": dtm_dir / "dtm_api_request.json",
        "API_SAMPLE_PATH": dtm_dir / "dtm_api_sample.json",
        "API_RESPONSE_SAMPLE_PATH": dtm_dir / "dtm_api_response_sample.json",
        "DISCOVERY_SNAPSHOT_PATH": dtm_dir / "discovery_countries.csv",
        "DISCOVERY_FAIL_PATH": dtm_dir / "discovery_fail.json",
        "DTM_HTTP_LOG_PATH": dtm_dir / "dtm_http.ndjson",
        "DISCOVERY_RAW_JSON_PATH": dtm_dir / "raw" / "dtm_countries.json",
        "PER_COUNTRY_METRICS_PATH": dtm_dir / "metrics" / "dtm_per_country.jsonl",
        "SAMPLE_ROWS_PATH": dtm_dir / "dtm_sample.csv",
        "DTM_CLIENT_LOG_PATH": dtm_dir / "logs" / "dtm_client.log",
        "RESCUE_PROBE_PATH": dtm_dir / "rescue_probe.json",
        "METRICS_SUMMARY_PATH": dtm_dir / "metrics" / "metrics.json",
        "SAMPLE_ADMIN0_PATH": dtm_dir / "samples" / "admin0_head.csv",
        "_LEGACY_DIAGNOSTICS_DIR": tmp_path / "legacy_diagnostics",
    }
    staging_dir = tmp_path / "staging"
    output_csv = staging_dir / "dtm_displacement.csv"
    replacements.update(
        {
            "OUT_DIR": staging_dir,
            "OUT_PATH": output_csv,
            "OUTPUT_PATH": output_csv,
            "META_PATH": output_csv.with_suffix(".meta.json"),
            "HTTP_TRACE_PATH": staging_dir / "dtm_http.ndjson",
        }
    )
    for attr, path in replacements.items():
        monkeypatch.setattr(dtm_client, attr, path)
    for directory in (
        diagnostics_root,
        dtm_dir,
        dtm_dir / "raw",
        dtm_dir / "metrics",
        dtm_dir / "samples",
        dtm_dir / "logs",
        staging_dir,
        replacements["_LEGACY_DIAGNOSTICS_DIR"],
    ):
        directory.mkdir(parents=True, exist_ok=True)
    yield


def test_explicit_countries_skip_discovery_and_fetch_admin_levels(monkeypatch):
    calls: Dict[str, List[str]] = {"admin0": [], "admin1": []}

    class _StubDTMClient:
        def __init__(self):
            self._http_counts: Dict[str, int] = {}
            self.rate_limit_delay = 0

        def _record_success(self, http_counts: Dict[str, int]) -> None:
            if http_counts is not None:
                http_counts["last_status"] = 200
                http_counts["2xx"] = http_counts.get("2xx", 0) + 1

        def _make_frame(self, country: str, level: str) -> pd.DataFrame:
            name_map = {
                "NGA": "Nigeria",
                "SSD": "South Sudan",
                "ETH": "Ethiopia",
                "SOM": "Somalia",
            }
            label = name_map.get(country, country)
            data = {
                "CountryName": label,
                "CountryISO3": country if len(country) == 3 else "",
                "Admin1Name": f"{label} Admin1",
                "ReportingDate": "2023-01-15",
                "TotalIDPs": 10 if level == "admin0" else 5,
                "Cause": "conflict",
            }
            return pd.DataFrame([data])

        def get_idp_admin0(self, *, country, from_date, to_date, http_counts):
            calls["admin0"].append(str(country))
            self._record_success(http_counts)
            return self._make_frame(country, "admin0")

        def get_idp_admin1(self, *, country, from_date, to_date, http_counts):
            calls["admin1"].append(str(country))
            self._record_success(http_counts)
            return self._make_frame(country, "admin1")

    class _StubDTMApiClient:
        def __init__(self, config):
            self.config = config
            self.client = _StubDTMClient()
            self.rate_limit_delay = 0
            self._http_counts: Dict[str, int] = {}

        def get_idp_admin0(self, **kwargs):
            return self.client.get_idp_admin0(**kwargs)

        def get_idp_admin1(self, **kwargs):
            return self.client.get_idp_admin1(**kwargs)

    monkeypatch.setattr(dtm_client, "DTMApiClient", _StubDTMApiClient)
    monkeypatch.setattr(dtm_client, "get_dtm_api_key", lambda: "dummy-key")
    discovery_calls: List[str] = []

    def _fail_discovery(*args, **kwargs):
        discovery_calls.append("http")
        raise AssertionError("explicit config should skip discovery")

    monkeypatch.setattr(dtm_client, "_get_country_list_via_http", _fail_discovery)
    monkeypatch.setattr(
        dtm_client,
        "_load_static_iso3",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("static roster should not be used")),
    )

    cfg = {
        "enabled": True,
        "api": {
            "countries": ["Nigeria", "South Sudan", "DR Congo"],
            "admin_levels": ["admin0", "admin1"],
        },
        "field_aliases": {"idp_count": ["TotalIDPs"]},
    }

    metrics: Dict[str, int] = {}
    discovery = dtm_client._perform_discovery(cfg, metrics=metrics)
    assert "NGA" in discovery.countries
    assert "SSD" in discovery.countries
    assert "COD" in discovery.countries
    assert discovery.report.get("used_stage") == "explicit_config"
    assert discovery.report.get("unresolved_labels") == []
    assert metrics["stage_used"] == "explicit_config"
    assert not discovery_calls

    # Patch runtime helpers to keep execution self-contained.
    monkeypatch.setattr(dtm_client, "load_config", lambda: cfg)
    monkeypatch.setattr(dtm_client, "resolve_ingestion_window", lambda: (None, None))
    monkeypatch.setattr(dtm_client, "diagnostics_start_run", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(dtm_client, "diagnostics_finalize_run", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(dtm_client, "diagnostics_append_jsonl", lambda *_args, **_kwargs: None)

    exit_code = dtm_client.main([])
    assert exit_code == 0

    # ensure admin0 and admin1 fetches were attempted
    assert calls["admin0"]
    assert calls["admin1"]
    expected_countries = {"NGA", "SSD", "COD"}
    assert expected_countries.issubset(set(calls["admin0"]))
    assert expected_countries.issubset(set(calls["admin1"]))

    run_payload = Path(dtm_client.RUN_DETAILS_PATH).read_text(encoding="utf-8")
    extras = json.loads(run_payload)["extras"]
    assert extras["config"]["countries_mode"] == "explicit_config"
    assert "admin1" in extras["config"]["admin_levels"]
    snapshot = Path(dtm_client.DISCOVERY_SNAPSHOT_PATH)
    assert snapshot.exists()
    content = snapshot.read_text(encoding="utf-8").strip().splitlines()
    assert content and content[0].startswith("country_label")
    assert any("explicit_config" in line for line in content)
