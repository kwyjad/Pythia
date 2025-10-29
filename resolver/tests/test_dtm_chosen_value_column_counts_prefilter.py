import json
from pathlib import Path
from typing import Dict

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


def test_chosen_value_columns_count_raw_frame(monkeypatch):
    class _StubDTMClient:
        def __init__(self):
            self._http_counts: Dict[str, int] = {}
            self.rate_limit_delay = 0

        @staticmethod
        def _record_success(http_counts: Dict[str, int] | None) -> None:
            if http_counts is not None:
                http_counts["last_status"] = 200
                http_counts["2xx"] = http_counts.get("2xx", 0) + 1

        def get_idp_admin0(self, *, country, from_date, to_date, http_counts):
            self._record_success(http_counts)
            label = str(country)
            if label == "DR Congo":
                return pd.DataFrame(
                    [
                        {
                            "CountryName": "Democratic Republic of the Congo",
                            "ReportingDate": "2023-03-15",
                            "TotalIDPs": 120,
                        },
                        {
                            "CountryName": "Atlantis",
                            "ReportingDate": "2023-03-15",
                            "TotalIDPs": 15,
                        },
                    ]
                )
            if label == "Côte d'Ivoire":
                return pd.DataFrame(
                    [
                        {
                            "CountryName": "Côte d'Ivoire",
                            "ReportingDate": "2023-03-01",
                            "OtherValue": 5,
                        }
                    ]
                )
            return pd.DataFrame()

    class _StubDTMApiClient:
        def __init__(self, config):
            self.client = _StubDTMClient()
            self.config = config
            self.rate_limit_delay = 0
            self._http_counts: Dict[str, int] = {}

        def get_idp_admin0(self, **kwargs):
            return self.client.get_idp_admin0(**kwargs)

    monkeypatch.setattr(dtm_client, "DTMApiClient", _StubDTMApiClient)
    monkeypatch.setattr(dtm_client, "get_dtm_api_key", lambda: "dummy-key")
    monkeypatch.setattr(
        dtm_client,
        "_preflight_dependencies",
        lambda: ({"missing": [], "packages": [], "python": "3.11", "executable": "python"}, True),
    )

    def _fake_discovery(cfg, metrics, *, api_key=None, client=None):
        metrics["stage_used"] = "explicit_config"
        return dtm_client.DiscoveryResult(
            countries=["DR Congo", "Côte d'Ivoire"],
            frame=pd.DataFrame(),
            stage_used="explicit_config",
            report={
                "used_stage": "explicit_config",
                "configured_labels": ["DR Congo", "Côte d'Ivoire"],
                "unresolved_labels": [],
            },
        )

    monkeypatch.setattr(dtm_client, "_perform_discovery", _fake_discovery)
    monkeypatch.setattr(
        dtm_client,
        "load_config",
        lambda: {
            "enabled": True,
            "api": {"countries": ["DR Congo", "Côte d'Ivoire"], "admin_levels": ["admin0"]},
            "field_aliases": {"idp_count": ["TotalIDPs"]},
        },
    )
    monkeypatch.setattr(dtm_client, "resolve_ingestion_window", lambda: (None, None))
    monkeypatch.setattr(dtm_client, "diagnostics_start_run", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(dtm_client, "diagnostics_finalize_run", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(dtm_client, "diagnostics_append_jsonl", lambda *_args, **_kwargs: None)

    exit_code = dtm_client.main([])
    assert exit_code == 0

    payload = json.loads(Path(dtm_client.RUN_DETAILS_PATH).read_text(encoding="utf-8"))
    extras = payload["extras"]
    drops = extras["normalize"]["drop_reasons"]
    assert drops["no_iso3"] == 1
    assert drops["no_value_col"] == 1

    chosen = {
        entry["column"]: entry["count"]
        for entry in extras["normalize"].get("chosen_value_columns", [])
    }
    assert chosen.get("TotalIDPs") == 2
