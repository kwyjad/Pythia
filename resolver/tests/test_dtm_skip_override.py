import json
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest

from resolver.ingestion import dtm_client


@pytest.fixture(autouse=True)
def _redirect_dtm_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
        "API_SAMPLE_PATH": dtm_dir / "dtm_api_sample.json",
        "API_RESPONSE_SAMPLE_PATH": dtm_dir / "dtm_api_response_sample.json",
        "DTM_HTTP_LOG_PATH": dtm_dir / "dtm_http.ndjson",
        "DISCOVERY_RAW_JSON_PATH": dtm_dir / "raw" / "dtm_countries.json",
        "PER_COUNTRY_METRICS_PATH": dtm_dir / "metrics" / "dtm_per_country.jsonl",
        "SAMPLE_ROWS_PATH": dtm_dir / "dtm_sample.csv",
        "DTM_CLIENT_LOG_PATH": dtm_dir / "logs" / "dtm_client.log",
        "RESCUE_PROBE_PATH": dtm_dir / "rescue_probe.json",
        "METRICS_SUMMARY_PATH": dtm_dir / "metrics" / "metrics.json",
        "SAMPLE_ADMIN0_PATH": dtm_dir / "samples" / "admin0_head.csv",
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
    for attr, value in replacements.items():
        monkeypatch.setattr(dtm_client, attr, value)
    for directory in [
        diagnostics_root,
        dtm_dir,
        dtm_dir / "raw",
        dtm_dir / "metrics",
        dtm_dir / "samples",
        dtm_dir / "logs",
        staging_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    yield


def test_force_run_overrides_skip(monkeypatch: pytest.MonkeyPatch):
    class _StubDTMClient:
        def __init__(self):
            self.rate_limit_delay = 0

        @staticmethod
        def _record_success(http_counts: Dict[str, int] | None) -> None:
            if http_counts is not None:
                http_counts["last_status"] = 200
                http_counts["2xx"] = http_counts.get("2xx", 0) + 1

        def get_idp_admin0(self, *, country, from_date, to_date, http_counts):  # noqa: ANN001
            self._record_success(http_counts)
            return pd.DataFrame(
                [
                    {
                        "CountryName": "Exampleland",
                        "ReportingDate": "2024-01-15",
                        "TotalIDPs": 42,
                    }
                ]
            )

    class _StubClientWrapper:
        def __init__(self, *_args, **_kwargs):  # noqa: ANN001
            self._client = _StubDTMClient()
            self.client = self
            self.rate_limit_delay = 0
            self.timeout = 0

        def get_idp_admin0(self, **kwargs):  # noqa: ANN003
            return self._client.get_idp_admin0(**kwargs)

        def get_idp_admin0_data(self, **kwargs):  # noqa: ANN003
            return self.get_idp_admin0(**kwargs)

    monkeypatch.setenv("RESOLVER_SKIP_DTM", "1")
    monkeypatch.setenv("DTM_FORCE_RUN", "1")
    monkeypatch.setattr(dtm_client, "DTMApiClient", _StubClientWrapper)
    monkeypatch.setenv("DTM_API_KEY", "dummy-key")
    monkeypatch.setattr(
        dtm_client,
        "_preflight_dependencies",
        lambda: ({"missing": [], "packages": [], "python": "3.11", "executable": "python"}, True),
    )

    def _fake_discovery(cfg, metrics, *, api_key=None, client=None):  # noqa: ANN001, ANN002
        metrics["stage_used"] = "explicit_config"
        return dtm_client.DiscoveryResult(
            countries=["Kenya"],
            frame=pd.DataFrame(),
            stage_used="explicit_config",
            report={
                "used_stage": "explicit_config",
                "configured_labels": ["Kenya"],
                "unresolved_labels": [],
            },
        )

    monkeypatch.setattr(dtm_client, "_perform_discovery", _fake_discovery)
    monkeypatch.setattr(
        dtm_client,
        "load_config",
        lambda: {
            "enabled": True,
            "api": {"countries": ["Kenya"], "admin_levels": ["admin0"]},
            "field_aliases": {"idp_count": ["TotalIDPs"]},
        },
    )
    monkeypatch.setattr(dtm_client, "resolve_ingestion_window", lambda: (None, None))
    monkeypatch.setattr(dtm_client, "diagnostics_start_run", lambda *_a, **_k: object())
    monkeypatch.setattr(dtm_client, "diagnostics_finalize_run", lambda *_a, **_k: {})
    monkeypatch.setattr(dtm_client, "diagnostics_append_jsonl", lambda *_a, **_k: None)

    sample_page = pd.DataFrame(
        [
            {
                "CountryName": "Kenya",
                "CountryISO3": "KEN",
                "ReportingDate": "2024-01-15",
                "TotalIDPs": 42,
            }
        ]
    )

    def _fake_fetch_level_pages(client, level, **kwargs):  # noqa: ANN001, ANN003
        if level == "admin0":
            return [sample_page], int(sample_page.shape[0]), "direct"
        return [], 0, None

    monkeypatch.setattr(
        dtm_client,
        "_fetch_level_pages_with_logging",
        _fake_fetch_level_pages,
    )

    exit_code = dtm_client.main([])
    assert exit_code == 0

    payload = json.loads(Path(dtm_client.RUN_DETAILS_PATH).read_text(encoding="utf-8"))
    extras = payload["extras"]
    skip_flags = extras.get("skip_flags", {})
    assert skip_flags.get("RESOLVER_SKIP_DTM") is True
    assert skip_flags.get("DTM_FORCE_RUN") is True
    assert payload.get("status") == "ok"
    normalize_block = extras.get("normalize", {})
    assert normalize_block.get("rows_written", 0) > 0
