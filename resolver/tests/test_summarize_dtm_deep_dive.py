# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import json
from pathlib import Path

from scripts.ci.summarize_connectors import _render_dtm_deep_dive


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_dtm_deep_dive_renders_sections(tmp_path):
    http_trace = tmp_path / "dtm_http.ndjson"
    http_trace.write_text(
        "\n".join(
            [
                json.dumps({"path": "/countries", "elapsed_ms": 120}),
                json.dumps({"path": "/countries", "elapsed_ms": 80}),
                json.dumps({"path": "/admin0", "elapsed_ms": 200}),
            ]
        ),
        encoding="utf-8",
    )
    discovery_fail = tmp_path / "discovery_fail.json"
    _write_json(discovery_fail, {"errors": [{"http_code": 403, "message": "forbidden"}]})
    samples = tmp_path / "samples.csv"
    samples.write_text(
        "Operation,admin0Name,admin0Pcode,CountryISO3,ReportingDate,idp_count\n"
        "OpA,CountryA,CAA,CAA,2023-01-01,10\n",
        encoding="utf-8",
    )
    run_json = tmp_path / "dtm_run.json"
    _write_json(run_json, {"status": "ok"})

    extras = {
        "dtm": {"sdk_version": "1.2.3", "base_url": "https://example", "python_version": "3.11"},
        "config": {
            "config_path_used": "resolver/ingestion/config/dtm.yml",
            "admin_levels": ["admin0"],
            "countries_mode": "explicit_config",
            "countries_count": 2,
            "no_date_filter": 0,
        },
        "window": {"start_iso": "2023-01-01", "end_iso": "2023-02-01"},
        "discovery": {
            "stages": [
                {"name": "sdk", "status": "ok", "http_code": 200, "attempts": 1, "latency_ms": 250}
            ],
            "used_stage": "sdk",
            "reason": None,
            "snapshot_path": "diagnostics/ingestion/dtm/discovery_countries.csv",
            "first_fail_path": str(discovery_fail),
            "total_countries": 2,
        },
        "http": {
            "count_2xx": 5,
            "count_4xx": 1,
            "count_5xx": 0,
            "retries": 2,
            "timeouts": 0,
            "last_status": 200,
            "endpoints_top": [],
        },
        "fetch": {"pages": 3, "max_page_size": 100, "total_received": 250},
        "normalize": {
            "rows_fetched": 250,
            "rows_normalized": 120,
            "rows_written": 0,
            "drop_reasons": {"no_country_match": 1, "no_iso3": 2, "other": 0},
            "chosen_value_columns": [{"column": "TotalIDPs", "count": 120}],
        },
        "rescue_probe": {
            "tried": [
                {"country": "Nigeria", "window": "no_date_filter", "rows": 5},
                {"country": "South Sudan", "window": "no_date_filter", "rows": 0, "error": "empty"},
            ]
        },
        "artifacts": {
            "http_trace": str(http_trace),
            "samples": str(samples),
            "run_json": str(run_json),
        },
    }

    lines = _render_dtm_deep_dive({"extras": extras})
    markdown = "\n".join(lines)

    assert "## DTM Deep Dive" in markdown
    assert "### Discovery" in markdown
    assert "sdk" in markdown
    assert "forbidden" in markdown
    assert "### HTTP Roll-up" in markdown
    assert "### Sample rows" in markdown
    assert "Nigeria" in markdown
    assert "Actionable next steps" in markdown


def test_dtm_deep_dive_handles_missing_artifacts(tmp_path):
    extras = {
        "dtm": {},
        "config": {"admin_levels": [], "countries_mode": "discovered", "countries_count": 0, "no_date_filter": 1},
        "window": {"start_iso": None, "end_iso": None},
        "discovery": {"stages": [], "used_stage": None, "snapshot_path": "", "first_fail_path": ""},
        "http": {"count_2xx": 0, "count_4xx": 0, "count_5xx": 0, "retries": 0, "timeouts": 0, "last_status": None},
        "fetch": {},
        "normalize": {"rows_fetched": 0, "rows_normalized": 0, "rows_written": 0, "drop_reasons": {}},
        "artifacts": {},
    }

    lines = _render_dtm_deep_dive({"extras": extras})
    markdown = "\n".join(lines)

    assert "DTM Deep Dive" in markdown
    assert "_No discovery stages recorded._" in markdown
    assert "Actionable next steps" in markdown
