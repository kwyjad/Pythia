# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import json
from pathlib import Path

import pytest

from scripts.ci import generate_summary


def _run_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    out_path = tmp_path / "SUMMARY.md"
    monkeypatch.setattr(generate_summary.sys, "argv", ["generate_summary", "--out", str(out_path)])
    rc = generate_summary.main()
    assert rc == 0
    return out_path.read_text(encoding="utf-8")


def test_summary_includes_emdat_probe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    probe_dir = Path("diagnostics/ingestion/emdat")
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_payload = {
        "ok": True,
        "status": 200,
        "latency_ms": 123.4,
        "api_version": "2024-05",
        "table_version": "dataset-v1",
        "metadata_timestamp": "2024-05-01T00:00:00Z",
        "requests": {"total": 1, "2xx": 1, "4xx": 0, "5xx": 0},
        "recorded_at": "2024-06-01T00:00:00Z",
    }
    (probe_dir / "probe.json").write_text(json.dumps(probe_payload), encoding="utf-8")

    effective_payload = {
        "recorded_at": "2024-06-01T01:00:00Z",
        "source_type": "api",
        "source_override": "api",
        "network": True,
        "network_env": "1",
        "api_key_present": True,
        "default_from_year": 2022,
        "default_to_year": 2023,
        "include_hist": False,
        "classif_count": 1,
        "classif_keys": ["foo"],
        "iso_filter_applied": False,
        "iso_count": 0,
    }
    (probe_dir / "effective.json").write_text(
        json.dumps(effective_payload), encoding="utf-8"
    )

    sample_payload = {
        "ok": True,
        "http_status": 200,
        "elapsed_ms": 42.0,
        "rows": 0,
        "total_available": 10,
        "filters": {"from": 2021, "to": 2022},
        "classif_histogram": [{"classif_key": "foo", "count": 1}],
        "recorded_at": "2024-06-01T02:00:00Z",
    }
    (probe_dir / "probe_sample.json").write_text(
        json.dumps(sample_payload), encoding="utf-8"
    )

    preview_dir = Path("diagnostics/ingestion/export_preview")
    preview_dir.mkdir(parents=True, exist_ok=True)
    (preview_dir / "facts.csv").write_text("iso3,as_of_date,value\n", encoding="utf-8")

    content = _run_summary(tmp_path, monkeypatch)

    assert "## EMDAT Probe" in content
    assert "- status: 200 (ok)" in content
    assert "- api_version: 2024-05   table_version: dataset-v1" in content
    assert "- metadata timestamp: 2024-05-01T00:00:00Z" in content
    assert "- requests: total=1  2xx=1  4xx=0  5xx=0" in content
    assert "## EMDAT Effective Mode" in content
    assert "- source: api (override=api)" in content
    assert "- network: on (env=1)" in content
    assert "- api key: present" in content
    assert "- default window: 2022–2023 (include_hist=false)" in content
    assert "- classif: 1  iso filter: none" in content
    assert "## EMDAT Probe Sample" in content
    assert "- classif histogram: foo: 1" in content
    assert "## Export Preview" in content
    assert "facts.csv: present (0 rows)" in content


def test_summary_handles_missing_probe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    preview_dir = Path("diagnostics/ingestion/export_preview")
    preview_dir.mkdir(parents=True, exist_ok=True)
    (preview_dir / "facts.csv").write_text("iso3\n", encoding="utf-8")

    content = _run_summary(tmp_path, monkeypatch)

    assert "## EMDAT Probe" in content
    assert "- probe.json: missing" in content
    assert "## EMDAT Effective Mode" in content
    assert "- effective.json: missing" in content
