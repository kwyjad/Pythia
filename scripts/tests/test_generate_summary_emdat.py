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
        "http_status": 200,
        "elapsed_ms": 123.4,
        "api_version": "2024-05",
        "info": {"version": "dataset-v1", "timestamp": "2024-05-01T00:00:00Z"},
        "recorded_at": "2024-06-01T00:00:00Z",
        "total_available": 0,
    }
    (probe_dir / "probe.json").write_text(json.dumps(probe_payload), encoding="utf-8")

    effective_payload = {
        "network": {"requested": True, "env_value": "1"},
        "api": {"base_url": "https://api.example", "key_present": True},
        "filters": {
            "from": 2022,
            "to": 2023,
            "classif": ["foo"],
            "include_hist": False,
        },
        "graphQL_vars": {"from": 2022, "to": 2023, "classif": ["foo"]},
        "recorded_at": "2024-06-01T01:00:00Z",
    }
    (probe_dir / "effective_params.json").write_text(
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
    assert "- status: ok (HTTP 200, 123 ms)" in content
    assert "- api_version: 2024-05" in content
    assert "- dataset version: dataset-v1" in content
    assert "- metadata timestamp: 2024-05-01T00:00:00Z" in content
    assert "## EMDAT Effective Params" in content
    assert "- network: on (env=1)" in content
    assert "- graphQL vars: {\"classif\": [\"foo\"], \"from\": 2022, \"to\": 2023}" in content
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
    assert "## EMDAT Effective Params" in content
    assert "- effective_params.json: missing" in content
