from __future__ import annotations

import json
from pathlib import Path

from resolver.ingestion import dtm_client
from scripts.ci import summarize_connectors


def _setup_paths(monkeypatch, tmp_path: Path) -> Path:
    out_path = tmp_path / "data" / "staging" / "dtm_displacement.csv"
    diagnostics_dir = tmp_path / "diagnostics" / "ingestion"

    monkeypatch.setattr(dtm_client, "OUT_PATH", out_path)
    monkeypatch.setattr(dtm_client, "OUT_DIR", out_path.parent)
    monkeypatch.setattr(dtm_client, "OUTPUT_PATH", out_path)
    monkeypatch.setattr(
        dtm_client,
        "META_PATH",
        out_path.with_suffix(out_path.suffix + ".meta.json"),
    )
    monkeypatch.setattr(dtm_client, "DEFAULT_OUTPUT", out_path)
    monkeypatch.setattr(dtm_client, "DIAGNOSTICS_DIR", diagnostics_dir)
    monkeypatch.setattr(
        dtm_client,
        "CONNECTORS_REPORT",
        diagnostics_dir / "connectors_report.jsonl",
    )
    monkeypatch.setattr(
        dtm_client,
        "CONFIG_ISSUES_PATH",
        diagnostics_dir / "dtm_config_issues.json",
    )
    monkeypatch.setattr(
        dtm_client,
        "RESOLVED_SOURCES_PATH",
        diagnostics_dir / "dtm_sources_resolved.json",
    )
    monkeypatch.setattr(
        dtm_client,
        "RUN_DETAILS_PATH",
        diagnostics_dir / "dtm_run.json",
    )
    return out_path


def _stub_build_rows(monkeypatch):
    def fake_build_rows(
        cfg,
        *,
        results=None,
        no_date_filter=False,
        window_start=None,
        window_end=None,
        mode="records",
        http_client=None,
        http_summary=None,
        http_stats=None,
    ):
        if results is not None:
            results.append(dtm_client.SourceResult(source_name="stub", status="ok"))
        return []

    monkeypatch.setattr(dtm_client, "build_rows", fake_build_rows)


def test_dtm_records_ok_empty_reason(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RESOLVER_SKIP_DTM", raising=False)
    monkeypatch.delenv("DTM_STRICT", raising=False)
    monkeypatch.delenv("DTM_STRICT_EMPTY", raising=False)

    out_path = _setup_paths(monkeypatch, tmp_path)
    _stub_build_rows(monkeypatch)
    monkeypatch.setattr(
        dtm_client,
        "load_config",
        lambda: {"enabled": True, "sources": [{"id_or_path": "stub.csv"}]},
    )

    exit_code = dtm_client.main([])
    assert exit_code == 0

    meta_path = dtm_client.META_PATH
    assert meta_path.exists()
    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta_payload.get("row_count") == 0

    run_payload = json.loads(dtm_client.RUN_DETAILS_PATH.read_text(encoding="utf-8"))
    assert run_payload["rows_written"] == 0
    assert run_payload["missing_id_or_path"] == 0
    assert run_payload["totals"]["kept"] == 0
    assert run_payload["totals"]["parse_errors"] == 0
    assert run_payload["outputs"]["csv"] == str(out_path)

    report_path = dtm_client.CONNECTORS_REPORT
    entries = summarize_connectors.load_report(report_path)
    dtm_entry = next(item for item in entries if item["connector_id"] == "dtm_client")
    assert dtm_entry["status"] == "ok"
    assert dtm_entry["reason"] == "header-only; kept=0"
    extras = dtm_entry["extras"]
    assert extras.get("status_raw") == "ok-empty"
    assert extras.get("rows_written") == 0
    assert extras.get("meta_path") == str(meta_path)
