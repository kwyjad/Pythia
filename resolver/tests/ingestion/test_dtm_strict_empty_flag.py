from __future__ import annotations

import json
from pathlib import Path

from resolver.ingestion import dtm_client
from scripts.ci import summarize_connectors

from resolver.tests.ingestion.test_dtm_ok_empty_status import (
    _setup_paths,
    _stub_build_rows,
)


def test_strict_empty_exits_with_code_three(monkeypatch, tmp_path: Path) -> None:
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

    exit_code = dtm_client.main(["--strict-empty"])
    assert exit_code == 3

    report_path = dtm_client.CONNECTORS_REPORT
    entries = summarize_connectors.load_report(report_path)
    dtm_entry = next(item for item in entries if item["connector_id"] == "dtm_client")
    extras = dtm_entry["extras"]
    assert extras.get("status_raw") == "ok-empty"
    assert extras.get("exit_code") == 3
    assert extras.get("kept_rows") == 0

    meta_payload = json.loads(dtm_client.META_PATH.read_text(encoding="utf-8"))
    assert meta_payload.get("row_count") == 0
    run_payload = json.loads(dtm_client.RUN_DETAILS_PATH.read_text(encoding="utf-8"))
    assert run_payload["rows_written"] == 0
    assert run_payload["totals"]["kept"] == 0
    assert run_payload["outputs"]["csv"] == str(out_path)
