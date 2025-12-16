# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import json
from dataclasses import dataclass
from pathlib import Path


def test_run_connectors_bootstraps_diagnostics(tmp_path, monkeypatch):
    from scripts.ci import run_connectors

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_connectors, "_resolve_connectors", lambda env: [])

    rc = run_connectors.main([])
    assert rc == 0

    diag_base = Path(tmp_path) / "diagnostics" / "ingestion"
    expected_dirs = ["logs", "raw", "metrics", "samples", "dtm"]
    for sub in expected_dirs:
        assert (diag_base / sub).is_dir()

    for sub in ("raw", "metrics", "samples"):
        keep = diag_base / sub / ".keep"
        assert keep.is_file()


def test_run_connectors_report_jsonable(tmp_path):
    from scripts.ci import run_connectors

    @dataclass
    class Record:
        status: str
        code: int

    report_path = tmp_path / "diagnostics" / "ingestion" / "connectors_report.jsonl"
    payload = {"connector": "dtm_client", "result": Record("ok", 0)}
    run_connectors._write_report(report_path, [payload])
    more = {"connector": "acled_client", "result": {"status": "ok", "details": Record("ok", 0)}}
    run_connectors._write_report(report_path, [more])

    lines = report_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first == {"connector": "dtm_client", "result": {"status": "ok", "code": 0}}
    assert second == {
        "connector": "acled_client",
        "result": {"status": "ok", "details": {"status": "ok", "code": 0}},
    }
