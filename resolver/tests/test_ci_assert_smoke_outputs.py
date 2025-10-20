from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci.assert_smoke_outputs import main as assert_smoke_main


def _write_csv(base: Path, name: str, rows: list[str]) -> Path:
    canonical_dir = base / "data" / "staging" / "ci-smoke" / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    csv_path = canonical_dir / name
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("col1\n")
        for row in rows:
            handle.write(f"{row}\n")
    return csv_path


def _run(tmp_path: Path, min_rows: int = 1) -> int:
    args = [
        "--staging",
        str(tmp_path / "data" / "staging"),
        "--period",
        "ci-smoke",
        "--min-rows",
        str(min_rows),
    ]
    return assert_smoke_main(args)


def test_counts_rows_excluding_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_csv(tmp_path, "example.csv", ["1", "2"])

    assert _run(tmp_path, min_rows=1) == 0
    assert _run(tmp_path, min_rows=3) == 1


def test_handles_header_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_csv(tmp_path, "header_only.csv", [])

    assert _run(tmp_path, min_rows=1) == 1


def test_writes_json_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_csv(tmp_path, "report.csv", ["a", "b", "c"])

    exit_code = _run(tmp_path, min_rows=1)
    assert exit_code == 0

    report_path = Path(".ci/diagnostics/smoke-assert.json")
    assert report_path.is_file()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["total_rows"] == 3
    assert any(entry["rows"] == 3 for entry in payload["files"])
    assert payload["canonical_dir"].endswith("canonical")
