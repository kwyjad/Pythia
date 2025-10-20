from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci.smoke_assert import main as smoke_assert_main


def _canonical_dir(base: Path) -> Path:
    return base / "data" / "staging" / "ci-smoke" / "canonical"


def _write_csv(base: Path, name: str, rows: list[str]) -> Path:
    canonical_dir = _canonical_dir(base)
    canonical_dir.mkdir(parents=True, exist_ok=True)
    csv_path = canonical_dir / name
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("col1\n")
        for row in rows:
            handle.write(f"{row}\n")
        handle.write("\n")
    return csv_path


def _run(tmp_path: Path, min_rows: int = 1) -> int:
    canonical_dir = _canonical_dir(tmp_path)
    args = [
        "--canonical-dir",
        str(canonical_dir),
        "--min-rows",
        str(min_rows),
    ]
    return smoke_assert_main(args)


def test_counts_rows_excluding_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_csv(tmp_path, "example.csv", ["1", "2"])

    assert _run(tmp_path, min_rows=1) == 0
    assert _run(tmp_path, min_rows=3) == 2


def test_trailing_blank_lines_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_csv(tmp_path, "blank.csv", ["1", "2", ""])

    assert _run(tmp_path, min_rows=2) == 0


def test_handles_header_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_csv(tmp_path, "header_only.csv", [])

    assert _run(tmp_path, min_rows=1) == 2


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
