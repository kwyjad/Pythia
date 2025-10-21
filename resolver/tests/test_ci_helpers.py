from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_script(script: Path, *args: str) -> str:
    cmd = [sys.executable, str(script), *args]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout


def read_json_block(output_path: Path) -> dict[str, object]:
    payload_lines = output_path.read_text(encoding="utf-8").splitlines()
    if not payload_lines:
        raise AssertionError("report file was empty")
    json_start = None
    for idx, line in enumerate(payload_lines):
        if line.strip().startswith("{"):
            json_start = idx
            break
    if json_start is None:
        raise AssertionError("no JSON payload found in diagnostics report")
    payload = json.loads("\n".join(payload_lines[json_start:]))
    assert isinstance(payload, dict)
    return payload


def test_list_canonical_reports_rows(tmp_path: Path) -> None:
    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir()
    csv_path = canonical_dir / "sample.csv"
    csv_path.write_text("col\n1\n2\n", encoding="utf-8")
    out_path = tmp_path / "report.txt"

    script = REPO_ROOT / "scripts" / "ci" / "list_canonical.py"
    run_script(script, "--dir", str(canonical_dir), "--out", str(out_path))

    text = out_path.read_text(encoding="utf-8")
    assert "sample.csv" in text
    assert "rows=2" in text

    payload = read_json_block(out_path)
    assert payload["exists"] is True
    assert payload.get("total_rows") == 2
    files = payload.get("files")
    assert isinstance(files, list)
    assert any(entry.get("rows") == 2 for entry in files)  # type: ignore[index]


def test_db_counts_handles_missing_db(tmp_path: Path) -> None:
    db_path = tmp_path / "resolver.duckdb"
    out_path = tmp_path / "duckdb-report.txt"

    script = REPO_ROOT / "scripts" / "ci" / "db_counts.py"
    run_script(script, "--db", str(db_path), "--out", str(out_path))

    payload = read_json_block(out_path)
    assert payload["exists"] is False
    assert payload.get("tables") == {}
