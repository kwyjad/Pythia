from __future__ import annotations

from pathlib import Path


def test_runtime_requirements_include_openpyxl() -> None:
    req_path = Path(__file__).resolve().parents[1] / "requirements.txt"
    lines = [line.strip().lower() for line in req_path.read_text().splitlines()]
    assert any(line.startswith("openpyxl") for line in lines)
