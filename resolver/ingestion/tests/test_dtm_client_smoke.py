from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

from resolver.ingestion.dtm_client import CANONICAL_COLUMNS
from resolver.tests.utils import run as run_proc

REPO_ROOT = Path(__file__).resolve().parents[3]
RESOLVER_ROOT = REPO_ROOT / "resolver"
STAGING_CSV = RESOLVER_ROOT / "staging" / "dtm_displacement.csv"
META_PATH = STAGING_CSV.with_suffix(STAGING_CSV.suffix + ".meta.json")
SUMMARY_PATH = REPO_ROOT / "diagnostics" / "ingestion" / "dtm" / "summary.json"
REQUEST_LOG_PATH = REPO_ROOT / "diagnostics" / "ingestion" / "dtm" / "request_log.jsonl"
SAMPLE_PATH = REPO_ROOT / "diagnostics" / "sample_dtm_displacement.csv"
CONNECTORS_REPORT = REPO_ROOT / "diagnostics" / "connectors_report.jsonl"
NEW_CONNECTORS_REPORT = REPO_ROOT / "diagnostics" / "ingestion" / "connectors_report.jsonl"


def _reset_outputs() -> None:
    STAGING_CSV.unlink(missing_ok=True)
    META_PATH.unlink(missing_ok=True)
    SUMMARY_PATH.unlink(missing_ok=True)
    REQUEST_LOG_PATH.unlink(missing_ok=True)
    SAMPLE_PATH.unlink(missing_ok=True)
    CONNECTORS_REPORT.unlink(missing_ok=True)
    NEW_CONNECTORS_REPORT.unlink(missing_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("DTM_API_KEY", None)
    env.pop("DTM_SUBSCRIPTION_KEY", None)
    pythonpath = env.get("PYTHONPATH")
    repo_str = str(REPO_ROOT)
    env["PYTHONPATH"] = repo_str if not pythonpath else f"{repo_str}:{pythonpath}"
    return env


def _read_connectors_report() -> list[dict[str, object]]:
    if not CONNECTORS_REPORT.exists():
        return []
    lines: list[dict[str, object]] = []
    with CONNECTORS_REPORT.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            lines.append(json.loads(raw))
    return lines


def test_offline_smoke_creates_csv_and_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_outputs()
    env = _clean_env()
    result = run_proc(
        [sys.executable, "-m", "resolver.ingestion.dtm_client", "--offline-smoke"],
        check=False,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0
    assert STAGING_CSV.exists(), "offline smoke should produce a CSV"

    with STAGING_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        assert header == list(CANONICAL_COLUMNS)
        rows = list(reader)
        assert not rows, "offline smoke should only write the header"

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["mode"] == "offline_smoke"
    assert summary["rows_out"] == 0
    assert summary["reason"] == "offline_smoke"

    report_lines = _read_connectors_report()
    assert len(report_lines) == 1
    record = report_lines[0]
    assert record["status"] == "ok"
    assert record["connector"] == "dtm"
    assert record["rows_out"] == 0
    assert record["reason"] == "offline_smoke"
    assert record["output_path"].endswith("dtm_displacement.csv")


def test_skips_gracefully_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_outputs()

    class _DummyModule:
        class Client:  # pragma: no cover - constructor never used
            def __init__(self, *_, **__):
                pass

    monkeypatch.setitem(sys.modules, "dtmapi", _DummyModule())

    env = _clean_env()
    result = run_proc(
        [sys.executable, "-m", "resolver.ingestion.dtm_client"],
        check=False,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0
    assert STAGING_CSV.exists(), "skip mode should emit a header-only CSV"

    with STAGING_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        assert header == list(CANONICAL_COLUMNS)
        assert not list(reader), "skip mode should only emit the header"

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["status"] == "skipped"
    assert summary["reason"] == "auth_missing"
    assert summary["rows_out"] == 0

    report_lines = _read_connectors_report()
    assert len(report_lines) == 1
    record = report_lines[0]
    assert record["status"] == "skipped"
    assert record["reason"] == "auth_missing"
    assert record["rows_out"] == 0
