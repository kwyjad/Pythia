from __future__ import annotations

import io
from pathlib import Path

import pytest

from scripts.ci import run_connectors
from scripts.ci import summarize_connectors


class StubProcess:
    def __init__(self, returncode: int, output: str) -> None:
        self.stdout = io.StringIO(output)
        self.returncode = returncode

    def wait(self) -> None:
        return None


def _set_fake_popen(monkeypatch: pytest.MonkeyPatch, factory):
    monkeypatch.setattr(run_connectors.subprocess, "Popen", factory)


def test_run_connectors_retries_debug(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ONLY_CONNECTOR", "demo_client")
    monkeypatch.delenv("CONNECTOR_LIST", raising=False)
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    calls: list[list[str]] = []

    def fake_popen(cmd, stdout=None, stderr=None, env=None, text=None, bufsize=None):
        calls.append(list(cmd))
        if "--debug" in cmd:
            return StubProcess(2, "usage error\n")
        return StubProcess(0, "all good\n")

    _set_fake_popen(monkeypatch, fake_popen)

    rc = run_connectors.main()
    assert rc == 0
    assert len(calls) == 2

    log_path = tmp_path / "diagnostics" / "ingestion" / "logs" / "demo_client.log"
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert log_text.count("$ ") == 2
    assert "--debug" in log_text

    report_path = tmp_path / "diagnostics" / "ingestion" / "connectors_report.jsonl"
    entries = summarize_connectors.load_report(report_path)
    assert entries == sorted(entries, key=lambda item: item.get("connector_id"))
    entry = entries[0]
    assert entry["connector_id"] == "demo_client"
    assert entry["status"] == "ok"
    assert entry["extras"].get("exit_code") == 0
    assert entry["extras"].get("log_path", "").endswith("demo_client.log")


def test_run_connectors_propagates_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ONLY_CONNECTOR", raising=False)
    monkeypatch.setenv("CONNECTOR_LIST", "alpha_client,beta_client")
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    calls: list[list[str]] = []
    results = {
        "resolver.ingestion.alpha_client": (1, "alpha failed\n"),
        "resolver.ingestion.beta_client": (0, "beta ok\n"),
    }

    def fake_popen(cmd, stdout=None, stderr=None, env=None, text=None, bufsize=None):
        module = cmd[2]
        calls.append(list(cmd))
        rc, output = results[module]
        return StubProcess(rc, output)

    _set_fake_popen(monkeypatch, fake_popen)

    rc = run_connectors.main()
    assert rc == 1
    assert [cmd[2] for cmd in calls] == ["resolver.ingestion.alpha_client", "resolver.ingestion.beta_client"]

    logs_dir = tmp_path / "diagnostics" / "ingestion" / "logs"
    assert (logs_dir / "alpha_client.log").exists()
    assert (logs_dir / "beta_client.log").exists()

    report_path = tmp_path / "diagnostics" / "ingestion" / "connectors_report.jsonl"
    entries = summarize_connectors.load_report(report_path)
    statuses = {entry["connector_id"]: entry["status"] for entry in entries}
    assert statuses["alpha_client"] == "error"
    assert statuses["beta_client"] == "ok"
