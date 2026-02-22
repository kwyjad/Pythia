# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import importlib
import logging
from pathlib import Path


def _prepare_runner(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("RUNNER_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("RESOLVER_FORCE_ENABLE", raising=False)
    monkeypatch.delenv("RESOLVER_INCLUDE_STUBS", raising=False)
    monkeypatch.delenv("RESOLVER_INGESTION_MODE", raising=False)
    monkeypatch.delenv("RESOLVER_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("RESOLVER_STAGING_DIR", raising=False)
    monkeypatch.delenv("RESOLVER_OUTPUT_PATH", raising=False)

    import resolver.ingestion.run_all_stubs as run_all_stubs

    module = importlib.reload(run_all_stubs)

    monkeypatch.setattr(module, "ROOT", tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(module, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(module, "LOGS_DIR", tmp_path / "logs")
    staging = tmp_path / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(module, "STAGING", staging)
    monkeypatch.setattr(module, "CONFIG_OVERRIDES", {
        "wfp_mvam": config_dir / "wfp_mvam_sources.yml",
    })
    monkeypatch.setattr(module, "SUMMARY_TARGETS", {})
    monkeypatch.setattr(module, "REAL", ["wfp_mvam_client.py"])
    monkeypatch.setattr(module, "STUBS", [])

    base_logger = logging.getLogger("resolver.ingestion.runner.test")
    base_logger.handlers.clear()
    base_logger.setLevel(logging.INFO)
    base_logger.propagate = True

    class _TestLoggerAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            extra = kwargs.get("extra") or {}
            merged = dict(self.extra)
            merged.update({k: v for k, v in dict(extra).items() if k != "name"})
            kwargs["extra"] = merged
            return msg, kwargs

    root_logger = _TestLoggerAdapter(base_logger, {"connector": "-"})

    def _init_logger(run_id, level=None, fmt=None, log_dir=None):
        return root_logger

    def _child_logger(connector_name: str):
        return _TestLoggerAdapter(base_logger, {"connector": connector_name})

    def _attach_handler(_logger, _connector):
        handler = logging.NullHandler()
        return handler

    def _detach_handler(_logger, handler):
        handler.close()

    monkeypatch.setattr(module, "init_logger", _init_logger)
    monkeypatch.setattr(module, "child_logger", _child_logger)
    monkeypatch.setattr(module, "attach_connector_handler", _attach_handler)
    monkeypatch.setattr(module, "detach_connector_handler", _detach_handler)

    for name in module.REAL:
        (tmp_path / name).write_text("", encoding="utf-8")

    return module


def test_wfp_mvam_disabled_by_config(monkeypatch, tmp_path, caplog):
    module = _prepare_runner(monkeypatch, tmp_path)

    config_path = module.CONFIG_DIR / "wfp_mvam_sources.yml"
    config_path.write_text("enable: false\n", encoding="utf-8")

    calls: list[str] = []

    def _record(spec, _logger):
        calls.append(spec.filename)
        return {"status": "ok", "rows": 0, "duration_ms": 0}

    monkeypatch.setattr(module, "_run_connector", _record)

    caplog.set_level(logging.INFO)

    exit_code = module.main([])

    assert exit_code == 0
    assert calls == []
    assert any(
        "connector=wfp_mvam" in record.message
        and f"config_path={config_path}" in record.message
        and "enable=False" in record.message
        and "gated_by=config" in record.message
        for record in caplog.records
    ), "expected enable log to show config gating"


def test_wfp_mvam_disabled_prints_skip_reason(monkeypatch, tmp_path, capsys):
    module = _prepare_runner(monkeypatch, tmp_path)

    config_path = module.CONFIG_DIR / "wfp_mvam_sources.yml"
    config_path.write_text("enable: false\n", encoding="utf-8")

    calls: list[str] = []

    def _record(spec, _logger):
        calls.append(spec.filename)
        return {"status": "ok", "rows": 0, "duration_ms": 0}

    monkeypatch.setattr(module, "_run_connector", _record)

    exit_code = module.main([])

    assert exit_code == 0
    assert calls == []

    captured = capsys.readouterr()
    assert "mode=skipped" in captured.out
    assert "reason=disabled: config" in captured.out
    assert "Connector totals" in captured.out


def test_wfp_mvam_forced_by_env(monkeypatch, tmp_path, caplog):
    module = _prepare_runner(monkeypatch, tmp_path)

    config_path = module.CONFIG_DIR / "wfp_mvam_sources.yml"
    config_path.write_text("enable: false\n", encoding="utf-8")

    monkeypatch.setenv("RESOLVER_FORCE_ENABLE", "wfp_mvam")

    calls: list[str] = []

    def _record(spec, _logger):
        calls.append(spec.filename)
        return {"status": "ok", "rows": 0, "duration_ms": 0}

    monkeypatch.setattr(module, "_run_connector", _record)

    caplog.set_level(logging.INFO)

    exit_code = module.main([])

    assert exit_code == 0
    assert calls == ["wfp_mvam_client.py"]
    assert any(
        "connector=wfp_mvam" in record.message
        and f"config_path={config_path}" in record.message
        and "enable=True" in record.message
        and "gated_by=forced_by_env" in record.message
        for record in caplog.records
    ), "expected enable log to show forced run"
