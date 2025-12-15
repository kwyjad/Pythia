# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from scripts.ci import run_connectors


def test_dtm_env_flags_before_user_flags(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ONLY_CONNECTOR", "dtm_client")
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    captured: Dict[str, List[str]] = {}

    def fake_try_with_optional_debug(cmd: List[str], log_path: Path, env: Dict[str, str]) -> int:
        captured["cmd"] = cmd
        captured["env"] = env
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("log", encoding="utf-8")
        return 0

    monkeypatch.setattr(run_connectors, "try_with_optional_debug", fake_try_with_optional_debug)

    exit_code = run_connectors.main(
        [
            "--extra-env",
            "dtm_client=DTM_SOFT_TIMEOUTS=1",
            "--extra-args",
            "dtm_client=--no-date-filter",
        ]
    )
    assert exit_code == 0

    cmd = captured["cmd"]
    assert cmd[-1] == "--no-date-filter"
    assert "--soft-timeouts" in cmd
    assert cmd.index("--soft-timeouts") < cmd.index("--no-date-filter")
