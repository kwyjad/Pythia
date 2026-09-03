# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""run_connectors must append to the connectors report, never blank it.

resolver_update.yml writes the IDMC HELIX record to the report one step
BEFORE this runner starts; truncating on entry erased that record on every
scheduled run, and the step summary then said IDMC was skipped beside a
successful HELIX pull.
"""

from __future__ import annotations

import importlib
import json
import os
import sys


def _reload(monkeypatch, tmp_path, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("scripts.ci.run_connectors", None)
    return importlib.import_module("scripts.ci.run_connectors")


def test_report_path_honours_the_env_var(monkeypatch, tmp_path):
    mod = _reload(monkeypatch, tmp_path, DIAGNOSTICS_REPORT_PATH="diag/custom.jsonl")
    assert str(mod.REPORT_PATH).endswith("diag/custom.jsonl")


def test_idmc_default_env_is_expanded_not_literal(monkeypatch, tmp_path):
    mod = _reload(monkeypatch, tmp_path, IDMC_REQ_PER_SEC="0.7")
    assert mod._IDMC_DEFAULT_ENV["IDMC_REQ_PER_SEC"] == "0.7"
    assert "${" not in mod._IDMC_DEFAULT_ENV["IDMC_MAX_CONCURRENCY"]


def test_existing_report_is_appended_to_unless_reset(monkeypatch, tmp_path):
    mod = _reload(monkeypatch, tmp_path, DIAGNOSTICS_REPORT_PATH="diagnostics/ingestion/connectors_report.jsonl")
    report = tmp_path / "diagnostics" / "ingestion" / "connectors_report.jsonl"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"connector_id": "idmc_helix", "status": "ok"}) + "\n")
    monkeypatch.setenv("CONNECTOR_LIST", "no_such_connector_xyz")
    monkeypatch.setenv("RESOLVER_SKIP_IDMC", "1")
    # Running a connector that does not exist still exercises the report reset.
    try:
        mod.main([])
    except SystemExit:
        pass
    except Exception:
        pass
    assert "idmc_helix" in report.read_text()
    try:
        mod.main(["--reset-report"])
    except SystemExit:
        pass
    except Exception:
        pass
    assert "idmc_helix" not in report.read_text()
