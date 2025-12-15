# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import runpy
import sys

import pytest


def test_dtm_offline_smoke_runs_without_dtmapi(monkeypatch):
    """Ensure offline smoke path doesn't require dtmapi."""
    monkeypatch.delitem(sys.modules, "dtmapi", raising=False)
    argv = sys.argv[:]
    try:
        sys.argv = ["resolver.ingestion.dtm_client", "--offline-smoke"]
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("resolver.ingestion.dtm_client", run_name="__main__")
    finally:
        sys.argv = argv
    assert excinfo.value.code == 0
