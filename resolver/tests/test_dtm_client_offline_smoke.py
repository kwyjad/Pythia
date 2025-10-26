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
