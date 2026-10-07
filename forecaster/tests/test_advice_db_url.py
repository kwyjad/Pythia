# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""The prompt's advice loaders read PYTHIA_DB_URL before config (Oct 2026).

They read ``app.db_url`` alone, so a caller pointing ``PYTHIA_DB_URL`` at
another file had the advice read from ``data/resolver.duckdb`` behind its back;
``scripts/ci/advice_arm_text.py`` monkeypatched the helper to get round it.
"""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from forecaster import prompts


def test_env_url_wins_over_config(monkeypatch):
    monkeypatch.setenv("PYTHIA_DB_URL", "duckdb:///elsewhere.duckdb")
    assert prompts._pythia_db_url_from_config() == "duckdb:///elsewhere.duckdb"


def test_config_still_answers_without_the_env(monkeypatch):
    monkeypatch.delenv("PYTHIA_DB_URL", raising=False)
    monkeypatch.setattr(prompts, "_PYTHIA_CFG_LOAD", lambda: {"app": {"db_url": "duckdb:///cfg.duckdb"}})
    assert prompts._pythia_db_url_from_config() == "duckdb:///cfg.duckdb"


def test_shared_advice_is_read_from_the_env_database(tmp_path, monkeypatch):
    path = tmp_path / "advice.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE calibration_advice (hazard_code TEXT, metric TEXT, model_name TEXT, "
        "as_of_month TEXT, advice TEXT, findings_json TEXT, created_at TIMESTAMP)"
    )
    con.close()
    seen = []
    import resolver.db.duckdb_io as duckdb_io

    real = duckdb_io.get_db

    def spy(url, *a, **k):
        seen.append(url)
        return real(url, *a, **k)

    monkeypatch.setattr(duckdb_io, "get_db", spy)
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{path}")
    monkeypatch.setattr(prompts, "_PYTHIA_CFG_LOAD", lambda: {"app": {"db_url": "duckdb:///nowhere.duckdb"}})
    prompts._load_calibration_advice_for_hazard("ACE", "FATALITIES")
    assert seen and all(str(path) in u for u in seen)
