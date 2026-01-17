import json

import pytest


duckdb = pytest.importorskip("duckdb")

from horizon_scanner import horizon_scanner as hs_mod
from pythia.db import schema as pythia_schema


def _triage_payload() -> dict:
    return {
        "hazards": {
            "ACE": {
                "triage_score": 0.10,
                "tier": "quiet",
                "drivers": ["driver"],
                "regime_shifts": [],
                "data_quality": {"resolution_source": "ACLED", "reliability": "low"},
                "scenario_stub": "stub",
                "regime_change": {
                    "likelihood": 0.80,
                    "magnitude": 0.60,
                    "direction": "up",
                    "window": "month_3-4",
                    "rationale_bullets": ["signal A"],
                    "trigger_signals": [
                        {
                            "signal": "trigger A",
                            "timeframe_months": 3,
                            "evidence_refs": ["E1"],
                        }
                    ],
                },
            }
        }
    }


def test_write_hs_triage_regime_change_persists_new_columns(tmp_path, monkeypatch):
    db_path = tmp_path / "hs_triage_regime_change.duckdb"
    con = duckdb.connect(str(db_path))
    pythia_schema.ensure_schema(con)
    con.close()

    def _connect(*_args, **_kwargs):
        return duckdb.connect(str(db_path))

    monkeypatch.setattr(hs_mod, "pythia_connect", _connect)

    hs_mod._write_hs_triage("run_1", "tst", _triage_payload())

    con = duckdb.connect(str(db_path))
    row = con.execute(
        """
        SELECT need_full_spd,
               regime_change_score,
               regime_change_level,
               regime_change_direction,
               regime_change_window,
               regime_change_json
        FROM hs_triage
        WHERE iso3 = 'TST' AND hazard_code = 'ACE'
        """
    ).fetchone()
    con.close()

    assert row is not None
    need_full_spd, score, level, direction, window, regime_json = row
    assert need_full_spd is True
    assert score == pytest.approx(0.48, abs=1e-6)
    assert level == 3
    assert direction == "up"
    assert window == "month_3-4"
    parsed = json.loads(regime_json)
    assert parsed.get("likelihood") == 0.8


def test_write_hs_triage_falls_back_when_columns_missing(tmp_path, monkeypatch):
    db_path = tmp_path / "hs_triage_legacy.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE hs_triage (
            run_id TEXT NOT NULL,
            iso3 TEXT NOT NULL,
            hazard_code TEXT NOT NULL,
            tier TEXT NOT NULL,
            triage_score DOUBLE NOT NULL,
            need_full_spd BOOLEAN NOT NULL,
            drivers_json TEXT,
            regime_shifts_json TEXT,
            data_quality_json TEXT,
            scenario_stub TEXT
        );
        """
    )
    con.close()

    def _connect(*_args, **_kwargs):
        return duckdb.connect(str(db_path))

    monkeypatch.setattr(hs_mod, "pythia_connect", _connect)

    hs_mod._write_hs_triage("run_2", "tst", _triage_payload())

    con = duckdb.connect(str(db_path))
    row = con.execute(
        """
        SELECT need_full_spd
        FROM hs_triage
        WHERE iso3 = 'TST' AND hazard_code = 'ACE'
        """
    ).fetchone()
    con.close()

    assert row is not None
    assert row[0] is True
