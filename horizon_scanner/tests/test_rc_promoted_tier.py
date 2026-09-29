# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""An RC-promoted hazard is stored as rc_promoted, never as a quiet one.

``_RC_PROMOTED_DEFAULTS`` has always said ``tier="rc_promoted"``, but
``_write_hs_triage`` derived the tier from the 0.0 placeholder score and
wrote "quiet" on every such row, so the August 2026 scored bundle described
105 Track-1 questions as quiet hazards.
"""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from horizon_scanner import horizon_scanner as hs_mod  # noqa: E402
from horizon_scanner.triage import _RC_PROMOTED_DEFAULTS  # noqa: E402
from pythia.db import schema as pythia_schema  # noqa: E402
from scripts.ai_bundle.common import triage_view  # noqa: E402


def _db(tmp_path, monkeypatch):
    db_path = tmp_path / "hs.duckdb"
    con = duckdb.connect(str(db_path))
    pythia_schema.ensure_schema(con)
    con.close()
    monkeypatch.setattr(hs_mod, "pythia_connect", lambda *a, **k: duckdb.connect(str(db_path)))
    return db_path


def _promoted() -> dict:
    hz = dict(_RC_PROMOTED_DEFAULTS)
    hz["regime_change"] = {"likelihood": 0.6, "magnitude": 0.5, "direction": "up", "window": "month_1-2"}
    return hz


def test_rc_promoted_hazard_is_written_as_rc_promoted(tmp_path, monkeypatch):
    db_path = _db(tmp_path, monkeypatch)
    hs_mod._write_hs_triage(
        "run_1", "ETH",
        {"hazards": {
            "ACE": _promoted(),
            "FL": {"triage_score": 0.1, "drivers": [], "data_quality": {},
                   "regime_change": {"likelihood": 0.0, "magnitude": 0.0}},
        }},
    )
    con = duckdb.connect(str(db_path))
    rows = dict(
        (hz, (tier, score, track))
        for hz, tier, score, track in con.execute(
            "SELECT hazard_code, tier, triage_score, track FROM hs_triage WHERE iso3 = 'ETH'"
        ).fetchall()
    )
    con.close()
    tier, score, track = rows["ACE"]
    assert tier == "rc_promoted"
    assert track == 1
    # A genuinely triaged quiet hazard is still quiet.
    assert rows["FL"][0] == "quiet"


def test_bundle_view_corrects_legacy_quiet_rows():
    legacy = {"tier": "quiet", "triage_score": 0.0,
              "data_quality_json": '{"status": "rc_promoted"}'}
    assert triage_view(legacy) == ("rc_promoted", None)
    new = {"tier": "rc_promoted", "triage_score": 0.0, "data_quality_json": None}
    assert triage_view(new) == ("rc_promoted", None)
    triaged = {"tier": "priority", "triage_score": 0.8, "data_quality_json": "{}"}
    assert triage_view(triaged) == ("priority", 0.8)
    assert triage_view(None) == (None, None)
