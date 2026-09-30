# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Family recalibration inside the forecaster: what is stored, what votes."""

from __future__ import annotations

import json
from datetime import datetime

import duckdb
import pytest

from forecaster import cli
from forecaster.providers import ModelSpec
from pythia.tools import family_recalibration as fr

MONTHS = ["2026-10", "2026-11", "2026-12", "2027-01", "2027-02", "2027-03"]
Q = {
    "question_id": "SOM_ACE_FATALITIES_2026-10",
    "iso3": "SOM",
    "hazard_code": "ACE",
    "metric": "FATALITIES",
    "window_start_date": "2026-10-01",
    "target_month": "2027-03",
}
PROBS = [0.3, 0.2, 0.2, 0.15, 0.1, 0.03, 0.02]


@pytest.fixture
def factors_db(tmp_path, monkeypatch):
    path = tmp_path / "factors.duckdb"
    con = duckdb.connect(str(path))
    fr.ensure_table(con)
    for b in range(1, 8):
        con.execute(
            "INSERT INTO family_recalibration VALUES "
            "('gpt','ACE','FATALITIES','spd',?,?,20,NULL,NULL,'2026-09',?,FALSE)",
            [b, 2.0 if b == 5 else 1.0, datetime(2026, 9, 28)],
        )
    con.execute(
        "INSERT INTO family_recalibration VALUES "
        "('gpt','FL','EVENT_OCCURRENCE','binary',0,1.0,20,NULL,NULL,'2026-09',?,FALSE)",
        [datetime(2026, 9, 28)],
    )
    con.close()
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{path}")
    fr.reset_factor_cache()
    yield
    fr.reset_factor_cache()


def _specs():
    return [
        ModelSpec(name="gpt-6-sol", provider="openai", model_id="gpt-6-sol", active=True),
        ModelSpec(name="claude-opus-5-5", provider="anthropic", model_id="claude-opus-5-5", active=True),
    ]


def _write(tmp_path, monkeypatch, *, brbv=None):
    db = str(tmp_path / "w.duckdb")
    monkeypatch.setattr(cli, "connect", lambda read_only=False: duckdb.connect(db))
    specs = _specs()
    spd = {m: list(PROBS) for m in MONTHS}
    cli._write_spd_members_v2_to_db(
        run_id="r1",
        question_row=dict(Q),
        specs_used=specs,
        per_model_spds=[dict(spd), dict(spd)],
        raw_calls=[{"usage": {}, "model_spec": s} for s in specs],
        resolution_source="ACLED",
        base_rate_block_version=brbv,
        recalibrate=True,
    )
    return duckdb.connect(db)


def _month1(con, model):
    rows = con.execute(
        "SELECT bucket_index, probability FROM forecasts_raw "
        "WHERE model_name = ? AND month_index = 1 ORDER BY bucket_index",
        [model],
    ).fetchall()
    return [p for _b, p in rows]


def test_off_writes_exactly_what_it_wrote_before(tmp_path, monkeypatch, factors_db):
    monkeypatch.delenv("PYTHIA_FAMILY_RECALIBRATION_MODE", raising=False)
    con = _write(tmp_path, monkeypatch)
    names = {r[0] for r in con.execute("SELECT DISTINCT model_name FROM forecasts_raw").fetchall()}
    assert names == {"gpt-6-sol", "claude-opus-5-5"}
    assert _month1(con, "gpt-6-sol") == pytest.approx(PROBS)
    assert con.execute(
        "SELECT COUNT(*) FROM forecasts_raw WHERE recalibration_json IS NOT NULL"
    ).fetchone()[0] == 0


def test_apply_stores_the_correction_and_keeps_the_raw_forecast(tmp_path, monkeypatch, factors_db):
    monkeypatch.setenv("PYTHIA_FAMILY_RECALIBRATION_MODE", "apply")
    con = _write(tmp_path, monkeypatch)
    names = {r[0] for r in con.execute("SELECT DISTINCT model_name FROM forecasts_raw").fetchall()}
    assert names == {"gpt-6-sol", "gpt-6-sol__raw", "claude-opus-5-5"}
    corrected = _month1(con, "gpt-6-sol")
    assert sum(corrected) == pytest.approx(1.0)
    assert corrected[4] == pytest.approx(0.2 / 1.1)
    assert _month1(con, "gpt-6-sol__raw") == pytest.approx(PROBS)
    # A family with no factors is written as it came.
    assert _month1(con, "claude-opus-5-5") == pytest.approx(PROBS)
    meta = json.loads(con.execute(
        "SELECT ANY_VALUE(recalibration_json) FROM forecasts_raw WHERE model_name='gpt-6-sol'"
    ).fetchone()[0])
    assert meta["applied"] is True and meta["family"] == "gpt" and meta["row"] == "corrected"
    raw_json = json.loads(con.execute(
        "SELECT ANY_VALUE(spd_json) FROM forecasts_raw WHERE model_name='gpt-6-sol__raw'"
    ).fetchone()[0])
    assert raw_json.get("shadow") is True  # the copy never votes


def test_shadow_stores_the_correction_beside_the_forecast(tmp_path, monkeypatch, factors_db):
    monkeypatch.setenv("PYTHIA_FAMILY_RECALIBRATION_MODE", "shadow")
    con = _write(tmp_path, monkeypatch)
    assert _month1(con, "gpt-6-sol") == pytest.approx(PROBS)
    assert _month1(con, "gpt-6-sol__recal")[4] == pytest.approx(0.2 / 1.1)


def test_another_prompt_version_is_shadowed_under_apply(tmp_path, monkeypatch, factors_db):
    monkeypatch.setenv("PYTHIA_FAMILY_RECALIBRATION_MODE", "apply")
    con = _write(tmp_path, monkeypatch, brbv="prior_anchor_v1")
    assert _month1(con, "gpt-6-sol") == pytest.approx(PROBS)
    assert _month1(con, "gpt-6-sol__recal")[4] == pytest.approx(0.2 / 1.1)
    meta = json.loads(con.execute(
        "SELECT ANY_VALUE(recalibration_json) FROM forecasts_raw WHERE model_name='gpt-6-sol__recal'"
    ).fetchone()[0])
    assert meta["mode"] == "auto_shadow"


def test_voting_members_are_corrected_only_under_apply(monkeypatch, factors_db):
    specs = _specs()
    spds = [{m: list(PROBS) for m in MONTHS} for _ in specs]
    for mode, expect in (("off", 0.1), ("shadow", 0.1), ("apply", 0.2 / 1.1)):
        monkeypatch.setenv("PYTHIA_FAMILY_RECALIBRATION_MODE", mode)
        out = cli._recalibrate_voting(spds, specs, "ACE", "FATALITIES",
                                      base_rate_block_version=None, rc_guidance=None)
        assert out[0]["2026-10"][4] == pytest.approx(expect)
        assert out[1]["2026-10"][4] == pytest.approx(0.1)


def test_a_derived_copy_never_votes():
    raw = ModelSpec(name="gpt-6-sol__raw", provider="openai", model_id="gpt-6-sol",
                    active=True, shadow=True)
    spds, specs = cli._voting_members([{"m": [1.0]}, {"m": [1.0]}], [_specs()[0], raw])
    assert [s.name for s in specs] == ["gpt-6-sol"]


def test_binary_members_are_shifted_before_pooling(monkeypatch, factors_db):
    monkeypatch.setenv("PYTHIA_FAMILY_RECALIBRATION_MODE", "apply")
    specs = _specs()
    probs = [{"2026-10": 0.2}, {"2026-10": 0.2}]
    members = [(s.name, dict(p), {}, None) for s, p in zip(specs, probs)]
    shadow: set = set()
    new_probs, new_members, metas = cli._recalibrate_binary_members(
        probs, specs, members, "FL", "EVENT_OCCURRENCE", shadow,
    )
    assert new_probs[0]["2026-10"] == pytest.approx(fr.apply_binary_shift(0.2, 1.0))
    assert new_probs[1]["2026-10"] == pytest.approx(0.2)
    assert "gpt-6-sol__raw" in shadow
    assert ("gpt-6-sol__raw", {"2026-10": 0.2}, {}, None) in new_members
    assert metas["gpt-6-sol"]["applied"] is True


def test_derived_copies_take_no_calibration_weight():
    from pythia.tools.compute_calibration_pythia import Sample, _compute_weights_for_group

    samples = []
    for i in range(30):
        for name, v in (("gpt-6-sol", 0.3), ("gpt-6-sol__raw", 0.1), ("gpt-6-sol__recal", 0.05)):
            samples.append(Sample((f"q{i}", "1", "", ""), "ACE", "FATALITIES", name, "brier", v, "2026-08"))
    rows, _msg = _compute_weights_for_group("2026-09", samples)
    names = {str(r.get("model_name")) for r in rows}
    assert names and not any(n.endswith(("__raw", "__recal")) for n in names)
