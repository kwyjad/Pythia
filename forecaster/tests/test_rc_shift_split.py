# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""The RC shift guidance as a split test (PYTHIA_RC_SHIFT_SHARE, Oct 2026).

Half of Track 1 SPD questions, chosen by a salted hash of the question id,
carry the shift guidance; the other half carry the legacy wording, byte for
byte. The arm is independent of the advice arm, recorded on the member rows,
and read by the scored and attribution bundles.
"""

from __future__ import annotations

import hashlib

import pytest

duckdb = pytest.importorskip("duckdb")

from forecaster import prompts

SHIFT_Q = "C01_ACE_FATALITIES_2026-12"
CONTROL_Q = "C00_ACE_FATALITIES_2026-12"
_Q = {
    "iso3": "SOM", "hazard_code": "ACE", "metric": "FATALITIES",
    "wording": "How many conflict deaths will Somalia record?",
    "window_start_date": "2026-12-01", "target_month": "2027-05",
}
_HIST = {"source": "acled", "summary": "24 months of data"}
_TRI = {"regime_change_level": 2, "regime_change_direction": "up",
        "regime_change_likelihood": 0.5, "regime_change_magnitude": 0.4, "triage_score": 0.6}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    missing = f"duckdb:///{tmp_path / 'absent.duckdb'}"
    monkeypatch.setenv("PYTHIA_DB_URL", missing)
    monkeypatch.setenv("RESOLVER_DB_URL", missing)
    monkeypatch.setenv("PYTHIA_PROMPT_V3_ORDER", "1")
    for var in ("PYTHIA_RC_SHIFT_GUIDANCE", "PYTHIA_RC_SHIFT_SHARE", "PYTHIA_ADVICE_EXPERIMENT_SHARE"):
        monkeypatch.delenv(var, raising=False)


def _prompt(qid, track=1):
    return prompts.build_spd_prompt_v2(dict(_Q, question_id=qid), _HIST, _TRI, {}, track=track)


def test_arms_follow_the_salted_hash(monkeypatch):
    assert prompts.rc_shift_arm(SHIFT_Q) is None  # test off
    monkeypatch.setenv("PYTHIA_RC_SHIFT_SHARE", "0.5")
    assert prompts.rc_shift_arm(SHIFT_Q) == "shift"
    assert prompts.rc_shift_arm(CONTROL_Q) == "control"
    assert prompts.rc_guidance_version(1, SHIFT_Q) == prompts.RC_SHIFT_GUIDANCE_VERSION
    assert prompts.rc_guidance_version(1, CONTROL_Q) is None
    assert prompts.rc_guidance_version(2, SHIFT_Q) is None


def test_the_rc_arm_is_independent_of_the_advice_arm(monkeypatch):
    monkeypatch.setenv("PYTHIA_RC_SHIFT_SHARE", "0.5")
    monkeypatch.setenv("PYTHIA_ADVICE_EXPERIMENT_SHARE", "0.5")
    cells: dict[tuple, int] = {}
    for i in range(2000):
        q = f"Q{i}_ACE_FATALITIES_2026-12"
        key = (prompts.advice_arm(q), prompts.rc_shift_arm(q))
        cells[key] = cells.get(key, 0) + 1
    assert len(cells) == 4
    assert min(cells.values()) > 400  # roughly 500 each; an unsalted hash would give two cells


def test_control_prompt_is_the_legacy_prompt_byte_for_byte(monkeypatch):
    legacy = _prompt(CONTROL_Q)
    monkeypatch.setenv("PYTHIA_RC_SHIFT_SHARE", "0.5")
    assert _prompt(CONTROL_Q) == legacy
    shifted = _prompt(SHIFT_Q)
    assert "MOVE THE DISTRIBUTION" in shifted
    assert "MOVE THE DISTRIBUTION" not in legacy


def test_the_flag_still_turns_it_on_for_everyone(monkeypatch):
    monkeypatch.setenv("PYTHIA_RC_SHIFT_SHARE", "0.5")
    monkeypatch.setenv("PYTHIA_RC_SHIFT_GUIDANCE", "1")
    assert prompts.rc_shift_arm(CONTROL_Q) is None
    assert "MOVE THE DISTRIBUTION" in _prompt(CONTROL_Q)


def test_the_stamp_writes_the_arm_on_track1_rows_only(monkeypatch, tmp_path):
    from forecaster import cli

    path = tmp_path / "f.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE forecasts_raw (run_id TEXT, question_id TEXT, model_name TEXT, rc_shift_arm TEXT)")
    con.execute("INSERT INTO forecasts_raw VALUES ('r1', ?, 'm', NULL), ('r1', ?, 'm', NULL)", [SHIFT_Q, CONTROL_Q])
    con.close()
    monkeypatch.setattr(cli, "connect", lambda read_only=False: duckdb.connect(str(path)))
    monkeypatch.setenv("PYTHIA_RC_SHIFT_SHARE", "0.5")
    cli._stamp_rc_shift_arm("r1", SHIFT_Q, {"track": 1, "metric": "FATALITIES"})
    cli._stamp_rc_shift_arm("r1", CONTROL_Q, {"track": 2})
    got = dict(duckdb.connect(str(path)).execute(
        "SELECT question_id, rc_shift_arm FROM forecasts_raw").fetchall())
    assert got == {SHIFT_Q: "shift", CONTROL_Q: None}


def test_a_binary_question_is_never_stamped(monkeypatch, tmp_path):
    """The shift guidance is in the SPD prompt only. The 7 Oct 2026 rehearsal
    stamped four EVENT_OCCURRENCE questions with an arm they never received."""
    from forecaster import cli

    path = tmp_path / "f.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE forecasts_raw (run_id TEXT, question_id TEXT, model_name TEXT, rc_shift_arm TEXT)")
    binary_q = "C01_FL_EVENT_OCCURRENCE_2026-12"
    con.execute("INSERT INTO forecasts_raw VALUES ('r1', ?, 'm', NULL)", [binary_q])
    con.close()
    monkeypatch.setattr(cli, "connect", lambda read_only=False: duckdb.connect(str(path)))
    monkeypatch.setenv("PYTHIA_RC_SHIFT_SHARE", "1.0")
    cli._stamp_rc_shift_arm("r1", binary_q, {"track": 1, "metric": "EVENT_OCCURRENCE"})
    assert duckdb.connect(str(path)).execute(
        "SELECT rc_shift_arm FROM forecasts_raw").fetchone()[0] is None


def test_the_assignment_is_balanced_over_a_large_set_of_question_ids(monkeypatch):
    """The 7 Oct 2026 rehearsal put 2 of 11 SPD questions in the shift arm.
    That is chance (P <= 0.033 one-sided for an unbiased coin at n = 11), not
    a biased hash: over 60,000 ids shaped like real ones the shift share sits
    within three standard errors of 0.5, overall and per hazard and metric."""
    import math

    monkeypatch.setenv("PYTHIA_RC_SHIFT_SHARE", "0.5")
    groups: dict[str, list[int]] = {}
    for i in range(1000):
        iso3 = f"{chr(65 + i // 676 % 26)}{chr(65 + i // 26 % 26)}{chr(65 + i % 26)}"
        for hz_metric in ("ACE_FATALITIES", "ACE_PA", "FL_PA", "TC_PA", "DR_PHASE3PLUS_IN_NEED"):
            for month in ("2026-11", "2026-12", "2027-01", "2027-02", "2027-03", "2027-04",
                          "2027-05", "2027-06", "2027-07", "2027-08", "2027-09", "2027-10"):
                qid = f"{iso3}_{hz_metric}_{month}"
                groups.setdefault(hz_metric, []).append(prompts.rc_shift_arm(qid) == "shift")
    every = [x for v in groups.values() for x in v]
    for flags in list(groups.values()) + [every]:
        n = len(flags)
        z = (sum(flags) - n / 2) / math.sqrt(n / 4)
        assert abs(z) < 3, (n, sum(flags), z)


def test_the_attribution_split_skips_binary_questions():
    from scripts.ai_bundle.build_forecast_attribution_bundle import build_rc_shift_arms

    binary_q = "C01_FL_EVENT_OCCURRENCE_2026-12"
    deviation = {binary_q: {"ensemble_mean_v2": {"baserate_json": '{"probs": [0.2, 0.8]}'}}}
    qs = [dict(_Q, question_id=binary_q, hazard_code="FL", metric="EVENT_OCCURRENCE")]
    assert build_rc_shift_arms(qs, deviation, {(binary_q, "m"): ([0.3, 0.7], "shift")}) == []


def test_attribution_table_per_arm():
    from scripts.ai_bundle.build_forecast_attribution_bundle import build_rc_shift_arms

    anchor = [0.1, 0.6, 0.3]
    deviation = {q: {"ensemble_mean_v2": {"baserate_json": '{"probs": [0.1, 0.6, 0.3]}'}}
                 for q in (SHIFT_Q, CONTROL_Q)}
    month1 = {(SHIFT_Q, "m"): ([0.0, 0.4, 0.6], "shift"), (CONTROL_Q, "m"): (anchor, "control")}
    qs = [dict(_Q, question_id=SHIFT_Q), dict(_Q, question_id=CONTROL_Q)]
    rows = {r["rc_shift_arm"]: r for r in build_rc_shift_arms(qs, deviation, month1)}
    assert rows["control"]["mean_delta_expected_bucket"] == 0.0
    assert rows["shift"]["mean_delta_expected_bucket"] == pytest.approx(0.4)
    assert rows["shift"]["mean_delta_modal_mass"] == pytest.approx(-0.2)
