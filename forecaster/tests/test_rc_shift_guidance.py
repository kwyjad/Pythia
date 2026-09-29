# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""PYTHIA_RC_SHIFT_GUIDANCE: Track 1 prompts ask members to MOVE, not WIDEN.

Pinned here:

* flag off (unset or "0") leaves every prompt exactly as it was — no new text,
  the legacy "widen posterior" lines still present;
* flag on changes Track 1 only, renders the direction rule for up / down /
  mixed, states the base-rate sharpness anchor, adds ``rc_shift`` to the JSON
  schema, and keeps the V3 static prefix identical across questions;
* ``rc_shift`` parsing survives absence and malformation;
* the sharpness check flags spread-without-shift and nothing else;
* the member writer stamps ``forecasts_raw.rc_guidance``.
"""

from __future__ import annotations

import json

import pytest

_Q = {
    "question_id": "SOM_ACE_FATALITIES_2026-10",
    "iso3": "SOM",
    "hazard_code": "ACE",
    "metric": "FATALITIES",
    "wording": "How many conflict deaths will Somalia record?",
    "window_start_date": "2026-10-01",
    "target_month": "2027-03",
}
_Q_YEM = dict(_Q, question_id="YEM_ACE_FATALITIES_2026-10", iso3="YEM",
              wording="How many conflict deaths will Yemen record?")
_HIST = {"source": "acled", "summary": "24 months of data"}


def _triage(direction: str = "up", level: int = 2) -> dict:
    return {
        "regime_change_level": level,
        "regime_change_direction": direction,
        "regime_change_likelihood": 0.5,
        "regime_change_magnitude": 0.4,
        "triage_score": 0.6,
    }


@pytest.fixture(autouse=True)
def _no_db(monkeypatch, tmp_path):
    # No database: every loader degrades, so prompts are deterministic.
    missing = f"duckdb:///{tmp_path / 'absent.duckdb'}"
    monkeypatch.setenv("PYTHIA_DB_URL", missing)
    monkeypatch.setenv("RESOLVER_DB_URL", missing)
    monkeypatch.delenv("PYTHIA_RC_SHIFT_GUIDANCE", raising=False)


def _build(monkeypatch, *, flag=None, v3="1", track=1, q=_Q, tri=None, parts=False):
    from forecaster import prompts

    if flag is None:
        monkeypatch.delenv("PYTHIA_RC_SHIFT_GUIDANCE", raising=False)
    else:
        monkeypatch.setenv("PYTHIA_RC_SHIFT_GUIDANCE", flag)
    monkeypatch.setenv("PYTHIA_PROMPT_V3_ORDER", v3)
    return prompts.build_spd_prompt_v2(q, _HIST, tri or _triage(), {}, track=track, return_parts=parts)


# ---------------------------------------------------------------------------
# Flag off: nothing moves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v3", ["0", "1"])
@pytest.mark.parametrize("track", [1, 2])
def test_flag_off_and_flag_zero_are_the_same_bytes(monkeypatch, v3, track):
    unset = _build(monkeypatch, flag=None, v3=v3, track=track)
    zero = _build(monkeypatch, flag="0", v3=v3, track=track)
    assert unset == zero


@pytest.mark.parametrize("v3", ["0", "1"])
def test_flag_off_keeps_the_legacy_guidance_and_adds_nothing(monkeypatch, v3):
    p = _build(monkeypatch, flag="0", v3=v3)
    assert "widen posterior" in p
    assert "(widened/shifted/rebutted)" in p
    for new_text in ("REGIME-CHANGE FLAG: MOVE", "rc_shift", "Sharpness anchor", "Direction UP"):
        assert new_text not in p


def test_flag_off_never_touches_the_database(monkeypatch):
    from forecaster import prompts

    def _boom(_q):  # pragma: no cover - must not be called
        raise AssertionError("base-rate anchor loaded with the flag off")

    monkeypatch.setattr(prompts, "_load_base_rate_modal", _boom)
    _build(monkeypatch, flag="0")


# ---------------------------------------------------------------------------
# Flag on
# ---------------------------------------------------------------------------


def test_flag_on_changes_track_1_only(monkeypatch):
    assert _build(monkeypatch, flag="1", track=2) == _build(monkeypatch, flag="0", track=2)
    assert _build(monkeypatch, flag="1", track=1) != _build(monkeypatch, flag="0", track=1)


def test_flag_on_replaces_widening_with_shifting(monkeypatch):
    p = _build(monkeypatch, flag="1")
    assert "widen posterior" not in p
    assert "REGIME-CHANGE FLAG: MOVE THE DISTRIBUTION, DO NOT WIDEN IT (months 1-3)" in p
    assert "(months 4-6)" in p  # later-month widening kept separate
    assert '"rc_shift": {"direction": "up or down or none or two_sided"' in p
    assert "`reasoning_trace.rc_shift` states how" in p
    assert "(shifted up/shifted down/two-sided/rebutted)" in p


def test_up_rule(monkeypatch):
    p = _build(monkeypatch, flag="1", tri=_triage("up"))
    assert "Direction UP: move mass to higher buckets" in p
    assert "Direction DOWN" not in p


def test_down_rule(monkeypatch):
    p = _build(monkeypatch, flag="1", tri=_triage("down"))
    assert "Direction DOWN: move mass to lower buckets" in p
    assert "Direction UP" not in p


@pytest.mark.parametrize("direction", ["mixed", "unclear", ""])
def test_two_sided_rule(monkeypatch, direction):
    p = _build(monkeypatch, flag="1", tri=_triage(direction))
    assert "Direction mixed or unclear" in p
    assert "only modestly" in p


def test_no_flag_level_zero_says_so(monkeypatch):
    p = _build(monkeypatch, flag="1", tri=_triage("up", level=0))
    assert "No regime change flagged: stay with the base rate" in p
    assert "Sharpness anchor" not in p


def test_anchor_states_the_base_rate_modal_mass(monkeypatch):
    from forecaster import prompts

    # FATALITIES buckets: 0, 1-<5, 5-<25, 25-<100, 100-<500, 500-<1000, >=1000
    probs = [0.02, 0.03, 0.10, 0.20, 0.45, 0.15, 0.05]
    monkeypatch.setattr(prompts, "_load_base_rate_modal", lambda q: (4, 0.45, probs))
    up = _build(monkeypatch, flag="1", tri=_triage("up"))
    assert 'puts 45% on its modal bucket "100-<500"' in up
    assert 'at least 45% on the "100-<500" and "500-<1000" buckets together' in up
    down = _build(monkeypatch, flag="1", tri=_triage("down"))
    assert 'on the "100-<500" and "25-<100" buckets together' in down
    mixed = _build(monkeypatch, flag="1", tri=_triage("mixed"))
    assert 'the "100-<500" bucket and its immediate neighbours together' in mixed


def test_anchor_falls_back_to_the_prior_without_a_base_rate(monkeypatch):
    p = _build(monkeypatch, flag="1")
    assert "your Step 1 prior's modal-bucket probability" in p


def test_v3_prefix_is_shared_across_questions_with_the_flag_on(monkeypatch):
    pre_som, suf_som = _build(monkeypatch, flag="1", q=_Q, parts=True)
    pre_yem, suf_yem = _build(monkeypatch, flag="1", q=_Q_YEM, tri=_triage("down"), parts=True)
    assert pre_som == pre_yem
    assert "REGIME-CHANGE FLAG: MOVE" in pre_som  # static half in the cached prefix
    assert "Direction UP" in suf_som and "Direction DOWN" in suf_yem  # per-question half in the tail


def test_rc_guidance_version(monkeypatch):
    from forecaster.prompts import RC_SHIFT_GUIDANCE_VERSION, rc_guidance_version

    monkeypatch.setenv("PYTHIA_RC_SHIFT_GUIDANCE", "0")
    assert rc_guidance_version(1) is None
    monkeypatch.setenv("PYTHIA_RC_SHIFT_GUIDANCE", "1")
    assert rc_guidance_version(1) == RC_SHIFT_GUIDANCE_VERSION == "shift_v1"
    assert rc_guidance_version(2) is None


# ---------------------------------------------------------------------------
# rc_shift parsing
# ---------------------------------------------------------------------------


def test_parse_rc_shift_ok():
    from forecaster.trace_validation import parse_rc_shift

    value, status = parse_rc_shift({
        "direction": "UP", "expected_bucket_change": "0.4", "mass_moved": 0.15,
        "sharpness_kept": "true", "why": "  Ceasefire collapsed.  ",
    })
    assert status == "ok"
    assert value == {"direction": "up", "expected_bucket_change": 0.4, "mass_moved": 0.15,
                     "sharpness_kept": True, "why": "Ceasefire collapsed."}


def test_parse_rc_shift_missing():
    from forecaster.trace_validation import normalise_rc_shift_in_trace, parse_rc_shift

    assert parse_rc_shift(None) == (None, "absent")
    trace = {"prior": {"spd": [1.0]}}
    assert normalise_rc_shift_in_trace(trace) == {"prior": {"spd": [1.0]}}  # untouched


@pytest.mark.parametrize(
    "raw, bad",
    [
        ("up", "not_an_object"),
        ({"direction": "sideways", "expected_bucket_change": 0, "mass_moved": 0.1,
          "sharpness_kept": True, "why": "x"}, "direction"),
        ({"direction": "down", "expected_bucket_change": "a lot", "mass_moved": 1.5,
          "sharpness_kept": "maybe", "why": ""},
         "expected_bucket_change,mass_moved,sharpness_kept,why"),
    ],
)
def test_parse_rc_shift_malformed(raw, bad):
    from forecaster.trace_validation import normalise_rc_shift_in_trace

    trace = normalise_rc_shift_in_trace({"rc_shift": raw})
    assert trace["rc_shift_status"] == f"malformed:{bad}"
    assert trace["rc_shift_raw"] == raw


def test_mixed_direction_normalises_to_two_sided():
    from forecaster.trace_validation import parse_rc_shift

    value, _ = parse_rc_shift({"direction": "mixed"})
    assert value["direction"] == "two_sided"


# ---------------------------------------------------------------------------
# The sharpness check
# ---------------------------------------------------------------------------

_PRIOR = [0.02, 0.05, 0.10, 0.20, 0.45, 0.13, 0.05]


def test_spread_without_shift_is_flagged():
    from forecaster.trace_validation import check_rc_sharpness

    # Modal mass 0.45 -> 0.25, moved to both sides so the mean stays put.
    spread = [0.02, 0.05, 0.20, 0.20, 0.25, 0.13, 0.15]
    out = check_rc_sharpness({"prior": {"spd": _PRIOR}}, 7, posterior=spread)
    assert out["checked"] and out["spread_without_shift"] is True
    assert out["modal_mass_loss"] > 0.25


def test_a_real_shift_is_not_flagged():
    from forecaster.trace_validation import check_rc_sharpness

    shifted = [0.01, 0.02, 0.05, 0.12, 0.30, 0.35, 0.15]
    out = check_rc_sharpness({"prior": {"spd": _PRIOR}}, 7, posterior=shifted)
    assert out["spread_without_shift"] is False
    assert out["expected_bucket_shift"] > 0.25


def test_an_unchanged_posterior_is_not_flagged_and_defaults_to_the_last_update():
    from forecaster.trace_validation import check_rc_sharpness

    trace = {"prior": {"spd": _PRIOR}, "updates": [{"post_update_spd": _PRIOR}]}
    out = check_rc_sharpness(trace, 7)
    assert out["spread_without_shift"] is False and out["modal_mass_loss"] == 0.0


def test_the_check_never_raises_on_junk():
    from forecaster.trace_validation import check_rc_sharpness, validate_reasoning_traces

    assert check_rc_sharpness({"prior": {"spd": "x"}}, 7)["checked"] is False
    assert check_rc_sharpness({"prior": {"spd": _PRIOR}}, 7)["checked"] is False
    res = validate_reasoning_traces(
        [{"model_spec": None, "reasoning_trace": {"prior": {"spd": _PRIOR}, "updates": []}}],
        {}, "ACE", "FATALITIES",
    )
    assert "rc_sharpness" in res[0]


# ---------------------------------------------------------------------------
# forecasts_raw.rc_guidance
# ---------------------------------------------------------------------------


def test_member_writer_stamps_rc_guidance(monkeypatch, tmp_path):
    import duckdb

    from forecaster import cli
    from forecaster.providers import ModelSpec

    db = str(tmp_path / "w.duckdb")
    monkeypatch.setattr(cli, "connect", lambda read_only=False: duckdb.connect(db))
    spec = ModelSpec(name="m1", provider="openai", model_id="m1", active=True)
    probs = [0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1]
    month_spds = {f"2026-{m:02d}": probs for m in range(10, 13)}
    month_spds.update({f"2027-{m:02d}": probs for m in range(1, 4)})
    trace = {"prior": {"spd": probs}, "rc_shift": {"direction": "up"}}
    for version, run in ((None, "r0"), ("shift_v1", "r1")):
        cli._write_spd_members_v2_to_db(
            run_id=run,
            question_row=dict(_Q),
            specs_used=[spec],
            per_model_spds=[month_spds],
            raw_calls=[{"usage": {}, "reasoning_trace": trace}],
            resolution_source="ACLED",
            rc_guidance=version,
        )
    con = duckdb.connect(db)
    got = dict(con.execute(
        "SELECT run_id, ANY_VALUE(rc_guidance) FROM forecasts_raw GROUP BY 1"
    ).fetchall())
    assert got == {"r0": None, "r1": "shift_v1"}
    stored = json.loads(con.execute(
        "SELECT reasoning_trace_json FROM forecasts_raw WHERE run_id='r1' LIMIT 1"
    ).fetchone()[0])
    assert stored["rc_shift"] == {"direction": "up"}
