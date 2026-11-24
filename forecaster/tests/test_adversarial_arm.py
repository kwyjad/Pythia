# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""The adversarial counter-evidence experiment (PYTHIA_ADVERSARIAL_WITHHELD_SHARE, Nov 2026).

Among questions whose SPD prompt would carry the adversarial block (a check
loaded and RC level 1 or above), a salted hash of the question id withholds
it from half. The withheld prompt is the shown prompt with the block taken
out and nothing else changed; the arm is stamped on the forecast rows, and
a question the experiment does not cover is stamped ``n/a``.
"""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from forecaster import prompts

SHOWN_Q = "C02_ACE_FATALITIES_2026-12"
WITHHELD_Q = "C00_ACE_FATALITIES_2026-12"
_Q = {
    "iso3": "SDN", "hazard_code": "ACE", "metric": "FATALITIES",
    "wording": "How many conflict deaths will Sudan record?",
    "window_start_date": "2026-12-01", "target_month": "2027-05",
}
_HIST = {"source": "acled", "summary": "24 months of data"}
_TRI = {"regime_change_level": 2, "regime_change_direction": "up",
        "regime_change_likelihood": 0.5, "regime_change_magnitude": 0.4, "triage_score": 0.6}
_TRI_RC0 = {"regime_change_level": 0, "triage_score": 0.6}
_CHECK = {
    "net_assessment": "strong_counter",
    "summary": "A proposed truce may reduce fighting.",
    "sources": [],
}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    missing = f"duckdb:///{tmp_path / 'absent.duckdb'}"
    monkeypatch.setenv("PYTHIA_DB_URL", missing)
    monkeypatch.setenv("RESOLVER_DB_URL", missing)
    monkeypatch.setenv("PYTHIA_PROMPT_V3_ORDER", "1")
    for var in ("PYTHIA_RC_SHIFT_GUIDANCE", "PYTHIA_RC_SHIFT_SHARE",
                "PYTHIA_ADVICE_EXPERIMENT_SHARE", "PYTHIA_ADVERSARIAL_WITHHELD_SHARE"):
        monkeypatch.delenv(var, raising=False)
    prompts.reset_adversarial_eligibility()
    yield
    prompts.reset_adversarial_eligibility()


def _prompt(qid, tri=_TRI, sd=None):
    return prompts.build_spd_prompt_v2(dict(_Q, question_id=qid), _HIST, tri, {},
                                       structured_data=sd, track=1)


def test_default_share_is_half_and_arms_follow_the_salted_hash(monkeypatch):
    assert prompts.adversarial_withheld_share() == 0.5
    assert prompts.adversarial_arm(SHOWN_Q) == "shown"
    assert prompts.adversarial_arm(WITHHELD_Q) == "withheld"
    monkeypatch.setenv("PYTHIA_ADVERSARIAL_WITHHELD_SHARE", "0")
    assert prompts.adversarial_arm(SHOWN_Q) is None
    assert prompts.adversarial_arm_applied(SHOWN_Q) is None


def test_the_arm_crosses_the_advice_and_rc_arms(monkeypatch):
    monkeypatch.setenv("PYTHIA_RC_SHIFT_SHARE", "0.5")
    monkeypatch.setenv("PYTHIA_ADVICE_EXPERIMENT_SHARE", "0.5")
    cells: dict[tuple, int] = {}
    for i in range(4000):
        q = f"Q{i}_ACE_FATALITIES_2026-12"
        key = (prompts.advice_arm(q), prompts.rc_shift_arm(q), prompts.adversarial_arm(q))
        cells[key] = cells.get(key, 0) + 1
    assert len(cells) == 8
    assert min(cells.values()) > 400  # about 500 each


def test_withheld_prompt_is_the_shown_prompt_less_the_block():
    sd = {"adversarial_check": _CHECK}
    withheld = _prompt(WITHHELD_Q, sd=sd)
    assert "ADVERSARIAL EVIDENCE CHECK" not in withheld
    # Byte-identical to the same question built with no check at all.
    assert withheld == _prompt(WITHHELD_Q, sd={})
    shown = _prompt(SHOWN_Q, sd=sd)
    assert "ADVERSARIAL EVIDENCE CHECK" in shown


def test_the_shown_arm_is_unchanged_from_the_experiment_off(monkeypatch):
    sd = {"adversarial_check": _CHECK}
    shown = _prompt(SHOWN_Q, sd=sd)
    monkeypatch.setenv("PYTHIA_ADVERSARIAL_WITHHELD_SHARE", "0")
    assert _prompt(SHOWN_Q, sd=sd) == shown


def test_eligibility_needs_a_check_and_rc_level_1():
    _prompt(SHOWN_Q, sd={"adversarial_check": _CHECK})
    _prompt(WITHHELD_Q, sd={"adversarial_check": _CHECK})
    no_check = "C03_ACE_FATALITIES_2026-12"
    rc0 = "C04_ACE_FATALITIES_2026-12"
    _prompt(no_check, sd={})
    _prompt(rc0, tri=_TRI_RC0, sd={"adversarial_check": _CHECK})
    assert prompts.adversarial_arm_applied(SHOWN_Q) == "shown"
    assert prompts.adversarial_arm_applied(WITHHELD_Q) == "withheld"
    assert prompts.adversarial_arm_applied(no_check) == "n/a"
    assert prompts.adversarial_arm_applied(rc0) == "n/a"


def test_the_stamp_writes_both_tables_and_counts_the_arms(monkeypatch, tmp_path):
    from forecaster import cli

    path = tmp_path / "s.duckdb"
    con = duckdb.connect(str(path))
    for table in ("forecasts_raw", "forecasts_ensemble"):
        con.execute(f"CREATE TABLE {table} (run_id TEXT, question_id TEXT, model_name TEXT, "
                    "adversarial_arm TEXT)")
    for qid in (SHOWN_Q, WITHHELD_Q, "C03_ACE_FATALITIES_2026-12"):
        con.execute("INSERT INTO forecasts_raw VALUES ('r1', ?, 'm', NULL)", [qid])
        con.execute("INSERT INTO forecasts_ensemble VALUES ('r1', ?, 'ensemble_mean_v2', NULL)", [qid])
    con.close()
    monkeypatch.setattr(cli, "connect", lambda read_only=False: duckdb.connect(str(path)))
    monkeypatch.setattr(cli, "_ADVERSARIAL_ARM_COUNTS", {})
    _prompt(SHOWN_Q, sd={"adversarial_check": _CHECK})
    _prompt(WITHHELD_Q, sd={"adversarial_check": _CHECK})
    for qid in (SHOWN_Q, WITHHELD_Q, "C03_ACE_FATALITIES_2026-12"):
        cli._stamp_adversarial_arm("r1", qid, {"track": 1})
    con = duckdb.connect(str(path))
    for table in ("forecasts_raw", "forecasts_ensemble"):
        got = dict(con.execute(f"SELECT question_id, adversarial_arm FROM {table}").fetchall())
        assert got == {SHOWN_Q: "shown", WITHHELD_Q: "withheld",
                       "C03_ACE_FATALITIES_2026-12": "n/a"}
    assert cli.adversarial_arm_summary_line() == (
        "[adversarial-arm] eligible=2 shown=1 withheld=1 not_eligible=1")


def test_nothing_is_stamped_when_the_experiment_is_off(monkeypatch, tmp_path):
    from forecaster import cli

    monkeypatch.setenv("PYTHIA_ADVERSARIAL_WITHHELD_SHARE", "0")
    calls = []
    monkeypatch.setattr(cli, "connect", lambda read_only=False: calls.append(1))
    assert cli._stamp_adversarial_arm("r1", SHOWN_Q, {"track": 1}) is None
    assert calls == []


def test_the_schema_declares_the_column(tmp_path):
    from pythia.db import schema

    con = duckdb.connect(str(tmp_path / "x.duckdb"))
    schema.ensure_schema(con)
    for table in ("forecasts_raw", "forecasts_ensemble"):
        cols = {r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}
        assert "adversarial_arm" in cols, table
