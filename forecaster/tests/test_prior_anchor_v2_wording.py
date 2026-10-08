# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""prior_anchor_v2: the Spread sentence describes the vector it sits beside.

The v1 block states the share of raw bucket moves that stayed put. The
vector below it clips moves past either end onto the end bucket and floors
every bucket, so at bucket 0 the two disagree: the November 2026 test run
told members Israel's count "stayed in the same bucket 42% of the time"
beside a month-1 row putting 71% on zero. v2 reads stay / up / down straight
off the vector. v1 stays the CODE default; v2 is selected by
``PYTHIA_PRIOR_ANCHOR_BLOCK_VERSION=v2`` and recorded as ``prior_anchor_v2``.
Both production workflows set v2 from the run on 13 October 2026
(owner decision 2026-10-06), and the experiment flags must agree between them.
"""

from __future__ import annotations

import re

import pytest

from forecaster import prompts
from pythia.tools import base_rate_spd as brs

ISRAEL_M1 = [0.71, 0.12, 0.08, 0.05, 0.02, 0.01, 0.01]
ISRAEL_M6 = [0.55, 0.18, 0.12, 0.08, 0.04, 0.02, 0.01]


def _anchor(version):
    return {
        "version": version,
        "spds": {1: list(ISRAEL_M1), 6: list(ISRAEL_M6)},
        "detail": {
            "level_month": "2026-08", "level_value": 0.0, "level_bucket": 0,
            "n_months_in_window": 13,
            "horizons": {"1": {"gap_months": 2, "share_same": 0.42, "share_one": 0.30,
                               "share_two_plus": 0.28, "pooled": True, "n_own_pairs": 11,
                               "n_band_countries": 9}},
        },
    }


def test_v2_states_the_vectors_own_stay_up_and_down():
    block = prompts.render_prior_anchor_block(_anchor(brs.LEVEL_VOLATILITY_VERSION_V2),
                                              ["2026-10"] + ["x"] * 5)
    assert "stays in bucket 0 with probability 71%" in block
    assert "moves to a higher bucket 29% and to a lower one 0.0%" in block
    assert "by month 6, 55%, 45% and 0.0%" in block
    assert "42%" not in block  # the raw move share is gone
    assert "pool 9 countries" in block
    assert len(block) <= prompts.PRIOR_ANCHOR_MAX_CHARS


def test_v2_sentence_matches_the_row_for_a_middle_bucket():
    anchor = _anchor(brs.LEVEL_VOLATILITY_VERSION_V2)
    anchor["detail"]["level_bucket"] = 3
    anchor["spds"][1] = [0.02, 0.03, 0.15, 0.5, 0.2, 0.07, 0.03]
    block = prompts.render_prior_anchor_block(anchor, ["2026-10"] + ["x"] * 5)
    stay = re.search(r"with probability (\d+)%", block).group(1)
    assert stay == "50"
    assert "higher bucket 30% and to a lower one 20%" in block


def test_v1_text_is_still_reachable_and_unchanged():
    block = prompts.render_prior_anchor_block(_anchor(brs.LEVEL_VOLATILITY_VERSION),
                                              ["2026-10"] + ["x"] * 5)
    assert "stayed in the same bucket 42% of the time" in block
    assert "Read off the distribution" not in block


def test_version_selector_defaults_to_v1(monkeypatch):
    monkeypatch.delenv("PYTHIA_PRIOR_ANCHOR_BLOCK_VERSION", raising=False)
    assert prompts.prior_anchor_version() == "prior_anchor_v1"
    for raw in ("v2", "V2", "prior_anchor_v2", "2"):
        monkeypatch.setenv("PYTHIA_PRIOR_ANCHOR_BLOCK_VERSION", raw)
        assert prompts.prior_anchor_version() == "prior_anchor_v2"
    monkeypatch.setenv("PYTHIA_PRIOR_ANCHOR_BLOCK_VERSION", "v3")
    assert prompts.prior_anchor_version() == "prior_anchor_v1"


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_flag_off_prompts_ignore_the_wording_selector(monkeypatch, version):
    """With PYTHIA_PRIOR_ANCHOR_SPD off no block is loaded, whatever the selector."""
    monkeypatch.delenv("PYTHIA_PRIOR_ANCHOR_SPD", raising=False)
    monkeypatch.setenv("PYTHIA_PRIOR_ANCHOR_BLOCK_VERSION", version)
    q = {"question_id": "ISR_ACE_FATALITIES_2026-10", "iso3": "ISR", "hazard_code": "ACE",
         "metric": "FATALITIES", "window_start_date": "2026-10-01"}
    assert prompts.load_prior_anchor(q) is None
    assert prompts.prior_anchor_block_version(q) is None


def test_flag_on_v2_reaches_the_prompt_and_the_stamp(tmp_path, monkeypatch):
    from forecaster.tests.test_prior_anchor_prompt import STABLE, _build, _q, _seed

    url = _seed(tmp_path / "v2.duckdb", {"STB": STABLE})
    monkeypatch.setenv("RESOLVER_DB_URL", url)
    monkeypatch.setenv("PYTHIA_DB_URL", url)
    monkeypatch.setenv("PYTHIA_PRIOR_ANCHOR_TODAY", "2026-10-01")
    monkeypatch.setenv("PYTHIA_PRIOR_ANCHOR_SPD", "1")
    prompts._prior_anchor_cached.cache_clear()
    try:
        monkeypatch.delenv("PYTHIA_PRIOR_ANCHOR_BLOCK_VERSION", raising=False)
        v1 = _build(_q("STB"))
        assert prompts.prior_anchor_block_version(_q("STB")) == "prior_anchor_v1"
        monkeypatch.setenv("PYTHIA_PRIOR_ANCHOR_BLOCK_VERSION", "v2")
        v2 = _build(_q("STB"))
        assert prompts.prior_anchor_block_version(_q("STB")) == "prior_anchor_v2"
    finally:
        prompts._prior_anchor_cached.cache_clear()
    assert "Read off the distribution below" in v2
    assert "Read off the distribution below" not in v1
    # Only the Spread line differs: the rows a member copies are identical.
    rows = lambda text: [l for l in text.splitlines() if l.startswith("  Month ")]  # noqa: E731
    assert rows(v1) == rows(v2) and rows(v1)


# --- the two production workflows set every experiment flag alike -----------

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENT_FLAGS = (
    "PYTHIA_PROMPT_V3_ORDER", "PYTHIA_PROMPT_CACHE_ENABLED",
    "PYTHIA_RC_SHIFT_GUIDANCE", "PYTHIA_RC_SHIFT_SHARE",
    "PYTHIA_PRIOR_ANCHOR_SPD", "PYTHIA_PRIOR_ANCHOR_BLOCK_VERSION",
    "PYTHIA_ADVICE_FAMILY_CARRYOVER", "PYTHIA_ADVICE_EXPERIMENT_SHARE",
    "PYTHIA_FAMILY_RECALIBRATION_MODE", "PYTHIA_ADVICE_BLOCK_GROUPS",
    "PYTHIA_MEMBER_ADVICE",
)


def _workflow_flags(name: str) -> dict:
    text = (_ROOT / ".github" / "workflows" / name).read_text()
    out = {}
    for flag in _EXPERIMENT_FLAGS:
        vals = re.findall(rf"^\s*{flag}:\s*\"?([^\"\n#]*)\"?\s*$", text, flags=re.M)
        out[flag] = sorted(set(v.strip() for v in vals)) or None
    return out


def test_both_production_workflows_set_the_experiment_flags_alike():
    stage = _workflow_flags("pythia_pipeline_stage.yml")
    legacy = _workflow_flags("run_horizon_scanner.yml")
    assert stage == legacy
    assert stage["PYTHIA_PRIOR_ANCHOR_SPD"] == ["1"]
    assert stage["PYTHIA_PRIOR_ANCHOR_BLOCK_VERSION"] == ["v2"]
    assert stage["PYTHIA_RC_SHIFT_GUIDANCE"] == ["0"]
    # The shared-advice repair (#974) renders observations and drops the
    # prior-anchoring line under the anchor, so no group is withheld.
    assert stage["PYTHIA_ADVICE_BLOCK_GROUPS"] is None
