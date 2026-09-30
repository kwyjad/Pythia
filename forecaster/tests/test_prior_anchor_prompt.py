# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""PYTHIA_PRIOR_ANCHOR_SPD: the level-and-volatility distribution as the prior.

On the resolved August 2026 conflict-death questions the Track 1 members'
priors put 0.38 on the realised bucket where the climatology SPD put 0.54;
the loss was in the prior. The flag shows ACE/FATALITIES members the
distribution and tells them to copy it. Off, every prompt is byte-identical.
"""

from __future__ import annotations

import json
from datetime import datetime

import duckdb
import pytest

from forecaster import prompts
from forecaster.trace_validation import prior_anchor_check
from pythia.tools import base_rate_spd as brs


def _months(first: str, n: int) -> list[str]:
    return [brs._add_months(first, i) for i in range(n)]


def _seed(path, iso_values: dict[str, dict[str, float]]) -> str:
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month DATE, fatalities BIGINT, "
        "source TEXT, updated_at TIMESTAMP)"
    )
    for iso, values in iso_values.items():
        for ym, v in values.items():
            nxt = brs._add_months(ym, 1)
            con.execute(
                "INSERT INTO acled_monthly_fatalities VALUES (?, ?, ?, 'ACLED', ?)",
                [iso, f"{ym}-01", int(v), datetime(int(nxt[:4]), int(nxt[5:7]), 28)],
            )
    con.close()
    return f"duckdb:///{path}"


STABLE = {ym: 200 + (i % 3) * 40 for i, ym in enumerate(_months("2023-05", 40))}
VOLATILE = {ym: [3, 40, 700, 150, 12, 1500][i % 6] for i, ym in enumerate(_months("2023-05", 40))}


def _q(iso: str, metric: str = "FATALITIES", hazard: str = "ACE") -> dict:
    return {
        "question_id": f"{iso}_{hazard}_{metric}_2026-10",
        "iso3": iso,
        "hazard_code": hazard,
        "metric": metric,
        "wording": "How many deaths?",
        "window_start_date": "2026-10-01",
        "target_month": "2027-03",
    }


@pytest.fixture
def anchor_db(tmp_path, monkeypatch):
    url = _seed(tmp_path / "a.duckdb", {"STB": STABLE, "VOL": VOLATILE})
    monkeypatch.setenv("RESOLVER_DB_URL", url)
    monkeypatch.setenv("PYTHIA_DB_URL", url)
    monkeypatch.setenv("PYTHIA_PRIOR_ANCHOR_TODAY", "2026-10-01")
    prompts._prior_anchor_cached.cache_clear()
    yield url
    prompts._prior_anchor_cached.cache_clear()


def _build(q, track=1, **kw):
    return prompts.build_spd_prompt_v2(
        q, {"source": "acled"}, {"regime_change_level": 1}, {}, track=track, **kw
    )


def test_flag_off_is_identical_to_unset(anchor_db, monkeypatch):
    monkeypatch.delenv("PYTHIA_PRIOR_ANCHOR_SPD", raising=False)
    for v3 in ("0", "1"):
        monkeypatch.setenv("PYTHIA_PROMPT_V3_ORDER", v3)
        for track in (1, 2):
            for metric in ("FATALITIES", "PA"):
                unset = _build(_q("VOL", metric), track)
                monkeypatch.setenv("PYTHIA_PRIOR_ANCHOR_SPD", "0")
                off = _build(_q("VOL", metric), track)
                monkeypatch.delenv("PYTHIA_PRIOR_ANCHOR_SPD")
                assert unset == off
                assert "BASE-RATE DISTRIBUTION" not in off
                assert "copied exactly" not in off
    assert prompts.load_prior_anchor(_q("VOL")) is None
    assert prompts.prior_anchor_block_version(_q("VOL")) is None


def test_block_reaches_the_assembled_prompt_for_both_tracks(anchor_db, monkeypatch):
    monkeypatch.setenv("PYTHIA_PRIOR_ANCHOR_SPD", "1")
    monkeypatch.setenv("PYTHIA_PROMPT_V3_ORDER", "1")
    for track in (1, 2):
        prefix, suffix = _build(_q("VOL"), track, return_parts=True)
        # The distribution is per-question data: after the cache prefix.
        assert "BASE-RATE DISTRIBUTION (your Step 1 prior)" in suffix
        assert "BASE-RATE DISTRIBUTION (your Step 1 prior)" not in prefix
        # The Step 1 instruction is static text: inside the prefix.
        assert "copied exactly" in prefix
        assert "Derive this prior from the Resolver history summary" not in prefix
        assert "Month 1 (2026-10):" in suffix and "Month 6 (2027-03):" in suffix
    assert prompts.prior_anchor_block_version(_q("VOL")) == brs.LEVEL_VOLATILITY_VERSION


def test_other_metrics_and_hazards_are_untouched_with_the_flag_on(anchor_db, monkeypatch):
    for metric, hazard in (("PA", "ACE"), ("PA", "FL")):
        monkeypatch.delenv("PYTHIA_PRIOR_ANCHOR_SPD", raising=False)
        off = _build(_q("VOL", metric, hazard))
        monkeypatch.setenv("PYTHIA_PRIOR_ANCHOR_SPD", "1")
        assert _build(_q("VOL", metric, hazard)) == off


def test_step_one_keeps_the_old_wording_when_no_anchor_exists(anchor_db, monkeypatch):
    monkeypatch.setenv("PYTHIA_PRIOR_ANCHOR_SPD", "1")
    text = _build(_q("XXX"))  # a country the table does not hold
    assert "BASE-RATE DISTRIBUTION" not in text
    assert "copied exactly" not in text
    assert "Derive this prior from the Resolver history summary" in text


def test_volatile_and_stable_render_differently_and_inside_budget(anchor_db, monkeypatch):
    monkeypatch.setenv("PYTHIA_PRIOR_ANCHOR_SPD", "1")
    keys = [f"2026-{m}" for m in ("10", "11", "12")] + ["2027-01", "2027-02", "2027-03"]
    blocks = {}
    for iso in ("STB", "VOL"):
        anchor = prompts.load_prior_anchor(_q(iso))
        assert anchor is not None
        blocks[iso] = prompts.render_prior_anchor_block(anchor, keys)
        assert len(blocks[iso]) <= prompts.PRIOR_ANCHOR_MAX_CHARS
        assert "Level: " in blocks[iso] and "the last complete month in the record" in blocks[iso]
        assert "stayed in the same bucket" in blocks[iso]
        assert "; >=1000 " in blocks[iso]  # every bucket, with its label
    assert "bucket 100% of the time" in blocks["STB"]
    assert "bucket 100% of the time" not in blocks["VOL"]
    assert "Level: 200 deaths in August 2026 (bucket 100-<500)" in blocks["STB"]


def test_render_names_a_pooled_spread():
    anchor = {
        "version": "prior_anchor_v1",
        "spds": {1: [0.005, 0.005, 0.01, 0.07, 0.8, 0.1, 0.01], 6: [1 / 7] * 7},
        "detail": {
            "level_month": "2026-08", "level_value": 360.0, "level_bucket": 4,
            "n_months_in_window": 13,
            "horizons": {"1": {"gap_months": 2, "share_same": 0.81, "share_one": 0.17,
                               "share_two_plus": 0.02, "pooled": True, "n_own_pairs": 11,
                               "n_band_countries": 11}},
        },
    }
    block = prompts.render_prior_anchor_block(anchor, ["2026-10"] + ["x"] * 5)
    assert "Level: 360 deaths in August 2026 (bucket 100-<500)" in block
    assert "the count 2 months later stayed in the same bucket 81% of the time" in block
    assert "moved one bucket 17%, and two or more 2%" in block
    assert "only 11 such pairs, so the shares pool 11 countries" in block
    assert "100-<500 80%" in block and "0 0.5%" in block


def test_divergence_diagnostic():
    shown = [0.005, 0.005, 0.01, 0.07, 0.8, 0.1, 0.01]
    copied = prior_anchor_check({"prior": {"spd": list(shown)}}, shown, "prior_anchor_v1")
    assert copied["status"] == "ok" and copied["js_distance"] == pytest.approx(0.0, abs=1e-6)
    moved = prior_anchor_check(
        {"prior": {"spd": [0.3, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02]}}, shown, "prior_anchor_v1"
    )
    assert 0.3 < moved["js_distance"] <= 1.0
    per_month = prior_anchor_check(
        {"prior": {"spd": {"2026-10": list(shown), "2026-11": [1 / 7] * 7}}}, shown, "v"
    )
    assert per_month["js_distance"] == pytest.approx(0.0, abs=1e-6)
    assert prior_anchor_check({}, shown, "v")["status"] == "no_declared_prior"
    assert prior_anchor_check({"prior": {"spd": [0.5, 0.5]}}, shown, "v")["status"].startswith(
        "length_mismatch"
    )
    json.dumps(moved)  # stored on the trace, so it must serialise


def test_member_writer_stamps_the_block_version_and_the_distance(anchor_db, monkeypatch, tmp_path):
    from forecaster import cli
    from forecaster.providers import ModelSpec

    monkeypatch.setenv("PYTHIA_PRIOR_ANCHOR_SPD", "1")
    db = str(tmp_path / "w.duckdb")
    monkeypatch.setattr(cli, "connect", lambda read_only=False: duckdb.connect(db))
    spec = ModelSpec(name="m1", provider="openai", model_id="m1", active=True)
    shown = prompts.load_prior_anchor(_q("VOL"))["spds"][1]
    month_spds = {k: list(shown) for k in
                  ("2026-10", "2026-11", "2026-12", "2027-01", "2027-02", "2027-03")}
    trace = {"prior": {"spd": list(shown)}}
    for version, run in ((None, "r0"), (brs.LEVEL_VOLATILITY_VERSION, "r1")):
        cli._write_spd_members_v2_to_db(
            run_id=run,
            question_row=_q("VOL"),
            specs_used=[spec],
            per_model_spds=[month_spds],
            raw_calls=[{"usage": {}, "reasoning_trace": trace}],
            resolution_source="ACLED",
            base_rate_block_version=version,
        )
    con = duckdb.connect(db)
    got = dict(con.execute(
        "SELECT run_id, ANY_VALUE(base_rate_block_version) FROM forecasts_raw GROUP BY 1"
    ).fetchall())
    assert got == {"r0": None, "r1": "prior_anchor_v1"}
    t0 = json.loads(con.execute(
        "SELECT reasoning_trace_json FROM forecasts_raw WHERE run_id='r0' LIMIT 1").fetchone()[0])
    t1 = json.loads(con.execute(
        "SELECT reasoning_trace_json FROM forecasts_raw WHERE run_id='r1' LIMIT 1").fetchone()[0])
    assert "prior_anchor_check" not in t0
    assert t1["prior_anchor_check"]["js_distance"] == pytest.approx(0.0, abs=1e-6)
