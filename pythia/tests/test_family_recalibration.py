# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Per-family recalibration: the arithmetic, the fit, and the version gate."""

from __future__ import annotations

import duckdb
import pytest

from pythia.tools import family_recalibration as fr


# --- arithmetic ------------------------------------------------------------

def test_shrinkage_pulls_toward_one_by_twenty_pseudo_questions():
    # Assigned 10%, observed 20%: raw ratio 2.0. With 20 questions and a
    # prior of 20 the factor is halfway: 1.5.
    (f,) = fr.shrunk_spd_factors([0.1], [0.2], n_questions=20)
    assert f == pytest.approx(1.5)
    (f,) = fr.shrunk_spd_factors([0.1], [0.2], n_questions=180)
    assert f == pytest.approx((180 * 2 + 20) / 200)


def test_factors_are_clipped():
    lo, hi = fr.shrunk_spd_factors([0.5, 0.01], [0.0, 0.9], n_questions=10_000)
    assert lo == fr.SPD_FACTOR_MIN and hi == fr.SPD_FACTOR_MAX


def test_apply_floors_and_renormalises():
    out = fr.apply_spd_factors([0.5, 0.5, 0.0], [2.0, 1.0, 0.5])
    assert sum(out) == pytest.approx(1.0)
    assert out[2] > 0  # floored, never zero
    assert out[0] == pytest.approx(2 * out[1], rel=1e-3)
    with pytest.raises(ValueError):
        fr.apply_spd_factors([0.5, 0.5], [1.0])


def test_binary_shift_moves_toward_the_outcomes_and_is_clipped():
    # Forecast 10% on events that happened half the time: shift up.
    d = fr.binary_logit_shift([0.1] * 40, [1, 0] * 20)
    assert 0 < d <= fr.BINARY_SHIFT_MAX
    # A handful of questions barely move it: the prior holds.
    small = fr.binary_logit_shift([0.1] * 2, [1, 0])
    assert 0 < small < d
    assert fr.binary_logit_shift([0.01] * 500, [1] * 500) == fr.BINARY_SHIFT_MAX
    assert fr.apply_binary_shift(0.2, 0.0) == pytest.approx(0.2)
    assert fr.apply_binary_shift(0.999, 1.0) == fr.BINARY_CLAMP[1]


# --- fitting ---------------------------------------------------------------

def _db(tmp_path, n_questions=12, *, raw_for=(), brbv=None):
    con = duckdb.connect(str(tmp_path / "f.duckdb"))
    con.execute(
        "CREATE TABLE questions (question_id TEXT, hazard_code TEXT, metric TEXT, is_test BOOLEAN)"
    )
    con.execute("CREATE TABLE resolutions (question_id TEXT, horizon_m INTEGER, value DOUBLE)")
    con.execute("CREATE TABLE forecasts_ensemble (question_id TEXT, run_id TEXT)")
    con.execute(
        "CREATE TABLE forecasts_raw (run_id TEXT, question_id TEXT, model_name TEXT, "
        "month_index INTEGER, bucket_index INTEGER, probability DOUBLE, "
        "base_rate_block_version TEXT, rc_guidance TEXT)"
    )
    # Every question resolves in bucket 5 (100-<500); the member puts 10% there.
    probs = [0.3, 0.2, 0.2, 0.15, 0.1, 0.03, 0.02]
    for i in range(n_questions):
        q = f"Q{i}"
        con.execute("INSERT INTO questions VALUES (?, 'ACE', 'FATALITIES', FALSE)", [q])
        con.execute("INSERT INTO resolutions VALUES (?, 1, 300)", [q])
        con.execute("INSERT INTO forecasts_ensemble VALUES (?, 'r1')", [q])
        for b, p in enumerate(probs, start=1):
            con.execute(
                "INSERT INTO forecasts_raw VALUES ('r1', ?, 'gpt-6-sol', 1, ?, ?, ?, NULL)",
                [q, b, p, brbv],
            )
            if q in raw_for:
                # The row under the model's name is a corrected copy; the
                # raw one says 50% on bucket 5.
                con.execute(
                    "INSERT INTO forecasts_raw VALUES ('r1', ?, 'gpt-6-sol__raw', 1, ?, ?, ?, NULL)",
                    [q, b, 0.5 if b == 5 else 0.5 / 6, brbv],
                )
            # Aggregates and references are never fitted.
            con.execute(
                "INSERT INTO forecasts_raw VALUES ('r1', ?, 'ensemble_mean_v2', 1, ?, ?, NULL, NULL)",
                [q, b, 0.9 if b == 1 else 0.1 / 6],
            )
    return con


def _factors(con):
    return {
        (r[0], r[1]): r[2]
        for r in con.execute(
            "SELECT family, bucket_index, factor FROM family_recalibration ORDER BY 1, 2"
        ).fetchall()
    }


def test_fit_raises_the_bucket_that_kept_happening(tmp_path):
    con = _db(tmp_path)
    summary = fr.fit_family_recalibration(con, "2026-09")
    assert [g["n_questions"] for g in summary["fitted"]] == [12]
    f = _factors(con)
    assert set(k[0] for k in f) == {"gpt"}  # the aggregate is not a family
    assert f[("gpt", 5)] == pytest.approx(fr.SPD_FACTOR_MAX)  # ratio 10, shrunk, clipped
    assert f[("gpt", 1)] == pytest.approx(20 / 32)  # never happened: (0 + 20) / (12 + 20)


def test_no_fit_below_ten_questions(tmp_path):
    con = _db(tmp_path, n_questions=9)
    summary = fr.fit_family_recalibration(con, "2026-09")
    assert summary["fitted"] == []
    assert summary["skipped"][0]["n_questions"] == 9
    assert con.execute("SELECT COUNT(*) FROM family_recalibration").fetchone()[0] == 0


def test_the_raw_row_is_fitted_never_the_corrected_one(tmp_path):
    con = _db(tmp_path, raw_for={f"Q{i}" for i in range(12)})
    fr.fit_family_recalibration(con, "2026-09")
    # Raw put 50% on the bucket that always happened: ratio 2 -> (12*2+20)/32.
    assert _factors(con)[("gpt", 5)] == pytest.approx((12 * 2 + 20) / 32)


def test_refit_of_the_same_month_replaces_it(tmp_path):
    con = _db(tmp_path)
    fr.fit_family_recalibration(con, "2026-09")
    fr.fit_family_recalibration(con, "2026-09")
    assert con.execute("SELECT COUNT(*) FROM family_recalibration").fetchone()[0] == 7


# --- lookup and the version gate --------------------------------------------

@pytest.fixture
def fitted(tmp_path, monkeypatch):
    con = _db(tmp_path)
    fr.fit_family_recalibration(con, "2026-09")
    con.close()
    url = f"duckdb:///{tmp_path / 'f.duckdb'}"
    monkeypatch.setenv("PYTHIA_DB_URL", url)
    fr.reset_factor_cache()
    yield url
    fr.reset_factor_cache()


def test_off_is_the_default_and_does_nothing(fitted, monkeypatch):
    monkeypatch.delenv("PYTHIA_FAMILY_RECALIBRATION_MODE", raising=False)
    info = fr.lookup("gpt-6-sol", "ACE", "FATALITIES", base_rate_block_version=None, rc_guidance=None)
    assert info == {"mode": "off"}


def test_apply_with_matching_versions(fitted, monkeypatch):
    monkeypatch.setenv("PYTHIA_FAMILY_RECALIBRATION_MODE", "apply")
    info = fr.lookup("gpt-6-sol", "ACE", "FATALITIES", base_rate_block_version=None, rc_guidance=None)
    assert info["mode"] == "apply" and info["family"] == "gpt" and len(info["factors"]) == 7
    # The family carries the factors to every version in it.
    assert fr.lookup("gpt-5.6-sol", "ACE", "FATALITIES",
                     base_rate_block_version=None, rc_guidance=None)["mode"] == "apply"


def test_another_prompt_version_shadows(fitted, monkeypatch):
    monkeypatch.setenv("PYTHIA_FAMILY_RECALIBRATION_MODE", "apply")
    info = fr.lookup("gpt-6-sol", "ACE", "FATALITIES",
                     base_rate_block_version="prior_anchor_v1", rc_guidance=None)
    assert info["mode"] == "auto_shadow" and info["factors"]
    assert info["fitted_versions"] == {"base_rate_block_version": None, "rc_guidance": None}


def test_missing_factors_fall_back_to_nothing(fitted, monkeypatch):
    monkeypatch.setenv("PYTHIA_FAMILY_RECALIBRATION_MODE", "apply")
    assert fr.lookup("gpt-6-sol", "FL", "PA", base_rate_block_version=None,
                     rc_guidance=None)["mode"] == "none"
    assert fr.lookup("unknown-model", "ACE", "FATALITIES", base_rate_block_version=None,
                     rc_guidance=None)["mode"] == "none"
    for name in ("sibyl", "track2_flash", "ensemble_mean_v2", "gpt-6-sol__raw", "gpt-6-sol__recal"):
        assert fr.lookup(name, "ACE", "FATALITIES", base_rate_block_version=None,
                         rc_guidance=None)["mode"] == "none"


def test_a_database_without_the_table_means_no_factors(tmp_path, monkeypatch):
    duckdb.connect(str(tmp_path / "empty.duckdb")).close()
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{tmp_path / 'empty.duckdb'}")
    monkeypatch.setenv("PYTHIA_FAMILY_RECALIBRATION_MODE", "apply")
    fr.reset_factor_cache()
    assert fr.lookup("gpt-6-sol", "ACE", "FATALITIES", base_rate_block_version=None,
                     rc_guidance=None)["mode"] == "none"
    fr.reset_factor_cache()


def test_recalibrate_spd_refuses_a_length_mismatch():
    assert fr.recalibrate_spd({"2026-10": [0.5, 0.5]}, {1: 1.0, 2: 1.0, 3: 1.0}) is None
    out = fr.recalibrate_spd({"2026-10": [0.5, 0.5]}, {1: 2.0, 2: 1.0})
    assert out["2026-10"][0] == pytest.approx(2 / 3)


def test_unknown_mode_reads_off(monkeypatch):
    monkeypatch.setenv("PYTHIA_FAMILY_RECALIBRATION_MODE", "sometimes")
    assert fr.recalibration_mode() == "off"
    assert fr.is_derived_name("x__raw") and fr.base_model_name("x__recal") == "x"
