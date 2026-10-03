# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Part 2 of the Oct 2026 Sibyl work: start from the reference, pool with it.

The reference functions in ``pythia.tools.base_rate_spd``, Sibyl's
``sibyl.reference``, the two-horizon distribution maths in
``sibyl.aggregate`` and the advice loop reading the raw month series.
"""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import duckdb
import numpy as np
import pytest

from pythia.tools import base_rate_spd as brs
from sibyl.aggregate import (
    MonthDist,
    dist_from_vector,
    mixture_weight,
    month_vector,
    pool_months,
    publish_vectors,
)
from sibyl.reference import build_reference


# --- fixtures ------------------------------------------------------------------

def _acled_db(tmp_path):
    con = duckdb.connect(str(tmp_path / "ref.duckdb"))
    con.execute(
        "CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month DATE, fatalities BIGINT, "
        "source TEXT, updated_at TIMESTAMP)"
    )
    return con


def _fill(con, iso: str, values: dict) -> None:
    for ym, v in values.items():
        nxt = brs._add_months(ym, 1)
        con.execute(
            "INSERT INTO acled_monthly_fatalities VALUES (?, ?, ?, 'ACLED', ?)",
            [iso, f"{ym}-01", int(v), datetime(int(nxt[:4]), int(nxt[5:7]), 20)],
        )


def _series(first: str, values: list) -> dict:
    return {brs._add_months(first, i): v for i, v in enumerate(values)}


def _q(iso="ETH", hz="ACE", metric="FATALITIES", qid="Q"):
    return SimpleNamespace(question_id=qid, iso3=iso, hazard_code=hz, metric=metric)


KEYS = [brs._add_months("2026-11", i) for i in range(6)]


# --- 2.1 the reference functions --------------------------------------------------

def test_conflictology_is_the_bucket_shares_of_the_last_12_months(tmp_path):
    con = _acled_db(tmp_path)
    # 24 months; the last 12 (Sep 2025 - Aug 2026 ... the level month is Sep 2026)
    vals = [3] * 12 + [0, 0, 0, 10, 10, 10, 30, 30, 30, 200, 200, 200]
    _fill(con, "ETH", _series("2024-10", vals))
    spds, src, d = brs.conflictology_spds(con, "ETH", "2026-11", known_at="2026-11-01")
    assert src == brs.CONFLICTOLOGY_MODEL_SOURCE
    assert d["level_month"] == "2026-09"
    assert d["n_months"] == 12
    v = spds[1]
    # 3 at zero, 3 in 5-<25, 3 in 25-<100, 3 in 100-<500: a quarter each,
    # less the floor that the four empty buckets carry.
    assert v[0] == pytest.approx(v[2]) == pytest.approx(v[3]) == pytest.approx(v[4])
    assert v[0] > 0.23 and min(v) >= brs.LEVEL_VOLATILITY_FLOOR / 1.03
    assert sum(v) == pytest.approx(1.0)
    assert all(spds[h] == v for h in range(1, 7))  # same at every horizon


def test_quiet_live_months_count_as_zero(tmp_path):
    con = _acled_db(tmp_path)
    _fill(con, "PEER", _series("2025-10", [5] * 12))
    _fill(con, "ETH", {"2025-10": 40})  # silent since: live months, no row
    spds, _, d = brs.conflictology_spds(con, "ETH", "2026-11", known_at="2026-11-01")
    assert d["values"].count(0.0) == 11
    assert spds[1][0] > 0.85


def test_ref_pool_is_the_stated_mixture(tmp_path):
    con = _acled_db(tmp_path)
    _fill(con, "ETH", _series("2023-08", ([0, 12, 40, 150] * 10)[:38]))
    c12, _, _ = brs.conflictology_spds(con, "ETH", "2026-11", known_at="2026-11-01")
    tr, _, _ = brs.level_transition_spds(con, "ETH", "2026-11", known_at="2026-11-01")
    pool, src, d = brs.reference_pool_spds(con, "ETH", "2026-11", known_at="2026-11-01")
    assert src == brs.REFERENCE_POOL_MODEL_SOURCE
    for h in range(1, 7):
        if h in tr:
            exp = [0.75 * a + 0.25 * b for a, b in zip(c12[h], tr[h])]
            assert pool[h] == pytest.approx([x / sum(exp) for x in exp])
        else:
            assert pool[h] == pytest.approx(c12[h])
            assert h in d["horizons_without_transition"]


def test_ref_pool_falls_back_to_the_12_month_vector(tmp_path, monkeypatch):
    con = _acled_db(tmp_path)
    _fill(con, "ETH", _series("2025-10", [7] * 12))
    monkeypatch.setattr(brs, "level_transition_spds", lambda *a, **k: ({}, "NONE", {}))
    c12, _, _ = brs.conflictology_spds(con, "ETH", "2026-11", known_at="2026-11-01")
    pool, _, d = brs.reference_pool_spds(con, "ETH", "2026-11", known_at="2026-11-01")
    assert pool == c12 and d["horizons_without_transition"] == [1, 2, 3, 4, 5, 6]


def _facts_db(tmp_path):
    con = duckdb.connect(str(tmp_path / "facts.duckdb"))
    con.execute(
        "CREATE TABLE facts_resolved (iso3 TEXT, hazard_code TEXT, metric TEXT, ym TEXT, "
        "value DOUBLE, publisher TEXT)"
    )
    return con


def test_seasonal_pa_gives_one_vector_per_forecast_month(tmp_path):
    con = _facts_db(tmp_path)
    # Ten years: August always floods (event + 60k affected), November never does.
    for y in range(2016, 2026):
        for m in range(1, 13):
            ym = f"{y}-{m:02d}"
            ev = 1.0 if m == 8 else 0.0
            con.execute("INSERT INTO facts_resolved VALUES ('BGD','FL','event_occurrence',?,?, 'GDACS')", [ym, ev])
        con.execute("INSERT INTO facts_resolved VALUES ('BGD','FL','affected',?,60000,'IFRC')", [f"{y}-08"])
    probs, src, d = brs.base_rate_spd(con, "BGD", "FL", "PA", "2026-06")
    by = d["probs_by_month"]
    assert set(by) == set(brs.forecast_months("2026-06"))
    assert by["2026-08"][0] < 0.1  # August: an event is near certain
    assert by["2026-11"][0] > 0.9  # November: nothing
    assert by["2026-08"][3] > 0.8  # 50k-<250k from August's own records
    # November has no records: the pooled severity shares stand in.
    nov_sev = [x / sum(by["2026-11"][1:]) for x in by["2026-11"][1:]]
    pooled_sev = [x / sum(probs[1:]) for x in probs[1:]]
    assert nov_sev == pytest.approx(pooled_sev, abs=1e-9)


# --- build_reference -------------------------------------------------------------------

def test_build_reference_for_conflict(tmp_path):
    con = _acled_db(tmp_path)
    _fill(con, "ETH", _series("2023-08", ([0, 12, 40, 150] * 10)[:38]))
    ref = build_reference(con, _q(), KEYS, date(2026, 11, 1))
    assert ref is not None and ref.source == brs.REFERENCE_POOL_MODEL_SOURCE
    assert set(ref.by_month) == {1, 2, 3, 4, 5, 6}
    assert len(ref.history) == 12
    text = ref.prompt_text
    assert "ACLED" in text and "may still rise" in text
    assert "month 1 = 2026-11" in text and "month 6 = 2027-04" in text
    assert "stays" in text and "rises" in text and "falls" in text


def test_the_shares_sentence_is_read_off_the_vector(tmp_path):
    con = _acled_db(tmp_path)
    _fill(con, "ETH", _series("2025-10", [7] * 12))
    ref = build_reference(con, _q(), KEYS, date(2026, 11, 1))
    v1 = ref.by_month[1]
    stay = round(100 * v1[2])  # 7 deaths sits in bucket 5-<25
    assert f"month 1 stays {stay}%" in ref.prompt_text


def test_build_reference_for_drought_pools_persistence_and_history(tmp_path):
    con = _facts_db(tmp_path)
    for i, v in enumerate([2e6, 2.2e6, 2.5e6, 3e6]):
        con.execute(
            "INSERT INTO facts_resolved VALUES ('SOM','DR','phase3plus_in_need',?,?, 'FEWS NET')",
            [brs._add_months("2026-05", i), v],
        )
    q = _q("SOM", "DR", "PHASE3PLUS_IN_NEED")
    ref = build_reference(con, q, KEYS, date(2026, 11, 1))
    assert ref is not None and ref.source.startswith("pool:persistence_0.5")
    assert all(ref.by_month[m] == ref.by_month[1] for m in range(1, 7))
    assert ref.current_value == 3e6
    assert ref.by_month[1][3] > 0.5  # 1M-<5M carries the persistence mass


def test_no_history_gives_no_reference(tmp_path):
    con = _acled_db(tmp_path)
    assert build_reference(con, _q("XXX"), KEYS, date(2026, 11, 1)) is None


# --- 2.3 the two-horizon distribution ----------------------------------------------------

def test_p_zero_is_the_zero_bucket_and_the_rest_is_positive():
    d = MonthDist(0.3, {0.05: 2, 0.25: 8, 0.5: 20, 0.75: 60, 0.95: 300})
    v = month_vector(d, "FATALITIES")
    assert v[0] == pytest.approx(0.3)
    assert sum(v) == pytest.approx(1.0)
    # The 0.5 positive quantile is 20: half the positive mass lies below it.
    assert sum(v[1:3]) / 0.7 == pytest.approx(0.5, abs=0.06)


def test_a_quantile_of_exactly_100_does_not_flip_a_bucket():
    # Half the positive mass at or below 100 means the 25-<100 edge is 99.5,
    # so 100 itself sits in 100-<500 and the CDF at 99.5 stays under 0.5.
    d = MonthDist(0.0, {0.05: 50, 0.25: 80, 0.5: 100, 0.75: 150, 0.95: 300})
    v = month_vector(d, "FATALITIES")
    assert sum(v[:4]) < 0.5
    assert sum(v[4:]) > 0.5


def test_the_curve_reaches_one_at_five_times_q95():
    d = MonthDist(0.0, {0.05: 2, 0.25: 4, 0.5: 10, 0.75: 20, 0.95: 100})
    assert d.cdf([499.0])[0] < 1.0
    assert d.cdf([500.0])[0] == pytest.approx(1.0)


def _trial(p1, q1, p6, q6):
    return {1: MonthDist(p1, q1), 6: MonthDist(p6, q6)}


def test_months_2_to_5_lie_between_months_1_and_6():
    q1 = {0.05: 1, 0.25: 3, 0.5: 8, 0.75: 20, 0.95: 60}
    q6 = {0.05: 20, 0.25: 60, 0.5: 150, 0.75: 400, 0.95: 2000}
    pool = pool_months([_trial(0.4, q1, 0.05, q6), _trial(0.3, q1, 0.1, q6)], "FATALITIES")
    a, b = np.array(pool.vectors[1]), np.array(pool.vectors[6])
    for m in range(2, 6):
        v = np.array(pool.vectors[m])
        assert np.all(v >= np.minimum(a, b) - 1e-9) and np.all(v <= np.maximum(a, b) + 1e-9)
        w = mixture_weight(m)
        assert v == pytest.approx((1 - w) * a + w * b, abs=1e-6)
    med = [pool.quantiles[m][0.5] for m in range(1, 7)]
    assert med == sorted(med)


def test_trials_are_linearly_pooled_per_month():
    q = {0.05: 1, 0.25: 3, 0.5: 8, 0.75: 20, 0.95: 60}
    t1, t2 = _trial(0.2, q, 0.2, q), _trial(0.6, q, 0.6, q)
    pool = pool_months([t1, t2], "FATALITIES")
    v1 = month_vector(t1[1], "FATALITIES")
    v2 = month_vector(t2[1], "FATALITIES")
    assert pool.vectors[1] == pytest.approx([(a + b) / 2 for a, b in zip(v1, v2)], abs=1e-9)


def test_the_published_vector_is_the_stated_pool():
    raw = {m: [0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.02] for m in range(1, 7)}
    ref = {m: [0.1, 0.1, 0.3, 0.3, 0.1, 0.05, 0.05] for m in range(1, 7)}
    out = publish_vectors(raw, ref, 0.5)
    assert out[3] == pytest.approx([0.5 * a + 0.5 * b for a, b in zip(ref[3], raw[3])])
    assert publish_vectors(raw, None, 0.5)[3] == pytest.approx(raw[3])


def test_seed_read_off_a_reference_vector_round_trips():
    vec = [0.3, 0.1, 0.3, 0.2, 0.05, 0.03, 0.02]
    d = dist_from_vector(vec, "FATALITIES")
    assert d.p_zero == pytest.approx(0.3)
    back = month_vector(d, "FATALITIES")
    assert back[0] == pytest.approx(0.3)
    assert back[2] == pytest.approx(0.3, abs=0.05)


# --- advice reads the raw month series --------------------------------------------------

def test_advice_compares_each_month_with_its_own_quantiles():
    from sibyl.advice import SibylRecord, diagnose

    rec = SibylRecord(
        question_id="Q", hazard_code="ACE", metric="FATALITIES",
        quantiles={0.1: 0, 0.5: 5, 0.9: 10, 0.99: 20},
        outcomes=[8.0, 300.0], outcome_horizons=[1, 6],
        month_quantiles={1: {0.1: 0, 0.5: 5, 0.9: 10, 0.99: 20},
                         6: {0.1: 100, 0.5: 250, 0.9: 400, 0.99: 900}},
        month_zero_mass={1: 0.2, 6: 0.0},
    )
    d = diagnose([rec])
    # Month 6's 300 sits inside month 6's own 10-90 range; against the
    # month-1 quantiles it would have been above q0.9.
    assert d["above_q90"].value == 0.0
    assert d["coverage_10_90"].value == 1.0
    assert d["zero_mass"].value == pytest.approx(0.1)
