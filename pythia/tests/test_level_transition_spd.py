# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""The transition reference: moves counted only from the level's own bucket.

The level-and-volatility reference pools every bucket move a country has
made over the same gap, wherever it started. A country at zero then
inherits the downward moves of months that started higher, and they pile up
on bucket 0 at the edge. ``__ext_level_transition`` counts only pairs whose
start month sits in the level's bucket; it is scored and never shown to a
model.
"""

from __future__ import annotations

from datetime import datetime

import duckdb
import pytest

from pythia.tools import base_rate_spd as brs


def _db(tmp_path):
    con = duckdb.connect(str(tmp_path / "t.duckdb"))
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


def _flicker(first="2023-05", n=40):
    """Mostly quiet, with a 300-death month every fourth month; ends quiet."""
    months = [brs._add_months(first, i) for i in range(n)]
    vals = {ym: (300 if i % 4 == 2 else 0) for i, ym in enumerate(months)}
    vals[months[-1]] = 0
    return vals


def test_a_country_at_zero_borrows_no_downward_moves(tmp_path):
    con = _db(tmp_path)
    _fill(con, "ZER", _flicker())
    lv, _, lv_d = brs.level_volatility_spds(con, "ZER", "2026-10", known_at="2026-10-01")
    lt, src, lt_d = brs.level_transition_spds(con, "ZER", "2026-10", known_at="2026-10-01")
    assert src == brs.LEVEL_TRANSITION_MODEL_SOURCE
    assert lt_d["level_bucket"] == 0 and lt_d["method"] == "level_plus_transition_moves"
    # The pooled recipe counts the 300 -> 0 drops and clips them onto zero.
    assert lv_d["horizons"]["1"]["share_down"] > 0
    # The transition recipe starts every pair in bucket 0: nothing goes down.
    for h, hd in lt_d["horizons"].items():
        assert hd["share_down"] == 0.0, h
        assert hd["share_same"] + hd["share_up"] == pytest.approx(1.0)
    # So it holds less on zero than the pooled recipe does.
    assert lt[1][0] < lv[1][0]
    for vec in lt.values():
        assert sum(vec) == pytest.approx(1.0)
        assert min(vec) >= brs.LEVEL_VOLATILITY_FLOOR / 1.2


def test_pooling_keeps_the_start_bucket_rule(tmp_path):
    con = _db(tmp_path)
    months = [brs._add_months("2025-11", i) for i in range(10)]
    # A short record: too few own pairs, so the band is pooled.
    _fill(con, "NEW", {ym: 0 for ym in months})
    _fill(con, "PEER", {ym: (300 if i % 3 == 0 else 0) for i, ym in enumerate(months)})
    _, _, d = brs.level_transition_spds(con, "NEW", "2026-10", known_at="2026-10-01")
    h1 = d["horizons"]["1"]
    assert h1["pooled"] is True and h1["n_band_countries"] == 1
    # PEER's 300 -> 0 drops start in bucket 4 and are not borrowed.
    assert h1["share_down"] == 0.0
    assert h1["share_up"] > 0


def test_no_pair_from_the_level_bucket_means_no_vector(tmp_path):
    con = _db(tmp_path)
    months = [brs._add_months("2023-05", i) for i in range(40)]
    # Always 300 except the very last month, so nothing ever started at zero.
    vals = {ym: 300 for ym in months}
    vals[months[-1]] = 0
    _fill(con, "ONE", vals)
    spds, src, d = brs.level_transition_spds(con, "ONE", "2026-10", known_at="2026-10-01")
    assert spds == {} and src == brs.NO_BASE_RATE_SOURCE
