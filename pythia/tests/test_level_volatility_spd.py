# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""The ACE/FATALITIES level-and-volatility distribution, and the complete-month rule.

On 1 August 2026 the prompt's "last month" was a row written on 15 July,
holding a median 28% of July's settled deaths. Every reader now takes complete
months only, and the level-and-volatility reference reads the level the
forecaster could have read at forecast time.
"""

from __future__ import annotations

from datetime import date, datetime

import duckdb
import pytest

from pythia.tools import base_rate_spd as brs


def _db(tmp_path):
    con = duckdb.connect(str(tmp_path / "a.duckdb"))
    con.execute(
        "CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month DATE, fatalities BIGINT, "
        "source TEXT, updated_at TIMESTAMP)"
    )
    return con


def _months(first: str, n: int) -> list[str]:
    return [brs._add_months(first, i) for i in range(n)]


def _written_after(ym: str, days: int = 28) -> datetime:
    nxt = brs._add_months(ym, 1)
    return datetime(int(nxt[:4]), int(nxt[5:7]), min(days, 28), 9, 0)


def _fill(con, iso: str, values: dict[str, float]) -> None:
    for ym, v in values.items():
        con.execute(
            "INSERT INTO acled_monthly_fatalities VALUES (?, ?, ?, 'ACLED', ?)",
            [iso, f"{ym}-01", int(v), _written_after(ym)],
        )


def _stable(n=24, first="2024-07"):
    # 200-300 deaths every month: bucket 100-<500 throughout.
    return {ym: 200 + (i % 3) * 40 for i, ym in enumerate(_months(first, n))}


def _volatile(n=24, first="2024-07"):
    # Swings across four buckets month to month.
    cycle = [3, 40, 700, 150, 12, 1500]
    return {ym: cycle[i % len(cycle)] for i, ym in enumerate(_months(first, n))}


def test_level_is_the_last_month_known_at_forecast_time(tmp_path):
    con = _db(tmp_path)
    _fill(con, "SOM", _stable(first="2024-08"))  # through 2026-07
    # Forecast on 1 October for a window opening in October: September has
    # not settled and is not in the table, and August is.
    spds, src, d = brs.level_volatility_spds(con, "SOM", "2026-10", known_at="2026-10-01")
    assert d["level_month"] == "2026-07"  # August is absent from this table
    _fill(con, "SOM", {"2026-08": 250})
    spds, src, d = brs.level_volatility_spds(con, "SOM", "2026-10", known_at="2026-10-01")
    assert d["level_month"] == "2026-08" and d["level_value"] == 250
    assert src == brs.LEVEL_VOLATILITY_MODEL_SOURCE
    # Scored later, when September is complete, the level is still what the
    # forecaster could have read on 1 October.
    _fill(con, "SOM", {"2026-09": 5000})
    _, _, d_later = brs.level_volatility_spds(con, "SOM", "2026-10", known_at="2026-10-01")
    assert d_later["level_month"] == "2026-08"
    # A mid-month forecast already has the month before it.
    _, _, d_mid = brs.level_volatility_spds(con, "SOM", "2026-10", known_at="2026-09-15")
    assert d_mid["level_month"] == "2026-08"
    # The gap from the level to month 1 is two months on the 1st.
    assert d["horizons"]["1"]["gap_months"] == 2
    assert d["horizons"]["6"]["gap_months"] == 7


def test_a_partial_month_row_is_never_the_level(tmp_path):
    con = _db(tmp_path)
    _fill(con, "AFG", _stable(first="2024-06"))  # through 2026-05
    _fill(con, "AFG", {"2026-06": 117})
    # July written on 15 July: a partial count.
    con.execute(
        "INSERT INTO acled_monthly_fatalities VALUES ('AFG','2026-07-01',9,'ACLED','2026-07-15 10:00')"
    )
    _, _, d = brs.level_volatility_spds(con, "AFG", "2026-08", known_at="2026-08-20")
    assert d["level_month"] == "2026-06" and d["level_value"] == 117
    assert brs.last_observed_value(con, "AFG", "ACE", "FATALITIES", "2026-08")[1] == "2026-06"
    probs, _src, detail = brs.base_rate_spd(con, "AFG", "ACE", "FATALITIES", "2026-08")
    assert 9.0 not in detail["values"]


def test_spread_widens_with_horizon_on_a_volatile_country_and_stays_tight_on_a_stable_one(tmp_path):
    con = _db(tmp_path)
    _fill(con, "STB", _stable(n=40, first="2023-05"))
    _fill(con, "VOL", _volatile(n=40, first="2023-05"))
    s_spds, _, s_d = brs.level_volatility_spds(con, "STB", "2026-10", known_at="2026-10-01")
    v_spds, _, v_d = brs.level_volatility_spds(con, "VOL", "2026-10", known_at="2026-10-01")
    assert not s_d["horizons"]["1"]["pooled"] and not v_d["horizons"]["1"]["pooled"]
    # Stable: nearly everything stays in the level's bucket.
    lb = s_d["level_bucket"]
    assert s_spds[1][lb] > 0.9 and s_spds[6][lb] > 0.9
    assert s_d["horizons"]["1"]["share_same"] == pytest.approx(1.0)
    # Volatile: mass spreads far from the level's bucket.
    assert max(v_spds[1]) < 0.6
    assert v_d["horizons"]["1"]["share_two_plus"] > 0.3


def test_pooled_fallback_when_the_country_record_is_short(tmp_path):
    con = _db(tmp_path)
    # The table itself is short (ACLED's history here began in August 2025),
    # so no country has twelve pairs of its own.
    _fill(con, "NEW", {ym: 250 for ym in _months("2025-11", 10)})
    _fill(con, "OLD1", _stable(n=10, first="2025-11"))
    _fill(con, "OLD2", _stable(n=10, first="2025-11"))
    _fill(con, "OTHER", {ym: 2 for ym in _months("2025-11", 10)})  # another band
    spds, _, d = brs.level_volatility_spds(con, "NEW", "2026-10", known_at="2026-10-01")
    h1 = d["horizons"]["1"]
    assert h1["pooled"] is True
    assert h1["n_own_pairs"] < brs.LEVEL_VOLATILITY_MIN_PAIRS
    assert h1["n_band_countries"] == 2  # the other band stays out
    assert h1["n_pairs"] > h1["n_own_pairs"]


def test_every_vector_sums_to_one_with_the_floor(tmp_path):
    con = _db(tmp_path)
    _fill(con, "VOL", _volatile(n=40, first="2023-05"))
    _fill(con, "STB", _stable(n=40, first="2023-05"))
    for iso in ("VOL", "STB"):
        spds, _, _ = brs.level_volatility_spds(con, iso, "2026-10", known_at="2026-10-01")
        assert set(spds) == {1, 2, 3, 4, 5, 6}
        for vec in spds.values():
            assert len(vec) == 7
            assert sum(vec) == pytest.approx(1.0)
            assert min(vec) >= brs.LEVEL_VOLATILITY_FLOOR / 1.2


def test_a_quiet_live_month_is_a_level_of_zero(tmp_path):
    con = _db(tmp_path)
    _fill(con, "QUI", {ym: 4 for ym in _months("2024-05", 26)})  # through 2026-06
    _fill(con, "OTH", {ym: 100 for ym in _months("2024-05", 28)})  # through 2026-08
    _, _, d = brs.level_volatility_spds(con, "QUI", "2026-10", known_at="2026-10-01")
    assert d["level_month"] == "2026-08" and d["level_value"] == 0.0 and d["level_bucket"] == 0


def test_no_country_row_means_no_anchor(tmp_path):
    con = _db(tmp_path)
    _fill(con, "OTH", _stable())
    spds, src, d = brs.level_volatility_spds(con, "XXX", "2026-10", known_at="2026-10-01")
    assert spds == {} and src == brs.NO_BASE_RATE_SOURCE and d["reason"]
    assert brs.level_volatility_spd(con, "XXX", "2026-10", 1)[0] == []


def test_a_table_without_updated_at_is_read_whole(tmp_path):
    con = duckdb.connect(str(tmp_path / "b.duckdb"))
    con.execute("CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month DATE, fatalities BIGINT)")
    con.execute("INSERT INTO acled_monthly_fatalities VALUES ('SOM','2026-07-01',300)")
    assert brs.acled_complete_month_clause(con) == "TRUE"
    assert brs.last_observed_value(con, "SOM", "ACE", "FATALITIES", "2026-08")[0] == 300.0


def test_known_at_accepts_dates_datetimes_and_strings():
    assert brs._as_date(date(2026, 10, 1)) == date(2026, 10, 1)
    assert brs._as_date(datetime(2026, 10, 1, 4, 5)) == date(2026, 10, 1)
    assert brs._as_date("2026-10-01 04:05:00") == date(2026, 10, 1)
    assert brs._usable_at("2026-08", date(2026, 9, 15))
    assert not brs._usable_at("2026-09", date(2026, 10, 1))
