# Pythia / Copyright (c) 2025 Kevin Wyjad
"""The ACLED history backfill reads one calendar year at a time.

``acled_monthly_fatalities`` began in 2025-08, so the 36-month windows the
climatology and level-volatility references read held about thirteen months.
The backfill reaches back years; splitting it by year is what lets the run
say what the account was actually served per year, and keeps one refused
year from costing the others.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from resolver.ingestion.acled_client import ACLEDClient


def _client(fetch):
    client = ACLEDClient.__new__(ACLEDClient)
    client.fields = list(ACLEDClient._DEFAULT_FIELDS)
    client.use_stub = False
    client.logger = logging.getLogger("test")
    client.fetch_events = fetch
    return client


def _events(year: int, n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "event_date": [pd.Timestamp(f"{year}-0{1 + i % 9}-15") for i in range(n)],
        "iso3": ["SOM" if i % 2 else "KEN" for i in range(n)],
        "country": ["x"] * n,
        "fatalities": [1] * n,
    })


def test_a_multi_year_window_is_fetched_year_by_year() -> None:
    calls = []

    def fetch(start, end, countries=None):
        calls.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
        return _events(start.year, 0 if start.year == 2018 else 4)

    client = _client(fetch)
    out = client.monthly_fatalities("2018-01-01", "2020-06-30")
    assert calls == [
        ("2018-01-01", "2018-12-31"),
        ("2019-01-01", "2019-12-31"),
        ("2020-01-01", "2020-06-30"),
    ]
    by_year = {r["year"]: r for r in client.year_summary}
    assert by_year[2018]["events"] == 0 and by_year[2018]["error"] is None
    assert by_year[2019]["events"] == 4 and by_year[2019]["countries"] == 2
    assert set(out["month"].dt.year) == {2019, 2020}


def test_one_refused_year_does_not_cost_the_others() -> None:
    def fetch(start, end, countries=None):
        if start.year == 2018:
            raise RuntimeError("HTTP 403 recency limit")
        return _events(start.year, 2)

    client = _client(fetch)
    out = client.monthly_fatalities("2018-01-01", "2019-12-31")
    assert "403" in client.year_summary[0]["error"]
    assert set(out["month"].dt.year) == {2019}


def test_every_year_refused_is_a_read_failure() -> None:
    def fetch(start, end, countries=None):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _client(fetch).monthly_fatalities("2018-01-01", "2019-12-31")


def test_a_window_inside_one_year_is_one_call() -> None:
    calls = []

    def fetch(start, end, countries=None):
        calls.append(start.year)
        return _events(2026, 2)

    _client(fetch).monthly_fatalities("2026-07-01", "2026-09-30")
    assert calls == [2026]
