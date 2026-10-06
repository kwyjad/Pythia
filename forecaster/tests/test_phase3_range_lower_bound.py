# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""The Phase 3+ prompt says each FEWS NET figure is a range's lower bound.

FEWS NET publishes "1.0 - 2.49 million" and the feed's ``value`` is the lower
bound (975 of 999 Current Situation rows, probe run 37315217164). The prompt
printed 1,000,000 as though it were the figure.
"""

from __future__ import annotations

from datetime import date

import pytest

duckdb = pytest.importorskip("duckdb")

import forecaster.cli as cli  # type: ignore
from forecaster.history_loaders import _format_base_rate_for_prompt


def _ym(back: int) -> str:
    d = date.today().replace(day=1)
    idx = d.year * 12 + d.month - 1 - back
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def _summary(monkeypatch, *, with_high_column: bool):
    con = duckdb.connect(":memory:")
    high = ", value_high DOUBLE" if with_high_column else ""
    con.execute(
        "CREATE TABLE facts_resolved (ym TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, "
        f"value DOUBLE, created_at TIMESTAMP, publisher TEXT{high})"
    )
    for back in range(1, 7):
        cols = "(ym, iso3, hazard_code, metric, value, publisher" + (", value_high)" if with_high_column else ")")
        vals = "(?, 'SOM', 'DR', 'phase3plus_in_need', 1000000, 'FEWS NET'" + (", 2490000)" if with_high_column else ")")
        con.execute(f"INSERT INTO facts_resolved {cols} VALUES {vals}", [_ym(back)])

    class _Con:
        def execute(self, *a, **k):
            return con.execute(*a, **k)

        def close(self):
            pass

    monkeypatch.setattr(cli, "connect", lambda read_only=False: _Con())
    return cli._load_fewsnet_phase3_history("SOM", months=36)


def test_the_block_shows_the_range_and_says_the_figure_is_its_lower_bound(monkeypatch):
    summary = _summary(monkeypatch, with_high_column=True)
    assert summary["last_observed"]["value_high"] == 2_490_000
    text = _format_base_rate_for_prompt(summary, "SOM", "DR", metric="PHASE3PLUS_IN_NEED")
    assert "LOWER bound of the population range FEWS NET publishes" in text
    assert "1,000,000 to 2,490,000" in text
    assert "resolves on the lower bound" in text


def test_a_db_without_the_column_still_says_lower_bound(monkeypatch):
    summary = _summary(monkeypatch, with_high_column=False)
    text = _format_base_rate_for_prompt(summary, "SOM", "DR", metric="PHASE3PLUS_IN_NEED")
    assert "LOWER bound" in text
    assert " to 2,490,000" not in text


def test_the_projection_table_carries_the_range(monkeypatch):
    from forecaster import prompts
    from resolver.db import duckdb_io

    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE facts_resolved (ym TEXT, iso3 TEXT, metric TEXT, value DOUBLE, "
        "as_of_date TEXT, value_high DOUBLE)"
    )
    con.execute(
        "INSERT INTO facts_resolved VALUES ('2026-12', 'SOM', 'phase3plus_projection', "
        "1500000, '2027-01-31', 2490000)"
    )
    monkeypatch.setattr(duckdb_io, "get_db", lambda url: con)
    monkeypatch.setattr(duckdb_io, "close_db", lambda c: None)
    out = prompts._load_fewsnet_projection("SOM", ["2026-12", "2027-01"])
    assert "1,500,000 to 2,490,000" in out
    assert "LOWER bound" in out
