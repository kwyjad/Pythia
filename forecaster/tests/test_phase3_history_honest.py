# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""The Phase 3+ history block counts the window it shows (Oct 2026).

Ethiopia's 1 October 2026 prompt read "reported in 36 of the last 36 months
(100% coverage) ... Data quality: high" above six null months: the loader
counted the 36 newest analyses wherever they fell, and its newest Current
Situation figure was 2026-01.
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


@pytest.fixture()
def eth(monkeypatch):
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE facts_resolved (ym TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, "
        "value DOUBLE, created_at TIMESTAMP, publisher TEXT)"
    )
    # 36 consecutive monthly analyses ending nine months ago.
    for back in range(9, 45):
        con.execute(
            "INSERT INTO facts_resolved VALUES (?, 'ETH', 'DR', 'phase3plus_in_need', 8000000, NULL, 'FEWS NET')",
            [_ym(back)],
        )
    con.execute(
        "INSERT INTO facts_resolved VALUES (?, 'ETH', 'DR', 'phase3plus_projection', 9000000, NULL, 'FEWS NET')",
        [_ym(-2)],
    )

    class _Con:
        def execute(self, *a, **k):
            return con.execute(*a, **k)

        def close(self):
            pass

    monkeypatch.setattr(cli, "connect", lambda read_only=False: _Con())
    return cli._load_fewsnet_phase3_history("ETH", months=36)


def test_coverage_counts_the_window_only(eth):
    assert eth["observed_months"] == 27
    assert eth["coverage_pct"] == pytest.approx(75.0)


def test_a_full_but_old_record_is_not_high_quality(eth):
    assert eth["data_quality"] == "low"
    text = _format_base_rate_for_prompt(eth, [], metric="PHASE3PLUS_IN_NEED")
    assert "27 of the last 36 months" in text
    assert f"Last observed value: {_ym(9)}: 8,000,000 (9 months before this forecast)" in text
    assert "Data quality: high" not in text
    assert f"no observation since {_ym(9)}" in text


def test_projections_are_shown_apart_and_labelled(eth):
    text = _format_base_rate_for_prompt(eth, [], metric="PHASE3PLUS_IN_NEED")
    assert "Projections (Most Likely scenario, a FORECAST" in text
    assert f"{_ym(-2)}: 9,000,000" in text
    # A projection never fills a Current Situation month.
    assert all(e["value"] is None for e in eth["last_6m_values"])
