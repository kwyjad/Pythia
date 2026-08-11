# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""What the conflict anchors rest on, after the v4 investigation.

The v3 report marked 138 of 185 anchors thin at a twelve-observation cutoff.
Two causes, both in the builder rather than in the data:

* the conflict window was six months against a twelve-observation cutoff, so
  no armed conflict anchor could ever clear the flag. Arithmetic, not
  evidence.
* months in which a source recorded nothing were dropped rather than counted
  as observed zeros, so an anchor built from a country IDMC reports twice a
  year said displacement happens every month.

Both are pinned here, together with the guard that stops the second fix
manufacturing a zero out of an ingestion outage.
"""

from __future__ import annotations

import duckdb
import pytest

from pythia.tools import base_rate_spd


@pytest.fixture()
def con(tmp_path):
    c = duckdb.connect(str(tmp_path / "anchors.duckdb"))
    c.execute(
        "CREATE TABLE acled_monthly_fatalities "
        "(iso3 TEXT, month TEXT, fatalities BIGINT)"
    )
    c.execute(
        "CREATE TABLE facts_deltas (iso3 TEXT, ym TEXT, hazard_code TEXT, "
        "metric TEXT, series_semantics TEXT, source_id TEXT, value_new DOUBLE)"
    )
    yield c
    c.close()


def _acled(con, rows):
    con.executemany(
        "INSERT INTO acled_monthly_fatalities VALUES (?, ?, ?)", rows
    )


class TestTheWindowIsLongerThanTheCutoff:
    def test_a_conflict_anchor_can_now_clear_the_thin_flag(self, con):
        # Two years of a country reporting every month. Under the old
        # six-month window this anchor rested on six observations against a
        # twelve-observation cutoff and was thin by construction.
        rows = []
        for year in (2024, 2025):
            for month in range(1, 13):
                rows.append(("ETH", f"{year}-{month:02d}", 40))
        _acled(con, rows)
        probs, source, detail = base_rate_spd.base_rate_spd(
            con, "ETH", "ACE", "FATALITIES", "2026-01"
        )
        assert probs
        assert detail["n_months_used"] == 24
        assert detail["n_months_used"] >= 12, "the cutoff must be reachable"
        assert base_rate_spd.CONFLICT_WINDOW_MONTHS > 12

    def test_history_strictly_before_the_window_only(self, con):
        # The leakage rule is unchanged by the wider window.
        _acled(con, [("ETH", "2025-12", 40), ("ETH", "2026-01", 9_999)])
        _, _, detail = base_rate_spd.base_rate_spd(
            con, "ETH", "ACE", "FATALITIES", "2026-01"
        )
        assert 9_999 not in detail["values"]


class TestQuietMonthsAreObservations:
    def test_a_month_the_source_covered_and_did_not_report_is_a_zero(self, con):
        # ETH reports every month; SOM reports in one of them. SOM's other
        # months are observed zeros, not absent observations.
        rows = [("ETH", f"2025-{m:02d}", 10) for m in range(1, 13)]
        rows.append(("SOM", "2025-06", 300))
        _acled(con, rows)
        probs, _, detail = base_rate_spd.base_rate_spd(
            con, "SOM", "ACE", "FATALITIES", "2026-01"
        )
        assert detail["n_months_reported"] == 1
        assert detail["n_months_quiet"] == 11
        # The anchor now says most months are quiet, which is what the record
        # shows. Built from present rows alone it said every month is violent.
        assert probs[0] > 0.5

    def test_a_month_the_source_never_covered_is_not_an_observation(self, con):
        # Only 2025-06 exists in the table at all: the ingest covered one
        # month. Counting the other eleven as quiet would manufacture zeros
        # out of an outage, which is the rule source_coverage exists for.
        _acled(con, [("SOM", "2025-06", 300)])
        _, _, detail = base_rate_spd.base_rate_spd(
            con, "SOM", "ACE", "FATALITIES", "2026-01"
        )
        assert detail["n_months_used"] == 1
        assert detail["n_months_quiet"] == 0

    def test_a_country_outside_the_source_universe_gets_no_anchor(self, con):
        # A country that never appears cannot be given a run of zeros: its
        # silence says nothing about it.
        _acled(con, [("ETH", f"2025-{m:02d}", 10) for m in range(1, 13)])
        probs, source, _ = base_rate_spd.base_rate_spd(
            con, "NIC", "ACE", "FATALITIES", "2026-01"
        )
        assert probs == []
        assert source == base_rate_spd.NO_BASE_RATE_SOURCE

    def test_the_displacement_anchor_counts_its_quiet_months_too(self, con):
        # IDMC reports a country only when it records displacement, so an
        # anchor from present rows alone said displacement every month.
        rows = []
        for month in range(1, 13):
            rows.append(("ETH", f"2025-{month:02d}", "ACE", "new_displacements",
                         "new", "idmc", 5_000.0))
        rows.append(("SOM", "2025-06", "ACE", "new_displacements", "new",
                     "idmc", 80_000.0))
        con.executemany(
            "INSERT INTO facts_deltas VALUES (?, ?, ?, ?, ?, ?, ?)", rows
        )
        probs, _, detail = base_rate_spd.base_rate_spd(
            con, "SOM", "ACE", "PA", "2026-01"
        )
        assert detail["n_months_reported"] == 1
        assert detail["n_months_quiet"] == 11
        assert probs[0] > 0.5

    def test_the_live_gate_reads_the_same_source_the_anchor_does(self, con):
        # A month in which some OTHER publisher wrote to facts_deltas is not
        # a month IDMC was live for.
        con.executemany(
            "INSERT INTO facts_deltas VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("SOM", "2025-06", "ACE", "new_displacements", "new", "idmc",
                 80_000.0),
                ("KEN", "2025-07", "FL", "affected", "new", "gdacs", 1.0),
            ],
        )
        _, _, detail = base_rate_spd.base_rate_spd(
            con, "SOM", "ACE", "PA", "2026-01"
        )
        # 2025-07 belongs to gdacs, so it is not an IDMC observation.
        assert detail["n_months_quiet"] == 0
        assert detail["n_months_used"] == 1


class TestTheAnchorCheckDiagnosesRatherThanTunes:
    def test_it_names_a_window_shorter_than_the_cutoff(self):
        from interpreter import anchorcheck

        rows = [
            {"hazard_code": "ACE", "metric": "FATALITIES", "n_obs": 6,
             "p_zero": 0.4, "method": "empirical_monthly_buckets",
             "window_months": 6},
            {"hazard_code": "ACE", "metric": "FATALITIES", "n_obs": 5,
             "p_zero": 0.4, "method": "empirical_monthly_buckets",
             "window_months": 6},
        ]
        summary = anchorcheck.summarise(rows, thin_below=12)
        notes = anchorcheck._diagnosis(summary, thin_below=12)
        assert any("EVERY anchor is thin" in n for n in notes)
        assert any("arithmetic, not data" in n for n in notes)

    def test_it_names_an_anchor_with_no_weight_on_a_quiet_month(self):
        from interpreter import anchorcheck

        rows = [
            {"hazard_code": "ACE", "metric": "PA", "n_obs": 40,
             "p_zero": 0.001, "method": "empirical_monthly_buckets",
             "window_months": 36},
        ]
        summary = anchorcheck.summarise(rows, thin_below=12)
        notes = anchorcheck._diagnosis(summary, thin_below=12)
        assert any("quiet months are being dropped" in n for n in notes)
