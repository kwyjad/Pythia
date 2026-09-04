# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Phase-4 acceptance tests for the drought path.

The Phase 4 acceptance criterion, in its own words: *run on two countries
with recent, well-known drought-driven IPC deterioration and two quiet
countries; resolutions match hand calculation from the published IPC
figures; the NO_DATA path fires where indicators show drought but IPC
coverage is absent.* :class:`TestAcceptance` at the bottom of this file is
exactly that run.

Everything here is network-free. IPC analyses and indicator snapshots are
seeded into their raw caches, and the two connectors are exercised through
their injectable transport seams — so the suite pins the deterministic
half of the drought rule without a network or a key.

The rule under test:

    people affected = max(0, Phase3+(analysis covering the month)
                             - Phase3+(previous current-period analysis))

admitted as drought impact only where the drought indicators say the
country was in drought.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from resolver.hazard_resolution import drought as drought_mod
from resolver.hazard_resolution import drought_indicators as ind_mod
from resolver.hazard_resolution import ipc as ipc_mod
from resolver.hazard_resolution import resolutions as res_mod
from resolver.hazard_resolution.rulebook import RulebookError, validate_rulebook
from resolver.hazard_resolution.rules import drought_delta_qualifies
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.tests.hazard_resolution_utils import (
    make_rulebook,
    seed_indicator_snapshot,
    seed_ipc_analysis,
    seed_population,
)

YM = "2024-03"
#: Long past 2024-03 + freeze_days, so an unresolvable cell is NO_DATA
#: rather than PENDING.
AFTER_FREEZE = dt.date(2024, 9, 1)
BEFORE_FREEZE = dt.date(2024, 4, 1)


#: Two fake routes for the legacy ASAP warnings feed, so the candidate-list
#: tests below can exercise "first route dead, second answers".
LEGACY_ASAP_URLS = [
    "https://example.test/asap/warnings_v1.json",
    "https://example.test/asap/warnings_v2.json",
]


def legacy_feed_rulebook(overrides: dict | None = None):
    """The shipped rulebook with the LEGACY single-feed mechanics restored.

    This module pins the drought rule's mechanics as they were built: one
    ASAP warnings feed, a zero resting on its silence. Since Sept 2026 the
    shipped rulebook (a) leaves the legacy ``asap`` entry dormant, because
    JRC no longer publishes the warnings file, and reads the hotspot time
    series through ``asap_hotspots`` instead, and (b) demands
    ``min_answered_for_zero: 2`` feeds before a zero may rest on absence.
    Both are tested on the SHIPPED values in
    ``test_hazard_resolution_drought_group_d.py``; here they are set back so
    each existing case keeps asserting the one thing it was written for.
    """

    data = make_rulebook()._data
    entries = []
    for entry in data["drought"]["indicators"]["entries"]:
        entry = dict(entry)
        if entry["name"] == "asap":
            entry["urls"] = list(LEGACY_ASAP_URLS)
        if entry["name"] == "asap_hotspots":
            entry["urls"] = []
            entry["url"] = ""
        entries.append(entry)
    base = {
        "drought": {
            "indicators": {"entries": entries, "min_answered_for_zero": 1}
        }
    }
    if overrides:
        from resolver.tests.hazard_resolution_utils import _deep_merge

        base = _deep_merge(base, overrides)
    return make_rulebook(base)


@pytest.fixture()
def rulebook():
    return legacy_feed_rulebook()


@pytest.fixture()
def con():
    con = duckdb.connect(":memory:")
    ensure_haz_schema(con)
    return con


def _drought_indicators(iso3s, *, ym: str = YM, quiet=()):
    """An ASAP snapshot marking ``iso3s`` in drought and ``quiet`` clear."""

    values = {code: "3" for code in iso3s}
    values.update({code: "0" for code in quiet})
    return values


def _verdict(con, iso3, rulebook, ym: str = YM):
    return ind_mod.evaluate_indicators(con, iso3, ym, rulebook)


def _raise_dead():
    """A transport that never answers — the ASAP outage, reproduced."""

    raise RuntimeError("404 Not Found")


# ---------------------------------------------------------------------------
# The rulebook block
# ---------------------------------------------------------------------------


class TestRulebook:
    def test_shipped_rulebook_configures_the_drought_rule(self):
        rulebook = make_rulebook()  # the SHIPPED values, not the legacy fixture
        assert rulebook.get("drought.rule") == "ipc_phase3plus_delta"
        assert rulebook.get("drought.month_attribution") in (
            "analysis_window",
            "start_month",
        )
        entries = rulebook.get("drought.indicators.entries")
        # A total indicator outage must never read as "no drought anywhere".
        # Two ways to guarantee that: nominate a required entry, or demand a
        # minimum number of ANSWERS. The shipped rulebook uses the second,
        # because naming one feed as required makes that feed a single point
        # of failure for every cell — which is how one dead ASAP URL left 756
        # drought cells inconclusive in a single run.
        assert (
            any(e["required"] for e in entries)
            or int(rulebook.get("drought.indicators.min_available")) >= 1
        )
        # The ASAP endpoint is configurable, never hard-coded in Python —
        # and since Sept 2026 it is a candidate LIST, because the portal has
        # moved its download route without moving the data. The warnings
        # feed (`asap`) is dormant; the hotspot time series carries the
        # addresses, and reaches HDX's mirror through the CKAN API.
        asap = next(e for e in entries if e["name"] == "asap_hotspots")
        addresses = list(asap.get("urls") or []) + [asap.get("url") or ""]
        assert any(a.startswith("https://") for a in addresses)
        assert any(a.startswith("hdx-ckan://") for a in addresses)
        assert asap.get("time_series") is True
        assert asap.get("absence_means_no_drought") is False
        assert int(rulebook.get("drought.indicators.min_answered_for_zero")) >= 2

    def test_a_required_indicator_with_nothing_to_read_is_rejected(self):
        """It could never be read, so it would suppress every drought zero."""

        data = make_rulebook().raw
        entry = data["drought"]["indicators"]["entries"][0]
        entry["required"] = True
        entry["url"] = ""
        entry["urls"] = []
        problems = validate_rulebook(data)
        assert any("has nothing to read from" in p for p in problems)

    def test_a_required_entry_is_satisfied_by_a_candidate_url_list(self):
        """`urls` addresses an entry just as `url` does.

        A portal that moves its download route has not lost the data, and the
        candidate list is how a moved route costs a YAML edit rather than an
        outage.
        """

        data = make_rulebook().raw
        entry = data["drought"]["indicators"]["entries"][0]
        entry["required"] = True
        entry["url"] = ""
        entry["urls"] = ["https://example.test/a.json", "https://example.test/b.json"]
        assert not any(
            "has nothing to read from" in p for p in validate_rulebook(data)
        )

    def test_no_required_entry_and_no_minimum_is_rejected(self):
        """A total indicator outage would otherwise zero the whole world."""

        data = make_rulebook().raw
        data["drought"]["indicators"]["min_available"] = 0
        for entry in data["drought"]["indicators"]["entries"]:
            entry["required"] = False
            entry["url"] = entry["url"] or "https://example.test/feed.json"
        problems = validate_rulebook(data)
        assert any("min_available >= 1" in p for p in problems)

    def test_a_minimum_larger_than_the_entry_list_is_rejected(self):
        """No verdict could ever be reached, so no drought could ever resolve."""

        data = make_rulebook().raw
        data["drought"]["indicators"]["min_available"] = 9
        assert any(
            "no verdict could ever be reached" in p for p in validate_rulebook(data)
        )

    def test_a_lookback_too_short_to_hold_a_baseline_is_rejected(self):
        data = make_rulebook().raw
        data["drought"]["ipc"]["lookback_months"] = 3
        problems = validate_rulebook(data)
        assert any("lookback_months must be >= 12" in p for p in problems)

    def test_unknown_month_attribution_is_rejected(self):
        data = make_rulebook().raw
        data["drought"]["month_attribution"] = "spread_evenly"
        problems = validate_rulebook(data)
        assert any("drought.month_attribution" in p for p in problems)

    def test_no_credential_may_live_in_the_drought_block(self):
        data = make_rulebook().raw
        data["drought"]["ipc"]["api_key"] = "secret-value"
        problems = validate_rulebook(data)
        assert any("looks like a credential" in p for p in problems)


class TestDeltaThresholds:
    def test_any_increase_qualifies_at_the_shipped_defaults(self, rulebook):
        assert drought_delta_qualifies(1.0, 100_000.0, rulebook) is True

    def test_a_flat_or_falling_analysis_never_qualifies(self, rulebook):
        assert drought_delta_qualifies(0.0, 100_000.0, rulebook) is False

    def test_absolute_and_relative_minimums_are_honoured(self):
        rb = make_rulebook({"drought": {"min_delta_people": 50_000}})
        assert drought_delta_qualifies(40_000, 1_000_000, rb) is False
        assert drought_delta_qualifies(60_000, 1_000_000, rb) is True

        rb = make_rulebook({"drought": {"min_delta_pct": 25.0}})
        # 10% of the baseline — a real increase, below the configured bar.
        assert drought_delta_qualifies(100_000, 1_000_000, rb) is False
        assert drought_delta_qualifies(300_000, 1_000_000, rb) is True


# ---------------------------------------------------------------------------
# The IPC connector
# ---------------------------------------------------------------------------


class TestIpcConnector:
    def test_the_borrowed_period_parser_still_exists(self):
        """Guard on the reuse contract with resolver/connectors/ipc_api.py."""

        parse = ipc_mod._ipc_connector_api()
        assert parse("Jun 2025 - Sep 2025") == ("2025-06-01", "2025-09-01")

    def test_api_path_stores_current_period_analyses_only(
        self, con, rulebook, monkeypatch
    ):
        monkeypatch.setenv(ipc_mod.API_KEY_ENV, "test-key")
        payload = [
            {
                "country": "SO",
                "country_name": "Somalia",
                "p3plus": 4_400_000,
                "p3plus_projected": 6_000_000,
                "current_period_dates": "Jan 2024 - Mar 2024",
                "projected_period_dates": "Apr 2024 - Jun 2024",
                "analysis_date": "2023-12-20",
            }
        ]
        outcome = ipc_mod.fetch_ipc(
            con, YM, rulebook, get=lambda url, params, timeout: payload
        )
        assert outcome.ok
        analyses = ipc_mod.analyses_for_country(con, "SOM", rulebook)
        assert len(analyses) == 1, "the projection must not enter the cache"
        analysis = analyses[0]
        assert analysis.value == 4_400_000
        assert analysis.months == ["2024-01", "2024-02", "2024-03"]

    def test_iso2_codes_convert_when_pycountry_is_available(
        self, con, rulebook, monkeypatch
    ):
        """The IPC API reports ISO2; the drought path stores ISO3."""

        pytest.importorskip("pycountry")
        monkeypatch.setenv(ipc_mod.API_KEY_ENV, "test-key")
        payload = [
            {
                "country": "SO",
                "p3plus": 4_400_000,
                "current_period_dates": "Jan 2024 - Mar 2024",
            }
        ]
        ipc_mod.fetch_ipc(con, YM, rulebook, get=lambda url, params, timeout: payload)
        assert ipc_mod.analyses_for_country(con, "SOM", rulebook)

    def test_a_missing_key_falls_back_to_facts_resolved(
        self, con, rulebook, monkeypatch
    ):
        """FEWS NET countries are exactly the ones the IPC API connector drops."""

        monkeypatch.delenv(ipc_mod.API_KEY_ENV, raising=False)
        con.execute(
            """
            CREATE TABLE facts_resolved (
                ym TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, value DOUBLE,
                as_of_date VARCHAR, publication_date VARCHAR, publisher TEXT,
                source_url TEXT
            )
            """
        )
        con.execute(
            """
            INSERT INTO facts_resolved VALUES
                ('2024-01', 'ETH', 'DR', 'phase3plus_in_need', 3200000,
                 '2024-04-30', '2023-12-15', 'FEWS NET', 'https://fdw.fews.net')
            """
        )
        outcome = ipc_mod.fetch_ipc(con, YM, rulebook)
        assert outcome.ok, "the facts_resolved path must work without a key"
        analyses = ipc_mod.analyses_for_country(con, "ETH", rulebook)
        assert len(analyses) == 1
        assert analyses[0].value == 3_200_000
        assert analyses[0].path == ipc_mod.PATH_FACTS
        # ym is the window's first month, as_of_date its last.
        assert analyses[0].months == ["2024-01", "2024-02", "2024-03", "2024-04"]

    def test_both_paths_failing_is_reported_not_swallowed(
        self, con, rulebook, monkeypatch
    ):
        monkeypatch.setenv(ipc_mod.API_KEY_ENV, "test-key")
        rb = make_rulebook({"drought": {"ipc": {"use_facts_resolved": False}}})

        def boom(url, params, timeout):
            raise RuntimeError("IPC API down")

        outcome = ipc_mod.fetch_ipc(con, YM, rb, get=boom)
        assert outcome.ok is False
        assert "no IPC acquisition path answered" in (outcome.error or "")

    def test_one_analysis_seen_by_both_paths_collapses_by_priority(
        self, con, rulebook
    ):
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2024-01-01", window_end="2024-03-31",
            value=4_400_000, path="ipc_api",
        )
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2024-01-01", window_end="2024-03-31",
            value=4_100_000, path="facts_resolved",
        )
        analyses = ipc_mod.analyses_for_country(con, "SOM", rulebook)
        assert len(analyses) == 1, "one real analysis must not become two"
        assert analyses[0].path == "ipc_api"

    def test_covering_and_previous_selection(self, con, rulebook):
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2023-07-01", window_end="2023-09-30",
            value=3_000_000,
        )
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2024-01-01", window_end="2024-03-31",
            value=4_400_000,
        )
        analyses = ipc_mod.analyses_for_country(con, "SOM", rulebook)
        current = ipc_mod.covering_analysis(analyses, YM)
        assert current is not None and current.value == 4_400_000
        previous = ipc_mod.previous_analysis(analyses, current)
        assert previous is not None and previous.value == 3_000_000
        assert ipc_mod.covering_analysis(analyses, "2023-11") is None


# ---------------------------------------------------------------------------
# The indicators
# ---------------------------------------------------------------------------


class TestIndicatorSourcesBeyondAsap:
    """The gate must not rest on a single URL.

    Until Sept 2026 ASAP was the only required indicator, so when its
    download route vanished every drought cell in the run came back
    INCONCLUSIVE — 756 of them. The repair is not a louder log: it is more
    ways to satisfy the same requirement, drawn from tables this repo
    already ingests.
    """

    @staticmethod
    def _refresh(con, rulebook):
        """Run the fetch so the seeded tables reach the indicator cache.

        The module stores VALUES and applies thresholds at evaluation time,
        so seeding a table is not enough on its own — the snapshot is what
        the verdict reads.
        """

        return ind_mod.fetch_indicators(
            con, YM, rulebook,
            get=lambda url, timeout: (_raise_dead()),
        )

    @staticmethod
    def _seed_hdx(con, values: dict[str, str], signal_date: str = "2024-03-15"):
        """HDX Signals rows, in the shape the connector writes them."""

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS hdx_signals (
                iso3 VARCHAR, hazard_code VARCHAR, indicator VARCHAR,
                concern_level VARCHAR, indicator_value DOUBLE,
                description VARCHAR, source_url VARCHAR, signal_date VARCHAR
            )
            """
        )
        for iso3, level in values.items():
            con.execute(
                "INSERT INTO hdx_signals (iso3, hazard_code, indicator, "
                "concern_level, signal_date) VALUES (?, 'DR', "
                "'jrc_agricultural_hotspots', ?, ?)",
                [iso3, level, signal_date],
            )

    @staticmethod
    def _seed_nmme(con, values: dict[str, float], issue_date: str = "2024-02-01"):
        """NMME rows as the ingest writes them.

        A lead-1 forecast issued in February is ABOUT March (leads are
        1-based and count from the issue month), so the row that speaks
        for the test month YM=2024-03 is issued the month before. Seeding it
        with a March issue date would describe April.
        """

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS seasonal_forecasts (
                iso3 TEXT, variable TEXT, lead_months INTEGER,
                anomaly_value DOUBLE, tercile_category TEXT,
                forecast_issue_date DATE
            )
            """
        )
        for iso3, anomaly in values.items():
            con.execute(
                "INSERT INTO seasonal_forecasts (iso3, variable, lead_months, "
                "anomaly_value, forecast_issue_date) VALUES (?, 'prate', 1, ?, ?)",
                [iso3, anomaly, issue_date],
            )

    def test_hdx_signals_satisfy_the_gate_when_asap_is_unavailable(
        self, con, rulebook
    ):
        """The headline case: no ASAP snapshot at all, and drought still resolves.

        HDX Signals' ``jrc_agricultural_hotspots`` indicator IS ASAP, reached
        through a connector this repo already maintains — so this is the same
        underlying evidence arriving by a route that did not break.
        """

        self._seed_hdx(con, {"SOM": "High concern"})
        self._refresh(con, rulebook)

        verdict = _verdict(con, "SOM", rulebook)
        assert verdict.state == ind_mod.STATE_DROUGHT
        assert verdict.available
        # ASAP itself contributed nothing.
        asap = next(r for r in verdict.readings if r.name == "asap")
        assert asap.state == ind_mod.STATE_UNAVAILABLE

    def test_a_resolution_row_is_written_on_hdx_evidence_alone(self, con, rulebook):
        """The whole point: a row that does not exist today, existing.

        A deterioration with a drought signal behind it resolves to the
        delta. Before this change the same cell was INCONCLUSIVE because the
        one required indicator could not be read.
        """

        self._seed_hdx(con, {"SOM": "High concern"})
        self._refresh(con, rulebook)
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2023-10-01", window_end="2023-12-31",
            value=1_000_000,
        )
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2024-02-01", window_end="2024-04-30",
            value=1_400_000,
        )
        seed_population(con, iso3="SOM", year=2024, population=17_000_000)

        outcome = _decide(con, rulebook, "SOM")
        assert outcome.status == drought_mod.STATUS_RESOLVED_VALUE
        assert outcome.value == pytest.approx(400_000)

    def test_absence_from_the_hdx_feed_is_the_statement_that_none_was_issued(
        self, con, rulebook
    ):
        """HDX Signals is an alerting feed, so absence is evidence of absence.

        It publishes signals, not a global status — the same property that
        makes ASAP usable for zeros.
        """

        self._seed_hdx(con, {"ETH": "High concern"})
        self._refresh(con, rulebook)
        verdict = _verdict(con, "SOM", rulebook)
        assert verdict.state == ind_mod.STATE_NO_DROUGHT
        hdx = next(r for r in verdict.readings if r.name == "hdx_agricultural_stress")
        assert hdx.state == ind_mod.STATE_NO_DROUGHT

    def test_an_nmme_anomaly_is_read_against_the_rulebook_threshold(
        self, con, rulebook
    ):
        self._seed_nmme(con, {"SOM": -1.6, "KEN": -0.2})
        self._refresh(con, rulebook)

        assert _verdict(con, "SOM", rulebook).state == ind_mod.STATE_DROUGHT
        # -0.2 sigma is an ordinary month. Absence of drought here, not
        # absence of information: NMME answered about Kenya.
        kenya = _verdict(con, "KEN", rulebook)
        nmme = next(r for r in kenya.readings if r.name == "nmme_precip_anomaly")
        assert nmme.state == ind_mod.STATE_NO_DROUGHT

    def test_nmme_is_not_thresholded_at_the_tercile_boundary(self, rulebook):
        """-0.43 sigma is a third of countries in any month, by construction.

        Under combine: any, that would attribute a third of the world's food
        insecurity to drought — the conflict-driven deterioration this gate
        exists to keep out.
        """

        entry = next(
            e for e in rulebook.get("drought.indicators.entries")
            if e["name"] == "nmme_precip_anomaly"
        )
        assert float(entry["threshold"]) <= -1.0

    def test_a_country_absent_from_an_anomaly_feed_is_unknown_not_dry(
        self, con, rulebook
    ):
        self._seed_nmme(con, {"ETH": -1.6})
        self._refresh(con, rulebook)
        verdict = _verdict(con, "SOM", rulebook)
        nmme = next(r for r in verdict.readings if r.name == "nmme_precip_anomaly")
        assert nmme.state == ind_mod.STATE_UNAVAILABLE

    def test_no_indicator_answering_is_still_inconclusive(self, con, rulebook):
        """More sources is not the same as fewer standards.

        With nothing seeded, nothing has been checked — and the machine may
        neither attribute a deterioration nor write a zero.
        """

        verdict = _verdict(con, "SOM", rulebook)
        assert verdict.state == ind_mod.STATE_UNAVAILABLE
        assert "min_available" in verdict.note

    def test_fews_net_phase3_is_deliberately_not_an_indicator(self, rulebook):
        """It is the quantity being differenced, so it cannot attribute itself.

        Admitting it would reduce the gate to "IPC rose, therefore drought,
        because IPC rose", and file conflict-driven food insecurity as
        drought impact.
        """

        for entry in rulebook.get("drought.indicators.entries"):
            table = str(entry.get("table") or "")
            where = str(entry.get("where") or "")
            assert "facts_resolved" not in table
            assert "phase3plus" not in where

    def test_a_missing_table_is_unavailable_not_no_drought(self, con, rulebook):
        """An absent table means we did not check, not that nothing happened."""

        outcome = ind_mod.fetch_indicators(
            con, YM, rulebook, get=lambda url, timeout: ("[]", "application/json")
        )
        hdx = outcome.detail["entries"]["hdx_agricultural_stress"]
        assert hdx["ok"] is False
        assert _verdict(con, "SOM", rulebook).state == ind_mod.STATE_UNAVAILABLE


class TestCandidateUrls:
    """A moved download route should cost a YAML edit, not an outage."""

    def test_the_second_candidate_answers_when_the_first_is_dead(
        self, con, rulebook
    ):
        body = json.dumps([{"iso3": "SOM", "asap_warning": "3", "ym": YM}])
        calls: list[str] = []

        def get(url: str, timeout: float):
            calls.append(url)
            if len(calls) == 1:
                raise RuntimeError("404 Not Found")
            return body, "application/json"

        outcome = ind_mod.fetch_indicators(con, YM, rulebook, get=get)
        asap = outcome.detail["entries"]["asap"]
        assert asap["ok"] is True
        assert asap["attempts"] and "404" in asap["attempts"][0]["error"]
        assert _verdict(con, "SOM", rulebook).state == ind_mod.STATE_DROUGHT

    def test_every_candidate_failing_names_the_last_reason(self, con, rulebook):
        """"The shape changed" and "the route is dead" want different repairs."""

        def get(url: str, timeout: float):
            raise RuntimeError("connection refused")

        outcome = ind_mod.fetch_indicators(con, YM, rulebook, get=get)
        asap = outcome.detail["entries"]["asap"]
        assert asap["ok"] is False
        assert "connection refused" in asap["error"]
        assert len(asap["attempts"]) >= 2


class TestIndicators:
    def test_absence_from_the_asap_feed_means_no_warning(self, con, rulebook):
        seed_indicator_snapshot(con, values={"SOM": "3"})
        assert _verdict(con, "SOM", rulebook).state == ind_mod.STATE_DROUGHT
        # URY is not in the feed at all: ASAP lists what it has warned about.
        assert _verdict(con, "URY", rulebook).state == ind_mod.STATE_NO_DROUGHT

    def test_thresholds_are_applied_at_evaluation_not_at_fetch(self, con):
        """Retuning the rulebook changes the verdict with no re-fetch."""

        seed_indicator_snapshot(con, values={"SOM": "1"})
        strict = legacy_feed_rulebook()  # shipped drought_classes: 2, 3, 4, …
        assert _verdict(con, "SOM", strict).state == ind_mod.STATE_NO_DROUGHT

        entries = legacy_feed_rulebook().get("drought.indicators.entries")
        loosened = [
            {**entry, "drought_classes": ["1", "2", "3", "4"]}
            if entry["name"] == "asap" else entry
            for entry in entries
        ]
        loose = legacy_feed_rulebook({"drought": {"indicators": {"entries": loosened}}})
        # Same cached snapshot, different rulebook, different verdict.
        assert _verdict(con, "SOM", loose).state == ind_mod.STATE_DROUGHT

    def test_a_stale_snapshot_does_not_speak_for_the_month(self, con, rulebook):
        """max_observation_age_months is what stops a backcast inventing zeros."""

        seed_indicator_snapshot(con, observed_ym="2023-01", values={"SOM": "3"})
        verdict = _verdict(con, "SOM", rulebook)
        assert verdict.state == ind_mod.STATE_UNAVAILABLE
        assert "no cached observation" in (verdict.readings[0].error or "")

    def test_a_later_snapshot_is_never_read_backwards(self, con, rulebook):
        """A September warning must not become evidence about March."""

        seed_indicator_snapshot(con, observed_ym="2024-09", values={"SOM": "3"})
        assert _verdict(con, "SOM", rulebook).state == ind_mod.STATE_UNAVAILABLE

    def test_a_required_indicator_that_cannot_be_read_blocks_every_verdict(
        self, con, rulebook
    ):
        # Nothing cached at all.
        verdict = _verdict(con, "SOM", rulebook)
        assert verdict.state == ind_mod.STATE_UNAVAILABLE
        assert verdict.available is False

    def test_an_anomaly_feed_absence_is_unknown_not_dry(self, con):
        rb = make_rulebook(
            {
                "drought": {
                    "indicators": {
                        "entries": [
                            {
                                "name": "spei3",
                                "provider": "tabular",
                                "url": "https://example.test/spei.csv",
                                "required": True,
                                "request_timeout_sec": 60,
                                "threshold": -1.0,
                                "direction": "below",
                            }
                        ]
                    }
                }
            }
        )
        seed_indicator_snapshot(
            con, name="spei3", provider="tabular", values={"SOM": -1.8}
        )
        assert _verdict(con, "SOM", rb).state == ind_mod.STATE_DROUGHT
        # A country the anomaly feed says nothing about is unknown, not wet.
        assert _verdict(con, "URY", rb).state == ind_mod.STATE_UNAVAILABLE

    def test_a_feed_whose_shape_changed_is_refused_not_cached(self, con, rulebook):
        """0 of N records resolving means the parser broke, not that the
        world is drought-free."""

        body = json.dumps([{"mystery_code": "SOM", "asap_warning": "3"}])
        outcome = ind_mod.fetch_indicators(
            con, YM, rulebook, get=lambda url, timeout: (body, "application/json")
        )
        assert outcome.ok is False
        asap = outcome.detail["entries"]["asap"]
        assert asap["ok"] is False and "records resolved" in asap["error"]
        assert _verdict(con, "SOM", rulebook).state == ind_mod.STATE_UNAVAILABLE

    def test_a_dateless_feed_is_stamped_with_its_retrieval_month(
        self, con, rulebook
    ):
        body = json.dumps([{"iso3": "SOM", "asap_warning": "3"}])
        this_month = dt.date.today().strftime("%Y-%m")
        outcome = ind_mod.fetch_indicators(
            con, this_month, rulebook,
            get=lambda url, timeout: (body, "application/json"),
        )
        assert outcome.ok
        assert outcome.detail["entries"]["asap"]["observed_ym_source"] == "retrieval"
        assert _verdict(con, "SOM", rulebook, ym=this_month).state == (
            ind_mod.STATE_DROUGHT
        )

    def test_csv_feeds_and_url_templating(self, con):
        rb = make_rulebook(
            {
                "drought": {
                    "indicators": {
                        "entries": [
                            {
                                "name": "chirps_spi3",
                                "provider": "tabular",
                                "url": "https://example.test/spi/{year}/{ym}.csv",
                                "required": True,
                                "request_timeout_sec": 60,
                                "threshold": -1.0,
                                "direction": "below",
                            }
                        ]
                    }
                }
            }
        )
        seen: list[str] = []

        def get(url, timeout):
            seen.append(url)
            return "iso3,ym,value\nSOM,2024-03,-2.4\nURY,2024-03,0.3\n", "text/csv"

        outcome = ind_mod.fetch_indicators(con, YM, rb, get=get)
        assert outcome.ok
        assert seen == ["https://example.test/spi/2024/2024-03.csv"]
        assert _verdict(con, "SOM", rb).state == ind_mod.STATE_DROUGHT
        assert _verdict(con, "URY", rb).state == ind_mod.STATE_NO_DROUGHT


# ---------------------------------------------------------------------------
# The rule itself — a pure function, exercised directly
# ---------------------------------------------------------------------------


def _decide(con, rulebook, iso3, *, today=AFTER_FREEZE, ym=YM):
    return drought_mod.decide_drought(
        iso3=iso3,
        ym=ym,
        analyses=ipc_mod.analyses_for_country(con, iso3, rulebook),
        indicators=_verdict(con, iso3, rulebook, ym=ym),
        rulebook=rulebook,
        national_population=None,
        today=today,
    )


class TestRule:
    def test_a_drought_driven_deterioration_resolves_to_the_delta(
        self, con, rulebook
    ):
        seed_indicator_snapshot(con, values=_drought_indicators(["SOM"]))
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2023-07-01", window_end="2023-09-30",
            value=3_000_000,
        )
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2024-01-01", window_end="2024-03-31",
            value=4_400_000,
        )
        decision = _decide(con, rulebook, "SOM")
        assert decision.status == drought_mod.STATUS_RESOLVED_VALUE
        assert decision.value == 1_400_000
        assert decision.rule_fired == drought_mod.RULE_IPC_DELTA
        # Provenance names the analysis windows the answer rests on.
        analysis = decision.provenance["decision"]["ipc_analysis"]
        assert analysis["current"]["window"] == {
            "start": "2024-01-01", "end": "2024-03-31"
        }
        assert analysis["previous"]["phase3plus_population"] == 3_000_000

    def test_a_falling_analysis_is_a_measured_zero_not_missing_data(
        self, con, rulebook
    ):
        seed_indicator_snapshot(con, values=_drought_indicators(["SOM"]))
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2023-07-01", window_end="2023-09-30",
            value=4_400_000,
        )
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2024-01-01", window_end="2024-03-31",
            value=3_000_000,
        )
        decision = _decide(con, rulebook, "SOM")
        assert decision.status == drought_mod.STATUS_RESOLVED_ZERO
        assert decision.value == 0.0
        assert decision.rule_fired == drought_mod.RULE_ZERO_NO_INCREASE
        assert decision.triggered is True

    def test_a_quiet_country_resolves_zero_on_evidence_of_absence(
        self, con, rulebook
    ):
        seed_indicator_snapshot(con, values=_drought_indicators([], quiet=["URY"]))
        decision = _decide(con, rulebook, "URY")
        assert decision.status == drought_mod.STATUS_RESOLVED_ZERO
        assert decision.rule_fired == drought_mod.RULE_ZERO_ABSENCE
        assert decision.triggered is False
        assert decision.evidence_of_absence is not None
        assert decision.evidence_of_absence["indicators"]["state"] == (
            ind_mod.STATE_NO_DROUGHT
        )

    def test_indicators_dry_but_no_ipc_analysis_is_the_no_data_path(
        self, con, rulebook
    ):
        seed_indicator_snapshot(con, values=_drought_indicators(["SOM"]))
        decision = _decide(con, rulebook, "SOM")
        assert decision.status == drought_mod.STATUS_NO_DATA
        assert decision.rule_fired == drought_mod.RULE_NO_ANALYSIS
        assert drought_mod.FLAG_NO_ANALYSIS in decision.flags

    def test_the_same_gap_before_the_freeze_deadline_waits_instead(
        self, con, rulebook
    ):
        seed_indicator_snapshot(con, values=_drought_indicators(["SOM"]))
        decision = _decide(con, rulebook, "SOM", today=BEFORE_FREEZE)
        assert decision.status == drought_mod.STATUS_PENDING
        assert decision.writes_row is False

    def test_no_baseline_analysis_is_never_published_as_a_stock(
        self, con, rulebook
    ):
        """The rule is a difference; the covering analysis alone is a STOCK."""

        seed_indicator_snapshot(con, values=_drought_indicators(["SOM"]))
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2024-01-01", window_end="2024-03-31",
            value=4_400_000,
        )
        decision = _decide(con, rulebook, "SOM")
        assert decision.status == drought_mod.STATUS_NO_DATA
        assert decision.rule_fired == drought_mod.RULE_NO_BASELINE
        assert decision.value is None

    def test_a_deterioration_without_a_drought_signal_is_not_drought_impact(
        self, con, rulebook
    ):
        """Attribution gate: IPC states Phase 3+, never its driver."""

        seed_indicator_snapshot(con, values=_drought_indicators([], quiet=["SSD"]))
        seed_ipc_analysis(
            con, iso3="SSD", window_start="2023-07-01", window_end="2023-09-30",
            value=5_000_000,
        )
        seed_ipc_analysis(
            con, iso3="SSD", window_start="2024-01-01", window_end="2024-03-31",
            value=7_000_000,
        )
        decision = _decide(con, rulebook, "SSD")
        assert decision.status == drought_mod.STATUS_NO_DATA
        assert decision.rule_fired == drought_mod.RULE_NOT_ATTRIBUTED
        assert decision.value is None, "a conflict-driven rise is not drought PA"

    def test_an_unreadable_indicator_is_inconclusive_never_a_zero(
        self, con, rulebook
    ):
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2024-01-01", window_end="2024-03-31",
            value=4_400_000,
        )
        decision = _decide(con, rulebook, "SOM")
        assert decision.status == drought_mod.STATUS_INCONCLUSIVE
        assert decision.writes_row is False
        assert decision.value is None

    def test_a_figure_above_the_national_population_flags_but_stands(
        self, con, rulebook
    ):
        seed_indicator_snapshot(con, values=_drought_indicators(["SOM"]))
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2023-07-01", window_end="2023-09-30",
            value=0,
        )
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2024-01-01", window_end="2024-03-31",
            value=99_000_000,
        )
        decision = drought_mod.decide_drought(
            iso3="SOM",
            ym=YM,
            analyses=ipc_mod.analyses_for_country(con, "SOM", rulebook),
            indicators=_verdict(con, "SOM", rulebook),
            rulebook=rulebook,
            national_population=17_000_000,
            today=AFTER_FREEZE,
        )
        assert decision.value == 99_000_000, "the rule's answer is never rewritten"
        assert drought_mod.FLAG_POPULATION_EXCEEDED in decision.flags

    def test_the_decision_is_a_pure_function(self, con, rulebook):
        seed_indicator_snapshot(con, values=_drought_indicators(["SOM"]))
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2023-07-01", window_end="2023-09-30",
            value=3_000_000,
        )
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2024-01-01", window_end="2024-03-31",
            value=4_400_000,
        )
        first = _decide(con, rulebook, "SOM")
        second = _decide(con, rulebook, "SOM")
        assert (first.status, first.value, first.rule_fired) == (
            second.status, second.value, second.rule_fired
        )


class TestMonthAttribution:
    """An analysis states one figure for a window of several months."""

    def _seed(self, con):
        seed_indicator_snapshot(
            con, observed_ym="2024-01", values=_drought_indicators(["SOM"])
        )
        seed_indicator_snapshot(
            con, observed_ym="2024-02", values=_drought_indicators(["SOM"])
        )
        seed_indicator_snapshot(
            con, observed_ym="2024-03", values=_drought_indicators(["SOM"])
        )
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2023-07-01", window_end="2023-09-30",
            value=3_000_000,
        )
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2024-01-01", window_end="2024-03-31",
            value=4_400_000,
        )

    def test_analysis_window_repeats_the_delta_across_covered_months(self, con):
        rb = make_rulebook({"drought": {"month_attribution": "analysis_window"}})
        self._seed(con)
        for ym in ("2024-01", "2024-02", "2024-03"):
            decision = _decide(con, rb, "SOM", ym=ym)
            assert decision.status == drought_mod.STATUS_RESOLVED_VALUE
            assert decision.value == 1_400_000
        # The repetition is declared on the row, because summing these
        # months would count the same people three times.
        note = _decide(con, rb, "SOM", ym="2024-02").provenance["note"]
        assert "must NOT be" in note and "summed" in note

    def test_start_month_puts_the_delta_in_the_first_month_only(self, con):
        rb = make_rulebook({"drought": {"month_attribution": "start_month"}})
        self._seed(con)
        first = _decide(con, rb, "SOM", ym="2024-01")
        assert first.status == drought_mod.STATUS_RESOLVED_VALUE
        assert first.value == 1_400_000
        for ym in ("2024-02", "2024-03"):
            later = _decide(con, rb, "SOM", ym=ym)
            assert later.status == drought_mod.STATUS_RESOLVED_ZERO
            assert later.rule_fired == drought_mod.RULE_ZERO_MID_WINDOW


# ---------------------------------------------------------------------------
# Persistence: same tables, same freeze logic as every other hazard
# ---------------------------------------------------------------------------


class TestPersistence:
    def _seed_resolvable(self, con):
        seed_indicator_snapshot(con, values=_drought_indicators(["SOM"], quiet=["URY"]))
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2023-07-01", window_end="2023-09-30",
            value=3_000_000,
        )
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2024-01-01", window_end="2024-03-31",
            value=4_400_000,
        )
        seed_population(con, iso3="SOM", year=2024, population=17_000_000)

    def test_a_resolution_lands_with_a_candidate_and_a_trigger_row(
        self, con, rulebook
    ):
        self._seed_resolvable(con)
        drought_mod.run_month(
            con, ym=YM, iso3s=["SOM", "URY"], rulebook=rulebook, today=AFTER_FREEZE
        )

        status, value, provenance, rule = con.execute(
            """
            SELECT status, value, provenance_json, rule_fired FROM haz_resolutions
            WHERE iso3 = 'SOM' AND hazard = 'DR' AND year = 2024 AND month = 3
            """
        ).fetchone()
        assert (status, value) == ("RESOLVED_VALUE", 1_400_000)
        assert rule == drought_mod.RULE_IPC_DELTA
        payload = json.loads(provenance)
        assert payload["source"] == "ipc"
        assert payload["source_record_ids"], "provenance must cite its records"
        assert payload["decision"]["ipc_analysis"]["current"]["window"]["start"]

        source, value_type, detail = con.execute(
            """
            SELECT source, value_type, detail_json FROM haz_impact_candidates
            WHERE iso3 = 'SOM' AND hazard = 'DR' AND year = 2024 AND month = 3
            """
        ).fetchone()
        assert (source, value_type) == ("ipc", "affected")
        assert json.loads(detail)["delta"] == 1_400_000

        triggered, trigger_source = con.execute(
            """
            SELECT triggered, trigger_source FROM haz_triggers
            WHERE iso3 = 'SOM' AND hazard = 'DR' AND year = 2024 AND month = 3
            """
        ).fetchone()
        assert triggered is True
        assert trigger_source == drought_mod.TRIGGER_SOURCE_INDICATORS

    def test_a_zero_carries_evidence_of_absence_on_both_rows(self, con, rulebook):
        self._seed_resolvable(con)
        drought_mod.run_month(
            con, ym=YM, iso3s=["SOM", "URY"], rulebook=rulebook, today=AFTER_FREEZE
        )
        status, value, provenance = con.execute(
            """
            SELECT status, value, provenance_json FROM haz_resolutions
            WHERE iso3 = 'URY' AND hazard = 'DR' AND year = 2024 AND month = 3
            """
        ).fetchone()
        assert (status, value) == ("RESOLVED_ZERO", 0.0)
        evidence = json.loads(provenance)["evidence_of_absence"]
        assert evidence["indicators"]["state"] == ind_mod.STATE_NO_DROUGHT
        assert "ipc" in evidence

        absence = con.execute(
            """
            SELECT evidence_of_absence_json FROM haz_triggers
            WHERE iso3 = 'URY' AND hazard = 'DR' AND year = 2024 AND month = 3
            """
        ).fetchone()[0]
        assert absence and json.loads(absence)["indicators"]

    def test_an_inconclusive_cell_writes_no_resolution_at_all(self, con, rulebook):
        """Fail-closed: no indicator read means no answer, not a zero."""

        seed_ipc_analysis(
            con, iso3="SOM", window_start="2024-01-01", window_end="2024-03-31",
            value=4_400_000,
        )
        run = drought_mod.run_month(
            con, ym=YM, iso3s=["SOM"], rulebook=rulebook, today=AFTER_FREEZE
        )
        assert run.inconclusive == 1
        assert con.execute("SELECT COUNT(*) FROM haz_resolutions").fetchone()[0] == 0
        # …but the trigger row records that the cell was looked at.
        assert con.execute("SELECT COUNT(*) FROM haz_triggers").fetchone()[0] == 1

    def test_a_frozen_row_is_never_altered_only_audited(self, con, rulebook):
        self._seed_resolvable(con)
        drought_mod.run_month(
            con, ym=YM, iso3s=["SOM"], rulebook=rulebook, today=AFTER_FREEZE
        )
        # A later analysis revises the figure upward, well past the freeze.
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2024-01-01", window_end="2024-03-31",
            value=9_000_000, analysis_date="2024-11-01",
        )
        run = drought_mod.run_month(
            con, ym=YM, iso3s=["SOM"], rulebook=rulebook, today=dt.date(2025, 6, 1)
        )
        assert run.frozen_skipped == 1
        value = con.execute(
            "SELECT value FROM haz_resolutions WHERE iso3 = 'SOM'"
        ).fetchone()[0]
        assert value == 1_400_000, "a frozen resolution is immutable"
        assert con.execute("SELECT COUNT(*) FROM haz_revisions").fetchone()[0] == 1

    def test_a_pre_freeze_answer_is_marked_provisional(self, con, rulebook):
        self._seed_resolvable(con)
        drought_mod.run_month(
            con, ym=YM, iso3s=["SOM"], rulebook=rulebook, today=BEFORE_FREEZE
        )
        provisional = con.execute(
            "SELECT provisional FROM haz_resolutions WHERE iso3 = 'SOM'"
        ).fetchone()[0]
        assert provisional is True


# ---------------------------------------------------------------------------
# Phase 4 acceptance
# ---------------------------------------------------------------------------


class TestAcceptance:
    """Two deteriorating drought countries, two quiet ones, and the gap.

    The figures below are the shape of a real IPC deterioration (a Horn of
    Africa country whose Phase 3+ population rises between consecutive
    analyses) rather than any single published analysis — the point of the
    test is that the machine's answer equals the hand calculation from the
    two figures it was given, and that each answer cites the analysis
    window it rests on.
    """

    # (iso3, previous Phase 3+, current Phase 3+, expected people affected)
    DETERIORATING = [
        ("SOM", 3_000_000, 4_400_000, 1_400_000),
        ("ETH", 11_800_000, 15_800_000, 4_000_000),
    ]
    QUIET = ["URY", "ISL"]
    #: Drought conditions indicated, no IPC analysis anywhere near it.
    NO_IPC = "AFG"

    def _seed(self, con):
        seed_indicator_snapshot(
            con,
            values=_drought_indicators(
                [iso3 for iso3, *_ in self.DETERIORATING] + [self.NO_IPC],
                quiet=self.QUIET,
            ),
        )
        for iso3, previous, current, _expected in self.DETERIORATING:
            seed_ipc_analysis(
                con, iso3=iso3, window_start="2023-07-01", window_end="2023-09-30",
                value=previous,
            )
            seed_ipc_analysis(
                con, iso3=iso3, window_start="2024-01-01", window_end="2024-03-31",
                value=current,
            )
            seed_population(con, iso3=iso3, year=2024, population=130_000_000)

    def test_acceptance(self, con, rulebook):
        self._seed(con)
        countries = (
            [iso3 for iso3, *_ in self.DETERIORATING] + self.QUIET + [self.NO_IPC]
        )
        run = drought_mod.run_month(
            con, ym=YM, iso3s=countries, rulebook=rulebook, today=AFTER_FREEZE
        )

        assert run.cells == 5
        assert run.resolved_value == 2
        assert run.resolved_zero == 2
        assert run.no_data == 1

        rows = {
            iso3: (status, value, rule, flagged, json.loads(prov))
            for iso3, status, value, rule, flagged, prov in con.execute(
                """
                SELECT iso3, status, value, rule_fired, flagged, provenance_json
                FROM haz_resolutions WHERE hazard = 'DR'
                """
            ).fetchall()
        }

        # 1. The deteriorating countries match the hand calculation…
        for iso3, previous, current, expected in self.DETERIORATING:
            status, value, rule, flagged, provenance = rows[iso3]
            assert status == "RESOLVED_VALUE"
            assert value == expected == current - previous
            assert rule == drought_mod.RULE_IPC_DELTA
            assert not flagged
            # …and each says which analysis window it rests on.
            analysis = provenance["decision"]["ipc_analysis"]
            assert analysis["current"]["phase3plus_population"] == current
            assert analysis["previous"]["phase3plus_population"] == previous
            assert analysis["current"]["months_covered"] == [
                "2024-01", "2024-02", "2024-03"
            ]
            assert provenance["source_urls"], "every answer cites its sources"

        # 2. The quiet countries resolve zero, with evidence of absence.
        for iso3 in self.QUIET:
            status, value, rule, _flagged, provenance = rows[iso3]
            assert (status, value) == ("RESOLVED_ZERO", 0.0)
            assert rule == drought_mod.RULE_ZERO_ABSENCE
            assert provenance["evidence_of_absence"]["indicators"]["readings"]

        # 3. The NO_DATA path fires exactly where the spec says it should.
        status, value, rule, flagged, provenance = rows[self.NO_IPC]
        assert (status, value, rule) == (
            "NO_DATA", None, drought_mod.RULE_NO_ANALYSIS
        )
        assert flagged is True, "an unresolved drought month wants a human"

        # 4. Every row carries provenance and a rule (hard rule 1).
        for iso3, (_s, _v, rule, _f, provenance) in rows.items():
            assert rule, iso3
            assert provenance.get("rule_fired") == rule, iso3
            assert "retrieved_at" in provenance, iso3

        # 5. Re-running changes nothing (determinism + idempotence).
        again = drought_mod.run_month(
            con, ym=YM, iso3s=countries, rulebook=rulebook, today=AFTER_FREEZE
        )
        assert (again.resolved_value, again.resolved_zero, again.no_data) == (
            run.resolved_value, run.resolved_zero, run.no_data
        )
        assert con.execute(
            "SELECT COUNT(*) FROM haz_resolutions WHERE hazard = 'DR'"
        ).fetchone()[0] == 5

    def test_coverage_of_the_country_month_universe(self, con, rulebook):
        """The machine's headline goal: a number (including zero) per cell."""

        self._seed(con)
        countries = (
            [iso3 for iso3, *_ in self.DETERIORATING] + self.QUIET + [self.NO_IPC]
        )
        run = drought_mod.run_month(
            con, ym=YM, iso3s=countries, rulebook=rulebook, today=AFTER_FREEZE
        )
        resolved = run.resolved_value + run.resolved_zero
        assert resolved / run.cells >= 0.8


class TestZeroRuleVocabulary:
    def test_drought_has_its_own_zero_rule_string(self):
        """A drought zero rests on different evidence from a cyclone zero."""

        assert res_mod.ZERO_RULE_BY_HAZARD["DR"] == drought_mod.RULE_ZERO_ABSENCE
        assert len(set(res_mod.ZERO_RULE_BY_HAZARD.values())) == 3

    def test_pending_and_inconclusive_map_to_the_writer_s_skip_status(self):
        """Neither writes a row, and the literal must match reconcile's."""

        assert drought_mod.STATUS_PENDING == res_mod.WRITE_PENDING_STATUS
        assert drought_mod.STATUS_INCONCLUSIVE != res_mod.WRITE_PENDING_STATUS
        for status in (drought_mod.STATUS_PENDING, drought_mod.STATUS_INCONCLUSIVE):
            decision = drought_mod.DroughtDecision(
                iso3="SOM", ym=YM, triggered=False,
                trigger_source=drought_mod.TRIGGER_SOURCE_NONE,
                status=status, value=None, rule_fired="x", provisional=True,
            )
            assert decision.writes_row is False
            assert decision.as_reconciliation().status == res_mod.WRITE_PENDING_STATUS


def test_rulebook_loader_rejects_a_broken_drought_block(tmp_path):
    """End-to-end: load_rulebook refuses, listing every problem at once."""

    import yaml

    from resolver.hazard_resolution.rulebook import load_rulebook

    data = make_rulebook().raw
    data["drought"]["month_attribution"] = "nonsense"
    data["drought"]["ipc"]["lookback_months"] = 1
    path = tmp_path / "rulebook.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(RulebookError) as excinfo:
        load_rulebook(path)
    message = str(excinfo.value)
    assert "month_attribution" in message and "lookback_months" in message


class TestCoverageBeforeAZero:
    """A country no feed looked at is "not looked at", never "quiet".

    An alerting feed answers for every country it does not list, so
    ``min_available`` alone was satisfied by one feed that had never
    monitored the country. With HDX Signals covering a few dozen countries,
    that would have zeroed every country outside its watch list on the
    strength of not being watched.
    """

    _seed_hdx = staticmethod(TestIndicatorSourcesBeyondAsap._seed_hdx)
    _refresh = staticmethod(TestIndicatorSourcesBeyondAsap._refresh)

    def test_absence_alone_is_not_coverage(self, con, rulebook):
        self._seed_hdx(con, {"ETH": "High concern"})
        self._refresh(con, rulebook)
        verdict = _verdict(con, "SOM", rulebook)
        assert verdict.available
        assert verdict.present_count == 0
        assert not verdict.has_coverage

    def test_a_reading_that_names_the_country_is_coverage(self, con, rulebook):
        self._seed_hdx(con, {"ETH": "High concern", "SOM": "Low concern"})
        self._refresh(con, rulebook)
        verdict = _verdict(con, "SOM", rulebook)
        assert verdict.present_count == 1
        assert verdict.has_coverage
        assert not verdict.shows_drought

    def test_no_coverage_resolves_inconclusive_never_zero(self, con, rulebook):
        self._seed_hdx(con, {"ETH": "High concern"})
        self._refresh(con, rulebook)
        decision = drought_mod.decide_drought(
            iso3="SOM", ym=YM, analyses=[],
            indicators=_verdict(con, "SOM", rulebook),
            rulebook=rulebook, today=AFTER_FREEZE,
        )
        assert decision.status == drought_mod.STATUS_INCONCLUSIVE
        assert decision.rule_fired == drought_mod.RULE_NO_COVERAGE
        assert not decision.writes_row

    def test_coverage_plus_quiet_ipc_still_resolves_zero(self, con, rulebook):
        self._seed_hdx(con, {"SOM": "Low concern"})
        self._refresh(con, rulebook)
        decision = drought_mod.decide_drought(
            iso3="SOM", ym=YM, analyses=[],
            indicators=_verdict(con, "SOM", rulebook),
            rulebook=rulebook, today=AFTER_FREEZE,
        )
        assert decision.status == drought_mod.STATUS_RESOLVED_ZERO
        assert decision.rule_fired == drought_mod.RULE_ZERO_ABSENCE

    def test_zero_min_present_readings_restores_the_old_behaviour(self, con):
        rulebook = legacy_feed_rulebook(
            {"drought": {"indicators": {"min_present_readings": 0}}}
        )
        self._seed_hdx(con, {"ETH": "High concern"})
        self._refresh(con, rulebook)
        decision = drought_mod.decide_drought(
            iso3="SOM", ym=YM, analyses=[],
            indicators=_verdict(con, "SOM", rulebook),
            rulebook=rulebook, today=AFTER_FREEZE,
        )
        assert decision.status == drought_mod.STATUS_RESOLVED_ZERO

    def test_the_pick_between_two_rows_in_one_month_is_the_warning(self, con, rulebook):
        """Several rows for one country-month must resolve the same way every run."""

        self._seed_hdx(con, {"SOM": "Low concern"}, signal_date="2024-03-02")
        self._seed_hdx(con, {"SOM": "High concern"}, signal_date="2024-03-20")
        self._refresh(con, rulebook)
        assert _verdict(con, "SOM", rulebook).state == ind_mod.STATE_DROUGHT

    def test_the_hdx_entry_reads_agricultural_hotspots_only(self, rulebook):
        entry = next(
            e for e in rulebook.get("drought.indicators.entries")
            if e["name"] == "hdx_agricultural_stress"
        )
        assert "jrc_agricultural_hotspots" in entry["where"]
        assert "ipc_food_insecurity" not in entry["where"]

    def test_a_circular_indicator_is_refused_by_the_validator(self, rulebook):
        data = json.loads(json.dumps(rulebook.raw))
        entries = data["drought"]["indicators"]["entries"]
        hdx = next(e for e in entries if e["name"] == "hdx_agricultural_stress")
        hdx["where"] = "indicator IN ('jrc_agricultural_hotspots', 'ipc_food_insecurity')"
        problems = validate_rulebook(data)
        assert any("ipc_food_insecurity" in p for p in problems), problems

    def test_the_snapshot_cache_answers_the_same_question_once(self, con, rulebook, monkeypatch):
        self._seed_hdx(con, {"SOM": "High concern"})
        self._refresh(con, rulebook)
        calls = []
        real = ind_mod.load_raw_records

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(ind_mod, "load_raw_records", counting)
        cache: dict = {}
        ind_mod.evaluate_indicators(con, "SOM", YM, rulebook, cache=cache)
        after_first = len(calls)
        assert after_first >= 1
        for iso3 in ("ETH", "KEN", "NGA"):
            ind_mod.evaluate_indicators(con, iso3, YM, rulebook, cache=cache)
        assert len(calls) == after_first
