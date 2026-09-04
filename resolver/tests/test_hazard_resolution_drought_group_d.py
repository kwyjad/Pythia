# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Group D of the run-33841370196 fault series: the drought path.

D1  28,728 drought cells carried ``unexplained_no_row``: the backcast wrote
    nothing and said nothing. Every no-row verdict now stamps its reason on
    the trigger row, an undecided cell is ``assessed: false`` and stays out
    of the occurrence base rate.
D2  NMME was ingested and never read: the lookup pinned ``lead_months = 1``.
    Any vintage that speaks for the month is a candidate, the shortest lead
    wins, and a miss describes the table.
D3  The ASAP warnings feed is gone. The hotspot ASSESSMENT time series
    (one CSV since October 2016, HDX-mirrored) is parsed into one snapshot
    per month; and a zero may no longer rest on a single surviving feed.
D4  159 September zeros on 4 September: a month still in progress is
    PENDING, and any provisional row an earlier run wrote is retracted.

Every case runs on the SHIPPED rulebook (``make_rulebook()`` with no
overrides) unless it says otherwise — these are the production values.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import zipfile

import duckdb
import pytest

from resolver.hazard_resolution import backcast as bc
from resolver.hazard_resolution import base_rates as br
from resolver.hazard_resolution import cell_ledger
from resolver.hazard_resolution import detect as detect_mod
from resolver.hazard_resolution import drought as drought_mod
from resolver.hazard_resolution import drought_indicators as ind_mod
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.tests.hazard_resolution_utils import (
    make_rulebook,
    seed_indicator_snapshot,
    seed_ipc_analysis,
    seed_resolution,
    seed_trigger,
)

YM = "2024-03"
AFTER_FREEZE = dt.date(2024, 9, 1)
BEFORE_FREEZE = dt.date(2024, 4, 15)
IN_PROGRESS = dt.date(2024, 3, 4)


@pytest.fixture()
def rulebook():
    ind_mod.reset_for_tests()
    return make_rulebook()


@pytest.fixture()
def con():
    con = duckdb.connect(":memory:")
    ensure_haz_schema(con)
    return con


def _verdict(con, iso3, rulebook, ym=YM):
    return ind_mod.evaluate_indicators(con, iso3, ym, rulebook)


def _decide(con, rulebook, iso3, *, today=AFTER_FREEZE, ym=YM, analyses=None):
    from resolver.hazard_resolution import ipc as ipc_mod

    return drought_mod.decide_drought(
        iso3=iso3, ym=ym,
        analyses=analyses if analyses is not None
        else ipc_mod.analyses_for_country(con, iso3, rulebook),
        indicators=_verdict(con, iso3, rulebook, ym=ym),
        rulebook=rulebook, national_population=None, today=today,
    )


def _seed_hdx(con, values: dict[str, str], signal_date: str = "2024-03-15"):
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
            "INSERT INTO hdx_signals (iso3, hazard_code, indicator, concern_level, "
            "signal_date) VALUES (?, 'DR', 'jrc_agricultural_hotspots', ?, ?)",
            [iso3, level, signal_date],
        )


def _seed_nmme(con, rows: list[tuple[str, float, int, str]]):
    """``(iso3, anomaly, lead_months, issue_date)`` rows as the ingest writes them."""

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS seasonal_forecasts (
            iso3 TEXT, variable TEXT, lead_months INTEGER,
            anomaly_value DOUBLE, tercile_category TEXT,
            forecast_issue_date DATE
        )
        """
    )
    for iso3, anomaly, lead, issue in rows:
        con.execute(
            "INSERT INTO seasonal_forecasts (iso3, variable, lead_months, "
            "anomaly_value, forecast_issue_date) VALUES (?, 'prate', ?, ?, ?)",
            [iso3, lead, anomaly, issue],
        )


def _refresh(con, rulebook, get=None, ym=YM):
    def dead(url, timeout):
        raise RuntimeError("404 Not Found")

    return ind_mod.fetch_indicators(con, ym, rulebook, get=get or dead)


def _hotspots_zip(rows: list[tuple[str, str, str, str]]) -> bytes:
    """The JRC hotspot time series as HDX Signals reads it: one CSV in a zip."""

    body = "asap0_id,asap0_name,date,hs_code,hs_name,comment\n"
    for asap_id, name, date, code in rows:
        label = {"0": "No hotspot", "1": "Hotspot", "2": "Major hotspot",
                 "3": "Not assessed"}[code]
        body += f"{asap_id},{name},{date},{code},{label},\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("hotspots_ts.csv", body)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# D4: a month still in progress
# ---------------------------------------------------------------------------


class TestMonthInProgress:
    def test_the_current_month_is_pending_whatever_the_feeds_say(self, con, rulebook):
        """4 September wrote 159 September zeros. Nothing has observed a
        month with 26 days to run."""

        _seed_hdx(con, {"SOM": "Low concern"})
        _seed_nmme(con, [("SOM", 0.3, 1, "2024-02-08")])
        _refresh(con, rulebook)
        decision = _decide(con, rulebook, "SOM", today=IN_PROGRESS)
        assert decision.status == drought_mod.STATUS_PENDING
        assert decision.rule_fired == drought_mod.RULE_MONTH_IN_PROGRESS
        assert not decision.writes_row
        assert decision.trigger_detail["no_row_reason"] == cell_ledger.REASON_MONTH_IN_PROGRESS
        assert decision.trigger_detail["assessed"] is False

    def test_the_month_after_it_ends_is_decided_normally(self, con, rulebook):
        _seed_hdx(con, {"SOM": "Low concern"})
        _seed_nmme(con, [("SOM", 0.3, 1, "2024-02-08")])
        _refresh(con, rulebook)
        decision = _decide(con, rulebook, "SOM", today=dt.date(2024, 4, 1))
        assert decision.status == drought_mod.STATUS_RESOLVED_ZERO

    def test_an_earlier_provisional_zero_for_the_month_is_retracted(self, con, rulebook):
        """The row a weaker rule wrote must not survive the rule that
        contradicts it — PENDING writes nothing, so it used to stand."""

        seed_resolution(
            con, iso3="SOM", ym=YM, hazard="DR", status="RESOLVED_ZERO", value=0.0,
            source="ipc", provisional=True,
        )
        con.execute(
            "UPDATE haz_resolutions SET rule_fired = ? WHERE iso3 = 'SOM'",
            [drought_mod.RULE_ZERO_ABSENCE],
        )
        _seed_hdx(con, {"SOM": "Low concern"})
        _refresh(con, rulebook)
        run = drought_mod.run_month(
            con, ym=YM, iso3s=["SOM"], rulebook=rulebook, today=IN_PROGRESS
        )
        assert run.pending == 1
        assert run.retracted == 1
        assert con.execute(
            "SELECT COUNT(*) FROM haz_resolutions WHERE iso3 = 'SOM' AND hazard = 'DR'"
        ).fetchone()[0] == 0
        reason = json.loads(con.execute(
            "SELECT trigger_detail_json FROM haz_triggers WHERE iso3 = 'SOM' AND hazard = 'DR'"
        ).fetchone()[0])
        assert reason["no_row_reason"] == cell_ledger.REASON_MONTH_IN_PROGRESS

    def test_a_frozen_row_is_never_retracted(self, con, rulebook):
        seed_resolution(
            con, iso3="SOM", ym=YM, hazard="DR", status="RESOLVED_ZERO", value=0.0,
            source="ipc", provisional=False, frozen_at="2024-05-30 00:00:00",
        )
        decision = drought_mod.DroughtDecision(
            iso3="SOM", ym=YM, triggered=False, trigger_source="none",
            status=drought_mod.STATUS_PENDING, value=None,
            rule_fired=drought_mod.RULE_MONTH_IN_PROGRESS, provisional=True,
        )
        assert drought_mod.retract_unsupported_row(
            con, decision, rulebook, today=AFTER_FREEZE
        ) == drought_mod.RETRACT_RETAINED
        assert con.execute("SELECT COUNT(*) FROM haz_resolutions").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# D3: a zero needs more than one feed
# ---------------------------------------------------------------------------


class TestTwoFeedsForAZero:
    def test_shipped_floor_is_two(self, rulebook):
        assert int(rulebook.get("drought.indicators.min_answered_for_zero")) == 2

    def test_one_surviving_feed_cannot_zero_a_country(self, con, rulebook):
        """The September run: ASAP 404, NMME unread, HDX alone zeroed 159."""

        _seed_hdx(con, {"SOM": "Low concern"})
        _refresh(con, rulebook)
        verdict = _verdict(con, "SOM", rulebook)
        assert verdict.available and verdict.has_coverage
        assert verdict.answered_count == 1
        assert not verdict.supports_zero
        decision = _decide(con, rulebook, "SOM")
        assert decision.status == drought_mod.STATUS_INCONCLUSIVE
        assert decision.rule_fired == drought_mod.RULE_TOO_FEW_FEEDS
        assert decision.trigger_detail["no_row_reason"] == cell_ledger.REASON_TOO_FEW_FEEDS
        assert decision.trigger_detail["assessed"] is False

    def test_two_answering_feeds_may_zero_a_country(self, con, rulebook):
        _seed_hdx(con, {"SOM": "Low concern"})
        _seed_nmme(con, [("SOM", 0.4, 1, "2024-02-08")])
        _refresh(con, rulebook)
        verdict = _verdict(con, "SOM", rulebook)
        assert verdict.answered_count == 2 and verdict.supports_zero
        decision = _decide(con, rulebook, "SOM")
        assert decision.status == drought_mod.STATUS_RESOLVED_ZERO
        assert decision.rule_fired == drought_mod.RULE_ZERO_ABSENCE
        assert decision.provenance["decision"]["indicators"]["answered_count"] == 2

    def test_a_measured_zero_rests_on_ipc_and_is_exempt(self, con, rulebook):
        """Both analyses read and no increase: that zero is IPC's, not the
        feeds', and one feed answering is not what it rests on."""

        _seed_hdx(con, {"SOM": "Low concern"})
        _refresh(con, rulebook)
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2023-07-01", window_end="2023-09-30",
            value=3_000_000,
        )
        seed_ipc_analysis(
            con, iso3="SOM", window_start="2024-01-01", window_end="2024-03-31",
            value=2_900_000,
        )
        decision = _decide(con, rulebook, "SOM")
        assert decision.status == drought_mod.STATUS_RESOLVED_ZERO
        # No signal and no IPC deterioration is the absence zero, as before;
        # what matters is that it is written with one feed because IPC
        # measured the month, and its provenance carries the delta.
        assert decision.rule_fired == drought_mod.RULE_ZERO_ABSENCE
        assert decision.provenance["decision"]["delta"] == 0.0

    def test_a_drought_signal_from_one_feed_still_attributes(self, con, rulebook):
        """The floor is on ZEROS. A deterioration with one feed showing
        drought is attributed exactly as before (combine: any)."""

        _seed_hdx(con, {"SOM": "High concern"})
        _refresh(con, rulebook)
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
        assert decision.value == pytest.approx(1_400_000)

    def test_a_provisional_absence_zero_below_the_floor_is_retracted(self, con, rulebook):
        seed_resolution(
            con, iso3="SOM", ym=YM, hazard="DR", status="RESOLVED_ZERO", value=0.0,
            source="ipc", provisional=True,
        )
        con.execute(
            "UPDATE haz_resolutions SET rule_fired = ? WHERE iso3 = 'SOM'",
            [drought_mod.RULE_ZERO_ABSENCE],
        )
        _seed_hdx(con, {"SOM": "Low concern"})
        _refresh(con, rulebook)
        run = drought_mod.run_month(
            con, ym=YM, iso3s=["SOM"], rulebook=rulebook, today=BEFORE_FREEZE
        )
        assert run.inconclusive == 1 and run.retracted == 1
        assert con.execute("SELECT COUNT(*) FROM haz_resolutions").fetchone()[0] == 0

    def test_a_provisional_value_row_is_not_retracted_by_a_zero_rule(self, con, rulebook):
        """The floor contradicts absence zeros only. A value from IPC stands."""

        seed_resolution(
            con, iso3="SOM", ym=YM, hazard="DR", status="RESOLVED_VALUE",
            value=1_400_000.0, source="ipc", provisional=True,
        )
        _seed_hdx(con, {"SOM": "Low concern"})
        _refresh(con, rulebook)
        run = drought_mod.run_month(
            con, ym=YM, iso3s=["SOM"], rulebook=rulebook, today=BEFORE_FREEZE
        )
        assert run.retracted == 0 and run.retained_unsupported == 1
        assert con.execute("SELECT COUNT(*) FROM haz_resolutions").fetchone()[0] == 1

    def test_an_unreadable_indicator_retracts_nothing(self, con, rulebook):
        """"We could not check today" does not overturn "we checked last month"."""

        seed_resolution(
            con, iso3="SOM", ym=YM, hazard="DR", status="RESOLVED_ZERO", value=0.0,
            source="ipc", provisional=True,
        )
        con.execute(
            "UPDATE haz_resolutions SET rule_fired = ? WHERE iso3 = 'SOM'",
            [drought_mod.RULE_ZERO_ABSENCE],
        )
        # No feed answered at all -> INCONCLUSIVE (unread), not too-few.
        run = drought_mod.run_month(
            con, ym=YM, iso3s=["SOM"], rulebook=rulebook, today=BEFORE_FREEZE
        )
        assert run.inconclusive == 1 and run.retracted == 0
        assert con.execute("SELECT COUNT(*) FROM haz_resolutions").fetchone()[0] == 1

    def test_the_floor_of_one_restores_the_old_behaviour(self, con):
        rulebook = make_rulebook(
            {"drought": {"indicators": {"min_answered_for_zero": 1}}}
        )
        _seed_hdx(con, {"SOM": "Low concern"})
        _refresh(con, rulebook)
        assert _decide(con, rulebook, "SOM").status == drought_mod.STATUS_RESOLVED_ZERO


# ---------------------------------------------------------------------------
# D3: the ASAP hotspot time series
# ---------------------------------------------------------------------------


class TestAsapHotspotSeries:
    ROWS = [
        ("1", "Somalia", "2024-02-28", "2"),
        ("1", "Somalia", "2024-03-28", "1"),
        ("2", "Kenya", "2024-02-28", "0"),
        ("2", "Kenya", "2024-03-28", "3"),
        ("3", "Ethiopia", "2024-03-28", "0"),
        ("4", "Uruguay", "2016-10-28", "0"),
    ]

    def _serve(self, blob: bytes):
        seen: list[str] = []

        def get(url, timeout):
            seen.append(url)
            if url.startswith(ind_mod.HDX_CKAN_API):
                raise RuntimeError("unreachable in this test")
            return blob, "application/zip"

        return get, seen

    def test_the_series_is_parsed_into_one_snapshot_per_month(self, con, rulebook):
        get, seen = self._serve(_hotspots_zip(self.ROWS))
        outcome = _refresh(con, rulebook, get=get)
        entry = outcome.detail["entries"]["asap_hotspots"]
        assert entry["ok"] is True
        assert entry["months"] == 3
        assert entry["month_range"] == "2016-10..2024-03"
        months = {
            r[0] for r in con.execute(
                "SELECT ym FROM haz_raw_drought_indicators WHERE payload_json LIKE "
                "'%\"name\":\"asap_hotspots\"%'"
            ).fetchall()
        }
        assert months == {"2016-10", "2024-02", "2024-03"}
        # The first https candidate answered; nothing fell through to HDX.
        assert seen[0].startswith("https://agricultural-production-hotspots")

    def test_a_backcast_month_reads_the_assessment_made_for_it(self, con, rulebook):
        get, _ = self._serve(_hotspots_zip(self.ROWS))
        _refresh(con, rulebook, get=get, ym="2024-02")
        feb = {r.name: r for r in _verdict(con, "SOM", rulebook, ym="2024-02").readings}
        mar = {r.name: r for r in _verdict(con, "SOM", rulebook, ym="2024-03").readings}
        assert feb["asap_hotspots"].state == ind_mod.STATE_DROUGHT
        assert feb["asap_hotspots"].value == "2"
        assert mar["asap_hotspots"].state == ind_mod.STATE_DROUGHT
        assert mar["asap_hotspots"].value == "1"
        # A month before the archive begins has no reading, not a zero.
        old = {r.name: r for r in _verdict(con, "SOM", rulebook, ym="2015-06").readings}
        assert old["asap_hotspots"].state == ind_mod.STATE_UNAVAILABLE

    def test_not_assessed_is_no_reading_and_absence_is_unknown(self, con, rulebook):
        get, _ = self._serve(_hotspots_zip(self.ROWS))
        _refresh(con, rulebook, get=get)
        readings = {
            iso3: {r.name: r for r in _verdict(con, iso3, rulebook).readings}
            for iso3 in ("KEN", "ETH", "URY", "FRA")
        }
        assert readings["KEN"]["asap_hotspots"].state == ind_mod.STATE_UNAVAILABLE
        assert "not assessed" in readings["KEN"]["asap_hotspots"].error
        assert readings["ETH"]["asap_hotspots"].state == ind_mod.STATE_NO_DROUGHT
        assert readings["ETH"]["asap_hotspots"].present is True
        # France is not one of the ~80 countries ASAP assesses: unknown.
        assert readings["FRA"]["asap_hotspots"].state == ind_mod.STATE_UNAVAILABLE
        assert readings["FRA"]["asap_hotspots"].present is False

    def test_a_series_with_no_resolvable_country_is_refused_with_its_columns(
        self, con, rulebook
    ):
        body = "asap0_id,date,hs_code\n1,2024-03-28,1\n2,2024-03-28,0\n"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("hotspots_ts.csv", body)
        get, _ = self._serve(buffer.getvalue())
        outcome = _refresh(con, rulebook, get=get)
        entry = outcome.detail["entries"]["asap_hotspots"]
        assert entry["ok"] is False
        refusals = [a["error"] for a in entry["attempts"] if "records resolved" in a["error"]]
        assert len(refusals) == 2  # both https candidates served the bad shape
        assert all("asap0_id" in r for r in refusals)

    def test_the_hdx_mirror_is_reached_through_ckan(self, con, rulebook):
        blob = _hotspots_zip(self.ROWS)
        seen: list[str] = []

        def get(url, timeout):
            seen.append(url)
            if url.startswith(ind_mod.HDX_CKAN_API):
                return json.dumps({"result": {"resources": [
                    {"url": "https://data.humdata.org/x/asap-hotspots-monthly.csv",
                     "format": "CSV"},
                ]}}), "application/json"
            if "humdata" in url:
                return blob, "application/zip"
            raise RuntimeError("404 Not Found")

        outcome = _refresh(con, rulebook, get=get)
        entry = outcome.detail["entries"]["asap_hotspots"]
        assert entry["ok"] is True
        assert entry["url"] == "hdx-ckan://asap-hotspots-monthly"
        assert any(u.startswith(ind_mod.HDX_CKAN_API) for u in seen)
        assert len(entry["attempts"]) == 2

    def test_a_time_series_is_fetched_once_per_process(self, con, rulebook):
        get, seen = self._serve(_hotspots_zip(self.ROWS))
        _refresh(con, rulebook, get=get, ym="2024-02")
        _refresh(con, rulebook, get=get, ym="2024-03")
        assert len(seen) == 1

    def test_the_backcast_preflight_counts_the_series_as_dated(self, rulebook):
        check = bc.check_backcastable("drought", rulebook)
        joined = " | ".join(check.warnings)
        assert "asap_hotspots" in joined
        assert "EVERY" not in joined


# ---------------------------------------------------------------------------
# D2: NMME is read for every month it forecasts
# ---------------------------------------------------------------------------


class TestNmmeLookup:
    def test_any_vintage_speaking_for_the_month_is_read(self, con, rulebook):
        """Only a lead-3 vintage exists for March. The pinned lead-1 lookup
        found nothing while the forecast about March sat in the table."""

        _seed_nmme(con, [("SOM", -1.4, 3, "2023-12-08")])
        outcome = _refresh(con, rulebook)
        assert outcome.detail["entries"]["nmme_precip_anomaly"]["ok"] is True
        reading = {r.name: r for r in _verdict(con, "SOM", rulebook).readings}
        assert reading["nmme_precip_anomaly"].state == ind_mod.STATE_DROUGHT

    def test_the_shortest_lead_wins(self, con, rulebook):
        _seed_nmme(con, [
            ("SOM", -1.4, 3, "2023-12-08"),   # about March, issued December
            ("SOM", 0.2, 1, "2024-02-08"),    # about March, issued February
        ])
        _refresh(con, rulebook)
        reading = {r.name: r for r in _verdict(con, "SOM", rulebook).readings}
        assert reading["nmme_precip_anomaly"].value == pytest.approx(0.2)
        assert reading["nmme_precip_anomaly"].state == ind_mod.STATE_NO_DROUGHT

    def test_a_vintage_about_a_later_month_never_answers(self, con, rulebook):
        _seed_nmme(con, [("SOM", -1.4, 1, "2024-03-08")])  # about April
        outcome = _refresh(con, rulebook)
        entry = outcome.detail["entries"]["nmme_precip_anomaly"]
        assert entry["ok"] is False

    def test_a_miss_describes_the_table(self, con, rulebook):
        _seed_nmme(con, [("SOM", -1.4, 1, "2024-03-08")])
        outcome = _refresh(con, rulebook)
        error = outcome.detail["entries"]["nmme_precip_anomaly"]["error"]
        assert "seasonal_forecasts: 1 rows" in error
        assert "forecast_issue_date spans 2024-03-08..2024-03-08" in error
        assert "lead_months in {1}" in error

    def test_the_shipped_where_no_longer_pins_a_lead(self, rulebook):
        entry = next(
            e for e in rulebook.get("drought.indicators.entries")
            if e["name"] == "nmme_precip_anomaly"
        )
        assert "lead_months" not in str(entry.get("where"))
        assert entry.get("date_offset_column") == "lead_months"


# ---------------------------------------------------------------------------
# D1: a cell with no row says why, and an undecided cell is not a quiet year
# ---------------------------------------------------------------------------


class TestNoRowReasons:
    def test_every_no_row_drought_verdict_stamps_its_reason(self, con, rulebook):
        # Unread feeds -> inconclusive (nothing seeded).
        run = drought_mod.run_month(
            con, ym=YM, iso3s=["SOM", "KEN"], rulebook=rulebook, today=AFTER_FREEZE
        )
        assert run.inconclusive == 2
        rows = con.execute(
            "SELECT iso3, trigger_detail_json FROM haz_triggers WHERE hazard = 'DR'"
        ).fetchall()
        assert len(rows) == 2
        for _iso3, detail_json in rows:
            detail = json.loads(detail_json)
            assert detail["no_row_reason"] == cell_ledger.REASON_INCONCLUSIVE
            assert detail["assessed"] is False

    def test_a_written_row_carries_no_reason_and_counts_as_assessed(self, con, rulebook):
        _seed_hdx(con, {"SOM": "Low concern"})
        _seed_nmme(con, [("SOM", 0.4, 1, "2024-02-08")])
        _refresh(con, rulebook)
        drought_mod.run_month(
            con, ym=YM, iso3s=["SOM"], rulebook=rulebook, today=AFTER_FREEZE
        )
        detail = json.loads(con.execute(
            "SELECT trigger_detail_json FROM haz_triggers WHERE hazard = 'DR'"
        ).fetchone()[0])
        assert "no_row_reason" not in detail
        assert detail["assessed"] is True

    def test_the_sweep_stamps_its_reason_on_the_trigger_row(self, con):
        seed_trigger(con, iso3="TON", ym="2024-03", hazard="TC", triggered=False)
        detect_mod.record_no_row_reason(
            con, hazard="TC", iso3="TON", ym="2024-03",
            reason=cell_ledger.REASON_SWEEP_INCONCLUSIVE, note="HTTP 503",
        )
        detail = json.loads(con.execute(
            "SELECT trigger_detail_json FROM haz_triggers WHERE iso3 = 'TON'"
        ).fetchone()[0])
        assert detail["no_row_reason"] == cell_ledger.REASON_SWEEP_INCONCLUSIVE
        assert detail["no_row_note"] == "HTTP 503"
        # Idempotent, and a missing trigger row is not an error.
        detect_mod.record_no_row_reason(
            con, hazard="TC", iso3="TON", ym="2024-03",
            reason=cell_ledger.REASON_SWEEP_INCONCLUSIVE,
        )
        detect_mod.record_no_row_reason(
            con, hazard="TC", ym="2024-03", iso3="FJI", reason="pending_before_freeze"
        )

    def test_every_reason_code_is_in_the_known_set(self):
        assert cell_ledger.REASON_MONTH_IN_PROGRESS in cell_ledger.NO_ROW_REASONS
        assert cell_ledger.REASON_TOO_FEW_FEEDS in cell_ledger.NO_ROW_REASONS
        for reason in drought_mod.NO_ROW_REASON_BY_RULE.values():
            assert reason in cell_ledger.NO_ROW_REASONS


class TestOccurrenceIgnoresUndecidedCells:
    TODAY = dt.date(2026, 8, 1)

    def test_an_inconclusive_year_is_not_a_quiet_year(self, con, rulebook):
        """Nine drought backcast years INCONCLUSIVE would have published a
        ~0% March drought rate for every country from cells nobody decided."""

        for year in range(2018, 2024):
            seed_trigger(con, iso3="SOM", ym=f"{year}-03", hazard="DR",
                         triggered=year == 2022)
        for year in (2018, 2019, 2020):
            con.execute(
                "UPDATE haz_triggers SET trigger_detail_json = ? "
                "WHERE iso3 = 'SOM' AND year = ?",
                [json.dumps({"no_row_reason": "indicator_inconclusive",
                             "assessed": False}), year],
            )
        br.compute_occurrence(con, rulebook, hazards=["DR"], today=self.TODAY)
        row = con.execute(
            "SELECT p_occurrence, n_years FROM haz_base_rates_occurrence "
            "WHERE iso3 = 'SOM' AND hazard = 'DR' AND calendar_month = 3"
        ).fetchone()
        assert row is not None
        assert row[1] == 3
        assert row[0] == pytest.approx(1 / 3)

    def test_all_undecided_publishes_no_rate(self, con, rulebook):
        for year in range(2018, 2024):
            seed_trigger(con, iso3="SOM", ym=f"{year}-03", hazard="DR")
            con.execute(
                "UPDATE haz_triggers SET trigger_detail_json = ? WHERE year = ?",
                [json.dumps({"assessed": False}), year],
            )
        br.compute_occurrence(con, rulebook, hazards=["DR"], today=self.TODAY)
        assert con.execute(
            "SELECT COUNT(*) FROM haz_base_rates_occurrence WHERE hazard = 'DR'"
        ).fetchone()[0] == 0


# ---------------------------------------------------------------------------
# The rulebook validator knows the new keys
# ---------------------------------------------------------------------------


class TestRulebookKeys:
    def test_min_answered_for_zero_is_validated(self):
        from resolver.hazard_resolution.rulebook import validate_rulebook

        data = make_rulebook().raw
        data["drought"]["indicators"]["min_answered_for_zero"] = 0
        problems = validate_rulebook(data)
        assert any("min_answered_for_zero" in p for p in problems)

    def test_an_hdx_ckan_address_is_a_valid_candidate(self):
        from resolver.hazard_resolution.rulebook import validate_rulebook

        data = make_rulebook().raw
        assert validate_rulebook(data) == []
        for entry in data["drought"]["indicators"]["entries"]:
            if entry["name"] == "asap_hotspots":
                entry["urls"] = ["ftp://example.test/x.zip"]
        assert any("hdx-ckan" in p for p in validate_rulebook(data))
