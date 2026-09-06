# Pythia / Copyright (c) 2025 Kevin Wyjad
"""Group H — housekeeping that cost runtime or lied in the logs.

Each test names the thing that was wrong in run 33946954189 and asserts the
row, the reason or the exit code that now stands in its place.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import duckdb
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    """Import a module by path so a test can reach a script under scripts/."""

    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# H1 — dead tables are no longer created and then dropped
# ---------------------------------------------------------------------------


class TestDeadTablesAreNotCreated:
    def test_ensure_schema_does_not_create_the_gtmc1_tables(self, tmp_path):
        from pythia.db import schema as schema_mod

        con = duckdb.connect(str(tmp_path / "t.duckdb"))
        try:
            schema_mod.ensure_schema(con)
            names = {
                str(r[0])
                for r in con.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'main'"
                ).fetchall()
            }
        finally:
            con.close()
        assert "gtmc1_runs" not in names
        assert "gtmc1_actors" not in names

    def test_they_stay_in_dead_tables_so_an_old_database_still_loses_them(self, tmp_path):
        from pythia.db import schema as schema_mod

        assert "gtmc1_runs" in schema_mod.DEAD_TABLES
        assert "gtmc1_actors" in schema_mod.DEAD_TABLES

        con = duckdb.connect(str(tmp_path / "old.duckdb"))
        try:
            con.execute("CREATE TABLE gtmc1_runs (run_id TEXT)")
            con.execute("CREATE TABLE gtmc1_actors (actor_name TEXT)")
            outcome = schema_mod.drop_dead_tables(con)
            names = {
                str(r[0])
                for r in con.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'main'"
                ).fetchall()
            }
        finally:
            con.close()
        assert outcome["gtmc1_runs"] == "dropped"
        assert outcome["gtmc1_actors"] == "dropped"
        assert "gtmc1_runs" not in names

    def test_the_schema_no_longer_declares_them_at_all(self):
        """The cost was churn, not a stray table.

        ``ensure_schema`` created both and ``drop_dead_tables`` dropped them
        again a moment later, on every single connection — so an
        end-state assertion cannot see the difference. This reads the
        declaration itself.
        """

        source = (REPO_ROOT / "pythia" / "db" / "schema.py").read_text(encoding="utf-8")
        assert 'CREATE TABLE IF NOT EXISTS gtmc1_runs' not in source
        assert 'CREATE TABLE IF NOT EXISTS gtmc1_actors' not in source

    def test_a_dead_table_carrying_rows_is_kept(self, tmp_path):
        from pythia.db import schema as schema_mod

        con = duckdb.connect(str(tmp_path / "rows.duckdb"))
        try:
            con.execute("CREATE TABLE gtmc1_runs (run_id TEXT)")
            con.execute("INSERT INTO gtmc1_runs VALUES ('r1')")
            outcome = schema_mod.drop_dead_tables(con)
        finally:
            con.close()
        assert outcome["gtmc1_runs"] == "kept_non_empty"


# ---------------------------------------------------------------------------
# H2 — a level counts only where it stands as its own word
# ---------------------------------------------------------------------------


class TestLogLevelClassifier:
    @pytest.fixture(scope="class")
    def bundle(self):
        return _load("_h_bundle", "scripts/build_resolver_debug_bundle.py")

    def test_a_counter_named_keyerror_is_not_an_error(self, bundle):
        line = "2026-09-05 12:00:00 INFO nmme | merged rows=2408 KeyError=0"
        assert bundle.log_level_of(line) == "INFO"

    def test_a_real_error_line_is_still_an_error(self, bundle):
        line = "2026-09-05 12:00:00 ERROR nmme | could not open the grid"
        assert bundle.log_level_of(line) == "ERROR"

    def test_the_leftmost_level_wins(self, bundle):
        # An INFO line whose message mentions a warning is an INFO line.
        line = "2026-09-05 INFO connector | 0 warnings raised"
        assert bundle.log_level_of(line) == "INFO"

    def test_warn_is_an_alias_and_fatal_is_critical(self, bundle):
        assert bundle.log_level_of("[warn] gateway slow") == "WARNING"
        assert bundle.log_level_of("FATAL: gave up") == "CRITICAL"

    def test_a_line_announcing_no_level_is_not_counted(self, bundle):
        assert bundle.log_level_of("  ...   at line 4") is None

    def test_workflow_annotations_are_read(self, bundle):
        assert bundle.log_level_of("::warning::route moved") == "WARNING"
        assert bundle.log_level_of("::error::key rejected") == "ERROR"


# ---------------------------------------------------------------------------
# H5 — the checkout's dirty flag stops crying wolf
# ---------------------------------------------------------------------------


class TestExpectedTrackedChanges:
    @pytest.fixture(scope="class")
    def bundle(self):
        return _load("_h_bundle2", "scripts/build_resolver_debug_bundle.py")

    def test_the_committed_fallbacks_a_run_refreshes_are_named(self, bundle):
        # These three are read by the pipeline AND rewritten by it. Removing
        # the rewrite would cost rows, so they are declared instead.
        assert "horizon_scanner/data/crisiswatch_latest.json" in bundle.EXPECTED_TRACKED_CHANGES
        assert "resolver/data/fewsnet_countries.json" in bundle.EXPECTED_TRACKED_CHANGES
        assert "data/hdx_signals/hdx_signals.csv" in bundle.EXPECTED_TRACKED_CHANGES

    def test_a_porcelain_line_yields_its_path(self, bundle):
        assert bundle._porcelain_path(" M resolver/data/fewsnet_countries.json") == (
            "resolver/data/fewsnet_countries.json"
        )
        assert bundle._porcelain_path("R  old/path.py -> new/path.py") == "new/path.py"

    def test_the_two_pure_artifacts_are_no_longer_tracked(self):
        import subprocess

        out = subprocess.run(
            ["git", "ls-files",
             "horizon_scanner/data/crisiswatch_debug.html",
             "resolver/diagnostics/ingestion/acled/http_diag.json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        ).stdout.strip()
        assert out == "", f"still tracked: {out}"


# ---------------------------------------------------------------------------
# H6 — a month that has not finished is not a failure
# ---------------------------------------------------------------------------


class TestMonthHasEnded:
    def test_the_current_month_has_not_ended(self):
        from resolver.hazard_resolution import cli as haz_cli

        assert haz_cli.month_has_ended("2026-09", date(2026, 9, 6)) is False
        assert haz_cli.month_has_ended("2026-09", date(2026, 9, 30)) is False

    def test_a_finished_month_has_ended(self):
        from resolver.hazard_resolution import cli as haz_cli

        assert haz_cli.month_has_ended("2026-08", date(2026, 9, 6)) is True
        assert haz_cli.month_has_ended("2026-09", date(2026, 10, 1)) is True

    def test_february_in_a_leap_year_is_measured_correctly(self):
        from resolver.hazard_resolution import cli as haz_cli

        assert haz_cli.month_has_ended("2024-02", date(2024, 2, 29)) is False
        assert haz_cli.month_has_ended("2024-02", date(2024, 3, 1)) is True

    def test_an_unparseable_label_does_not_claim_the_month_is_running(self):
        from resolver.hazard_resolution import cli as haz_cli

        # Refusing to decide must not turn into "still in progress", which
        # would suppress a real failure forever.
        assert haz_cli.month_has_ended("not-a-month") is True

    def test_the_unreadable_source_reason_is_unchanged(self):
        from resolver.hazard_resolution import cli as haz_cli

        class Run:
            cells = 12
            resolved_value = 0
            resolved_zero = 0
            no_data = 0
            frozen_skipped = 0
            unavailable_sources = ["emdat"]
            fetches: dict = {}

        assert "12 cell(s) assessed" in haz_cli.wrote_nothing_because_a_source_was_unreadable(Run())


# ---------------------------------------------------------------------------
# Out of scope upstream, in scope here: the ladder must not short-circuit
# ---------------------------------------------------------------------------


class TestLadderRunsOnWithoutEmdat:
    def test_build_candidates_still_collects_the_lower_rungs(self, tmp_path, monkeypatch):
        """An unreadable EM-DAT costs its own rung and nothing else.

        EM-DAT's credential is the owner's to renew, so the machine's job is
        to keep the remaining rungs answering. This pins that it does.
        """

        from resolver.hazard_resolution import candidates as cand_mod

        monkeypatch.setattr(
            cand_mod.emdat_mod, "records_for_country_month",
            lambda *a, **k: [],
        )
        monkeypatch.setattr(
            cand_mod.go_mod, "records_for_country_month",
            lambda *a, **k: [
                {
                    "record_id": "go-1",
                    "num_affected": 4200,
                    "source_url": "https://go.ifrc.org/1",
                    "as_of_date": "2026-03-14",
                }
            ],
        )
        monkeypatch.setattr(
            cand_mod.idu_mod, "records_for_country_month", lambda *a, **k: []
        )
        monkeypatch.setattr(
            cand_mod.gdacs_mod, "events_for_country_month", lambda *a, **k: []
        )

        con = duckdb.connect(str(tmp_path / "cand.duckdb"))
        try:
            found = cand_mod.build_candidates(con, "PHL", "2026-03", "flood", None)
        finally:
            con.close()

        sources = {c.source for c in found}
        assert "ifrc_go" in sources, (
            "a dead EM-DAT rung must not stop the ladder collecting the rest"
        )
        assert "emdat" not in sources


# ---------------------------------------------------------------------------
# H4 — the INFORM severity trend, frozen since January 2024
# ---------------------------------------------------------------------------


class TestInformSeverityTrend:
    def test_month_labels_are_calendar_months(self):
        from pythia import acaps

        # Asked on 15 March the old thirty-day arithmetic returned
        # Mar, Jan, Dec, Dec — February never requested, December twice.
        labels = acaps._month_labels_back(6, date(2026, 3, 15))
        assert labels == [
            "Mar2026", "Feb2026", "Jan2026", "Dec2025", "Nov2025", "Oct2025",
        ]
        assert len(set(labels)) == len(labels), "a month must not be asked twice"

    def test_labels_cross_a_year_boundary(self):
        from pythia import acaps

        assert acaps._month_labels_back(3, date(2026, 1, 10)) == [
            "Jan2026", "Dec2025", "Nov2025",
        ]

    def test_the_trend_parses_a_renamed_field(self, monkeypatch):
        from pythia import acaps

        monkeypatch.setattr(
            acaps, "_fetch_paginated",
            lambda *a, **k: [
                {"log_date": "2026-08-01", "severity_index_score": 4.1},
                {"log_date": "2026-07-01", "severity_index_score": 3.9},
            ],
        )
        entries = acaps._trend_from_country_log("SDN", "tok")
        assert [e["score"] for e in entries] == [4.1, 3.9]

    def test_a_total_parse_failure_names_the_keys_it_saw(self, monkeypatch, caplog):
        from pythia import acaps

        monkeypatch.setattr(
            acaps, "_fetch_paginated",
            lambda *a, **k: [{"observed_on": "2026-08-01", "index": 4.1}],
        )
        with caplog.at_level("WARNING"):
            entries = acaps._trend_from_country_log("SDN", "tok")
        assert entries == []
        assert "0 of 1 rows parsed" in caplog.text
        assert "observed_on" in caplog.text and "index" in caplog.text

    def test_a_failing_country_log_does_not_raise(self, monkeypatch):
        from pythia import acaps

        def _boom(*a, **k):
            raise RuntimeError("gateway said no")

        monkeypatch.setattr(acaps, "_fetch_paginated", _boom)
        assert acaps._trend_from_country_log("SDN", "tok") == []

    def test_the_monthly_snapshots_supply_a_trend_when_the_log_cannot(self, monkeypatch):
        """The row-landing half: a dead country-log is not a dead trend."""

        from pythia import acaps

        def _fake(path, params=None, max_pages=1, token=None):
            # One snapshot per month label, the endpoint that already answers.
            label = path.rstrip("/").rsplit("/", 1)[-1]
            scores = {"Aug2026": 4.2, "Jul2026": 4.0, "Jun2026": 3.7}
            if label in scores:
                return [{"country_level": True, "severity_index_score": scores[label]}]
            return []

        monkeypatch.setattr(acaps, "_fetch_paginated", _fake)
        monkeypatch.setattr(
            acaps, "_month_labels_back",
            lambda n, today=None: ["Aug2026", "Jul2026", "Jun2026"],
        )
        entries = acaps._trend_from_monthly_snapshots("SDN", "tok", months_back=3)
        assert [e["score"] for e in entries] == [4.2, 4.0, 3.7]
        assert entries[0]["date"] == "2026-08-01"

    def test_the_snapshot_fallback_skips_the_label_already_used(self, monkeypatch):
        from pythia import acaps

        monkeypatch.setattr(
            acaps, "_fetch_paginated",
            lambda *a, **k: [{"country_level": True, "severity_index_score": 4.0}],
        )
        monkeypatch.setattr(
            acaps, "_month_labels_back",
            lambda n, today=None: ["Aug2026", "Jul2026"],
        )
        entries = acaps._trend_from_monthly_snapshots(
            "SDN", "tok", months_back=2, skip_label="Aug2026",
        )
        assert [e["date"] for e in entries] == ["2026-07-01"]

    def test_a_month_with_no_snapshot_is_absent_not_a_zero(self, monkeypatch):
        from pythia import acaps

        monkeypatch.setattr(acaps, "_fetch_paginated", lambda *a, **k: [])
        monkeypatch.setattr(
            acaps, "_month_labels_back", lambda n, today=None: ["Aug2026"],
        )
        assert acaps._trend_from_monthly_snapshots("SDN", "tok") == []
