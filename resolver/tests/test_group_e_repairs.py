# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Group E of the run-33841370196 faults: EM-DAT lockout, empty tables,
publication dates in the future.

Every test here pins a row in the database, not a log line: the fix is done
when the next Resolver Update leaves correct data behind.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from resolver.diagnostics import run_log  # noqa: E402
from resolver.hazard_resolution import cli as haz_cli  # noqa: E402
from resolver.hazard_resolution import emdat as emdat_mod  # noqa: E402
from resolver.hazard_resolution.rulebook import load_rulebook  # noqa: E402
from resolver.hazard_resolution.schema import ensure_haz_schema  # noqa: E402
from resolver.tools import enrich as enrich_mod  # noqa: E402
from resolver.tools import repair_publication_dates as repair_mod  # noqa: E402
from resolver.tools.make_deltas import process_group  # noqa: E402

TODAY = dt.date(2026, 9, 4)


# --------------------------------------------------------------------------
# E3: publication_date is never in the future
# --------------------------------------------------------------------------


class TestEnrichPublicationRule:
    def test_a_stated_date_on_a_projection_is_kept(self):
        # FEWS NET reporting_date 2026-08-28, window end 2027-01-31: the old
        # rule raised the date to the window end and then clamped it to
        # today, so the row said it was published on the run date every run.
        pub, why = enrich_mod.fix_publication_date("2026-08-28", "2027-01-31", TODAY)
        assert (pub, why) == ("2026-08-28", "supplied")

    def test_a_missing_date_on_a_future_period_is_today_not_the_period_end(self):
        pub, why = enrich_mod.fix_publication_date("", "2027-03-01", TODAY)
        assert (pub, why) == ("2026-09-04", "filled")

    def test_a_missing_date_on_a_past_observation_is_the_observation(self):
        pub, why = enrich_mod.fix_publication_date(None, "2026-01-31", TODAY)
        assert (pub, why) == ("2026-01-31", "filled")

    def test_an_observation_published_before_it_was_made_is_raised(self):
        pub, why = enrich_mod.fix_publication_date("2026-01-01", "2026-01-31", TODAY)
        assert (pub, why) == ("2026-01-31", "raised_to_as_of")

    def test_nothing_is_dated_after_today(self):
        pub, why = enrich_mod.fix_publication_date("2027-03-01", "2027-03-01", TODAY)
        assert (pub, why) == ("2026-09-04", "clamped_to_today")

    def test_enrich_applies_the_rule_to_a_frame(self):
        frame = pd.DataFrame({
            "event_id": ["", ""], "country_name": ["", ""], "iso3": ["swz", "ken"],
            "hazard_code": ["DR", "DR"], "hazard_label": ["", ""], "hazard_class": ["", ""],
            "metric": ["phase3plus_projection", "phase3plus_in_need"],
            "series_semantics": ["stock", "stock"], "value": [1.0, 2.0], "unit": ["", ""],
            "as_of_date": ["2027-03-01", "2026-05-31"],
            "publication_date": ["2026-08-28", ""],
            "publisher": ["IPC", "IPC"], "source_type": ["", ""], "source_url": ["", ""],
            "doc_title": ["", ""], "definition_text": ["", ""], "method": ["", ""],
            "confidence": ["", ""], "revision": ["", ""], "ingested_at": ["", ""],
        })
        out = enrich_mod.enrich(frame, today=TODAY)
        by_iso = dict(zip(out["iso3"], out["publication_date"]))
        assert by_iso == {"SWZ": "2026-08-28", "KEN": "2026-05-31"}


def test_a_delta_inherits_the_publication_date_of_its_fact():
    group = pd.DataFrame({
        "ym": ["2027-03"], "iso3": ["SWZ"], "hazard_code": ["DR"],
        "metric": ["phase3plus_projection"], "value": [10.0], "as_of": ["2027-03-01"],
        "source_name": ["IPC"], "source_url": [""], "series_semantics": ["stock"],
        "publication_date": ["2026-08-28"],
    })
    (record,) = process_group(group)
    assert record["publication_date"] == "2026-08-28"


def test_a_delta_without_a_publication_date_carries_none_for_the_store_to_fill():
    group = pd.DataFrame({
        "ym": ["2026-09"], "iso3": ["IDN"], "hazard_code": ["FL"], "metric": ["in_need"],
        "value": [10.0], "as_of": ["2026-09-30"], "source_name": ["GDACS"],
        "source_url": [""], "series_semantics": ["stock"],
    })
    (record,) = process_group(group)
    assert "publication_date" not in record


def test_the_store_fallback_never_dates_a_row_after_today():
    from resolver.db import duckdb_io

    frame = pd.DataFrame({"as_of_date": ["2027-03-01", "2026-01-31", "not-a-date"]})
    out = duckdb_io._ensure_frame_has_columns(frame, "facts_deltas", ["publication_date"])
    today = dt.date.today().isoformat()
    assert list(out["publication_date"]) == [today, "2026-01-31", "not-a-date"]


@pytest.fixture()
def facts_db():
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE facts_resolved (iso3 TEXT, hazard_code TEXT, metric TEXT, ym TEXT, "
        "publication_date VARCHAR, created_at TIMESTAMP)"
    )
    con.execute(
        "CREATE TABLE facts_deltas (iso3 TEXT, hazard_code TEXT, metric TEXT, ym TEXT, "
        "publication_date VARCHAR, created_at TIMESTAMP)"
    )
    con.executemany(
        "INSERT INTO facts_resolved VALUES (?,?,?,?,?,?)",
        [
            ("SWZ", "DR", "phase3plus_projection", "2027-03", "2027-03-01", "2026-08-28 14:47:17"),
            ("KEN", "DR", "phase3plus_in_need", "2026-01", "2026-01-31", "2026-02-01 00:00:00"),
        ],
    )
    con.executemany(
        "INSERT INTO facts_deltas VALUES (?,?,?,?,?,?)",
        [
            ("SWZ", "DR", "phase3plus_projection", "2027-03", "2027-03-01", "2026-08-28 14:47:17"),
            ("IDN", "FL", "in_need", "2026-09", "2026-09-30", None),
            ("KEN", "DR", "phase3plus_in_need", "2026-01", "2026-01-31", "2026-02-01 00:00:00"),
        ],
    )
    yield con
    con.close()


class TestRepairPass:
    def test_it_repairs_in_place_and_counts_what_it_did(self, facts_db):
        report = repair_mod.repair(facts_db, today=TODAY)
        assert report.repaired == 3 and report.untouched == 2
        facts = dict(
            facts_db.execute("SELECT iso3, publication_date FROM facts_resolved").fetchall()
        )
        deltas = dict(
            facts_db.execute("SELECT iso3, publication_date FROM facts_deltas").fetchall()
        )
        # created_at is the latest the projection row can have been published.
        assert facts["SWZ"] == "2026-08-28" and facts["KEN"] == "2026-01-31"
        # The delta takes its fact's repaired date; a row nothing can date
        # gets today; a correct row is untouched.
        assert deltas == {"SWZ": "2026-08-28", "IDN": "2026-09-04", "KEN": "2026-01-31"}
        assert report.tables["facts_deltas"]["from_fact"] == 1
        assert report.tables["facts_deltas"]["from_today"] == 1

    def test_it_is_idempotent(self, facts_db):
        repair_mod.repair(facts_db, today=TODAY)
        second = repair_mod.repair(facts_db, today=TODAY)
        assert second.repaired == 0
        assert repair_mod.count_future(facts_db, today=TODAY) == {
            "facts_resolved": 0, "facts_deltas": 0,
        }

    def test_a_dry_run_writes_nothing(self, facts_db):
        report = repair_mod.repair(facts_db, today=TODAY, dry_run=True)
        assert report.tables["facts_resolved"]["future"] == 1
        assert report.repaired == 0
        assert repair_mod.count_future(facts_db, today=TODAY)["facts_deltas"] == 2

    def test_it_never_deletes_a_row(self, facts_db):
        before = facts_db.execute("SELECT COUNT(*) FROM facts_deltas").fetchone()[0]
        repair_mod.repair(facts_db, today=TODAY)
        assert facts_db.execute("SELECT COUNT(*) FROM facts_deltas").fetchone()[0] == before


# --------------------------------------------------------------------------
# E1: EM-DAT lockout is named, recorded and checked
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error, expected",
    [
        ('EM-DAT GraphQL errors: [{"message": "Invalid key passed or insufficient user access"}]',
         emdat_mod.FAILURE_AUTH),
        ("HTTP 403 from https://api.emdat.be/v1: <empty body>", emdat_mod.FAILURE_AUTH),
        ("HTTP 500 from https://api.emdat.be/v1: Internal Server Error", emdat_mod.FAILURE_SERVER),
        ("HTTPSConnectionPool: Read timed out", emdat_mod.FAILURE_NETWORK),
        ("something else entirely", emdat_mod.FAILURE_OTHER),
        ("", emdat_mod.FAILURE_OTHER),
    ],
)
def test_emdat_failures_are_classified(error, expected):
    assert emdat_mod.classify_failure(error) == expected


@pytest.fixture()
def haz_con():
    con = duckdb.connect()
    ensure_haz_schema(con)
    yield con
    con.close()


def test_a_rejected_key_is_recorded_on_the_outcome_the_stream_and_the_job(
    haz_con, monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("EMDAT_API_KEY", "dead-key")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv(run_log.ENV_DIR, str(tmp_path / "streams"))
    run_log.reset_for_tests()

    def rejected(*args):
        return {"errors": [{"message": "Invalid key passed or insufficient user access"}]}

    outcome = emdat_mod.fetch_emdat(haz_con, "2026-08", "TC", load_rulebook(), post=rejected)
    assert outcome.ok is False
    assert outcome.detail["failure_class"] == emdat_mod.FAILURE_AUTH

    out = capsys.readouterr().out
    assert "::error::" in out and "EMDAT_API_KEY" in out

    records = list(run_log.read_stream(tmp_path / "streams" / f"{emdat_mod.STREAM_SOURCE_FETCHES}.jsonl"))
    assert len(records) == 1
    assert records[0]["source"] == "emdat"
    assert records[0]["failure_class"] == emdat_mod.FAILURE_AUTH
    assert records[0]["ok"] is False
    run_log.reset_for_tests()


def test_a_server_error_does_not_raise_the_job_annotation(haz_con, monkeypatch, capsys):
    monkeypatch.setenv("EMDAT_API_KEY", "key")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    def boom(*args):
        raise RuntimeError("HTTP 500 from https://api.emdat.be/v1: Internal Server Error")

    outcome = emdat_mod.fetch_emdat(haz_con, "2026-08", "TC", load_rulebook(), post=boom)
    assert outcome.detail["failure_class"] == emdat_mod.FAILURE_SERVER
    assert "::error::" not in capsys.readouterr().out


def test_the_unreadable_source_reason_names_a_rejected_credential():
    class Run:
        cells = 12
        resolved_value = 0
        resolved_zero = 0
        no_data = 0
        frozen_skipped = 0
        unavailable_sources = ["emdat"]
        fetches = {"emdat": {"ok": False, "error": "x", "detail": {"failure_class": "auth_rejected"}}}

    reason = haz_cli.wrote_nothing_because_a_source_was_unreadable(Run())
    assert "emdat" in reason and "EMDAT_API_KEY" in reason and "re-run will not fix" in reason


def test_the_unreadable_source_reason_is_plain_for_a_server_error():
    class Run:
        cells = 12
        resolved_value = 0
        resolved_zero = 0
        no_data = 0
        frozen_skipped = 0
        unavailable_sources = ["emdat"]
        fetches = {"emdat": {"ok": False, "error": "500", "detail": {"failure_class": "server_error"}}}

    reason = haz_cli.wrote_nothing_because_a_source_was_unreadable(Run())
    assert "emdat" in reason and "EMDAT_API_KEY" not in reason


# --------------------------------------------------------------------------
# E2: empty tables — a failing writer and the dead ones
# --------------------------------------------------------------------------


def test_score_views_matches_the_source_name_the_connector_writes():
    """``views_scored_forecasts`` held 0 rows through every scoring round.

    The connector writes ``source = 'VIEWS'`` and the scorer filtered on
    ``'views'``, so the step logged "nothing to score" and went green.
    """

    from pathlib import Path

    from pythia.tools.score_views import _load_views_forecast_pairs

    connector = (Path(__file__).resolve().parents[1] / "connectors" / "views.py").read_text()
    assert '"source": "VIEWS"' in connector, "the connector's literal moved; update this test"

    con = duckdb.connect()
    con.execute("CREATE TABLE hs_runs (hs_run_id TEXT)")
    con.execute("INSERT INTO hs_runs VALUES ('hs_1')")
    con.execute(
        "CREATE TABLE questions (question_id TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, hs_run_id TEXT)"
    )
    con.execute("INSERT INTO questions VALUES ('SOM_ACE_FATALITIES_2026-08','SOM','ACE','FATALITIES','hs_1')")
    con.execute(
        "CREATE TABLE resolutions (question_id TEXT, horizon_m INTEGER, value DOUBLE, observed_month TEXT)"
    )
    con.execute("INSERT INTO resolutions VALUES ('SOM_ACE_FATALITIES_2026-08', 1, 415.0, '2026-08')")
    con.execute(
        "CREATE TABLE conflict_forecasts (source TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, "
        "lead_months INTEGER, value DOUBLE, forecast_issue_date DATE, target_month DATE, model_version TEXT)"
    )
    con.execute(
        "INSERT INTO conflict_forecasts VALUES "
        "('VIEWS','SOM','ACE','fatalities',1,380.0,DATE '2026-07-01',DATE '2026-08-01','v1')"
    )
    pairs = _load_views_forecast_pairs(con)
    assert len(pairs) == 1 and pairs[0]["views_value"] == 380.0


def test_the_pythia_schema_drops_dead_tables_only_when_empty(tmp_path):
    from pythia.db import schema

    con = duckdb.connect(str(tmp_path / "p.duckdb"))
    con.execute("CREATE TABLE ipc_phases (iso3 TEXT)")
    con.execute("CREATE TABLE pm_checks (pm_check_id TEXT)")
    con.execute("INSERT INTO pm_checks VALUES ('kept')")
    con.execute("CREATE TABLE meta_runs (run_id TEXT)")
    con.execute("CREATE TABLE gtmc1_runs (run_id TEXT)")
    schema.ensure_schema(con)
    tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    # Dropped, and not recreated by the schema that used to declare them.
    for dead in ("ipc_phases", "meta_runs", "gtmc1_runs", "gtmc1_actors"):
        assert dead not in tables, dead
    # A row somebody wrote is evidence; the migration keeps it.
    assert "pm_checks" in tables
    con.close()


def test_the_resolver_schema_drops_an_empty_meta_runs_and_never_recreates_it(tmp_path):
    from resolver.db import duckdb_io

    con = duckdb.connect(str(tmp_path / "r.duckdb"))
    con.execute("CREATE TABLE meta_runs (run_id TEXT PRIMARY KEY, started_at TIMESTAMP)")
    duckdb_io.init_schema(con)
    tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert "meta_runs" not in tables
    assert {"facts_resolved", "facts_deltas", "manifests", "snapshots"} <= tables
    duckdb_io.init_schema(con)  # idempotent on the second pass
    con.close()


def test_the_ingest_orchestrator_no_longer_offers_the_dead_ipc_group():
    from pythia.tools import ingest_structured_data as ingest

    assert "ipc" not in ingest._SOURCE_GROUPS
    assert not any("ipc_phases" in tables for tables in ingest._SOURCE_GROUPS.values())
    assert "fewsnet_ipc" in ingest._SOURCE_GROUPS and "ipc_api" in ingest._SOURCE_GROUPS


def test_the_run_summary_reader_reads_the_source_fetch_stream(tmp_path, monkeypatch):
    """The stream the bundle's EM-DAT check reads is the one the connector writes."""

    from scripts import build_resolver_debug_bundle as bundle

    assert bundle.BundleBuilder.SOURCE_FETCH_STREAM == emdat_mod.STREAM_SOURCE_FETCHES
    _ = json  # the stream is JSONL; read_stream is exercised in the bundle suite
