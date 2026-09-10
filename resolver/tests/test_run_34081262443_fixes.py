# Pythia / Copyright (c) 2025 Kevin Wyjad
"""Faults found in run 34081262443 — the first green Resolver Update after
Groups A-H.

Three of them are checks that could never pass, which is worse than a check
that is absent: a reader who sees the same red line every run stops reading
the report. The rest are writers and readers that lost a row or a field
quietly. Each test names the thing that was wrong and asserts what now
stands in its place; each fails against the code that produced that run.
"""

from __future__ import annotations

import importlib.util
import json
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
# 1 — git status --porcelain keeps its leading status column
# ---------------------------------------------------------------------------


class TestPorcelainFirstLine:
    """``_run_git`` stripped the whole output, so the FIRST porcelain line
    lost its leading status space and ``line[3:]`` cut a character off its
    path. ``data/hdx_signals/hdx_signals.csv`` became
    ``ata/hdx_signals/hdx_signals.csv``, matched nothing in
    EXPECTED_TRACKED_CHANGES, and the checkout check reported an unexpected
    change to a file that is declared expected — on every run.
    """

    def test_run_git_keeps_leading_whitespace(self, tmp_path, monkeypatch):
        bundle = _load("bundle_porcelain", "scripts/build_resolver_debug_bundle.py")

        class _Result:
            returncode = 0
            stdout = (
                " M data/hdx_signals/hdx_signals.csv\n"
                " M horizon_scanner/data/crisiswatch_latest.json\n"
            )

        monkeypatch.setattr(bundle.subprocess, "run", lambda *a, **k: _Result())
        lines = bundle._run_git(["status", "--porcelain"]).splitlines()
        assert lines[0] == " M data/hdx_signals/hdx_signals.csv"
        assert bundle._porcelain_path(lines[0]) == "data/hdx_signals/hdx_signals.csv"

    def test_every_declared_expected_change_is_recognised(self):
        """The path the first line yields must be one the declaration knows."""

        bundle = _load("bundle_porcelain2", "scripts/build_resolver_debug_bundle.py")
        for path in sorted(bundle.EXPECTED_TRACKED_CHANGES):
            # Both the unstaged (" M ") and staged ("M  ") spellings.
            for line in (f" M {path}", f"M  {path}"):
                assert bundle._porcelain_path(line) == path
                assert bundle._porcelain_path(line) in bundle.EXPECTED_TRACKED_CHANGES


# ---------------------------------------------------------------------------
# 2 — the drought absence-zero check reads the provenance those rows carry
# ---------------------------------------------------------------------------


class TestDroughtZeroFeedCount:
    """An absence zero is written by ``write_zero_resolution``, whose
    provenance has no ``decision`` key at all — the readings sit under
    ``evidence_of_absence``. Reading only ``decision`` scored every such row
    at zero answering feeds, so the check named 203 country-months that had
    three feeds behind them and could never pass.
    """

    @staticmethod
    def _db(tmp_path, provenance: dict) -> Path:
        path = tmp_path / "haz.duckdb"
        con = duckdb.connect(str(path))
        con.execute(
            """
            CREATE TABLE haz_resolutions (
                iso3 TEXT, year INTEGER, month INTEGER, hazard TEXT,
                status TEXT, rule_fired TEXT, provenance_json TEXT
            )
            """
        )
        con.execute(
            "INSERT INTO haz_resolutions VALUES "
            "('ETH', 2026, 8, 'DR', 'RESOLVED_ZERO', "
            "'drought_zero:no_indicator_signal+no_ipc_deterioration', ?)",
            [json.dumps(provenance)],
        )
        con.close()
        return path

    @staticmethod
    def _verdict(bundle, db_path: Path):
        builder = bundle.BundleBuilder.__new__(bundle.BundleBuilder)
        builder._con = duckdb.connect(str(db_path), read_only=True)
        builder.checks = []
        builder._check = lambda *a, **k: builder.checks.append((a, k))
        builder.tables = lambda: {"haz_resolutions"}
        builder.query = lambda sql, params=None: (
            [], builder._con.execute(sql, params or []).fetchall()
        )
        builder._check_no_zero_rests_on_one_feed()
        return builder.checks[0][0]

    def test_an_absence_zero_with_enough_feeds_passes(self, tmp_path):
        bundle = _load("bundle_drought", "scripts/build_resolver_debug_bundle.py")
        provenance = {
            "rule_fired": "drought_zero:no_indicator_signal+no_ipc_deterioration",
            "evidence_of_absence": {
                "indicators": {
                    "answered_count": 3,
                    "min_answered_for_zero": 2,
                    "readings": [
                        {"name": "a", "state": "no_drought"},
                        {"name": "b", "state": "no_drought"},
                        {"name": "c", "state": "no_drought"},
                    ],
                },
                "ipc": {"delta": None},
            },
        }
        verdict = self._verdict(bundle, self._db(tmp_path, provenance))
        assert verdict[1] == "PASS", verdict

    def test_an_absence_zero_on_one_feed_still_fails(self, tmp_path):
        """The check must keep catching the fault it was built for."""

        bundle = _load("bundle_drought2", "scripts/build_resolver_debug_bundle.py")
        provenance = {
            "evidence_of_absence": {
                "indicators": {
                    "answered_count": 1,
                    "readings": [{"name": "a", "state": "no_drought"}],
                },
                "ipc": {"delta": None},
            },
        }
        verdict = self._verdict(bundle, self._db(tmp_path, provenance))
        assert verdict[1] == "FAIL", verdict

    def test_a_measured_zero_is_exempt_under_either_shape(self, tmp_path):
        """A zero resting on two read IPC analyses does not need the feeds."""

        bundle = _load("bundle_drought3", "scripts/build_resolver_debug_bundle.py")
        provenance = {
            "evidence_of_absence": {
                "indicators": {"answered_count": 0, "readings": []},
                "ipc": {"delta": 0.0},
            },
        }
        verdict = self._verdict(bundle, self._db(tmp_path, provenance))
        assert verdict[1] == "PASS", verdict


# ---------------------------------------------------------------------------
# 4 — the GDELT writer names its own write stamp
# ---------------------------------------------------------------------------


class TestGdeltWriteStamp:
    """``INSERT OR REPLACE`` keeps every column the statement does not name,
    so ``fetched_at`` left to its DEFAULT never moved for a row that already
    existed: 16,588 rows rewritten, none stamped, and the reconciliation
    correctly answered that this run had touched nothing.
    """

    @staticmethod
    def _con(tmp_path):
        con = duckdb.connect(str(tmp_path / "g.duckdb"))
        con.execute(
            """
            CREATE TABLE gdelt_conflict_indicators (
                iso3 TEXT, event_date DATE, total_events INTEGER,
                material_conflict_events INTEGER, verbal_conflict_events INTEGER,
                tier1_events INTEGER, tier2_events INTEGER, tier3_events INTEGER,
                avg_goldstein DOUBLE, avg_tone_conflict DOUBLE,
                top_codes_json TEXT, is_test BOOLEAN,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (iso3, event_date)
            )
            """
        )
        return con

    def test_rewriting_an_existing_row_moves_fetched_at(self, tmp_path):
        from pythia.gdelt import _store_day

        con = self._con(tmp_path)
        day = date(2026, 9, 1)
        payload = {"ETH": {"total_events": 5, "top_codes_json": "{}"}}

        _store_day(con, day, payload)
        first = con.execute("SELECT fetched_at FROM gdelt_conflict_indicators").fetchone()[0]

        con.execute(
            "UPDATE gdelt_conflict_indicators SET fetched_at = TIMESTAMP '2000-01-01 00:00:00'"
        )
        _store_day(con, day, payload)
        second = con.execute("SELECT fetched_at FROM gdelt_conflict_indicators").fetchone()[0]

        assert second.year > 2000, (
            "a rewritten row kept its old fetched_at — the stamp must be named "
            "explicitly, because INSERT OR REPLACE keeps unnamed columns"
        )
        assert first is not None


# ---------------------------------------------------------------------------
# 5 — a content-guarded writer that changes nothing has not failed
# ---------------------------------------------------------------------------


class TestUnchangedIsNotAFailure:
    """``store_crisiswatch_entries`` leaves an identical row — and its
    ``fetched_at`` — alone on purpose. The reconciliation demanded a stamp
    since run start and called the guard a fault.
    """

    @staticmethod
    def _builder(bundle, stream_records, monkeypatch):
        """A builder whose only stream is the crisiswatch store accounting.

        ``run_log`` is shared with every other reader in the process, so the
        stub is installed through monkeypatch and undone at the end of the
        test — a bare assignment here leaked a fake stream into
        ``bulk_store_crisiswatch``'s own accounting read two files away.
        """

        builder = bundle.BundleBuilder.__new__(bundle.BundleBuilder)
        builder._stream_file = (
            lambda name: name if name == "crisiswatch_store" else None
        )
        monkeypatch.setattr(
            bundle.run_log, "read_stream", lambda _s: iter(stream_records)
        )
        return builder

    def test_all_unchanged_reads_as_declined_not_failed(self, monkeypatch):
        bundle = _load("bundle_unchanged", "scripts/build_resolver_debug_bundle.py")
        builder = self._builder(
            bundle, [{"inserted": 0, "updated": 0, "unchanged": 78}], monkeypatch
        )
        assert "unchanged=78" in builder._writer_declined_to_change("crisiswatch")

    def test_a_writer_that_wrote_nothing_at_all_is_still_a_fault(self, monkeypatch):
        bundle = _load("bundle_unchanged2", "scripts/build_resolver_debug_bundle.py")
        builder = self._builder(
            bundle, [{"inserted": 0, "updated": 0, "unchanged": 0}], monkeypatch
        )
        assert builder._writer_declined_to_change("crisiswatch") == ""

    def test_a_writer_with_no_content_guard_is_never_excused(self, monkeypatch):
        bundle = _load("bundle_unchanged3", "scripts/build_resolver_debug_bundle.py")
        builder = self._builder(
            bundle, [{"inserted": 0, "updated": 0, "unchanged": 99}], monkeypatch
        )
        assert builder._writer_declined_to_change("gdelt_conflict_indicators") == ""


# ---------------------------------------------------------------------------
# 6 — a cached extraction is re-joined to its document
# ---------------------------------------------------------------------------


class TestCachedExtractionKeepsItsDocument:
    """The cache key is (doc, model, prompt_version, cell), so a payload
    written the day before ``doc_publisher`` existed replays that field empty
    forever. 555 of 955 accepted figures read "(unattributed)" and lost the
    authority ranking that puts a government figure above an unnamed one.
    """

    def test_a_payload_written_before_the_field_gets_it_from_the_document(self):
        from resolver.hazard_resolution.extract import _figures_from_cache

        cached = {
            "payload": {
                "figures": [
                    {
                        # A real payload from the day before doc_publisher
                        # existed: everything the model answered, and every
                        # document field the code knew about at the time.
                        "value": 1200.0,
                        "unit": "people",
                        "quote": "1,200 people were affected",
                        "stated_by": "",
                        "area": "",
                        "date": "",
                        "cumulative_or_new": "unstated",
                        "doc_id": "rw-1",
                        "doc_url": "https://reliefweb.int/node/1",
                        "doc_title": "Flood update",
                        "doc_date": "2026-08-02",
                        "doc_source_rank": 0,
                    }
                ]
            }
        }
        document = {
            "doc_id": "rw-1",
            "url": "https://reliefweb.int/node/1",
            "title": "Flood update",
            "sources": ["UN Office for the Coordination of Humanitarian Affairs"],
            "primary_country_iso3": "eth",
            "source_rank": 0,
        }
        figures = _figures_from_cache(cached, "haiku", document)
        assert figures[0].doc_publisher.startswith("UN Office")
        assert figures[0].doc_primary_country == "ETH"

    def test_the_model_own_attribution_is_never_overwritten(self):
        from resolver.hazard_resolution.extract import _figures_from_cache
        from resolver.hazard_resolution.figures import _attribute

        cached = {
            "payload": {
                "figures": [
                    {
                        "value": 5.0,
                        "unit": "people",
                        "quote": "five people",
                        "stated_by": "UNHCR",
                        "area": "",
                        "date": "",
                        "cumulative_or_new": "unstated",
                        "doc_id": "rw-2",
                        "doc_url": "",
                        "doc_title": "",
                        "doc_date": "",
                        "doc_source_rank": 0,
                    }
                ]
            }
        }
        document = {"doc_id": "rw-2", "sources": ["OCHA"]}
        figure = _figures_from_cache(cached, "haiku", document)[0]
        stated_by, source = _attribute(figure)
        assert stated_by == "UNHCR"
        assert source != "document"

    def test_a_document_with_no_publisher_blanks_nothing(self):
        from resolver.hazard_resolution.extract import _figures_from_cache

        cached = {
            "payload": {
                "figures": [
                    {
                        "value": 5.0,
                        "unit": "people",
                        "quote": "five people",
                        "stated_by": "",
                        "area": "",
                        "date": "",
                        "cumulative_or_new": "unstated",
                        "doc_id": "rw-3",
                        "doc_url": "",
                        "doc_title": "",
                        "doc_date": "",
                        "doc_source_rank": 0,
                        "doc_publisher": "IFRC",
                    }
                ]
            }
        }
        figure = _figures_from_cache(cached, "haiku", {"doc_id": "rw-3"})[0]
        assert figure.doc_publisher == "IFRC"


# ---------------------------------------------------------------------------
# 9 — GDACS is asked once per event, and asked properly
# ---------------------------------------------------------------------------


class TestGdacsAsksOncePerEvent:
    """The machine walks the same events once per hazard-month pass. Run
    34081262443 made 2,678 per-event requests for 294 distinct events — up to
    24 for one event — and 2,132 were refused.
    """

    def test_the_session_names_itself(self):
        from resolver.connectors.gdacs import _build_session

        headers = _build_session().headers
        assert "python-requests" not in headers.get("User-Agent", "")
        assert headers.get("Accept")

    def test_a_second_pass_asks_nothing(self, monkeypatch):
        from resolver.connectors import gdacs as gdacs_mod

        gdacs_mod.reset_exposure_memo()
        calls: list[str] = []

        def _fetch(self, session, etype, eid, name_to_iso3):
            calls.append(f"{etype}/{eid}")
            return ({"population": 4200.0, "todate": "2026-08-31"}, None)

        monkeypatch.setattr(gdacs_mod.GdacsConnector, "_fetch_event_exposure", _fetch)
        connector = gdacs_mod.GdacsConnector()

        first = connector._enrich_one_event(None, {"eventtype": "FL", "eventid": 1}, {})
        second = connector._enrich_one_event(None, {"eventtype": "FL", "eventid": 1}, {})

        assert calls == ["FL/1"], "the second pass asked GDACS again"
        assert first["population"] == 4200.0
        assert second["population"] == 4200.0
        assert second["population_enriched"] is True

    def test_a_refusal_is_remembered_and_still_recorded(self, monkeypatch):
        from resolver.connectors import gdacs as gdacs_mod

        gdacs_mod.reset_exposure_memo()
        calls: list[str] = []

        def _fetch(self, session, etype, eid, name_to_iso3):
            calls.append(f"{etype}/{eid}")
            return (None, 403)

        monkeypatch.setattr(gdacs_mod.GdacsConnector, "_fetch_event_exposure", _fetch)
        connector = gdacs_mod.GdacsConnector()

        first = connector._enrich_one_event(None, {"eventtype": "TC", "eventid": 7}, {})
        second = connector._enrich_one_event(None, {"eventtype": "TC", "eventid": 7}, {})

        assert calls == ["TC/7"], "a refused event was re-hammered on the next pass"
        assert first["population_refused"] == 403
        assert second["population_refused"] == 403

    def test_the_memo_does_not_outlive_a_reset(self, monkeypatch):
        from resolver.connectors import gdacs as gdacs_mod

        gdacs_mod.reset_exposure_memo()
        calls: list[str] = []

        def _fetch(self, session, etype, eid, name_to_iso3):
            calls.append(f"{etype}/{eid}")
            return ({"population": 1.0, "todate": "2026-08-31"}, None)

        monkeypatch.setattr(gdacs_mod.GdacsConnector, "_fetch_event_exposure", _fetch)
        connector = gdacs_mod.GdacsConnector()
        connector._enrich_one_event(None, {"eventtype": "FL", "eventid": 2}, {})
        gdacs_mod.reset_exposure_memo()
        connector._enrich_one_event(None, {"eventtype": "FL", "eventid": 2}, {})
        assert calls == ["FL/2", "FL/2"]


# ---------------------------------------------------------------------------
# 8 — New Caledonia resolves
# ---------------------------------------------------------------------------


class TestCrisisWatchNewCaledonia:
    def test_the_bracketed_heading_resolves(self):
        from horizon_scanner.crisiswatch import _resolve_iso3

        assert _resolve_iso3("New Caledonia (France)") == "NCL"
        assert _resolve_iso3("New Caledonia") == "NCL"


# ---------------------------------------------------------------------------
# 7 — the ACAPS trend fallback fires on staleness, not only on count
# ---------------------------------------------------------------------------


class TestAcapsTrendStaleness:
    """acaps_inform_severity_trend stood at 2024-01-29 — 952 days — while
    ACAPS served 3,000 country-log records. The fallback was gated on
    "fewer than two entries", so a log full of OLD rows satisfied it and the
    monthly snapshots were never asked. A trend whose newest point predates
    the window it describes is not a trend.
    """

    def test_a_log_full_of_old_rows_counts_as_stale(self):
        from pythia.acaps import _trend_is_stale

        old = [{"date": "2024-01-29", "score": 4.1} for _ in range(6)]
        assert _trend_is_stale(old, 6, today=date(2026, 9, 7)) is True

    def test_a_current_log_is_not_stale(self):
        from pythia.acaps import _trend_is_stale

        recent = [
            {"date": "2026-07-01", "score": 4.1},
            {"date": "2026-08-01", "score": 4.2},
        ]
        assert _trend_is_stale(recent, 6, today=date(2026, 9, 7)) is False

    def test_an_empty_trend_is_stale(self):
        from pythia.acaps import _trend_is_stale

        assert _trend_is_stale([], 6, today=date(2026, 9, 7)) is True

    def test_newer_beats_longer(self):
        from pythia.acaps import _trend_is_better

        stale_long = [{"date": f"2024-0{i}-01", "score": 4.0} for i in range(1, 7)]
        fresh_short = [
            {"date": "2026-08-01", "score": 4.4},
            {"date": "2026-09-01", "score": 4.5},
        ]
        assert _trend_is_better(fresh_short, stale_long) is True
        assert _trend_is_better(stale_long, fresh_short) is False

    def test_nothing_never_replaces_something(self):
        from pythia.acaps import _trend_is_better

        assert _trend_is_better([], [{"date": "2024-01-29", "score": 4.0}]) is False


# ---------------------------------------------------------------------------
# 3 — the unexplained-no-row check separates a regression from the backlog
# ---------------------------------------------------------------------------


class TestUnexplainedNoRowScope:
    """Every one of the 29,035 unexplained cells was a backcast row dated
    2007-03..2026-06 — history written before the reason was stamped, which
    no re-run can put right because nothing records why a 2013 cell wrote
    nothing. Counting them made the check permanently red, and a check that
    can never pass is one a reader learns to skip.
    """

    @staticmethod
    def _verdict(bundle, summary):
        builder = bundle.BundleBuilder.__new__(bundle.BundleBuilder)
        builder.cell_ledger_summary = summary
        builder.checks = []
        builder._check = lambda *a, **k: builder.checks.append((a, k))
        builder._check_no_unexplained_no_row()
        return builder.checks[0][0]

    def test_a_historical_backlog_alone_does_not_fail(self):
        bundle = _load("bundle_backlog", "scripts/build_resolver_debug_bundle.py")
        verdict = self._verdict(
            bundle,
            {
                "rows": 156213,
                "no_row": 29952,
                "unexplained_no_row": 0,
                "unexplained_no_row_backlog": 29035,
                "backlog_by_hazard": {"DR": 28728, "FL": 218, "TC": 89},
                "backlog_month_range": "2007-03..2026-06",
                "cells_assessed_this_run": 2223,
            },
        )
        assert verdict[1] == "PASS", verdict

    def test_a_cell_this_run_assessed_still_fails(self):
        bundle = _load("bundle_backlog2", "scripts/build_resolver_debug_bundle.py")
        verdict = self._verdict(
            bundle,
            {
                "rows": 10,
                "no_row": 4,
                "unexplained_no_row": 1,
                "unexplained_no_row_backlog": 3,
                "backlog_by_hazard": {"DR": 3},
                "backlog_month_range": "2019-01..2019-03",
                "cells_assessed_this_run": 6,
            },
        )
        assert verdict[1] == "FAIL", verdict

    def test_the_backlog_is_named_not_merely_counted(self):
        bundle = _load("bundle_backlog3", "scripts/build_resolver_debug_bundle.py")
        builder = bundle.BundleBuilder.__new__(bundle.BundleBuilder)
        builder.cell_ledger_summary = {
            "rows": 10, "no_row": 4, "unexplained_no_row": 0,
            "unexplained_no_row_backlog": 29035,
            "backlog_by_hazard": {"DR": 28728},
            "backlog_month_range": "2007-03..2026-06",
            "cells_assessed_this_run": 2223,
        }
        builder.checks = []
        builder._check = lambda *a, **k: builder.checks.append((a, k))
        builder._check_no_unexplained_no_row()
        detail = builder.checks[0][0][4]
        assert "2007-03..2026-06" in detail
        assert "28728" in detail or "DR" in detail


# ---------------------------------------------------------------------------
# Follow-ups from the scoped re-run 34093339513
# ---------------------------------------------------------------------------


class TestScopedRunDoesNotCryWolf:
    """A scoped ``only_connector`` run never reaches the PA machine. Two
    checks failed anyway, one of them contradicting itself in the same
    sentence: "29,035 unexplained (0 cell(s) assessed this run; 0 historical)".
    """

    def test_no_cells_assessed_means_the_backlog_owns_them_all(self):
        bundle = _load("bundle_scoped", "scripts/build_resolver_debug_bundle.py")
        builder = bundle.BundleBuilder.__new__(bundle.BundleBuilder)
        builder.cell_ledger_summary = {
            "rows": 156213,
            "no_row": 30813,
            "unexplained_no_row": 0,
            "unexplained_no_row_backlog": 29035,
            "backlog_by_hazard": {"DR": 28728},
            "backlog_month_range": "2007-03..2026-06",
            "cells_assessed_this_run": 0,
            "cell_stream_recorded": True,
            "machine_ran_this_job": False,
        }
        builder.checks = []
        builder._check = lambda *a, **k: builder.checks.append((a, k))
        builder._check_no_unexplained_no_row()
        verdict = builder.checks[0][0]
        assert verdict[1] == "PASS", verdict
        assert "did not run in this job" in verdict[4]

    def test_a_machine_run_with_no_stream_must_still_be_explained(self):
        """Group A's contract: the nightly backcast records no run stream, so
        the DATABASE alone has to carry the reason. Excusing that case would
        undo the fix, so the discriminator is whether the machine ran in this
        job, never whether a stream happened to be written.
        """

        bundle = _load("bundle_scoped2", "scripts/build_resolver_debug_bundle.py")
        builder = bundle.BundleBuilder.__new__(bundle.BundleBuilder)
        builder.cell_ledger_summary = {
            "rows": 10, "no_row": 4, "unexplained_no_row": 4,
            "unexplained_no_row_backlog": 0,
            "backlog_by_hazard": {}, "backlog_month_range": "",
            "cells_assessed_this_run": 0,
            "cell_stream_recorded": False,
            "machine_ran_this_job": True,
        }
        builder.checks = []
        builder._check = lambda *a, **k: builder.checks.append((a, k))
        builder._check_no_unexplained_no_row()
        verdict = builder.checks[0][0]
        assert verdict[1] == "FAIL", verdict
        assert "database alone must explain" in verdict[4]

    def test_the_emdat_skip_guard_is_not_a_truthy_generator(self, tmp_path):
        """``any(path.glob(...) for hz in ...)`` is True for any non-empty
        tuple, because each glob is a generator object. The guard never fired.
        """

        bundle = _load("bundle_emdat", "scripts/build_resolver_debug_bundle.py")
        builder = bundle.BundleBuilder.__new__(bundle.BundleBuilder)
        builder.env = {"EMDAT_API_KEY": "set"}
        builder.diagnostics_dir = tmp_path  # no haz_run_*.json: no ladder ran
        # EM-DAT is stood down in the shipped rulebook, and that guard runs
        # first — correctly, since if we are not asking nothing else matters.
        # This test is about the guard BELOW it, which has to keep working
        # for the day the switch comes off.
        builder._declared_source_states = lambda: {}
        builder._stream_file = lambda _name: None
        builder.checks = []
        builder._check = lambda *a, **k: builder.checks.append((a, k))
        builder._check_emdat_read_when_enabled()
        verdict = builder.checks[0][0]
        assert verdict[1] == "SKIP", verdict
        assert "no flood or cyclone pass ran" in verdict[4]

    def test_a_ladder_run_is_still_checked(self, tmp_path):
        bundle = _load("bundle_emdat2", "scripts/build_resolver_debug_bundle.py")
        (tmp_path / "haz_run_flood.json").write_text("{}", encoding="utf-8")
        builder = bundle.BundleBuilder.__new__(bundle.BundleBuilder)
        builder.env = {"EMDAT_API_KEY": "set"}
        builder.diagnostics_dir = tmp_path
        builder._declared_source_states = lambda: {}  # see the test above
        builder._stream_file = lambda _name: None
        builder.tables = lambda: set()
        builder.checks = []
        builder._check = lambda *a, **k: builder.checks.append((a, k))
        builder._check_emdat_read_when_enabled()
        verdict = builder.checks[0][0]
        assert verdict[1] == "FAIL", verdict


# ---------------------------------------------------------------------------
# 7b — the ACAPS trend fix must land in the path production actually runs
# ---------------------------------------------------------------------------


class TestAcapsBulkTrendIsTheProductionPath:
    """The workflow runs ``_bulk_fetch_inform_severity``, which carries its
    OWN country-log parsing and never calls ``pythia.acaps.fetch_inform_severity``.
    Group H4's fallback and its successor both went into the function
    production does not use, which is why acaps_inform_severity_trend was
    still frozen at 2024-01-29 after both.
    """

    def test_the_bulk_path_month_labels_step_in_calendar_months(self):
        from pythia.tools.ingest_structured_data import _month_labels_back

        labels = _month_labels_back(6, today=date(2026, 3, 15))
        assert labels == [
            "Mar2026", "Feb2026", "Jan2026", "Dec2025", "Nov2025", "Oct2025",
        ], "February was skipped and December requested twice"

    def test_a_stale_country_log_is_replaced_by_the_monthly_snapshots(
        self, monkeypatch
    ):
        from pythia.tools import ingest_structured_data as ingest

        monkeypatch.setattr(ingest, "_get_acaps_token", lambda: "token")

        def _global(endpoint, max_pages=10, token=None):
            if "country-log" in endpoint:
                # Plenty of rows, all of them two years old — the shape that
                # satisfied "fewer than two entries" and froze the table.
                return [
                    {"iso3": "ETH", "date": "2024-01-29", "value": 4.1},
                    {"iso3": "ETH", "date": "2024-01-15", "value": 4.0},
                ]
            if "impact-of-crisis" in endpoint or "conditions" in endpoint or "complexity" in endpoint:
                return []
            # A monthly snapshot: the current month and the ones behind it.
            return [{"iso3": "ETH", "severity_index_score": 4.6, "country_level": True}]

        monkeypatch.setattr(ingest, "_fetch_paginated_global", _global)
        out = ingest._bulk_fetch_inform_severity({"ETH"})

        trend = out["ETH"]["trend_6m"]
        assert trend, "no trend was built at all"
        newest = max(e["date"] for e in trend)
        assert newest > "2024-01-29", (
            f"the trend is still the stale country-log (newest {newest})"
        )

    def test_a_current_country_log_is_left_alone(self, monkeypatch):
        """The fallback must not fire, or cost requests, when the log is fine."""

        from pythia.tools import ingest_structured_data as ingest

        monkeypatch.setattr(ingest, "_get_acaps_token", lambda: "token")
        this_month = date.today().replace(day=1).isoformat()
        calls: list[str] = []

        def _global(endpoint, max_pages=10, token=None):
            calls.append(endpoint)
            if "country-log" in endpoint:
                return [
                    {"iso3": "ETH", "date": this_month, "value": 4.4},
                    {"iso3": "ETH", "date": "2026-07-01", "value": 4.3},
                ]
            if any(k in endpoint for k in ("impact-of-crisis", "conditions", "complexity")):
                return []
            return [{"iso3": "ETH", "severity_index_score": 9.9, "country_level": True}]

        monkeypatch.setattr(ingest, "_fetch_paginated_global", _global)
        out = ingest._bulk_fetch_inform_severity({"ETH"})

        scores = {e["score"] for e in out["ETH"]["trend_6m"]}
        assert 9.9 not in scores, "the snapshot fallback overrode a healthy log"


# ---------------------------------------------------------------------------
# 7c — an empty snapshot fallback must name the keys it looked for
# ---------------------------------------------------------------------------


class TestSnapshotFallbackNamesWhatItSaw:
    """Run 34099296877 fetched five months of snapshots successfully and
    produced a trend for ZERO countries. "0 countries" is not a diagnosis: a
    reader cannot tell an empty response from a renamed field. The failure
    now names the keys it looked for and the keys the records carried.
    """

    def test_a_record_with_an_unexpected_score_key_is_reported(
        self, monkeypatch, caplog
    ):
        from pythia.tools import ingest_structured_data as ingest

        monkeypatch.setattr(ingest, "_get_acaps_token", lambda: "token")

        def _global(endpoint, max_pages=10, token=None):
            if "country-log" in endpoint:
                return [
                    {"iso3": "ETH", "date": "2024-01-29", "value": 4.1},
                    {"iso3": "ETH", "date": "2024-01-15", "value": 4.0},
                ]
            if any(k in endpoint for k in ("impact-of-crisis", "conditions", "complexity")):
                return []
            # A snapshot whose score lives under a name nobody looked for.
            return [{"iso3": "ETH", "country_level": True, "gravity_rating": 4.6}]

        monkeypatch.setattr(ingest, "_fetch_paginated_global", _global)
        with caplog.at_level("WARNING"):
            ingest._bulk_fetch_inform_severity({"ETH"})

        text = caplog.text
        assert "produced NO trend" in text
        assert "gravity_rating" in text, "the keys the records carried are not named"

    def test_a_known_alternate_key_is_simply_read(self, monkeypatch):
        """The lookup is a named list, so widening it is a one-line change."""

        from pythia.tools import ingest_structured_data as ingest

        monkeypatch.setattr(ingest, "_get_acaps_token", lambda: "token")

        def _global(endpoint, max_pages=10, token=None):
            if "country-log" in endpoint:
                return [{"iso3": "ETH", "date": "2024-01-29", "value": 4.1}]
            if any(k in endpoint for k in ("impact-of-crisis", "conditions", "complexity")):
                return []
            return [{"iso3": "ETH", "country_level": True, "severity": 4.6}]

        monkeypatch.setattr(ingest, "_fetch_paginated_global", _global)
        out = ingest._bulk_fetch_inform_severity({"ETH"})
        assert any(e["score"] == 4.6 for e in out["ETH"]["trend_6m"])


# ---------------------------------------------------------------------------
# 7d — the real INFORM key names, established by the diagnostic
# ---------------------------------------------------------------------------


class TestInformSeverityRealKeyNames:
    """Run 34103799256's diagnostic named them: the endpoint returns DISPLAY
    labels — "INFORM Severity Index", "Impact of the crisis" — not snake_case
    fields. Every snake_case name the connector read had matched nothing
    since it was written, so the severity score, its category and all three
    dimension scores were NULL for every row this path wrote.
    """

    #: Exactly what the records carried, from the run's own warning.
    RECORD = {
        "iso3": "ETH",
        "country_level": True,
        "crisis_id": "c1",
        "crisis_name": "Ethiopia",
        "INFORM Severity Index": 4.6,
        "INFORM Severity category": "High",
        "Impact of the crisis": 4.2,
        "Conditions of affected people": 4.8,
        "Complexity": 3.9,
        "People in need": 21000000,
    }

    def test_the_score_is_read(self):
        from pythia.tools.ingest_structured_data import (
            _SEVERITY_SCORE_KEYS, _first_present, _safe_float,
        )

        assert _safe_float(_first_present(self.RECORD, _SEVERITY_SCORE_KEYS)) == 4.6

    def test_the_category_and_dimensions_are_read(self):
        from pythia.tools.ingest_structured_data import (
            _COMPLEXITY_SCORE_KEYS,
            _CONDITIONS_SCORE_KEYS,
            _IMPACT_SCORE_KEYS,
            _SEVERITY_CATEGORY_KEYS,
            _first_present,
        )

        assert _first_present(self.RECORD, _SEVERITY_CATEGORY_KEYS) == "High"
        assert _first_present(self.RECORD, _IMPACT_SCORE_KEYS) == 4.2
        assert _first_present(self.RECORD, _CONDITIONS_SCORE_KEYS) == 4.8
        assert _first_present(self.RECORD, _COMPLEXITY_SCORE_KEYS) == 3.9

    def test_the_worst_crisis_wins_not_the_last_one(self):
        """_pick_country_crisis scored every crisis 0.0 on the old keys, so
        the pick was whichever the API listed last."""

        from pythia.tools.ingest_structured_data import _pick_country_crisis

        worst = dict(self.RECORD, crisis_id="worst")
        worst["INFORM Severity Index"] = 4.9
        mild = dict(self.RECORD, crisis_id="mild")
        mild["INFORM Severity Index"] = 2.0
        mild["country_level"] = False
        picked = _pick_country_crisis([worst, mild])
        assert picked["crisis_id"] == "worst"

    def test_the_snapshot_trend_now_builds(self, monkeypatch):
        from pythia.tools import ingest_structured_data as ingest

        monkeypatch.setattr(ingest, "_get_acaps_token", lambda: "token")

        def _global(endpoint, max_pages=10, token=None):
            if "country-log" in endpoint:
                return [{"iso3": "ETH", "date": "2024-01-29", "value": 4.1}]
            if any(k in endpoint for k in ("impact-of-crisis", "conditions", "complexity")):
                return []
            return [dict(self.RECORD)]

        monkeypatch.setattr(ingest, "_fetch_paginated_global", _global)
        out = ingest._bulk_fetch_inform_severity({"ETH"})

        assert out["ETH"]["severity_score"] == 4.6
        assert out["ETH"]["severity_category"] == "High"
        trend = out["ETH"]["trend_6m"]
        assert trend and max(e["date"] for e in trend) > "2024-01-29"
