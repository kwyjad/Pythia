# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The 11 Nov 2026 Phase 2.6 crash, and the three things that missed it.

Resolver Update run 42999258374 (scheduled) set ``LOG_LEVEL`` to the empty
string, because a scheduled run has no inputs; both Phase 2.6 scripts called
``logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))``, got ``''``
and died with ``ValueError: Unknown level: ''`` under ``continue-on-error``.
The log index filed the crash as INFO=1, no issue named the missing summary
JSONs, and no check saw that the four conflict displacement tables left the
run with the counts they entered it with. The fixture under
``fixtures/phase26_nov2026/`` is that run's two logs and its signatures,
trimmed to the tables ``writer_steps`` declares.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from resolver.common.logs import env_log_level
from resolver.diagnostics import issue_sources, writer_steps
from scripts import build_resolver_debug_bundle as bundle

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures" / "phase26_nov2026"


# -- A: env_log_level -------------------------------------------------------


class TestEnvLogLevel:
    def test_unset_reads_as_the_default(self, monkeypatch):
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        assert env_log_level() == "INFO"

    def test_empty_reads_as_the_default(self, monkeypatch):
        """The scheduled-run case: set, and empty."""

        monkeypatch.setenv("LOG_LEVEL", "")
        assert env_log_level() == "INFO"
        monkeypatch.setenv("LOG_LEVEL", "   ")
        assert env_log_level() == "INFO"

    def test_lower_case_is_upper_cased(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "debug")
        assert env_log_level() == "DEBUG"

    def test_unknown_falls_back_and_warns_once(self, monkeypatch, caplog):
        monkeypatch.setenv("LOG_LEVEL", "LOUD")
        with caplog.at_level(logging.WARNING, logger="resolver.common.logs"):
            assert env_log_level() == "INFO"
        warnings = [r for r in caplog.records if "not a log level" in r.getMessage()]
        assert len(warnings) == 1

    def test_valid_values_pass_through(self, monkeypatch):
        for value in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            monkeypatch.setenv("LOG_LEVEL", value)
            assert env_log_level() == value

    def test_other_names_and_defaults(self, monkeypatch):
        monkeypatch.setenv("RESOLVER_LOG_LEVEL", "")
        assert env_log_level("RESOLVER_LOG_LEVEL", "WARNING") == "WARNING"
        assert env_log_level("LOG_LEVEL", environ={"LOG_LEVEL": ""}) == "INFO"
        assert env_log_level("LOG_LEVEL", environ={"LOG_LEVEL": "debug"}) == "DEBUG"

    def test_the_result_is_accepted_by_basicconfig(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "")
        logging.getLogger().setLevel(env_log_level())  # raises on ''


@pytest.mark.parametrize(
    ("module", "extra"),
    [
        ("resolver.ingestion.conflict_displacement_extract",
         ["--plan-only", "--countries", "SOM", "--months", "2026-10"]),
        ("resolver.ingestion.dtm_conflict", ["--countries", "SOM"]),
    ],
)
def test_phase26_scripts_start_with_an_empty_log_level(tmp_path, module, extra):
    """Run each script as the workflow does, with LOG_LEVEL=''.

    A subprocess, because pytest has already put handlers on the root
    logger and ``basicConfig`` would then return before it read the level.
    The scripts write their summary on the way out, error or not, so a
    summary file is the proof the process got past its logging setup.
    """

    env = {k: v for k, v in os.environ.items()
           if k not in ("DTM_API_KEY", "DTM_API_PRIMARY_KEY")}
    env["LOG_LEVEL"] = ""
    env["PYTHONPATH"] = str(REPO_ROOT)
    summary = tmp_path / "summary.json"
    proc = subprocess.run(
        [sys.executable, "-m", module, "--db", str(tmp_path / "x.duckdb"),
         "--summary-out", str(summary), *extra],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180,
    )
    assert "Unknown level" not in proc.stderr, proc.stderr[-2000:]
    assert summary.is_file(), proc.stderr[-2000:]
    json.loads(summary.read_text(encoding="utf-8"))


# -- B: a traceback is an error ---------------------------------------------


def _levels(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for level, _ in bundle.log_events(text.splitlines()):
        out[level] = out.get(level, 0) + 1
    return out


class TestTracebacksAreErrors:
    def test_the_november_phase26_log_is_one_error(self):
        text = (FIXTURE / "phase26_conflict_extract.log").read_text(encoding="utf-8")
        assert _levels(text) == {"ERROR": 1}
        events = bundle.log_events(text.splitlines())
        assert events[0][1] == "Traceback: ValueError: Unknown level: ''"

    def test_the_old_reader_filed_it_as_info(self):
        """What the index said before: the default quoted in the source."""

        text = (FIXTURE / "phase26_dtm_conflict.log").read_text(encoding="utf-8")
        old = [bundle.log_level_of(line) for line in text.splitlines()]
        assert [lv for lv in old if lv] == ["INFO"]
        assert _levels(text) == {"ERROR": 1}

    def test_an_exception_line_with_a_level_word_is_not_read(self):
        text = "Traceback (most recent call last):\n  File \"x\", line 1\nKeyError: 'INFO'\n"
        assert _levels(text) == {"ERROR": 1}

    def test_a_chained_traceback_counts_once_by_its_last_exception(self):
        text = (
            "Traceback (most recent call last):\n"
            "  File \"a\", line 1\n"
            "KeyError: 'x'\n"
            "\n"
            "During handling of the above exception, another exception occurred:\n"
            "\n"
            "Traceback (most recent call last):\n"
            "  File \"b\", line 2\n"
            "RuntimeError: gave up\n"
        )
        events = bundle.log_events(text.splitlines())
        assert events == [("ERROR", "Traceback: RuntimeError: gave up")]

    def test_log_exception_keeps_its_own_line_and_counts_once(self):
        text = (
            "2026-11-11 ERROR [x] run failed\n"
            "Traceback (most recent call last):\n"
            "  File \"a\", line 1\n"
            "ValueError: nope\n"
            "2026-11-11 INFO [x] next\n"
        )
        events = bundle.log_events(text.splitlines())
        assert events == [("ERROR", "2026-11-11 ERROR [x] run failed"),
                          ("INFO", "2026-11-11 INFO [x] next")]

    def test_ordinary_lines_are_unchanged(self):
        text = "INFO a\nWARNING b\n::error::c\nplain\n"
        assert _levels(text) == {"INFO": 1, "WARNING": 1, "ERROR": 1}


# -- C: a step that should write and did not --------------------------------


def _diag(tmp_path: Path) -> Path:
    diag = tmp_path / "diagnostics"
    shutil.copytree(FIXTURE, diag)
    return diag


def _no_stamps(_spec):
    return None


class TestMissingSummaries:
    def test_both_phase26_steps_are_named_with_their_last_error(self, tmp_path):
        diag = _diag(tmp_path)
        missing = writer_steps.missing_summaries(
            diag, lambda p: "Traceback: ValueError: Unknown level: ''"
        )
        assert {m["step"] for m in missing} == {
            "phase26_conflict_extract", "phase26_dtm_conflict"
        }
        issues = issue_sources.issues_from_missing_summaries(missing)
        ids = {i.id for i in issues}
        assert ids == {"step_left_no_summary_phase26_conflict_extract",
                       "step_left_no_summary_phase26_dtm_conflict"}
        assert all(i.severity == "degraded" for i in issues)
        assert all("Unknown level" in i.evidence for i in issues)

    def test_a_step_that_left_its_summary_is_not_named(self, tmp_path):
        diag = _diag(tmp_path)
        (diag / "dtm_conflict.json").write_text('{"status": "ok"}', encoding="utf-8")
        missing = writer_steps.missing_summaries(diag, lambda p: "")
        assert [m["step"] for m in missing] == ["phase26_conflict_extract"]

    def test_a_step_that_did_not_run_is_not_named(self, tmp_path):
        missing = writer_steps.missing_summaries(tmp_path, lambda p: "")
        assert missing == []


def _signatures(tmp_path: Path, before: dict, after: dict) -> Path:
    diag = tmp_path / "d"
    diag.mkdir()
    (diag / "db_signature_before.json").write_text(json.dumps({"all_counts": before}))
    (diag / "db_signature_after.json").write_text(json.dumps({"all_counts": after}))
    return diag


class TestWriterStepsCheck:
    STEP = writer_steps.WriterStep(
        name="phase26_dtm_conflict",
        log="phase26_dtm_conflict.log",
        summary="dtm_conflict.json",
        tables=(writer_steps.TableSpec("dtm_conflict_rounds"),
                writer_steps.TableSpec("dtm_conflict_lower_bounds")),
        nothing_to_do=lambda s: "no DTM key configured"
        if s.get("status") == "no_credentials" else None,
    )

    def _run(self, diag, touched=_no_stamps):
        (diag / "phase26_dtm_conflict.log").write_text("ran\n")
        rows = writer_steps.evaluate(
            diag, json.loads((diag / "db_signature_before.json").read_text()),
            json.loads((diag / "db_signature_after.json").read_text()),
            touched, steps=(self.STEP,),
        )
        assert len(rows) == 1
        return rows[0]

    def test_unchanged_is_the_finding(self, tmp_path):
        counts = {"dtm_conflict_rounds": 346, "dtm_conflict_lower_bounds": 106}
        row = self._run(_signatures(tmp_path, counts, counts))
        assert row["verdict"] == "unchanged"

    def test_a_moved_count_is_a_change(self, tmp_path):
        row = self._run(_signatures(
            tmp_path,
            {"dtm_conflict_rounds": 346, "dtm_conflict_lower_bounds": 106},
            {"dtm_conflict_rounds": 350, "dtm_conflict_lower_bounds": 106},
        ))
        assert row["verdict"] == "changed"

    def test_a_rewrite_in_place_is_a_change(self, tmp_path):
        counts = {"dtm_conflict_rounds": 346, "dtm_conflict_lower_bounds": 106}
        row = self._run(_signatures(tmp_path, counts, counts),
                        touched=lambda spec: 106 if spec.table.endswith("bounds") else 0)
        assert row["verdict"] == "changed"

    def test_nothing_to_do_is_not_a_finding(self, tmp_path):
        counts = {"dtm_conflict_rounds": 346, "dtm_conflict_lower_bounds": 106}
        diag = _signatures(tmp_path, counts, counts)
        (diag / "dtm_conflict.json").write_text('{"status": "no_credentials"}')
        row = self._run(diag)
        assert row["verdict"] == "nothing_to_do"
        assert row["reason"] == "no DTM key configured"

    def test_a_shared_table_count_says_nothing_about_the_step(self, tmp_path):
        step = writer_steps.WriterStep(
            name="s", log="s.log",
            tables=(writer_steps.TableSpec("facts_resolved", where="publisher = 'IDMC'"),),
        )
        diag = _signatures(tmp_path, {"facts_resolved": 1}, {"facts_resolved": 9})
        (diag / "s.log").write_text("ran\n")
        rows = writer_steps.evaluate(diag, {"all_counts": {"facts_resolved": 1}},
                                     {"all_counts": {"facts_resolved": 9}},
                                     _no_stamps, steps=(step,))
        assert rows[0]["verdict"] == "unknown"
        rows = writer_steps.evaluate(diag, {"all_counts": {"facts_resolved": 1}},
                                     {"all_counts": {"facts_resolved": 9}},
                                     lambda spec: 0, steps=(step,))
        assert rows[0]["verdict"] == "unchanged"

    def test_the_november_fixture_flags_phase26(self, tmp_path):
        diag = _diag(tmp_path)
        rows = writer_steps.evaluate(
            diag,
            json.loads((diag / "db_signature_before.json").read_text()),
            json.loads((diag / "db_signature_after.json").read_text()),
            _no_stamps,
        )
        by_step = {r["step"]: r["verdict"] for r in rows}
        assert by_step == {"phase26_conflict_extract": "unchanged",
                           "phase26_dtm_conflict": "unchanged"}


def test_the_november_bundle_flags_phase26(tmp_path):
    """The whole builder over the fixture: check, issue register, log index."""

    diag = _diag(tmp_path)
    out = tmp_path / "bundle.zip"
    bundle.build_bundle(
        out_path=out, db_path=None, diagnostics_dir=diag, run_log_dir=None,
        staging=tmp_path / "staging", environ={},
    )
    import zipfile

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()

        def read(suffix: str) -> str:
            name = next(n for n in names if n.endswith(suffix))
            return zf.read(name).decode("utf-8")

        contradictions = read("checks/contradictions.md")
        issues = json.loads(read("checks/issues.json"))
        log_index = read("logs/log_index.md")

    assert "no_writer_step_left_its_tables_unchanged" in contradictions
    line = next(ln for ln in contradictions.splitlines()
                if "no_writer_step_left_its_tables_unchanged" in ln)
    assert "FAIL" in line
    ids = {i["id"] for i in issues["issues"]}
    assert "step_left_no_summary_phase26_conflict_extract" in ids
    assert "step_left_no_summary_phase26_dtm_conflict" in ids
    assert "writer_step_left_tables_unchanged_phase26_conflict_extract" in ids
    section = log_index.split("## phase26_conflict_extract.log", 1)[1].split("##", 1)[0]
    assert "ERROR=1" in section and "INFO=" not in section


def test_dtm_stops_itself_and_names_the_countries_it_did_not_reach(tmp_path):
    """23 Nov 2026: killed by the step timeout with no summary. Now it stops."""

    import duckdb

    from resolver.ingestion import dtm_conflict as dtm

    ticks = iter([0.0, 0.0, 30.0, 400.0, 400.0, 700.0, 700.0, 700.0])
    con = duckdb.connect(str(tmp_path / "d.duckdb"))
    summary = dtm.run(
        con, ["SDN", "COD", "YEM"], key="k", delay=0, max_minutes=10,
        get=lambda url, params, headers, timeout: (200, {"result": []}, ""),
        clock=lambda: next(ticks),
    )
    assert summary["not_reached"] == ["YEM"]
    assert set(summary["per_country"]) == {"SDN", "COD"}
