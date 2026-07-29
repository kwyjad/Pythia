# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the batch-economics reporter.

The load-bearing property is that this script can NEVER fail a pipeline stage:
stages 2..N of a live pipeline run whatever is on main, so a diagnostics bug
here would take out real forecasting work.
"""

from __future__ import annotations

import json
import os

import duckdb
import pytest

from scripts.ci import batch_economics


PIPE = "pl_test_1"


def _db(tmp_path, *, with_requests=True, with_batches=True):
    path = str(tmp_path / "t.duckdb")
    con = duckdb.connect(path)
    if with_batches:
        con.execute(
            """
            CREATE TABLE llm_batches (
                batch_id TEXT, provider TEXT, family TEXT, stage TEXT,
                model_id TEXT, status TEXT, pipeline_id TEXT,
                n_requests INTEGER, n_succeeded INTEGER, n_errored INTEGER,
                n_expired INTEGER, n_fallback_sync INTEGER, submitted_at TIMESTAMP
            )
            """
        )
    if with_requests:
        con.execute(
            """
            CREATE TABLE llm_batch_requests (
                custom_id TEXT, family TEXT, status TEXT, pipeline_id TEXT
            )
            """
        )
    con.close()
    return path


def _run(path, tmp_path, stage="hs_submit"):
    out = str(tmp_path / f"econ_{stage}.json")
    rc = batch_economics.main(
        ["--db", path, "--pipeline-id", PIPE, "--stage", stage, "--out", out]
    )
    assert rc == 0
    return json.load(open(out))


def test_reports_fallback_rate_from_request_rows(tmp_path):
    path = _db(tmp_path)
    con = duckdb.connect(path)
    con.execute(
        "INSERT INTO llm_batches VALUES "
        "('b1','google','hs_rc','hs_submit','gemini','collected',?,10,7,1,0,2,NULL)",
        [PIPE],
    )
    for i in range(7):
        con.execute("INSERT INTO llm_batch_requests VALUES (?, 'hs_rc', 'succeeded', ?)", [f"s{i}", PIPE])
    for i in range(2):
        con.execute("INSERT INTO llm_batch_requests VALUES (?, 'hs_rc', 'fallback_sync', ?)", [f"f{i}", PIPE])
    con.execute("INSERT INTO llm_batch_requests VALUES ('e0', 'hs_rc', 'errored', ?)", [PIPE])
    con.close()

    report = _run(path, tmp_path)
    s = report["summary"]
    assert s["n_terminal_requests"] == 10
    assert s["fallback_sync_pct"] == 20.0
    assert s["batched_pct"] == 70.0
    assert s["batch_totals"]["n_fallback_sync"] == 2


def test_other_pipelines_are_excluded(tmp_path):
    """The DB travels between runs — a report must be scoped to its own pipeline."""
    path = _db(tmp_path)
    con = duckdb.connect(path)
    con.execute("INSERT INTO llm_batch_requests VALUES ('a', 'hs_rc', 'succeeded', ?)", [PIPE])
    con.execute("INSERT INTO llm_batch_requests VALUES ('b', 'hs_rc', 'fallback_sync', 'pl_other')")
    con.execute(
        "INSERT INTO llm_batches VALUES ('bx','google','hs_rc','s','m','collected','pl_other',9,0,0,0,9,NULL)"
    )
    con.close()

    report = _run(path, tmp_path)
    assert report["summary"]["n_terminal_requests"] == 1
    assert report["summary"]["fallback_sync_pct"] == 0.0
    assert report["batches"] == []


def test_warns_when_fallback_rate_exceeds_threshold(tmp_path, capsys):
    path = _db(tmp_path)
    con = duckdb.connect(path)
    for i in range(4):
        con.execute("INSERT INTO llm_batch_requests VALUES (?, 'spd_v2', 'fallback_sync', ?)", [f"f{i}", PIPE])
    con.execute("INSERT INTO llm_batch_requests VALUES ('s0', 'spd_v2', 'succeeded', ?)", [PIPE])
    con.close()

    _run(path, tmp_path)
    assert "::warning title=Batch discount not realised::" in capsys.readouterr().out


def test_no_warning_when_fully_batched(tmp_path, capsys):
    path = _db(tmp_path)
    con = duckdb.connect(path)
    for i in range(5):
        con.execute("INSERT INTO llm_batch_requests VALUES (?, 'spd_v2', 'succeeded', ?)", [f"s{i}", PIPE])
    con.close()

    _run(path, tmp_path)
    assert "::warning" not in capsys.readouterr().out


# --- never-fail-a-stage guarantees -------------------------------------------------


def test_missing_tables_still_exit_zero(tmp_path):
    """A pre-batch DB has neither table; the stage must still pass."""
    path = _db(tmp_path, with_requests=False, with_batches=False)
    report = _run(path, tmp_path)
    assert report["batches"] == []
    assert report["summary"]["n_terminal_requests"] == 0


def test_missing_db_exits_zero(tmp_path):
    assert batch_economics.main(
        ["--db", str(tmp_path / "nope.duckdb"), "--pipeline-id", PIPE, "--stage", "s"]
    ) == 0


def test_empty_pipeline_id_exits_zero(tmp_path):
    path = _db(tmp_path)
    assert batch_economics.main(["--db", path, "--pipeline-id", "", "--stage", "s"]) == 0


def test_unwritable_out_path_exits_zero(tmp_path):
    """Report generation failing must not take the stage down with it."""
    path = _db(tmp_path)
    rc = batch_economics.main(
        ["--db", path, "--pipeline-id", PIPE, "--stage", "s", "--out", "/proc/nope/x.json"]
    )
    assert rc == 0


def test_writes_github_step_summary(tmp_path, monkeypatch):
    path = _db(tmp_path)
    con = duckdb.connect(path)
    con.execute("INSERT INTO llm_batch_requests VALUES ('s0', 'hs_rc', 'succeeded', ?)", [PIPE])
    con.close()
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    _run(path, tmp_path)
    assert "Batch economics" in summary.read_text()


def test_zero_batches_is_called_out_explicitly(tmp_path):
    """The silent-full-price case is the whole point — it must be legible."""
    path = _db(tmp_path)
    report = _run(path, tmp_path)
    md = batch_economics._markdown(report)
    assert "No provider batches recorded" in md
