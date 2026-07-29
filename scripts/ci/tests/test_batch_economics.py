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
                n_expired INTEGER, n_fallback_sync INTEGER, submitted_at TIMESTAMP,
                error_text TEXT
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
        "('b1','google','hs_rc','hs_submit','gemini','collected',?,10,7,1,0,2,NULL,NULL)",
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
        "INSERT INTO llm_batches VALUES ('bx','google','hs_rc','s','m','collected','pl_other',9,0,0,0,9,NULL,NULL)"
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


def test_empty_batch_is_named_with_its_recorded_cause(tmp_path, capsys):
    """A batch that yields nothing is WHY a fallback rate is high.

    On 2026-07-29 both OpenAI batches returned zero results and the cause had
    been discarded, so a 32% fallback rate had no explanation attached.
    """
    path = _db(tmp_path)
    con = duckdb.connect(path)
    con.execute(
        "INSERT INTO llm_batches VALUES "
        "('bo','openai','spd_v2','fc','gpt-5.6-sol','collected',?,12,0,0,12,0,NULL,"
        "'{\"provider_state\": \"failed\"}')",
        [PIPE],
    )
    for i in range(12):
        con.execute("INSERT INTO llm_batch_requests VALUES (?, 'spd_v2', 'fallback_sync', ?)", [f"f{i}", PIPE])
    con.close()

    report = _run(path, tmp_path)
    assert report["batches"][0]["error_text"] == '{"provider_state": "failed"}'
    md = batch_economics._markdown(report)
    assert "returned no results" in md
    assert "provider_state" in md
    out = capsys.readouterr().out
    assert "::warning title=Batch returned no results::" in out
    assert "openai" in out


def test_healthy_batch_emits_no_empty_batch_warning(tmp_path, capsys):
    path = _db(tmp_path)
    con = duckdb.connect(path)
    con.execute(
        "INSERT INTO llm_batches VALUES "
        "('bg','google','spd_v2','fc','gemini','collected',?,6,6,0,0,0,NULL,NULL)",
        [PIPE],
    )
    for i in range(6):
        con.execute("INSERT INTO llm_batch_requests VALUES (?, 'spd_v2', 'succeeded', ?)", [f"s{i}", PIPE])
    con.close()
    _run(path, tmp_path)
    assert "Batch returned no results" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# Cost weighting
#
# The reference run's fallback rate was 32% of requests but 67% of forecaster
# spend, because the OpenAI members are the expensive ones. A request count
# understates a cost problem by 2x, so the cost view is what the warning fires
# on. The ledger was also unscoped, reporting the travelling DB's cumulative
# spend ("$1.37 of $65.26") for a run that had spent $3.29.
# --------------------------------------------------------------------------

HS_RUN = "hs_20260729T071137"
FC_RUN = "fc_1785315556"


def _db_with_calls(tmp_path):
    path = _db(tmp_path)
    con = duckdb.connect(path)
    con.execute(
        """
        CREATE TABLE llm_calls (
            run_id TEXT, hs_run_id TEXT, provider TEXT,
            cost_usd DOUBLE, usage_json TEXT
        )
        """
    )
    batch = '{"service_tier": "batch"}'
    plain = "{}"

    def add(run_id, hs_run_id, provider, cost, usage):
        con.execute("INSERT INTO llm_calls VALUES (?,?,?,?,?)", [run_id, hs_run_id, provider, cost, usage])

    # This run: openai batching failed (full price), google batched.
    add(FC_RUN, None, "openai", 2.00, plain)
    add(FC_RUN, None, "google", 1.00, batch)
    # Grounding: never batchable, must not count as lost discount.
    add(None, HS_RUN, "brave", 0.50, plain)
    # A different pipeline in the same travelling DB — must be excluded.
    add("fc_other", None, "openai", 60.00, plain)
    con.close()
    return path


def _run_scoped(path, tmp_path, **extra):
    out = str(tmp_path / "econ_cost.json")
    argv = [
        "--db", path, "--pipeline-id", PIPE, "--stage", "fc_collect_finalize",
        "--out", out, "--hs-run-id", HS_RUN, "--fc-run-id", FC_RUN,
    ]
    for k, v in extra.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    assert batch_economics.main(argv) == 0
    return json.load(open(out))


def test_cost_section_is_scoped_to_this_run(tmp_path):
    path = _db_with_calls(tmp_path)
    con = duckdb.connect(path)
    con.execute(
        "INSERT INTO llm_batches VALUES "
        "('bo','openai','spd_v2','fc','gpt','collected',?,4,0,0,4,0,NULL,NULL)",
        [PIPE],
    )
    con.execute(
        "INSERT INTO llm_batches VALUES "
        "('bg','google','spd_v2','fc','gemini','collected',?,4,4,0,0,0,NULL,NULL)",
        [PIPE],
    )
    con.close()

    cost = _run_scoped(path, tmp_path)["cost"]
    # $60 from the foreign run must not appear anywhere.
    assert cost["run_total_cost_usd"] == pytest.approx(3.50)
    assert "fc_other" not in json.dumps(cost)


def test_lost_discount_excludes_providers_we_never_tried_to_batch(tmp_path):
    path = _db_with_calls(tmp_path)
    con = duckdb.connect(path)
    con.execute(
        "INSERT INTO llm_batches VALUES "
        "('bo','openai','spd_v2','fc','gpt','collected',?,4,0,0,4,0,NULL,NULL)",
        [PIPE],
    )
    con.execute(
        "INSERT INTO llm_batches VALUES "
        "('bg','google','spd_v2','fc','gemini','collected',?,4,4,0,0,0,NULL,NULL)",
        [PIPE],
    )
    con.close()

    cost = _run_scoped(path, tmp_path)["cost"]
    # Batchable = openai $2 (lost) + google $1 (batched). Brave's $0.50 is not
    # batchable and must not inflate the loss.
    assert cost["batchable_cost_usd"] == pytest.approx(3.00)
    assert cost["lost_discount_cost_usd"] == pytest.approx(2.00)
    assert cost["lost_discount_pct_of_batchable"] == pytest.approx(66.7)
    assert cost["by_provider"]["brave"]["batch_attempted"] is False
    assert cost["by_provider"]["openai"]["batch_attempted"] is True


def test_cost_share_warning_fires_on_spend_not_request_count(tmp_path, capsys):
    path = _db_with_calls(tmp_path)
    con = duckdb.connect(path)
    con.execute(
        "INSERT INTO llm_batches VALUES "
        "('bo','openai','spd_v2','fc','gpt','collected',?,4,0,0,4,0,NULL,NULL)",
        [PIPE],
    )
    con.execute(
        "INSERT INTO llm_batches VALUES "
        "('bg','google','spd_v2','fc','gemini','collected',?,4,4,0,0,0,NULL,NULL)",
        [PIPE],
    )
    # Requests split evenly, so the request-count rate alone reads as ~50%
    # while the real cost impact is 67%.
    for i in range(4):
        con.execute("INSERT INTO llm_batch_requests VALUES (?, 'spd_v2', 'fallback_sync', ?)", [f"f{i}", PIPE])
        con.execute("INSERT INTO llm_batch_requests VALUES (?, 'spd_v2', 'succeeded', ?)", [f"s{i}", PIPE])
    con.close()

    _run_scoped(path, tmp_path)
    out = capsys.readouterr().out
    assert "::warning title=Batch discount not realised (cost)::" in out
    assert "$2.0 of $3.0" in out


def test_no_cost_warning_when_everything_batched(tmp_path, capsys):
    path = _db(tmp_path)
    con = duckdb.connect(path)
    con.execute(
        """
        CREATE TABLE llm_calls (
            run_id TEXT, hs_run_id TEXT, provider TEXT,
            cost_usd DOUBLE, usage_json TEXT
        )
        """
    )
    con.execute(
        "INSERT INTO llm_calls VALUES (?, NULL, 'google', 1.0, '{\"service_tier\": \"batch\"}')",
        [FC_RUN],
    )
    con.execute(
        "INSERT INTO llm_batches VALUES "
        "('bg','google','spd_v2','fc','gemini','collected',?,4,4,0,0,0,NULL,NULL)",
        [PIPE],
    )
    con.close()
    report = _run_scoped(path, tmp_path)
    assert report["cost"]["lost_discount_cost_usd"] == pytest.approx(0.0)
    assert "not realised (cost)" not in capsys.readouterr().out


def test_cost_section_omitted_rather_than_unscoped_without_run_ids(tmp_path):
    """Better to report nothing than to report the travelling DB's total."""
    path = _db_with_calls(tmp_path)
    report = _run(path, tmp_path)
    assert report["cost"]["available"] is False
    assert "no run id" in report["cost"]["reason"]
    assert report["ledger"]["available"] is False
    md = batch_economics._markdown(report)
    assert "Cost section unavailable" in md


def test_missing_llm_calls_table_still_returns_zero(tmp_path):
    """Never fail a stage: a DB with no llm_calls at all must still report."""
    path = _db(tmp_path)
    report = _run_scoped(path, tmp_path)
    assert report["cost"]["available"] is False
