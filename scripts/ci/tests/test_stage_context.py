# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The stage context carried from fc_collect_finalize into the Sibyl job.

Two halves: the stage writes ``stage_context.json`` (env snapshot, both DB
signatures, conclusion, identifiers), and the Sibyl job resolves whatever
identifiers it was not given from the DB and downloads the artifact, or
writes a stub saying it could not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.ci import sibyl_bundle_context as sbc
from scripts.ci import write_stage_context as wsc


@pytest.fixture
def db(tmp_path: Path) -> str:
    path = tmp_path / "db.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE hs_runs (hs_run_id TEXT, generated_at TIMESTAMP, is_test BOOLEAN)")
    con.execute("INSERT INTO hs_runs VALUES ('hs_old', TIMESTAMP '2026-07-01', FALSE), ('hs_new', TIMESTAMP '2026-08-01', FALSE)")
    con.execute("CREATE TABLE questions (question_id TEXT, hs_run_id TEXT, status TEXT)")
    con.execute("INSERT INTO questions VALUES ('Q_OLD', 'hs_old', 'active'), ('Q_NEW', 'hs_new', 'active')")
    con.execute("CREATE TABLE forecasts_raw (run_id TEXT, question_id TEXT, model_name TEXT)")
    con.execute("INSERT INTO forecasts_raw VALUES ('fc_1000', 'Q_OLD', 'm'), ('fc_2000', 'Q_NEW', 'm')")
    con.execute(
        "CREATE TABLE llm_batches (batch_id TEXT, pipeline_id TEXT, run_id TEXT, hs_run_id TEXT, submitted_at TIMESTAMP)"
    )
    con.execute("INSERT INTO llm_batches VALUES ('b1', 'pl_old', 'fc_1000', 'hs_old', TIMESTAMP '2026-07-01')")
    con.execute("INSERT INTO llm_batches VALUES ('b2', 'pl_new', 'fc_2000', 'hs_new', TIMESTAMP '2026-08-01')")
    con.close()
    return f"duckdb:///{path}"


class TestResolveIdentifiers:
    def test_everything_resolved_from_the_db(self, db):
        con = duckdb.connect(sbc._db_path(db), read_only=True)
        try:
            out = sbc.resolve_identifiers(con, pipeline_id="", hs_run_id="", forecaster_run_id="")
        finally:
            con.close()
        assert out["hs_run_id"] == "hs_new"
        assert out["forecaster_run_id"] == "fc_2000"
        assert out["pipeline_id"] == "pl_new"

    def test_supplied_values_win(self, db):
        con = duckdb.connect(sbc._db_path(db), read_only=True)
        try:
            out = sbc.resolve_identifiers(con, pipeline_id="pl_given", hs_run_id="hs_old", forecaster_run_id="")
        finally:
            con.close()
        assert out["pipeline_id"] == "pl_given"
        assert out["hs_run_id"] == "hs_old"
        assert out["forecaster_run_id"] == "fc_1000"  # follows the supplied HS run

    def test_sync_pipeline_has_no_pipeline_id(self, tmp_path):
        path = tmp_path / "sync.duckdb"
        con = duckdb.connect(str(path))
        con.execute("CREATE TABLE questions (question_id TEXT, hs_run_id TEXT, status TEXT)")
        con.execute("INSERT INTO questions VALUES ('Q', 'hs_1', 'active')")
        con.execute("CREATE TABLE forecasts_raw (run_id TEXT, question_id TEXT, model_name TEXT)")
        con.execute("INSERT INTO forecasts_raw VALUES ('fc_1', 'Q', 'm')")
        con.close()
        con = duckdb.connect(str(path), read_only=True)
        try:
            out = sbc.resolve_identifiers(con, pipeline_id="", hs_run_id="", forecaster_run_id="")
        finally:
            con.close()
        assert out["hs_run_id"] == "hs_1"
        assert out["forecaster_run_id"] == "fc_1"
        assert out["pipeline_id"] is None


class TestFetchStageContext:
    def test_no_run_id_writes_a_stub(self, tmp_path):
        target = tmp_path / "stage_context"
        result = sbc.fetch_stage_context(run_id="", target=target)
        assert result["downloaded"] is False
        stub = json.loads((target / "MISSING.json").read_text())
        assert stub["missing"] is True and stub["artifact"] == "pythia-stage-context"

    def test_failed_download_writes_a_stub_with_the_reason(self, tmp_path):
        target = tmp_path / "stage_context"
        result = sbc.fetch_stage_context(
            run_id="123", target=target, runner=lambda cmd: (1, "no artifact matches pythia-stage-context")
        )
        assert result["downloaded"] is False
        stub = json.loads((target / "MISSING.json").read_text())
        assert "no artifact matches" in stub["gh_output"] and stub["run_id"] == "123"

    def test_successful_download_leaves_no_stub(self, tmp_path):
        target = tmp_path / "stage_context"

        def runner(cmd):
            assert cmd[:4] == ["gh", "run", "download", "123"]
            target.mkdir(parents=True, exist_ok=True)
            (target / "stage_context.json").write_text("{}")
            return 0, ""

        result = sbc.fetch_stage_context(run_id="123", target=target, runner=runner)
        assert result["downloaded"] is True and result["files"] == ["stage_context.json"]
        assert not (target / "MISSING.json").exists()


class TestMain:
    def test_main_writes_env_and_stub_and_never_fails(self, db, tmp_path):
        env_file = tmp_path / "github_env"
        rc = sbc.main([
            "--db", db, "--github-env", str(env_file),
            "--stage-context-dir", str(tmp_path / "sc"), "--stage-context-run-id", "",
        ])
        assert rc == 0
        text = env_file.read_text()
        assert "PIPELINE_ID=pl_new\n" in text and "HS_RUN_ID=hs_new\n" in text
        assert "FORECASTER_RUN_ID=fc_2000\n" in text
        assert (tmp_path / "sc" / "MISSING.json").exists()

    def test_main_survives_a_missing_db(self, tmp_path):
        rc = sbc.main(["--db", str(tmp_path / "nope.duckdb"), "--github-env", str(tmp_path / "e"),
                       "--skip-download", "--hs-run-id", "hs_x"])
        assert rc == 0
        assert "HS_RUN_ID=hs_x\n" in (tmp_path / "e").read_text()


class TestWriteStageContext:
    def test_context_carries_everything(self, tmp_path):
        before = tmp_path / "before.json"
        before.write_text(json.dumps({"tables": {"questions": 3}}))
        out = tmp_path / "stage_context.json"
        rc = wsc.main([
            "--out", str(out), "--pipeline-id", "pl_x", "--hs-run-id", "hs_x",
            "--forecaster-run-id", "fc_x", "--conclusion", "success",
            "--signature-before", str(before), "--signature-after", str(tmp_path / "absent.json"),
        ])
        assert rc == 0
        payload = json.loads(out.read_text())
        assert payload["pipeline_id"] == "pl_x" and payload["forecaster_run_id"] == "fc_x"
        assert payload["stage_conclusion"] == "success"
        assert payload["db_signature_before"] == {"tables": {"questions": 3}}
        assert payload["db_signature_after"]["missing"] is True
        assert "env" in payload["env_snapshot"]  # the env_config collector ran
