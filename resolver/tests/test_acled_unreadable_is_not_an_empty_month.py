# Pythia / Copyright (c) 2025 Kevin Wyjad
"""ACLED blocked by bot-protection is an unread source, not a quiet month.

Run 34054067660 (2026-09-06) lost an entire monthly ingest to one 403.
ACLED's WAF answered every OAuth request with

    {"message": "Access denied by Imunify360 bot-protection.
                 IPs used for automation should be whitelisted"}

and two separate defects turned that into a total loss:

* ``collect_rows`` caught it under ``except RuntimeError`` — and
  ``AcledResponseError`` IS a ``RuntimeError`` — so the connector printed
  "rows=0 (no data collected)", exited 0, and the fatal Phase 1 connectors
  step went GREEN having written no tier-0 conflict rows at all.
* The run then died one step later on a bare traceback, and because Phase 1
  is fatal every later phase was skipped: FEWS NET, IPC, GDACS, NMME, HDX,
  the whole PA machine, the conflict forecasts, ACAPS, ReliefWeb, GDELT,
  ENSO, seasonal TC and CrisisWatch, none of which depend on ACLED. No
  canonical DB was uploaded, so the run left nothing behind.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from resolver.ingestion import acled_auth, acled_client

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "resolver_update.yml"

#: The body ACLED actually served on 2026-09-06.
WAF_BODY = (
    "ACLED OAuth password grant failed: status=403 "
    "content_type='application/json' body='{ \"message\": \"Access denied by "
    "Imunify360 bot-protection. IPs used for automation should be whitelisted\" }'"
)


class TestAnUnreadableSourceIsNeverAnEmptyWindow:
    def test_a_waf_block_makes_the_connector_exit_non_zero(self, monkeypatch, tmp_path):
        def _blocked(_config):
            raise acled_auth.AcledResponseError(WAF_BODY)

        monkeypatch.setattr(acled_client, "fetch_events", _blocked)
        monkeypatch.setattr(acled_client, "OUT_PATH", tmp_path / "acled.csv")
        monkeypatch.delenv("RESOLVER_SKIP_ACLED", raising=False)

        rc = acled_client.cli_main()

        assert rc == 1, (
            "a source we could not read must exit non-zero — exiting 0 is what "
            "let Phase 1 go green with no tier-0 conflict rows"
        )
        assert acled_client._UNREAD_REASON is not None
        assert "Imunify360" in acled_client._UNREAD_REASON

    def test_the_error_names_the_cause_a_reader_has_to_act_on(
        self, monkeypatch, tmp_path, capsys
    ):
        monkeypatch.setattr(
            acled_client,
            "fetch_events",
            lambda _c: (_ for _ in ()).throw(acled_auth.AcledResponseError(WAF_BODY)),
        )
        monkeypatch.setattr(acled_client, "OUT_PATH", tmp_path / "acled.csv")
        monkeypatch.delenv("RESOLVER_SKIP_ACLED", raising=False)

        acled_client.cli_main()
        out = capsys.readouterr().out

        assert "::error title=ACLED could not be read::" in out
        assert "Imunify360" in out
        assert "not an empty month" in out

    def test_an_html_response_is_treated_the_same_way(self, monkeypatch, tmp_path):
        """AcledHtmlResponse subclasses AcledResponseError, so it must too."""

        monkeypatch.setattr(
            acled_client,
            "fetch_events",
            lambda _c: (_ for _ in ()).throw(
                acled_auth.AcledHtmlResponse(
                    what="events fetch",
                    status=200,
                    url="https://acleddata.com/api/acled/read",
                    snippet="<html><title>Unauthorized</title>",
                )
            ),
        )
        monkeypatch.setattr(acled_client, "OUT_PATH", tmp_path / "acled.csv")
        monkeypatch.delenv("RESOLVER_SKIP_ACLED", raising=False)

        assert acled_client.cli_main() == 1

    def test_a_genuinely_empty_window_still_exits_zero(self, monkeypatch, tmp_path):
        """The other half of the contract: silence is an answer."""

        monkeypatch.setattr(acled_client, "fetch_events", lambda _c: ([], "u", {}))
        monkeypatch.setattr(acled_client, "OUT_PATH", tmp_path / "acled.csv")
        monkeypatch.delenv("RESOLVER_SKIP_ACLED", raising=False)

        assert acled_client.cli_main() == 0
        assert acled_client._UNREAD_REASON is None

    def test_a_deliberate_skip_still_exits_zero(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RESOLVER_SKIP_ACLED", "1")
        monkeypatch.setattr(acled_client, "OUT_PATH", tmp_path / "acled.csv")

        assert acled_client.cli_main() == 0
        assert acled_client._UNREAD_REASON is None


class TestTheFatalitiesCliDescribesTheFailure:
    def test_an_auth_failure_is_named_rather_than_raised(self, monkeypatch, capsys):
        """It died at ``ACLEDClient()``, which sits outside the try below it."""

        from resolver.cli import acled_to_duckdb

        def _blocked(*_a, **_k):
            raise acled_auth.AcledResponseError(WAF_BODY)

        monkeypatch.setattr(acled_to_duckdb, "ACLEDClient", _blocked)

        rc = acled_to_duckdb.run(
            [
                "--start", "2026-07-01",
                "--end", "2026-09-30",
                "--db", "duckdb:///:memory:",
                "--dry-run",
            ]
        )

        assert rc == 1
        out = capsys.readouterr().out
        assert "::error title=ACLED could not be read::" in out
        assert "Imunify360" in out


class TestOneBlockedSourceDoesNotCostTheWholeIngest:
    @pytest.fixture(scope="class")
    def workflow(self) -> str:
        return WORKFLOW.read_text(encoding="utf-8")

    def test_the_acled_steps_do_not_abort_the_job(self, workflow):
        for step_id in ("phase1_connectors", "phase1_acled_fatalities"):
            marker = f"id: {step_id}"
            assert marker in workflow, f"{step_id} is not declared in the workflow"
            block = workflow.split(marker)[0].rsplit("- name:", 1)[-1]
            assert "continue-on-error: true" in block, (
                f"{step_id} must not abort the job: on 2026-09-06 it took every "
                "other phase down with it and no canonical DB was uploaded"
            )

    def test_a_terminal_gate_still_turns_the_run_red(self, workflow):
        assert "Phase 1 gate: ACLED must have been readable" in workflow
        assert "steps.phase1_connectors.outcome" in workflow
        assert "steps.phase1_acled_fatalities.outcome" in workflow

    def test_the_gate_runs_after_the_canonical_upload(self, workflow):
        upload = workflow.index("name: pythia-resolver-db")
        gate = workflow.index("Phase 1 gate: ACLED must have been readable")
        assert gate > upload, (
            "the gate must fail the run AFTER the canonical DB is uploaded, so a "
            "red run is still a complete ingest minus ACLED"
        )

    def test_the_red_artifact_stays_discoverable(self, workflow):
        # A gate that goes red after the upload is only safe because
        # discovery is told to accept this workflow's failed runs.
        assert re.search(
            r"include-failed-runs-from:\s*\|\s*\n\s*Resolver Update", workflow
        ), "include-failed-runs-from: Resolver Update must stay set"

    def test_the_gate_ignores_a_scoped_run_that_skipped_acled(self, workflow):
        """`only_connector: ifrc_go_client` leaves both outcomes `skipped`."""

        gate = workflow.split("Phase 1 gate: ACLED must have been readable")[1]
        gate = gate.split("exit 1")[0]
        # The gate raises FAILED only on an outcome of exactly `failure`, so
        # `skipped` (a scoped run) and `success` both leave the job green.
        failure_tests = re.findall(r'if \[ "\$\{[A-Z]+\}" = "(\w+)" \]', gate)
        assert failure_tests, "the gate must test the step outcomes explicitly"
        assert set(failure_tests) == {"failure"}, (
            f"the gate must fire on `failure` alone, not {sorted(set(failure_tests))} "
            "— a skipped step is a scoped run, not an unreadable source"
        )
