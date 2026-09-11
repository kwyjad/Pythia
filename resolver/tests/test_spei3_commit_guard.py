# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The refusal-based guard on the SPEI-3 producer's commit credential.

SPEI3_COMMIT_TOKEN was issued with no expiry, so the register entry that
warned about its expiry was warning about a date that does not exist and
reported overdue on every run. A token with no expiry cannot lapse on a
schedule, but it can be revoked, narrowed, or lose access — and then the
producer stops committing and the drought feed silently stops extending.

Three properties these tests hold:

* a refusal is classified from git's OWN words, auth before transport,
  because git prints its ``remote:`` lines and then a transport error, so a
  permission refusal routinely ends "the remote end hung up unexpectedly";
* only the credential class reaches the register, because a transport failure
  is a re-run rather than a rotation and crying wolf over one is how a reader
  learns to skip the report;
* the step NAMES are a contract with the workflow, since they are the only
  channel a refused push has — it cannot commit the record of its own refusal.
"""

from __future__ import annotations

import pathlib

import pytest

yaml = pytest.importorskip("yaml")

from resolver.diagnostics import producer_commit as mod  # noqa: E402

WORKFLOW = pathlib.Path(__file__).resolve().parents[2] / ".github" / "workflows" / "spei3_refresh.yml"


class TestClassifyingARefusal:
    """Three answers, because three remedies."""

    @pytest.mark.parametrize("body", [
        "remote: error: GH006: Protected branch update failed for refs/heads/main.",
        "remote: error: Changes must be made through a pull request.",
        "remote: Permission to kwyjad/Pythia.git denied to github-actions[bot].",
        "fatal: Authentication failed for 'https://github.com/kwyjad/Pythia/'",
        "remote: Repository not found.\nfatal: repository not found",
        "fatal: unable to access '...': The requested URL returned error: 403",
        "remote: Invalid username or token. Password authentication is not supported.",
        "! [remote rejected] HEAD -> main (pre-receive hook declined)",
    ])
    def test_a_credential_refusal_is_recognised(self, body):
        assert mod.classify_git_failure(body) == mod.CLASS_REFUSED_AUTH

    @pytest.mark.parametrize("body", [
        "send-pack: unexpected disconnect while reading sideband packet\n"
        "fatal: the remote end hung up unexpectedly",
        "error: RPC failed; curl 92 HTTP/2 stream 5 was not closed cleanly",
        "fatal: unable to access '...': Could not resolve host: github.com",
        "ssh: connect to host github.com port 22: Connection timed out",
        "fatal: early EOF",
    ])
    def test_a_transport_failure_is_recognised(self, body):
        assert mod.classify_git_failure(body) == mod.CLASS_REFUSED_NETWORK

    def test_a_permission_refusal_that_ends_in_a_disconnect_is_still_auth(self):
        """The ordering is the whole point.

        Git prints the remote's refusal and THEN its own transport error, so
        almost every real permission refusal carries a disconnect at the end.
        Reading the last line first would classify every refusal as a blip and
        retry it for ever — which is how GDACS came to spend 552 refused
        requests in one run.
        """

        body = (
            "remote: Permission to kwyjad/Pythia.git denied to github-actions[bot].\n"
            "fatal: unable to access ...: The requested URL returned error: 403\n"
            "send-pack: unexpected disconnect while reading sideband packet\n"
            "fatal: the remote end hung up unexpectedly"
        )
        assert mod.classify_git_failure(body) == mod.CLASS_REFUSED_AUTH

    def test_an_unread_failure_is_neither_of_the_actionable_classes(self):
        """Guessing "network" retries a bug; guessing "auth" sends somebody
        to rotate a working credential. Neither is better than saying so."""

        assert mod.classify_git_failure("error: pathspec 'x' did not match") == (
            mod.CLASS_REFUSED_OTHER
        )
        assert mod.classify_git_failure("") == mod.CLASS_REFUSED_OTHER

    def test_only_a_transport_failure_earns_another_attempt(self):
        assert mod.is_retryable(mod.CLASS_REFUSED_NETWORK) is True
        assert mod.is_retryable(mod.CLASS_REFUSED_AUTH) is False
        assert mod.is_retryable(mod.CLASS_REFUSED_OTHER) is False
        assert mod.is_retryable(mod.CLASS_NO_SECRET) is False

    def test_the_cli_prints_the_class_for_the_workflow(self, monkeypatch, capsys):
        """The patterns have one home; the workflow's shell owns no regexes."""

        import io
        monkeypatch.setattr("sys.stdin", io.StringIO("remote: error: GH006: nope"))
        assert mod._main(["--classify"]) == 0
        assert capsys.readouterr().out.strip() == mod.CLASS_REFUSED_AUTH


def _run(run_id="9", conclusion="failure", **extra):
    body = {"id": run_id, "conclusion": conclusion,
            "html_url": f"https://example/{run_id}", "run_started_at": "2026-09-11T05:40:00Z"}
    body.update(extra)
    return body


def _jobs(*step_names):
    return [{"steps": [{"name": n, "conclusion": "failure"} for n in step_names]}]


class TestTheRegistersVerdict:
    """What the register concludes from the producer's own last run."""

    def test_a_successful_newest_run_is_ok(self):
        state = mod.decide_producer_state([_run(conclusion="success")], lambda _: None)
        assert state.state == mod.CLASS_OK
        assert state.is_refused is False

    def test_a_refused_credential_is_reported(self):
        state = mod.decide_producer_state(
            [_run()], lambda _: _jobs(mod.STEP_BY_CLASS[mod.CLASS_REFUSED_AUTH])
        )
        assert state.state == mod.CLASS_REFUSED_AUTH
        assert state.is_refused is True
        assert state.run_url == "https://example/9"

    @pytest.mark.parametrize("cls", [
        mod.CLASS_REFUSED_NETWORK, mod.CLASS_NO_SECRET, mod.CLASS_REFUSED_OTHER,
    ])
    def test_the_other_three_classes_do_not_cry_wolf(self, cls):
        """Each is a red run with its own named error. Reporting a transport
        blip as a dead credential would send somebody to rotate a working
        token, and reporting an absent secret twice says nothing new."""

        state = mod.decide_producer_state(
            [_run()], lambda _: _jobs(mod.STEP_BY_CLASS[cls])
        )
        assert state.state == cls
        assert state.is_refused is False

    def test_the_newest_run_decides_not_the_newest_failure(self):
        """A credential refused in August and rotated in September is not a
        live fault, and reporting it teaches the reader to skip the register."""

        runs = [_run(run_id="new", conclusion="success"), _run(run_id="old")]
        state = mod.decide_producer_state(runs, lambda _: _jobs(
            mod.STEP_BY_CLASS[mod.CLASS_REFUSED_AUTH]))
        assert state.state == mod.CLASS_OK
        assert state.run_id == "new"

    def test_a_failure_in_some_other_step_is_not_a_commit_problem(self):
        state = mod.decide_producer_state([_run()], lambda _: _jobs("Gate the candidate"))
        assert state.state == mod.CLASS_REFUSED_OTHER
        assert state.is_refused is False
        assert "Gate the candidate" in state.detail

    def test_the_commit_step_failing_itself_reads_as_a_failed_commit(self):
        """It is written to exit 0 whatever happens, so a failure there is a
        bug in the step — but it still has to read as "the feed was not
        committed" rather than as an unrelated step going red."""

        state = mod.decide_producer_state([_run()], lambda _: _jobs(mod.STEP_COMMIT))
        assert state.state == mod.CLASS_REFUSED_OTHER

    def test_an_unreadable_api_is_unavailable_and_never_a_fault(self):
        """A guard that goes red because it saw nothing is a guard somebody
        switches off."""

        assert mod.decide_producer_state(None, lambda _: None).state == "unavailable"
        assert mod.decide_producer_state([], lambda _: None).state == "unavailable"
        unfinished = mod.decide_producer_state([_run(conclusion="")], lambda _: None)
        assert unfinished.state == "unavailable"
        blind = mod.decide_producer_state([_run()], lambda _: None)
        assert blind.state == "unavailable"
        assert "could not be read" in blind.detail

    def test_no_repository_slug_is_unavailable_rather_than_an_exception(self, monkeypatch):
        monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
        state = mod.read_producer_state(fetch=lambda _: None)
        assert state.state == "unavailable"
        assert "GITHUB_REPOSITORY" in state.detail

    def test_the_live_reader_filters_to_this_workflow(self):
        """The runs endpoint returns every workflow's runs, and the newest of
        those is usually somebody else's."""

        calls = []

        def fetch(path):
            calls.append(path)
            if "/jobs" in path:
                return {"jobs": _jobs(mod.STEP_BY_CLASS[mod.CLASS_REFUSED_AUTH])}
            return {"workflow_runs": [
                {"id": "1", "name": "Resolver Update", "conclusion": "success"},
                {"id": "2", "name": mod.SPEI3_WORKFLOW, "conclusion": "failure",
                 "html_url": "https://example/2"},
            ]}

        state = mod.read_producer_state(repo="kwyjad/Pythia", fetch=fetch)
        assert state.run_id == "2"
        assert state.state == mod.CLASS_REFUSED_AUTH
        assert any("/jobs" in c for c in calls)


class TestTheWorkflowHonoursTheContract:
    """The step names are the only channel a refused push has."""

    def _steps(self):
        doc = yaml.safe_load(WORKFLOW.read_text("utf-8"))
        return doc["jobs"]["refresh"]["steps"]

    def test_every_class_has_a_step_with_exactly_its_name(self):
        """A renamed step here would blind the guard silently rather than
        break it, which is why this is a test and not a comment."""

        names = {str(step.get("name") or "") for step in self._steps()}
        for cls, step_name in mod.STEP_BY_CLASS.items():
            assert step_name in names, f"{cls} has no step named {step_name!r}"
        assert mod.STEP_COMMIT in names

    def test_each_fail_step_fires_on_its_own_class(self):
        by_name = {str(s.get("name") or ""): s for s in self._steps()}
        for cls, step_name in mod.STEP_BY_CLASS.items():
            cond = str(by_name[step_name].get("if") or "")
            assert f"steps.commit.outputs.class == '{cls}'" in cond, step_name
            # always(), or the first failure above takes the message with it.
            assert "always()" in cond, step_name

    def test_the_commit_step_classifies_rather_than_dying(self):
        """If it exited non-zero on a refusal the fail steps below could not
        run, and the class would never leave the run."""

        commit = next(s for s in self._steps() if s.get("id") == "commit")
        body = str(commit["run"])
        assert "producer_commit --classify" in body
        assert 'echo "class=' in body
        # An empty class fires no step below, so the run would go green having
        # pushed nothing — the one outcome this arrangement forbids.
        assert 'CLASS="refused_other"' in body

    def test_only_a_transport_failure_is_retried_in_the_workflow(self):
        commit = next(s for s in self._steps() if s.get("id") == "commit")
        body = str(commit["run"])
        assert '"${CLASS}" != "refused_network"' in body
        assert "sleep $(( 2 ** attempt ))" in body


class TestTheBundleCheck:
    """The check that carries the refusal into the run issue register."""

    def _builder(self, tmp_path, name):
        import scripts.build_resolver_debug_bundle as bundle

        return bundle.BundleBuilder(
            out_path=tmp_path / f"{name}.zip", db_path=None,
            diagnostics_dir=tmp_path / "diag", run_log_dir=None,
            staging=tmp_path / name, max_bytes=bundle.DEFAULT_MAX_BYTES,
            environ={},
        )

    def _run_check(self, tmp_path, monkeypatch, name, state):
        import scripts.build_resolver_debug_bundle as bundle

        monkeypatch.setattr(mod, "read_producer_state", lambda *a, **k: state)
        builder = self._builder(tmp_path, name)
        builder._check_spei3_producer_can_commit()
        assert isinstance(builder, bundle.BundleBuilder)
        return builder

    def test_the_check_is_registered_so_it_actually_runs(self):
        """A check nothing calls is a check that passes for ever."""

        import scripts.build_resolver_debug_bundle as bundle

        src = pathlib.Path(bundle.__file__).read_text("utf-8")
        assert "self._check_spei3_producer_can_commit," in src

    def test_a_refusal_fails_the_check_and_declares_the_issue(self, tmp_path, monkeypatch):
        state = mod.ProducerCommitState(
            workflow=mod.SPEI3_WORKFLOW, state=mod.CLASS_REFUSED_AUTH,
            run_id="42", run_url="https://example/42",
            detail="the newest run (42) failed at 'Fail: SPEI3_COMMIT_TOKEN was refused'",
        )
        builder = self._run_check(tmp_path, monkeypatch, "refused", state)
        check = builder.checks[-1]
        assert check["verdict"] == "FAIL", check
        ids = [issue["id"] for issue in check.get("issues") or []]
        assert ids == ["spei3_commit_refused"], ids
        issue = (check["issues"])[0]
        assert issue["severity"] == "degraded"
        assert "SPEI3_COMMIT_TOKEN" in issue["title"]
        assert issue["recovers_on_rerun"] is False

    @pytest.mark.parametrize("state_name", [
        mod.CLASS_OK, mod.CLASS_REFUSED_NETWORK,
        mod.CLASS_NO_SECRET, mod.CLASS_REFUSED_OTHER,
    ])
    def test_the_other_classes_pass_and_still_say_what_they_saw(
        self, tmp_path, monkeypatch, state_name
    ):
        """A PASS a reader cannot interpret is a PASS they stop trusting, so
        the detail names the class even where nothing is reported."""

        state = mod.ProducerCommitState(
            workflow=mod.SPEI3_WORKFLOW, state=state_name, run_id="7",
            detail=f"the newest run (7) reported {state_name}",
        )
        builder = self._run_check(tmp_path, monkeypatch, f"p_{state_name}", state)
        check = builder.checks[-1]
        assert check["verdict"] == "PASS", check
        assert not (check.get("issues") or [])
        assert state_name in check["left"] or state_name in check["detail"]

    def test_an_unreadable_api_skips_rather_than_failing(self, tmp_path, monkeypatch):
        state = mod.ProducerCommitState(
            workflow=mod.SPEI3_WORKFLOW, state="unavailable",
            detail="the Actions API could not be read",
        )
        builder = self._run_check(tmp_path, monkeypatch, "skip", state)
        assert builder.checks[-1]["verdict"] == "SKIP"

    def test_a_reader_that_raises_costs_the_check_and_not_the_phase(
        self, tmp_path, monkeypatch
    ):
        """A collector added to the bundle must never fail the phase."""

        def explode(*_a, **_k):
            raise RuntimeError("gh went missing")

        monkeypatch.setattr(mod, "read_producer_state", explode)
        builder = self._builder(tmp_path, "raise")
        builder._check_spei3_producer_can_commit()
        check = builder.checks[-1]
        assert check["verdict"] == "SKIP"
        assert "gh went missing" in check["detail"]
