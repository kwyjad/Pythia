# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Can the SPEI-3 producer still commit, and if not, whose problem is it?

The SPEI-3 feed is written by a scheduled workflow that pushes
``resolver/data/spei3_country_means.csv`` straight to ``main`` with a
fine-grained ``SPEI3_COMMIT_TOKEN``. The default ``GITHUB_TOKEN`` cannot do
that — branch protection refuses a CI push with GH006 — and a pull request is
not an alternative, because one opened by ``GITHUB_TOKEN`` never triggers its
own checks and so can never merge.

**That token was issued with no expiry.** It cannot lapse on a schedule, so
there is no date to warn about and no scheduled death to nag about. What it
CAN do is be revoked, have its permissions narrowed, or lose access when the
account that owns it changes — and then the commit step is refused, the feed
stops extending, and the only downstream symptom is a drought gate that
quietly loses its only observation of the years before the HDX and NMME
ingests began. The feed-staleness check catches that eventually, but only
after the product's own lag plus a missed cycle; the watchdog catches the red
run after forty days. Neither is fast, and neither says which of the several
ways a push can fail actually happened.

So the guard is REFUSAL-BASED rather than date-based, and it has two halves.

The producer's half, ``classify_git_failure``. A push that fails is
classified from what git actually said, because the three cases want three
responses and they are not interchangeable: a credential refused is somebody
rotating a token, a transport failure is a retry, and anything else is a bug
to read. The producer then fails through a step whose NAME carries the class,
which is what makes the class readable from outside the run.

The register's half, ``read_producer_state``. A refusal cannot be committed —
the push is the thing being refused — so the committed status file cannot
carry it and the register has to look at the producer's own last run. The
step names above are the channel: one call for the newest run, one more for
its jobs when that run failed. Only the credential refusal reaches the
register, at ``degraded``. A transport failure is not the same problem and
must not cry wolf, and an absent secret is already a red run with its own
named error.

Pure decision core plus an injectable transport, so the classification and the
verdict are both tested without a network and without ``gh``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

#: The producer this module speaks about. Matched against a workflow's `name:`
#: exactly as `check_workflow_freshness.py` does.
SPEI3_WORKFLOW = "SPEI-3 Feed Refresh"

# -- classes ------------------------------------------------------------
#
# Four outcomes of an attempt to commit, and they are deliberately not a
# boolean. "The push did not land" is the least useful true statement
# available: rotating a credential, retrying a transport, configuring a
# secret and reading a traceback are four different actions.

CLASS_OK = "ok"
CLASS_NO_SECRET = "no_secret"
CLASS_REFUSED_AUTH = "refused_auth"
CLASS_REFUSED_NETWORK = "refused_network"
CLASS_REFUSED_OTHER = "refused_other"

#: The step names the workflow uses to carry each class out of the run. These
#: are a CONTRACT with ``.github/workflows/spei3_refresh.yml`` — the register
#: reads them off the jobs API, so a renamed step silently blinds the guard.
#: A test holds the workflow to them.
STEP_BY_CLASS: dict[str, str] = {
    CLASS_NO_SECRET: "Fail: SPEI3_COMMIT_TOKEN is not configured",
    CLASS_REFUSED_AUTH: "Fail: SPEI3_COMMIT_TOKEN was refused",
    CLASS_REFUSED_NETWORK: "Fail: the push could not reach GitHub",
    CLASS_REFUSED_OTHER: "Fail: the commit failed for an unread reason",
}

CLASS_BY_STEP: dict[str, str] = {step: cls for cls, step in STEP_BY_CLASS.items()}

#: The commit step's own name. It is written to exit 0 whatever happens, so a
#: failure HERE is a bug in the step rather than one of the four classes — but
#: it still has to read as "the commit did not happen" rather than as some
#: unrelated step going red, or the verdict's detail says the feed is probably
#: fine when it plainly is not.
STEP_COMMIT = "Commit the feed"
CLASS_BY_STEP[STEP_COMMIT] = CLASS_REFUSED_OTHER

# -- what git says when a credential is refused -------------------------
#
# Checked BEFORE the transport patterns, and the order is load-bearing: git
# prints its `remote:` lines and THEN a transport error, so a permission
# refusal routinely ends with "the remote end hung up unexpectedly". Reading
# the last line first would classify every refusal as a network blip and
# retry it forever.
_AUTH_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # Branch protection declining the push. This is the refusal the
        # default GITHUB_TOKEN gets, and the one a narrowed token gets back.
        r"\bGH006\b",
        r"protected branch (?:update failed|hook declined)",
        r"pre-receive hook declined",
        r"changes must be made through a pull request",
        # The token no longer carries write access.
        r"remote: permission to .* denied",
        r"\bpermission denied\b",
        # Revoked, expired, or the wrong account.
        r"invalid username or (?:password|token)",
        r"authentication failed",
        r"could not read Username",
        r"\bbad credentials\b",
        # GitHub answers 404 for a repository a token cannot see, so
        # "not found" from a repo that plainly exists is an access refusal.
        r"remote: repository not found",
        # A fine-grained token without the workflow scope.
        r"refusing to allow a(?:n OAuth App|.*Personal Access Token)",
        # Raw statuses from the smart-HTTP transport.
        r"requested URL returned error: 40[13]",
        r"\bHTTP (?:401|403)\b",
    )
)

# -- what git says when it could not reach GitHub -----------------------
#
# A retry, not a rotation. The repository's own guidance is up to four
# attempts with exponential backoff, and these are the shapes that earn one.
_NETWORK_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"send-pack: unexpected disconnect",
        r"the remote end hung up unexpectedly",
        r"RPC failed",
        r"early EOF",
        r"connection reset by peer",
        r"could not resolve host",
        r"(?:connection|operation) timed out",
        r"failed to connect to",
        r"unable to access .*: (?:SSL|TLS|GnuTLS|OpenSSL)",
        r"\bunexpected EOF\b",
        r"\btemporary failure in name resolution\b",
    )
)


def classify_git_failure(text: str) -> str:
    """Which kind of refusal is this, read from git's own words?

    Auth first, deliberately — see the note above ``_AUTH_PATTERNS``. An
    unrecognised failure is ``refused_other`` rather than either of the
    actionable classes: guessing "network" would retry a bug, and guessing
    "auth" would send somebody to rotate a working credential.
    """

    body = str(text or "")
    if not body.strip():
        return CLASS_REFUSED_OTHER
    for pattern in _AUTH_PATTERNS:
        if pattern.search(body):
            return CLASS_REFUSED_AUTH
    for pattern in _NETWORK_PATTERNS:
        if pattern.search(body):
            return CLASS_REFUSED_NETWORK
    return CLASS_REFUSED_OTHER


def is_retryable(cls: str) -> bool:
    """Only a transport failure earns another attempt.

    Retrying a refused credential spends four more requests to be told the
    same thing — the lesson GDACS taught this repository at 552 refusals in
    one run.
    """

    return cls == CLASS_REFUSED_NETWORK


# -- the register's half ------------------------------------------------


@dataclass
class ProducerCommitState:
    """The producer's own last word on whether it could commit."""

    workflow: str = ""
    state: str = "unavailable"
    run_id: str = ""
    run_url: str = ""
    conclusion: str = ""
    started_at: str = ""
    failed_steps: list[str] = field(default_factory=list)
    detail: str = ""

    @property
    def is_refused(self) -> bool:
        """Is the credential itself the problem?

        Deliberately narrow. A transport failure and an absent secret are
        both red runs with their own messages, and neither is a reason to
        tell somebody their token has stopped working.
        """

        return self.state == CLASS_REFUSED_AUTH

    def as_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "state": self.state,
            "run_id": self.run_id,
            "run_url": self.run_url,
            "conclusion": self.conclusion,
            "started_at": self.started_at,
            "failed_steps": list(self.failed_steps),
            "detail": self.detail,
        }


#: ``(path, query) -> parsed JSON``, or None when the API cannot be reached.
Fetch = Callable[[str], Any]


def _gh_fetch(path: str) -> Any:
    """Read one GitHub API path with ``gh``.

    ``gh`` and not ``requests``: same constraint the other CI-side readers in
    this repository work under, and it means the step needs no dependency
    beyond what the runner already has. Any failure returns None, which the
    decision core reports as ``unavailable`` — a diagnostic that raises is a
    diagnostic that takes the run with it.
    """

    try:
        proc = subprocess.run(
            ["gh", "api", path],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout or "null")
    except ValueError:
        return None


def decide_producer_state(
    runs: Sequence[Mapping[str, Any]] | None,
    jobs_for_run: Callable[[str], Sequence[Mapping[str, Any]] | None],
    workflow: str = SPEI3_WORKFLOW,
) -> ProducerCommitState:
    """Pure verdict over the producer's newest run and that run's steps.

    The newest run is the one that matters, not the newest failure: a
    credential refused in August and rotated in September is not a live
    fault, and reporting it would teach the reader to skip the register.
    """

    state = ProducerCommitState(workflow=workflow)
    if runs is None:
        state.detail = "the Actions API could not be read, so the producer's last run is unknown"
        return state
    if not runs:
        state.state = "unavailable"
        state.detail = f"no runs of {workflow!r} were found"
        return state

    newest = runs[0]
    state.run_id = str(newest.get("id") or "")
    state.run_url = str(newest.get("html_url") or "")
    state.conclusion = str(newest.get("conclusion") or "")
    state.started_at = str(newest.get("run_started_at") or newest.get("created_at") or "")

    if state.conclusion == "success":
        state.state = CLASS_OK
        state.detail = f"the newest run ({state.run_id}) committed successfully"
        return state
    if not state.conclusion:
        state.state = "unavailable"
        state.detail = f"the newest run ({state.run_id}) has not finished"
        return state

    jobs = jobs_for_run(state.run_id)
    if jobs is None:
        state.detail = (
            f"the newest run ({state.run_id}) concluded {state.conclusion} but its "
            "steps could not be read, so the reason is unknown"
        )
        return state

    state.failed_steps = [
        str(step.get("name") or "")
        for job in jobs
        for step in (job.get("steps") or [])
        if str(step.get("conclusion") or "") == "failure"
    ]
    for step in state.failed_steps:
        cls = CLASS_BY_STEP.get(step)
        if cls:
            state.state = cls
            state.detail = (
                f"the newest run ({state.run_id}) failed at {step!r}"
            )
            return state

    state.state = CLASS_REFUSED_OTHER
    state.detail = (
        f"the newest run ({state.run_id}) concluded {state.conclusion}, failing at "
        + (", ".join(repr(s) for s in state.failed_steps[:4]) or "no named step")
        + " — none of which is a commit step, so the feed is not necessarily blocked"
    )
    return state


def read_producer_state(
    workflow: str = SPEI3_WORKFLOW,
    repo: str | None = None,
    fetch: Fetch | None = None,
) -> ProducerCommitState:
    """``decide_producer_state`` over the live Actions API.

    Needs ``GITHUB_REPOSITORY`` and a ``gh`` with ``actions: read`` — which
    ``resolver_update.yml`` already grants. Without either it reports
    ``unavailable`` and the caller SKIPs, because a guard that fails red when
    it cannot see anything is a guard somebody switches off.
    """

    get = fetch or _gh_fetch
    slug = repo or os.getenv("GITHUB_REPOSITORY", "")
    if not slug:
        state = ProducerCommitState(workflow=workflow)
        state.detail = "GITHUB_REPOSITORY is unset, so the producer's runs cannot be read"
        return state

    payload = get(f"/repos/{slug}/actions/runs?per_page=20")
    runs: list[Mapping[str, Any]] | None
    if isinstance(payload, Mapping):
        runs = [
            run
            for run in (payload.get("workflow_runs") or [])
            if isinstance(run, Mapping) and str(run.get("name") or "") == workflow
        ]
    else:
        runs = None

    def jobs_for_run(run_id: str) -> Sequence[Mapping[str, Any]] | None:
        body = get(f"/repos/{slug}/actions/runs/{run_id}/jobs?per_page=50")
        if not isinstance(body, Mapping):
            return None
        return [job for job in (body.get("jobs") or []) if isinstance(job, Mapping)]

    return decide_producer_state(runs, jobs_for_run, workflow=workflow)


# -- the producer's entry point -----------------------------------------


def _main(argv: Iterable[str] | None = None) -> int:
    """``python -m resolver.diagnostics.producer_commit --classify``.

    Reads git's output on stdin and prints the class on stdout, so the
    workflow's shell needs no regular expressions of its own and the patterns
    have exactly one home.
    """

    args = list(argv if argv is not None else sys.argv[1:])
    if "--classify" not in args:
        print("usage: python -m resolver.diagnostics.producer_commit --classify", file=sys.stderr)
        return 2
    print(classify_git_failure(sys.stdin.read()))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI
    raise SystemExit(_main())
