# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The cycle's Actions logs, captured into the bundle.

A staged pipeline is four workflow runs and however many poller ticks came
between them, and the bundle was built from the last of those. So the
questions that matter after a bad run — when did the poller first come
back, what did the submit stage say when it created the batches, which
stage did the CrisisWatch refresh actually run in — were answerable only
from job logs, which expire, and only by somebody with repo access.

What is captured:

* every job in the CURRENT workflow run (the run building the bundle);
* the hs_submit / hs_rc_collect / hs_finalize runs of this pipeline,
  matched on the pipeline id the stage workflow puts in its run name;
* every poll_llm_batches run in the window from the first batch submission
  to now, which is the set that could have touched this pipeline.

One plain text file per run. Anything over ``MAX_LOG_BYTES`` is cut from
the middle, because the head carries the setup and the tail carries the
failure, and the middle is the repetitive part. A fetch failure writes a
stub naming the run and the error: the phase never fails over a log.
"""

from __future__ import annotations

import io
import json
import os
import re
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from scripts.debug_bundle.redaction import redact_text

MAX_LOG_BYTES = 5 * 1024 * 1024
_HTTP_TIMEOUT = 60.0
_API = "https://api.github.com"
# A cycle is four stages and a few dozen poller ticks; the cap is a guard
# against a pathological chain, not a design target.
MAX_POLLER_RUNS = 40
STAGE_WORKFLOW = "pythia_pipeline_stage.yml"
POLLER_WORKFLOW = "poll_llm_batches.yml"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(text: str) -> str:
    return _SAFE_NAME.sub("-", text).strip("-")[:80] or "run"


class GitHubApi:
    """The three calls this collector makes, behind one injectable seam."""

    def __init__(self, token: str, repo: str) -> None:
        self.token = token
        self.repo = repo

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        import requests  # noqa: PLC0415

        resp = requests.get(
            f"{_API}{path}", headers=self._headers(), params=params or {}, timeout=_HTTP_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()

    def get_logs_zip(self, run_id: str) -> bytes:
        import requests  # noqa: PLC0415

        resp = requests.get(
            f"{_API}/repos/{self.repo}/actions/runs/{run_id}/logs",
            headers=self._headers(),
            timeout=_HTTP_TIMEOUT,
            allow_redirects=True,
        )
        resp.raise_for_status()
        return resp.content


def _flatten_logs_zip(blob: bytes) -> str:
    """Actions serves logs as a zip of per-step text files. Flatten in order."""

    parts: list[str] = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in sorted(zf.namelist()):
            if name.endswith("/"):
                continue
            try:
                text = zf.read(name).decode("utf-8", errors="replace")
            except Exception as exc:  # noqa: BLE001
                text = f"<<could not read {name}: {exc}>>"
            parts.append(f"\n===== {name} =====\n{text}")
    return "".join(parts)


def truncate_middle(text: str, limit: int = MAX_LOG_BYTES) -> str:
    """Keep head and tail, say so in the middle.

    The note is inline rather than in a sidecar because whoever reads this
    file is reading it in a text viewer, and a truncation they cannot see
    is worse than one they can.
    """

    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= limit:
        return text
    keep = (limit - 400) // 2
    head = raw[:keep].decode("utf-8", errors="replace")
    tail = raw[-keep:].decode("utf-8", errors="replace")
    dropped = len(raw) - keep * 2
    note = (
        "\n\n"
        "================================================================\n"
        f"[bundle] TRUNCATED: {dropped:,} bytes removed from the middle of this\n"
        f"[bundle] log ({len(raw):,} bytes original, limit {limit:,}). Head and\n"
        "[bundle] tail are verbatim. Fetch the full log from the Actions run\n"
        "[bundle] while it is still retained.\n"
        "================================================================\n\n"
    )
    return head + note + tail


def _resolve_repo() -> str:
    return (os.getenv("GITHUB_REPOSITORY") or "").strip()


def _resolve_token() -> str:
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def _iso_to_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _discover_runs(
    api: GitHubApi,
    *,
    pipeline_id: str | None,
    since: datetime | None,
    problems: list[str],
) -> list[dict[str, Any]]:
    """The runs worth capturing, newest first, each tagged with its role."""

    runs: list[dict[str, Any]] = []
    current = (os.getenv("GITHUB_RUN_ID") or "").strip()
    if current:
        # The current run is still in progress, so its own job's log is not
        # yet in the archive; the earlier jobs of the run are, which is what
        # this entry is for. A partial or refused archive writes a stub.
        runs.append({"run_id": current, "role": "current", "workflow": "current-run"})

    if pipeline_id:
        try:
            payload = api.get_json(
                f"/repos/{api.repo}/actions/workflows/{STAGE_WORKFLOW}/runs",
                {"per_page": 100},
            )
            for run in payload.get("workflow_runs") or []:
                rid = str(run.get("id"))
                if rid == current:
                    continue
                # The stage workflow names its runs "<pipeline_id> — <stage>",
                # which is also what the poller's dispatch-once guard matches
                # on, so it is the reliable way to find a pipeline's stages.
                title = f"{run.get('name') or ''} {run.get('display_title') or ''}"
                if pipeline_id in title:
                    runs.append(
                        {
                            "run_id": rid,
                            "role": "pipeline_stage",
                            "workflow": "pythia_pipeline_stage",
                            "title": run.get("display_title") or run.get("name"),
                            "conclusion": run.get("conclusion"),
                            "created_at": run.get("created_at"),
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            problems.append(f"stage run discovery failed: {redact_text(str(exc))}")

    try:
        payload = api.get_json(
            f"/repos/{api.repo}/actions/workflows/{POLLER_WORKFLOW}/runs", {"per_page": 100}
        )
        n = 0
        for run in payload.get("workflow_runs") or []:
            created = _iso_to_dt(run.get("created_at"))
            # A poller tick before the first batch was submitted cannot have
            # touched this pipeline; one after it might have, and the tick
            # that did nothing is exactly the one worth reading.
            if since is not None and created is not None and created < since:
                continue
            runs.append(
                {
                    "run_id": str(run.get("id")),
                    "role": "poller",
                    "workflow": "poll_llm_batches",
                    "conclusion": run.get("conclusion"),
                    "created_at": run.get("created_at"),
                }
            )
            n += 1
            if n >= MAX_POLLER_RUNS:
                problems.append(
                    f"poller runs capped at {MAX_POLLER_RUNS}; older ticks not captured"
                )
                break
    except Exception as exc:  # noqa: BLE001
        problems.append(f"poller run discovery failed: {redact_text(str(exc))}")

    return runs


def collect(
    out_dir: Path,
    *,
    pipeline_id: str | None,
    earliest_batch_submitted_at: Any = None,
    api: GitHubApi | None = None,
    fetch_logs: Callable[[str], bytes] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Write ``out_dir/workflow_logs/*.txt``; return an index of what landed."""

    logs_dir = out_dir / "workflow_logs"
    index: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_id": pipeline_id,
        "runs": [],
        "problems": [],
    }
    if not enabled:
        index["problems"].append("workflow log capture disabled (PYTHIA_BUNDLE_WORKFLOW_LOGS=0)")
        _write_index(logs_dir, index)
        return index

    repo = _resolve_repo()
    token = _resolve_token()
    if api is None:
        if not repo or not token:
            index["problems"].append(
                "GITHUB_REPOSITORY / GITHUB_TOKEN not set — not running in Actions, "
                "so there are no workflow logs to fetch"
            )
            _write_index(logs_dir, index)
            return index
        api = GitHubApi(token, repo)

    since = _iso_to_dt(earliest_batch_submitted_at)
    if since is None and earliest_batch_submitted_at is not None:
        try:
            since = earliest_batch_submitted_at.replace(tzinfo=timezone.utc)
        except Exception:
            since = None
    if since is not None:
        # A little slack: the submit stage dispatches the poller right after
        # writing its batch state, and clocks are not identical.
        since = since - timedelta(minutes=15)

    runs = _discover_runs(api, pipeline_id=pipeline_id, since=since, problems=index["problems"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    fetcher = fetch_logs or api.get_logs_zip

    seen: set[str] = set()
    for run in runs:
        rid = str(run.get("run_id") or "")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        name = f"{_slug(str(run.get('workflow') or 'run'))}__{rid}.txt"
        path = logs_dir / name
        entry = dict(run)
        entry["file"] = f"workflow_logs/{name}"
        try:
            blob = fetcher(rid)
            text = _flatten_logs_zip(blob)
            original = len(text.encode("utf-8", errors="replace"))
            # Redact BEFORE truncating: a secret echoed into a log is the
            # one thing that must never reach the artifact, and it is as
            # likely to sit in the head as in the tail.
            text = truncate_middle(redact_text(text))
            path.write_text(text, encoding="utf-8")
            entry["ok"] = True
            entry["bytes"] = path.stat().st_size
            entry["truncated"] = original > MAX_LOG_BYTES
        except Exception as exc:  # noqa: BLE001
            message = redact_text(f"{type(exc).__name__}: {exc}")
            path.write_text(
                f"[bundle] Could not fetch logs for run {rid}\n"
                f"[bundle] role={run.get('role')} workflow={run.get('workflow')}\n"
                f"[bundle] error: {message}\n",
                encoding="utf-8",
            )
            entry["ok"] = False
            entry["error"] = message
        index["runs"].append(entry)

    _write_index(logs_dir, index)
    return index


def _write_index(logs_dir: Path, index: dict[str, Any]) -> None:
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "INDEX.json").write_text(
            json.dumps(index, indent=2, default=str), encoding="utf-8"
        )
    except Exception:  # pragma: no cover - defensive
        pass
