# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Has every published chain reached the Drive archive? Run by the Cron Watchdog.

Every successful Publish run that finished more than ``GRACE_HOURS`` ago must
be named in the ``archive_report.json`` of a successful "Archive Run to
Drive" run, with a status that accounts for it (archived, already present,
or deliberately skipped as a test chain). A missing one is a ``::warning``
naming the Publish run; one older than ``BLOCKING_DAYS`` is an error and the
job goes red, because its artifacts expire within thirty days of that.

The newest archive report's storage check is read too: a Drive that is
nearly full is an error here, since the next archive may not fit.

Stdlib and ``gh`` only. Exit 1 on an error, 0 otherwise. An Actions API that
cannot be read is an error: the watchdog fails loud.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ci.drive_archive import ACCOUNTED  # noqa: E402

PUBLISH_FILE = "publish_latest_data.yml"
ARCHIVE_FILE = "archive_to_drive.yml"
GRACE_HOURS = 48
BLOCKING_DAYS = 60
LOOKBACK_DAYS = 90
TRUSTED_EVENTS = frozenset({"schedule", "workflow_dispatch", "workflow_run", "push"})


def _ts(raw: Any) -> dt.datetime | None:
    if not raw:
        return None
    try:
        t = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _ours(run: Mapping[str, Any], repo: str) -> bool:
    if str(run.get("event") or "") not in TRUSTED_EVENTS:
        return False
    head = run.get("head_repository") or {}
    if head and str(head.get("full_name") or "") != repo:
        return False
    return run.get("head_branch") in (None, "", "main")


class Gh:
    def api(self, path: str) -> Any:
        out = subprocess.run(["gh", "api", path], capture_output=True, text=True, check=True,
                             timeout=120).stdout
        return json.loads(out or "null")

    def download(self, repo: str, run_id: str, name: str, dest: Path) -> bool:
        proc = subprocess.run(["gh", "run", "download", str(run_id), "-n", name, "-D", str(dest),
                               "-R", repo], capture_output=True, text=True, timeout=300)
        return proc.returncode == 0


def _runs(gh: Any, repo: str, file: str, now: dt.datetime) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for page in range(1, 6):
        listing = gh.api(f"repos/{repo}/actions/workflows/{file}/runs?per_page=100&page={page}") or {}
        runs = listing.get("workflow_runs") or []
        old = False
        for r in runs:
            t = _ts(r.get("created_at"))
            if t is None or (now - t).days > LOOKBACK_DAYS + 5:
                old = True
                continue
            if _ours(r, repo):
                out.append(r)
        if old or len(runs) < 100:
            break
    return out


def archive_reports(gh: Any, repo: str, archive_runs: list[dict[str, Any]],
                    workdir: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(run, archive_report.json) for every completed archive run, newest first."""
    found = []
    for run in sorted(archive_runs, key=lambda r: str(r.get("created_at")), reverse=True):
        # A run whose one leg failed still reports every other leg's success,
        # so read any completed run; each entry carries its own status.
        if run.get("status") != "completed":
            continue
        rid = str(run.get("id"))
        dest = workdir / rid
        if not gh.download(repo, rid, f"chain-diagnostics-archive_to_drive-{rid}", dest):
            continue
        for path in dest.rglob("archive_report.json"):
            try:
                found.append((run, json.loads(path.read_text(encoding="utf-8"))))
            except Exception:  # noqa: BLE001
                pass
            break
    return found


def evaluate(publish_runs: list[dict[str, Any]],
             reports: list[tuple[dict[str, Any], dict[str, Any]]],
             now: dt.datetime) -> dict[str, Any]:
    """Pure: which Publish runs lack an archive, and is Drive nearly full."""
    accounted: dict[str, str] = {}
    for run, rep in reports:
        for e in rep.get("publish_runs") or []:
            if e.get("status") in ACCOUNTED:
                accounted.setdefault(str(e.get("publish_run_id")), f"{e.get('status')} by archive run {run.get('id')}")
    warnings, errors, ok = [], [], []
    for p in publish_runs:
        if p.get("status") != "completed" or p.get("conclusion") != "success":
            continue
        done = _ts(p.get("updated_at")) or _ts(p.get("created_at"))
        if done is None or (now - done).total_seconds() < GRACE_HOURS * 3600:
            continue
        pid = str(p.get("id"))
        if pid in accounted:
            ok.append(pid)
            continue
        age = (now - (_ts(p.get("created_at")) or done)).days
        msg = (f"Publish run {pid} ({p.get('created_at')}, {age} days old) has no Drive archive. "
               f"Dispatch 'Archive Run to Drive' with publish_run_id={pid}.")
        (errors if age > BLOCKING_DAYS else warnings).append(msg)
    storage = (reports[0][1].get("storage") if reports else None) or {}
    if storage.get("low"):
        errors.append(f"Google Drive is nearly full: {storage.get('free_bytes')} bytes free, "
                      f"{storage.get('used_fraction')} used (archive run {reports[0][0].get('id')}).")
    return {"ok": ok, "warnings": warnings, "errors": errors, "storage": storage,
            "reports_read": len(reports)}


def main(argv: list[str] | None = None, *, gh: Any = None,
         now: dt.datetime | None = None, write: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    args = parser.parse_args(argv)
    gh = gh or Gh()
    now = now or dt.datetime.now(dt.timezone.utc)
    try:
        publish = _runs(gh, args.repo, PUBLISH_FILE, now)
        archive = _runs(gh, args.repo, ARCHIVE_FILE, now)
        with tempfile.TemporaryDirectory() as tmp:
            reports = archive_reports(gh, args.repo, archive, Path(tmp))
    except Exception as exc:  # noqa: BLE001
        write(f"::error title=Drive archive check::the Actions API could not be read: {exc}")
        return 1
    verdict = evaluate(publish, reports, now)
    for msg in verdict["warnings"]:
        write(f"::warning title=Drive archive missing::{msg}")
    for msg in verdict["errors"]:
        write(f"::error title=Drive archive::{msg}")
    write(f"Drive archive: {len(verdict['ok'])} Publish run(s) archived, "
          f"{len(verdict['warnings'])} missing, {len(verdict['errors'])} error(s); "
          f"{verdict['reports_read']} archive report(s) read.")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("## Drive archive\n\n")
            for msg in verdict["errors"] + verdict["warnings"] or ["Every published chain is archived."]:
                fh.write(f"- {msg}\n")
    return 1 if verdict["errors"] else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
