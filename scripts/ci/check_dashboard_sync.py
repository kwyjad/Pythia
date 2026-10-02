# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Is the live dashboard API serving the database the release publishes?

Written after the 1 October 2026 cycle: every workflow ran green, the
``pythia-data-latest`` release carried the new DB, and the dashboard still
showed September. Nothing in CI could see that, because the only end-to-end
step (the publish workflow's ``force_sync`` poke) needs two settings that
were never configured, and it skipped itself with a warning nobody read.
The same lag had recurred month after month.

This check needs NO secret. It compares two public facts:

* the release's ``manifest.json`` ``db_sha256`` — what was published;
* the API's ``GET /v1/version`` ``sync_status.served_key`` (falling back to
  ``downloaded_key``, then ``db_sha256``) — what the API actually answers from.

The GET is also what wakes an idle API and starts its sync, so the check
polls until the two agree or a deadline passes, and reports the API's own
``sync_status.last_error`` when they do not — "insufficient disk", "sha256
mismatch" and "manifest unreachable" want different repairs.

Modes
-----
* ``--mode publish``: after a release upload. Always exits 0 (everything
  after a release upload is diagnostics and non-fatal), but annotates a
  ``::warning`` and writes the step summary.
* ``--mode watchdog``: in the daily Cron Watchdog. Exits 1 when the API is
  behind a release older than ``--grace-min``, because a watchdog that fails
  open is decorative.

stdlib only, so the watchdog job needs no setup-python and no pip install.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

DEFAULT_API_BASE = "https://pythia-vdu3.onrender.com"
DEFAULT_REPO = "kwyjad/Pythia"
DEFAULT_TAG = "pythia-data-latest"


def _get_json(url: str, timeout: float, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https hosts
        body = resp.read()
    data = json.loads(body.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    return data


def release_manifest(repo: str, tag: str, timeout: float = 30) -> Dict[str, Any]:
    url = f"https://github.com/{repo}/releases/download/{tag}/manifest.json?t={int(time.time())}"
    headers = {"Cache-Control": "no-cache"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return _get_json(url, timeout, headers)


def api_version(api_base: str, timeout: float = 90) -> Dict[str, Any]:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return _get_json(f"{base}/v1/version?t={int(time.time())}", timeout)


def served_key(version: Dict[str, Any]) -> Optional[str]:
    """The DB the API answers from: served, else downloaded, else its manifest's sha.

    Served comes first because downloaded is not served — the October 2026
    API had downloaded the new file and still answered from the old one.
    """
    sync = version.get("sync_status") or {}
    key = sync.get("served_key") or sync.get("downloaded_key") or version.get("db_sha256")
    return str(key) if key else None


def compare(manifest: Dict[str, Any], version: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """Return (in_sync, one-line explanation)."""
    published = manifest.get("db_sha256")
    pub_run = manifest.get("latest_hs_run_id") or "?"
    if not published:
        return False, "the release manifest carries no db_sha256, so nothing can be compared"
    if version is None:
        return False, "the API did not answer /v1/version"
    served = served_key(version)
    sync = version.get("sync_status") or {}
    if served == published:
        return True, (
            f"API serves the published DB ({str(published)[:12]}..., HS run {pub_run})"
        )
    detail = (
        f"API serves {str(served)[:12] if served else 'nothing'}... but the release "
        f"publishes {str(published)[:12]}... (HS run {pub_run})"
    )
    err = sync.get("last_error")
    if err:
        detail += f"; the API's last sync error: {err}"
    elif sync.get("last_attempt_at"):
        detail += f"; last sync attempt {sync.get('last_attempt_at')}, no error recorded yet"
    return False, detail


def _age_minutes(created_utc: Optional[str]) -> Optional[float]:
    if not created_utc:
        return None
    try:
        dt = datetime.strptime(str(created_utc)[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0


def _summary(lines: list[str]) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n\n")
    except OSError:
        pass


def run(args: argparse.Namespace, sleep=time.sleep, now=time.monotonic) -> int:
    api_base = args.api_base or os.environ.get("PYTHIA_API_BASE_URL") or DEFAULT_API_BASE
    try:
        manifest = release_manifest(args.repo, args.tag)
    except Exception as exc:  # noqa: BLE001
        msg = f"Could not read the release manifest: {exc}"
        print(f"::warning::{msg}")
        _summary(["#### Dashboard sync check", "", msg])
        return 1 if args.mode == "watchdog" else 0

    deadline = now() + args.wait_min * 60
    version: Optional[Dict[str, Any]] = None
    in_sync, why = False, ""
    attempt = 0
    while True:
        attempt += 1
        try:
            version = api_version(api_base)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            version = None
            print(f"attempt {attempt}: {api_base} /v1/version failed: {exc}")
        in_sync, why = compare(manifest, version)
        print(f"attempt {attempt}: {why}")
        if in_sync or now() >= deadline:
            break
        sleep(args.poll_sec)

    if in_sync:
        _summary(["#### Dashboard sync check: OK", "", why])
        return 0

    age = _age_minutes(manifest.get("created_utc"))
    msg = f"Dashboard is behind the release: {why}. API: {api_base}"
    if args.mode == "watchdog" and age is not None and age < args.grace_min:
        print(f"::notice::{msg} (release is {age:.0f} min old, inside the {args.grace_min} min grace)")
        _summary(["#### Dashboard sync check: still catching up", "", msg])
        return 0
    level = "error" if args.mode == "watchdog" else "warning"
    print(f"::{level}::{msg}")
    _summary([
        "#### Dashboard sync check: BEHIND",
        "",
        msg,
        "",
        "Recovery: open `/v1/health` on the API for the sync error; if it names "
        "disk space, enlarge the Render disk (it must hold two copies of the DB); "
        "otherwise restart the API service, which syncs on boot.",
    ])
    return 1 if args.mode == "watchdog" else 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=("publish", "watchdog"), default="publish")
    p.add_argument("--api-base", default=None, help=f"API base URL (default env PYTHIA_API_BASE_URL, else {DEFAULT_API_BASE})")
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY") or DEFAULT_REPO)
    p.add_argument("--tag", default=DEFAULT_TAG)
    p.add_argument("--wait-min", type=float, default=20.0)
    p.add_argument("--poll-sec", type=float, default=30.0)
    p.add_argument("--grace-min", type=float, default=90.0)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
