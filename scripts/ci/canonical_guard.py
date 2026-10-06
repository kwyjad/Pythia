# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Two checks around the canonical ``pythia-resolver-db`` artifact.

``trigger-did-work`` — a chained workflow (Compute Resolutions, Compute SPD
Scores, Calibration, Publish) starts on ``workflow_run`` with conclusion
``success``. A run whose gate skipped it for an in-flight pipeline also
concludes ``success`` and uploads nothing, so on 5 October 2026 a skipped
Resolver Update started the whole chain on whatever DB discovery found. The
chain now asks whether its trigger uploaded a canonical DB in that run, and
exits early with a notice when it did not.

``upload-guard`` — every canonical producer downloads a DB, works on it and
uploads it as the newest canonical artifact. On 5 October the chain started
by a gated Resolver Update carried a DB from before an NMME refetch; its
Calibration uploaded at 14:36 UTC over the refetch's chain and Publish
released that DB without the 19,055 new probability rows. A later chain put
them back at 14:56, by luck of ordering.
The ``pythia-resolver-db`` concurrency group runs these workflows one at a
time and promises nothing about what each one downloaded. So before upload a
producer checks that no canonical artifact newer than the one it downloaded
was uploaded by another run meanwhile, and stops RED when one was.

``record`` — the staged pipeline downloads the canonical DB at its FIRST
stage and uploads it at its last, days later, from a staged copy. The first
stage records which canonical run it forked from in the DB itself
(``canonical_lineage``), so the last stage's ``upload-guard --db`` can ask
the same question every other producer asks.

Both read the Actions API through ``gh`` with retries. When the API cannot
answer after the retries, both WARN and let the run continue: a blip must
not discard a day's ingest, and the cost of being wrong is the narrow race
these checks exist for.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

ARTIFACT = "pythia-resolver-db"

GhFn = Callable[[list[str]], Any]


def _gh(args: list[str]) -> Any:
    out = subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout
    return json.loads(out or "null")


def with_retries(fn: Callable[[], Any], *, attempts: int = 4, backoff: float = 15.0,
                 sleep: Callable[[float], None] = time.sleep) -> Any:
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if i + 1 < attempts:
                sleep(backoff * (2 ** i))
    raise RuntimeError(f"gh failed after {attempts} attempt(s): {last}")


def _ts(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        t = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# --- trigger-did-work --------------------------------------------------------


def run_uploaded_canonical(artifacts: Iterable[dict]) -> bool:
    """True when a run's artifact list carries an unexpired canonical DB."""
    return any(a.get("name") == ARTIFACT and not a.get("expired") for a in artifacts or [])


# --- upload-guard -------------------------------------------------------------


def newer_canonical(
    artifacts: Iterable[dict], *, downloaded_run_id: str, this_run_id: str,
    repo_id: Any,
) -> tuple[str, dict | None]:
    """``(verdict, newer)``: ``ok`` | ``newer_exists`` | ``unknown``.

    ``newer`` is the newest canonical artifact uploaded by a run other than
    this one and the one we downloaded from, AFTER the artifact we downloaded.
    ``unknown`` when the downloaded artifact is not in the listing."""
    ours = [
        a for a in artifacts or []
        if a.get("name") == ARTIFACT and not a.get("expired")
        and str((a.get("workflow_run") or {}).get("head_repository_id")) == str(repo_id)
        and (a.get("workflow_run") or {}).get("head_branch") == "main"
    ]
    downloaded = [a for a in ours if str((a.get("workflow_run") or {}).get("id")) == str(downloaded_run_id)]
    if not downloaded:
        return "unknown", None
    base = max(_ts(a.get("created_at")) for a in downloaded)
    later = [
        a for a in ours
        if str((a.get("workflow_run") or {}).get("id")) not in (str(downloaded_run_id), str(this_run_id))
        and (_ts(a.get("created_at")) or base) > base
    ]
    if not later:
        return "ok", None
    return "newer_exists", max(later, key=lambda a: a.get("created_at") or "")


LINEAGE_TABLE = "canonical_lineage"
#: A lineage row older than this cannot belong to the pipeline uploading now.
LINEAGE_MAX_AGE_DAYS = 6


def record_lineage(con, *, run_id: str, downloaded_run_id: str, source: str) -> None:
    con.execute(
        f"CREATE TABLE IF NOT EXISTS {LINEAGE_TABLE} (recorded_at TIMESTAMP, run_id TEXT, "
        "downloaded_run_id TEXT, source TEXT)"
    )
    con.execute(
        f"INSERT INTO {LINEAGE_TABLE} VALUES (CAST(now() AS TIMESTAMP), ?, ?, ?)",
        [str(run_id), str(downloaded_run_id), str(source)],
    )


def read_lineage(con, *, now: datetime | None = None) -> tuple[str, str] | None:
    """``(downloaded_run_id, source)`` of the newest recent lineage row, or None."""
    try:
        row = con.execute(
            f"SELECT downloaded_run_id, source, recorded_at FROM {LINEAGE_TABLE} "
            "ORDER BY recorded_at DESC LIMIT 1"
        ).fetchone()
    except Exception:  # noqa: BLE001 - no table: nothing recorded
        return None
    if not row:
        return None
    recorded = row[2]
    if recorded is not None:
        recorded = recorded if recorded.tzinfo else recorded.replace(tzinfo=timezone.utc)
        age = (now or datetime.now(timezone.utc)) - recorded
        if age.total_seconds() > LINEAGE_MAX_AGE_DAYS * 86400:
            return None
    return str(row[0] or ""), str(row[1] or "")


def main(argv: list[str] | None = None, *, gh: GhFn = _gh,
         sleep: Callable[[float], None] = time.sleep) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trigger-did-work")
    t.add_argument("--run-id", default=os.getenv("TRIGGER_RUN_ID", ""))
    u = sub.add_parser("upload-guard")
    u.add_argument("--downloaded-run-id", default=os.getenv("DOWNLOADED_RUN_ID", ""))
    u.add_argument("--source", default=os.getenv("DOWNLOAD_SOURCE", ""))
    u.add_argument("--db", default="", help="read the downloaded run id from canonical_lineage")
    r = sub.add_parser("record")
    r.add_argument("--db", required=True)
    r.add_argument("--downloaded-run-id", default=os.getenv("DOWNLOADED_RUN_ID", ""))
    r.add_argument("--source", default=os.getenv("DOWNLOAD_SOURCE", ""))
    args = parser.parse_args(argv)

    if args.cmd == "record":
        import duckdb

        con = duckdb.connect(args.db)
        try:
            record_lineage(con, run_id=os.getenv("GITHUB_RUN_ID", ""),
                           downloaded_run_id=args.downloaded_run_id, source=args.source)
        finally:
            con.close()
        print(f"recorded: this DB was forked from canonical run {args.downloaded_run_id} "
              f"({args.source or 'discovery'})")
        return 0
    repo = os.getenv("GITHUB_REPOSITORY", "")
    attempts = int(os.getenv("PYTHIA_GATE_ATTEMPTS", "4") or 4)
    backoff = float(os.getenv("PYTHIA_GATE_BACKOFF_SEC", "15") or 15)
    out_path = os.getenv("GITHUB_OUTPUT")

    def emit(key: str, value: str) -> None:
        if out_path:
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(f"{key}={value}\n")

    if args.cmd == "trigger-did-work":
        event = os.getenv("GITHUB_EVENT_NAME", "")
        if event != "workflow_run" or not args.run_id:
            print(f"trigger-did-work: event {event or '?'}; nothing to check")
            emit("did_work", "true")
            return 0
        try:
            listing = with_retries(
                lambda: gh(["api", f"repos/{repo}/actions/runs/{args.run_id}/artifacts?per_page=100"]),
                attempts=attempts, backoff=backoff, sleep=sleep,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"::warning title=Trigger check unknown::could not list run {args.run_id}'s "
                  f"artifacts ({exc}); continuing")
            emit("did_work", "true")
            return 0
        did = run_uploaded_canonical((listing or {}).get("artifacts") or [])
        if not did:
            print(f"::notice title=Trigger did no work::run {args.run_id} uploaded no "
                  f"{ARTIFACT} (its gate skipped it, or it stopped before the upload); "
                  "this chained run does nothing")
        emit("did_work", "true" if did else "false")
        return 0

    # upload-guard
    if args.db and not args.downloaded_run_id:
        import duckdb

        con = duckdb.connect(args.db, read_only=True)
        try:
            got = read_lineage(con)
        finally:
            con.close()
        if got:
            args.downloaded_run_id, args.source = got
    if args.source == "forced":
        print("upload-guard: the operator chose this DB (forced); not comparing")
        return 0
    if not args.downloaded_run_id:
        print("::warning title=Upload guard unknown::no downloaded run id recorded; continuing")
        return 0
    try:
        repo_id = (with_retries(lambda: gh(["api", f"repos/{repo}"]), attempts=attempts,
                                backoff=backoff, sleep=sleep) or {}).get("id")
        listing = with_retries(
            lambda: gh(["api", f"repos/{repo}/actions/artifacts?name={ARTIFACT}&per_page=50"]),
            attempts=attempts, backoff=backoff, sleep=sleep,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"::warning title=Upload guard unknown::{exc}; uploading")
        return 0
    verdict, newer = newer_canonical(
        (listing or {}).get("artifacts") or [], downloaded_run_id=args.downloaded_run_id,
        this_run_id=os.getenv("GITHUB_RUN_ID", ""), repo_id=repo_id,
    )
    if verdict == "newer_exists":
        wr = (newer or {}).get("workflow_run") or {}
        print(
            f"::error title=A newer canonical DB exists::run {wr.get('id')} uploaded "
            f"{ARTIFACT} at {newer.get('created_at')}, after the DB this run downloaded "
            f"(run {args.downloaded_run_id}). Uploading would discard its rows. Re-run this "
            "workflow so it starts from the newest canonical DB."
        )
        return 1
    if verdict == "unknown":
        print(f"::warning title=Upload guard unknown::run {args.downloaded_run_id}'s "
              f"{ARTIFACT} is not in the listing; uploading")
    else:
        print("upload-guard: no newer canonical DB; uploading")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
