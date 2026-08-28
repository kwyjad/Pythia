# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Watchdog: has every critical scheduled workflow succeeded recently enough?

Written after Resolver Update silently stopped running for 44 days.

GitHub delivers ``schedule`` events on a best-effort basis, and this repo is
demonstrably throttled: measured on 2026-08-28, the daily ``0 6 * * *``
Resolver CI tick landed 11h32m late, the nightly ``30 20 * * *`` Hazard
Backcast tick 7h54m late, and the monthly ``0 3 28 * *`` Resolver Update tick
had not arrived 11h36m in. A monthly cron therefore gets ONE firing
opportunity that may simply be dropped, and because compute_resolutions ->
compute_scores -> compute_calibration -> publish all hang off Resolver Update
by ``workflow_run``, one dropped tick costs a month of ground truth, scoring
and calibration.

That is what happened between 2026-07-15 and 2026-08-28, and nothing anywhere
noticed. This script is the thing that notices.

Deliberate design choices
-------------------------
* **stdlib + ``gh`` only.** Same constraint as ``check_pipeline_active.py``, so
  the workflow needs no setup-python and no pip install — checkout, then run.
* **It fails LOUD.** The gates in this repo fail OPEN, and correctly so: a
  broken gate must never silence the weekly ingest or stall a live pipeline.
  A watchdog is the opposite. A watchdog that fails open is decorative, so a
  workflow over its age AND a workflow whose age cannot be determined both
  fail the run. A red scheduled run emails the repo owner; that email is the
  entire mechanism.
* **Age is measured from the last SUCCESS**, never the last run. Hazard
  Backcast failed 11 nights running in August while firing perfectly on time;
  "it ran" is not the property anyone cares about.

Usage: ``python -m scripts.ci.check_workflow_freshness [--workflow NAME ...]``
Requires ``GITHUB_REPOSITORY`` and a ``gh`` authenticated with ``actions: read``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Iterable, NamedTuple


class Watched(NamedTuple):
    """A workflow and how long it may go without a successful run."""

    name: str
    max_age_days: float
    cadence: str


# One line per workflow, carrying the cadence it encodes so the limit can be
# checked against the cron without opening another file. Limits are the cadence
# plus room for the observed multi-hour scheduling drift and one manual
# recovery cycle — they answer "is this workflow still alive?", not "did it run
# exactly on time?". Tightening them to the cadence itself would make the
# watchdog cry wolf on every late tick, and a validator that cries wolf gets
# switched off.
WATCHED: tuple[Watched, ...] = (
    Watched("Resolver Update", 35, "monthly, 28th"),
    Watched("Pythia — Compute Resolutions", 35, "chained off Resolver Update"),
    Watched("Pythia — Compute SPD Scores", 35, "chained off Compute Resolutions"),
    Watched("Pythia — Compute Calibration Weights & Advice", 35, "chained off Compute SPD Scores"),
    Watched("Pythia Pipeline Stage", 35, "monthly, 1st"),
    Watched("Publish Latest Data (Release)", 35, "after Sibyl / after Calibration"),
    Watched("Ingest Structured Data", 10, "weekly, Sunday"),
    Watched("Hazard Backcast", 3, "nightly"),
    Watched("Refresh CrisisWatch Data", 35, "monthly, days 3/5/7/10"),
)

UNKNOWN = "unknown"


class Result(NamedTuple):
    name: str
    last_success: str | None
    age_days: float | None
    max_age_days: float
    cadence: str
    stale: bool
    detail: str

    @property
    def verdict(self) -> str:
        if self.stale:
            return "STALE" if self.last_success else "NO SUCCESSFUL RUN"
        return "ok"


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def evaluate(
    watched: Watched,
    runs: list[dict] | None,
    now: datetime,
) -> Result:
    """Pure decision core (unit-tested without gh).

    ``runs`` is None when the lookup itself failed — distinct from an empty
    list, which means gh answered and the workflow has genuinely never
    succeeded. Both are stale, but only one of them is a broken watchdog, and
    the operator reading the email needs to be told which.
    """

    if runs is None:
        return Result(
            watched.name, None, None, watched.max_age_days, watched.cadence, True,
            "could not determine last success (gh lookup failed)",
        )

    timestamps = [
        ts
        for ts in (
            _parse_ts(run.get("createdAt"))
            for run in runs
            if run.get("conclusion") == "success"
        )
        if ts is not None
    ]
    if not timestamps:
        return Result(
            watched.name, None, None, watched.max_age_days, watched.cadence, True,
            "no successful run found in the queried window",
        )

    newest = max(timestamps)
    age_days = (now - newest).total_seconds() / 86400.0
    stale = age_days > watched.max_age_days
    detail = (
        f"last success {age_days:.1f}d ago (limit {watched.max_age_days:g}d)"
        if stale
        else f"{age_days:.1f}d since last success"
    )
    return Result(
        watched.name, newest.isoformat(), age_days, watched.max_age_days,
        watched.cadence, stale, detail,
    )


def _gh_runs(repo: str, name: str, limit: int = 40) -> list[dict] | None:
    """Recent runs for one workflow, or None when the lookup failed.

    Not restricted to --branch main: the compute workflows are triggered by
    workflow_run and a branch filter has historically hidden runs that plainly
    happened. A workflow whose only recent successes were on a side branch is
    a question for a human, not a reason for the watchdog to stay quiet.
    """

    try:
        out = subprocess.run(
            [
                "gh", "run", "list",
                "--repo", repo,
                "--workflow", name,
                "--json", "createdAt,conclusion",
                "--limit", str(limit),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        parsed = json.loads(out or "[]")
        return parsed if isinstance(parsed, list) else None
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not list runs for {name!r}: {exc}", file=sys.stderr)
        return None


def render(results: Iterable[Result]) -> str:
    rows = [
        "| Workflow | Cadence | Last success | Age | Limit | Verdict |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for r in results:
        last = (r.last_success or "—")[:19].replace("T", " ")
        age = f"{r.age_days:.1f}d" if r.age_days is not None else "—"
        rows.append(
            f"| {r.name} | {r.cadence} | {last} | {age} "
            f"| {r.max_age_days:g}d | {r.verdict} |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow",
        action="append",
        default=None,
        help="Check only these workflow names (repeatable). Default: all watched.",
    )
    parser.add_argument(
        "--limit", type=int, default=40,
        help="Runs to inspect per workflow when hunting for the last success.",
    )
    args = parser.parse_args(argv)

    watched = WATCHED
    if args.workflow:
        wanted = {w.strip() for w in args.workflow}
        watched = tuple(w for w in WATCHED if w.name in wanted)
        missing = wanted - {w.name for w in watched}
        if missing:
            print(f"::error::unknown workflow name(s): {', '.join(sorted(missing))}")
            return 2

    repo = os.getenv("GITHUB_REPOSITORY", "")
    if not repo:
        print("::error::GITHUB_REPOSITORY is unset; cannot query workflow runs")
        return 1

    now = datetime.now(timezone.utc)
    results = [evaluate(w, _gh_runs(repo, w.name, args.limit), now) for w in watched]

    table = render(results)
    print(table)

    stale = [r for r in results if r.stale]
    for r in stale:
        print(f"::error title=Workflow stale: {r.name}::{r.detail} — {r.cadence}")

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        headline = (
            f"{len(stale)} of {len(results)} watched workflows are stale"
            if stale
            else f"All {len(results)} watched workflows are fresh"
        )
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(f"### Cron watchdog\n\n{headline}.\n\n{table}\n")

    if stale:
        names = ", ".join(r.name for r in stale)
        print(f"\nFAIL: {len(stale)} stale workflow(s): {names}")
        return 1

    print(f"\nOK: all {len(results)} watched workflows have succeeded within their limits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
