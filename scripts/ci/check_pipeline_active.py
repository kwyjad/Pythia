# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Gate: is a staged Batch-API pipeline currently in flight?

Used by ingest-structured-data.yml before touching the canonical
``pythia-resolver-db`` artifact. The staged pipeline forks the DB at
hs_submit into ``pythia-resolver-db-staged`` and re-uploads the CANONICAL
artifact days later at fc_collect_finalize — the concurrency group only
serializes individual stage runs, not the multi-day pipeline. A weekly
ingest landing inside that window would add rows to canonical that the
pipeline's final upload (built from the pre-ingest fork) silently discards
— last writer wins, whole-file artifact swap.

Detection (mirrors the poller's activity gate):
1. Newest ``pythia-batch-state`` artifact. None, or older than
   ``PYTHIA_PIPELINE_ACTIVE_WINDOW_H`` (default 72h) → not in flight.
2. Otherwise, a SUCCESSFUL "Pythia Pipeline Stage" run whose title carries
   ``fc_collect_finalize`` created AFTER that artifact means the pipeline
   already published canonical → not in flight. (The final stage emits no
   batch-state, so while a pipeline is running the newest state artifact
   always belongs to an earlier stage.)
3. Else → in flight; the ingest should skip (next Sunday's cron covers it,
   and this workflow is documented as supplementary freshness).

Writes ``proceed=true|false`` to $GITHUB_OUTPUT. FORCE=true (the
force_during_pipeline dispatch input) always proceeds. Fails OPEN on gh
errors — a broken gate must not silence the weekly refresh forever.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

STAGE_WORKFLOW_NAME = "Pythia Pipeline Stage"
FINAL_STAGE_MARKER = "fc_collect_finalize"


def _window_hours() -> float:
    try:
        return float(os.getenv("PYTHIA_PIPELINE_ACTIVE_WINDOW_H", "72") or 72)
    except ValueError:
        return 72.0


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def pipeline_in_flight(
    latest_state_created_at: str | None,
    stage_runs: list[dict],
    now: datetime,
    *,
    window_hours: float = 72.0,
) -> bool:
    """Pure decision core (unit-tested without gh)."""

    state_ts = _parse_ts(latest_state_created_at)
    if state_ts is None:
        return False
    age_h = (now - state_ts).total_seconds() / 3600.0
    if age_h > window_hours:
        return False
    for run in stage_runs:
        if run.get("conclusion") != "success":
            continue
        if FINAL_STAGE_MARKER not in str(run.get("displayTitle") or ""):
            continue
        run_ts = _parse_ts(run.get("createdAt"))
        if run_ts is not None and run_ts > state_ts:
            return False
    return True


def _gh_json(*args: str) -> object:
    out = subprocess.run(
        ["gh", *args], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out or "null")


def main() -> int:
    force = (os.getenv("FORCE", "false") or "false").strip().lower() in ("1", "true", "yes")
    proceed = True
    reason = "no in-flight pipeline detected"

    if force:
        reason = "force_during_pipeline set — skipping the gate"
    else:
        try:
            repo = os.environ["GITHUB_REPOSITORY"]
            artifacts = _gh_json(
                "api", f"repos/{repo}/actions/artifacts?name=pythia-batch-state&per_page=1"
            )
            arts = (artifacts or {}).get("artifacts") or []
            latest_created = arts[0].get("created_at") if arts else None
            runs = _gh_json(
                "run", "list", "--workflow", STAGE_WORKFLOW_NAME, "--branch", "main",
                "--json", "createdAt,conclusion,displayTitle", "--limit", "30",
            ) or []
            if pipeline_in_flight(
                latest_created, runs, datetime.now(timezone.utc),
                window_hours=_window_hours(),
            ):
                proceed = False
                reason = (
                    f"staged pipeline in flight (batch-state artifact from "
                    f"{latest_created}, no newer successful {FINAL_STAGE_MARKER}) — "
                    "an ingest now would be silently discarded by the pipeline's "
                    "final canonical upload"
                )
        except Exception as exc:  # noqa: BLE001
            # Fail OPEN: a broken gate must not silence the weekly refresh.
            print(f"[warn] pipeline-active check failed ({exc}); proceeding")

    print(f"proceed={proceed} — {reason}")
    if not proceed:
        print(f"::notice title=Ingest skipped::{reason}")
    out_path = os.getenv("GITHUB_OUTPUT")
    if out_path:
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(f"proceed={'true' if proceed else 'false'}\n")
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"### Pipeline-active gate\n- proceed: {proceed}\n- {reason}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
