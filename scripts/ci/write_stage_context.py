# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Write the forecaster stage's own context for the bundle built later.

Since Sept 2026 the operational debug bundle is built in the Sibyl job, at
the end of the forecast cycle, so that every artifact a cycle produces is
produced in one place. Two of the bundle's collectors read the job they run
in: the environment snapshot and the DB signatures. Moved as they were, they
would describe the Sibyl job and quietly lose the forecaster stage's flags.

So the stage writes ``diagnostics/stage_context.json`` at its end — the env
snapshot, both DB signatures, its conclusion and its identifiers — and
uploads it as ``pythia-stage-context``. The Sibyl job downloads it into the
bundle's ``stage_context/`` directory; when it is missing there, a stub says
so. A reader must never have to guess whether a collector failed or a file
was never produced.

Usage:
    python -m scripts.ci.write_stage_context --out diagnostics/stage_context.json \
        --pipeline-id pl_x --hs-run-id hs_x --forecaster-run-id fc_x \
        --conclusion success \
        --signature-before diagnostics/db_signature_before.json \
        --signature-after diagnostics/db_signature_after.json

Never fails: on any error the file carries the error instead. Exit 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def _load_json(path: str | None) -> Any:
    """The file's content, or a record of why there is none."""
    if not path:
        return {"missing": True, "reason": "no path given"}
    p = Path(path)
    if not p.exists():
        return {"missing": True, "reason": f"{p} does not exist", "path": str(p)}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"missing": True, "reason": f"unreadable: {type(exc).__name__}: {exc}", "path": str(p)}


def build_context(
    *,
    pipeline_id: str | None,
    hs_run_id: str | None,
    forecaster_run_id: str | None,
    conclusion: str | None,
    signature_before: str | None,
    signature_after: str | None,
    repo_root: Path,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    env = dict(environ if environ is not None else os.environ)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "workflow": env.get("GITHUB_WORKFLOW"),
        "job": env.get("GITHUB_JOB"),
        "github_run_id": env.get("GITHUB_RUN_ID"),
        "github_run_attempt": env.get("GITHUB_RUN_ATTEMPT"),
        "pipeline_id": pipeline_id or None,
        "hs_run_id": hs_run_id or None,
        "forecaster_run_id": forecaster_run_id or None,
        "stage_conclusion": conclusion or None,
        "db_signature_before": _load_json(signature_before),
        "db_signature_after": _load_json(signature_after),
    }
    try:
        from scripts.debug_bundle import env_config  # noqa: PLC0415

        payload["env_snapshot"] = env_config.collect(repo_root=repo_root, environ=env)
    except Exception as exc:  # noqa: BLE001
        payload["env_snapshot"] = {"error": f"{type(exc).__name__}: {exc}"}
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pipeline-id", default="")
    parser.add_argument("--hs-run-id", default="")
    parser.add_argument("--forecaster-run-id", default="")
    parser.add_argument("--conclusion", default="", help="The stage's job status at write time.")
    parser.add_argument("--signature-before", default="diagnostics/db_signature_before.json")
    parser.add_argument("--signature-after", default="diagnostics/db_signature_after.json")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = build_context(
            pipeline_id=args.pipeline_id,
            hs_run_id=args.hs_run_id,
            forecaster_run_id=args.forecaster_run_id,
            conclusion=args.conclusion,
            signature_before=args.signature_before,
            signature_after=args.signature_after,
            repo_root=Path(args.repo_root),
        )
    except Exception as exc:  # noqa: BLE001
        payload = {
            "schema_version": SCHEMA_VERSION,
            "written_at": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"stage context written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
