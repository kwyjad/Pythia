# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Emit the tiny pythia-batch-state JSON a submit stage uploads.

The staged pipeline's cross-stage state lives in the DuckDB artifact
(llm_batches / llm_batch_requests), but the poller workflow must not
download a ~1GB DB every 15 minutes just to read a handful of provider
batch ids — so each submit stage also uploads this KB-sized JSON:

    {
      "pipeline_id": ..., "stage": ..., "next_stage": ...,
      "db_run_id": <the submit stage's own Actions run id>,
      "test_mode": bool, "fc_run_id": ..., "hs_run_id": ...,
      "created_at": iso8601,
      "dispatch_inputs": {"batch_providers", "only_countries",
                          "grounding_primary", "disable_brave"},
      "pending": [{"batch_id", "provider", "provider_batch_id", "family",
                    "status", "submitted_at"}]
    }

``dispatch_inputs`` mirrors the run-shaping workflow_dispatch inputs as this
stage ACTUALLY resolved them (read back out of the env the workflow set), so
the poller can re-supply them when it dispatches the next stage. Without it a
smoke run (e.g. only_countries=IRN,SOM + disable_brave) silently reverted to
full production settings from stage 2 onwards.

Usage:
    python -m scripts.ci.emit_batch_state --db data/resolver.duckdb \
        --pipeline-id pl_X --stage fc_submit --next-stage fc_collect_finalize \
        --db-run-id "$GITHUB_RUN_ID" --out diagnostics/batch_state.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes")


def dispatch_inputs_from_env(env: dict[str, str]) -> dict[str, str]:
    """Effective run-shaping inputs, as strings a `gh workflow run -f` accepts.

    Derived from the env the stage workflow set (not from the raw inputs), so
    what carries forward is what this stage actually ran with. ``disable_brave``
    is inferred from an empty BRAVE_SEARCH_API_KEY — the workflow implements
    that input by withholding the secret, and re-supplying it on the next stage
    would let a 402ing key trip the breaker and block ungrounded hazards.
    """

    return {
        "batch_providers": (env.get("PYTHIA_BATCH_PROVIDERS") or "").strip(),
        "only_countries": (env.get("PYTHIA_HS_ONLY_COUNTRIES") or "").strip(),
        "grounding_primary": (env.get("PYTHIA_GROUNDING_PRIMARY_BACKEND") or "").strip(),
        "disable_brave": "true" if not (env.get("BRAVE_SEARCH_API_KEY") or "").strip() else "false",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--pipeline-id", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--next-stage", required=True)
    ap.add_argument("--db-run-id", required=True, help="this workflow run's id")
    ap.add_argument("--fc-run-id", default="")
    ap.add_argument("--hs-run-id", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import duckdb

    from pythia.llm_batch import pending_batches

    db_path = args.db
    if db_path.startswith("duckdb:///"):
        db_path = db_path[len("duckdb:///"):]
    con = duckdb.connect(db_path, read_only=True)
    try:
        pending = pending_batches(con, args.pipeline_id)
    finally:
        con.close()

    state = {
        "pipeline_id": args.pipeline_id,
        "stage": args.stage,
        "next_stage": args.next_stage,
        "db_run_id": str(args.db_run_id),
        "test_mode": _truthy(os.getenv("PYTHIA_TEST_MODE", "0")),
        "fc_run_id": args.fc_run_id,
        "hs_run_id": args.hs_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dispatch_inputs": dispatch_inputs_from_env(dict(os.environ)),
        "pending": [
            {
                "batch_id": p["batch_id"],
                "provider": p["provider"],
                "provider_batch_id": p["provider_batch_id"],
                "family": p["family"],
                "status": p["status"],
                "submitted_at": str(p["submitted_at"]) if p["submitted_at"] else None,
            }
            for p in pending
        ],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    print(
        f"batch state written: {args.out} "
        f"(pipeline={args.pipeline_id}, next_stage={args.next_stage}, "
        f"pending={len(state['pending'])})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
