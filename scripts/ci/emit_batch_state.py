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
      "pending": [{"batch_id", "provider", "provider_batch_id", "family",
                    "status", "submitted_at"}]
    }

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
        "test_mode": os.getenv("PYTHIA_TEST_MODE", "0").strip().lower() in ("1", "true", "yes"),
        "fc_run_id": args.fc_run_id,
        "hs_run_id": args.hs_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
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
