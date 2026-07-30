# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Dump an OpenAI Batch object in full, including the ``errors`` field.

Why this exists: on 2026-07-29 and again on 2026-07-30 every OpenAI batch came
back ``status=failed`` with ``request_counts={total: 0, completed: 0, failed:
0}`` and no ``error_file_id`` — i.e. OpenAI registered ZERO requests from our
input file. ``_OpenAIBatch.poll`` recorded only ``output_file_id`` and
``error_file_id``, so the one field that explains an input-validation failure —
the batch object's ``errors`` list, which carries ``code``/``message``/``line``
— was discarded on both occasions and the cause could not be recovered.

Batch objects are retained provider-side, so this can be pointed at an
already-failed batch id after the fact. It also fetches the input file's status
and (optionally) its first lines, because "zero requests ingested" is equally
consistent with a file that never became readable.

Read-only: it performs GETs against the OpenAI API and nothing else. Costs
nothing — batch objects and file metadata are not billed.

    python -m scripts.ci.inspect_openai_batch batch_abc123 [batch_def456 ...]
    python -m scripts.ci.inspect_openai_batch --recent 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests

_TIMEOUT = 60


def _base() -> str:
    return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def _headers() -> Dict[str, str]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set")
    return {"Authorization": f"Bearer {key}"}


def _get(path: str, params: Optional[dict] = None) -> Dict[str, Any]:
    resp = requests.get(
        f"{_base()}{path}", headers=_headers(), params=params or {}, timeout=_TIMEOUT
    )
    if resp.status_code >= 400:
        return {"_http_status": resp.status_code, "_body": resp.text[:2000]}
    return resp.json()


def _describe(batch: Dict[str, Any], show_input_head: int) -> None:
    bid = batch.get("id")
    print(f"\n=== batch {bid} ===")
    print(f"status            : {batch.get('status')}")
    print(f"endpoint          : {batch.get('endpoint')}")
    print(f"completion_window : {batch.get('completion_window')}")
    print(f"request_counts    : {json.dumps(batch.get('request_counts') or {})}")
    print(f"input_file_id     : {batch.get('input_file_id')}")
    print(f"output_file_id    : {batch.get('output_file_id')}")
    print(f"error_file_id     : {batch.get('error_file_id')}")

    # THE field. For a validation failure this is the only place the reason
    # exists — there is no error_file when nothing was ingested.
    errors = batch.get("errors")
    if errors:
        print("errors            :")
        print(json.dumps(errors, indent=2)[:4000])
    else:
        print("errors            : (none reported)")

    input_file_id = batch.get("input_file_id")
    if input_file_id:
        meta = _get(f"/files/{input_file_id}")
        print(
            "input file        : "
            f"status={meta.get('status')} bytes={meta.get('bytes')} "
            f"purpose={meta.get('purpose')} filename={meta.get('filename')!r} "
            f"status_details={meta.get('status_details')!r}"
        )
        if show_input_head:
            content = requests.get(
                f"{_base()}/files/{input_file_id}/content",
                headers=_headers(),
                timeout=_TIMEOUT,
            )
            if content.status_code < 400:
                lines = content.text.splitlines()
                print(f"input lines       : {len(lines)}")
                for line in lines[:show_input_head]:
                    # Truncate: request bodies carry multi-KB prompts.
                    print(f"  | {line[:600]}")
            else:
                print(f"input content     : HTTP {content.status_code}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("batch_ids", nargs="*", help="Provider batch ids (batch_...)")
    p.add_argument("--recent", type=int, default=0, help="Also list the N most recent batches")
    p.add_argument(
        "--input-head",
        type=int,
        default=0,
        help="Print the first N lines of each batch's input file (truncated).",
    )
    args = p.parse_args(argv)

    ids = list(args.batch_ids)

    if args.recent:
        listing = _get("/batches", {"limit": args.recent})
        rows = listing.get("data") or []
        print(f"=== {len(rows)} most recent batches ===")
        for b in rows:
            print(
                f"{b.get('id')}  {b.get('status'):<12} "
                f"counts={json.dumps(b.get('request_counts') or {})}  "
                f"created={b.get('created_at')}"
            )
            if b.get("id") not in ids:
                ids.append(b.get("id"))

    if not ids:
        print("No batch ids supplied and --recent not set; nothing to do.")
        return 0

    for bid in ids:
        batch = _get(f"/batches/{bid}")
        if batch.get("_http_status"):
            print(f"\n=== batch {bid} ===")
            print(f"HTTP {batch['_http_status']}: {batch['_body']}")
            continue
        _describe(batch, args.input_head)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
