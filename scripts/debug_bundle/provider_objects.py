# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The raw provider batch objects, captured while they still exist.

A provider batch object carries the only account of a validation failure
there is. OpenAI reports it in an ``errors`` list on the batch and nowhere
else; there are no per-request results when nothing is ingested, so the
error file id is null and the output file id is null and the batch's own
counters are zeros. On 2026-09-01 that made four whole-batch rejections
unrecoverable twice over, because these objects and their files age out on
the provider side long before anyone opens an artifact.

So we fetch them at collect time and keep them verbatim: the batch object,
and the input, output and error files it names. Nothing is redacted except
auth headers — which are never in the response body anyway; the redaction
pass here exists for the case where a provider quotes a request back.

A fetch failure is a RECORD, not a dropped entry: the status code and body
say whether the object expired, the key was wrong, or the network broke,
and those want three different repairs.
"""

from __future__ import annotations

import json
import os
from typing import Any

from scripts.debug_bundle.redaction import redact_mapping, redact_text

# A provider file can be tens of MB (the batch input JSONL is every request
# body). Keep the head, which carries the shape, and say what was cut.
_MAX_FILE_CHARS = 512 * 1024
_HTTP_TIMEOUT = 30.0


def _http_get(url: str, headers: dict[str, str], *, timeout: float = _HTTP_TIMEOUT) -> dict[str, Any]:
    """GET returning a record, never an exception."""

    try:
        import requests  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - requests is a hard dep in prod
        return {"ok": False, "error": f"requests unavailable: {exc}"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": redact_text(str(exc)), "status_code": None}
    record: dict[str, Any] = {"ok": resp.status_code < 400, "status_code": resp.status_code}
    text = resp.text or ""
    try:
        record["json"] = redact_mapping(json.loads(text)) if text.strip().startswith("{") else None
    except Exception:
        record["json"] = None
    if record["json"] is None:
        truncated = len(text) > _MAX_FILE_CHARS
        record["text"] = redact_text(text[:_MAX_FILE_CHARS])
        if truncated:
            record["truncated_chars"] = len(text) - _MAX_FILE_CHARS
    return record


def _openai_headers() -> dict[str, str] | None:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    return {"Authorization": f"Bearer {key}"}


def _anthropic_headers() -> dict[str, str] | None:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return None
    return {
        "x-api-key": key,
        "anthropic-version": os.getenv("ANTHROPIC_API_VERSION", "2023-06-01"),
    }


def _fetch_openai(batch: dict[str, Any]) -> dict[str, Any]:
    headers = _openai_headers()
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    pid = batch.get("provider_batch_id")
    if not headers:
        return {"skipped": "OPENAI_API_KEY not set in this job"}
    out: dict[str, Any] = {"batch_object": _http_get(f"{base}/batches/{pid}", headers)}
    # File ids come from the batch object when it has them, and from our own
    # row when the object could not be read at all.
    obj = (out["batch_object"].get("json") or {}) if isinstance(out["batch_object"], dict) else {}
    files: dict[str, Any] = {}
    for label, key in (
        ("input_file_id", "input_file_id"),
        ("output_file_id", "output_file_id"),
        ("error_file_id", "error_file_id"),
    ):
        file_id = obj.get(key) or batch.get(label)
        if not file_id:
            continue
        files[label] = {
            "file_id": file_id,
            "metadata": _http_get(f"{base}/files/{file_id}", headers),
            "content": _http_get(f"{base}/files/{file_id}/content", headers),
        }
    out["files"] = files
    return out


def _fetch_anthropic(batch: dict[str, Any]) -> dict[str, Any]:
    headers = _anthropic_headers()
    if not headers:
        return {"skipped": "ANTHROPIC_API_KEY not set in this job"}
    base = "https://api.anthropic.com/v1"
    pid = batch.get("provider_batch_id")
    out: dict[str, Any] = {"batch_object": _http_get(f"{base}/messages/batches/{pid}", headers)}
    obj = (out["batch_object"].get("json") or {}) if isinstance(out["batch_object"], dict) else {}
    results_url = obj.get("results_url") or batch.get("results_url")
    if results_url:
        # The results stream is JSONL, one line per request — the Anthropic
        # equivalent of the OpenAI output file.
        out["files"] = {"results_url": {"url": results_url, "content": _http_get(results_url, headers)}}
    return out


def _fetch_google(batch: dict[str, Any]) -> dict[str, Any]:
    key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        return {"skipped": "GEMINI_API_KEY / GOOGLE_API_KEY not set in this job"}
    base = "https://generativelanguage.googleapis.com/v1beta"
    name = str(batch.get("provider_batch_id") or "").lstrip("/")
    if not name:
        return {"skipped": "no provider batch name recorded"}
    # Google carries the operation, its metadata and its inlined responses in
    # one object; there are no separate files to fetch.
    return {"batch_object": _http_get(f"{base}/{name}?key={key}", {})}


_FETCHERS = {
    "openai": _fetch_openai,
    "anthropic": _fetch_anthropic,
    "google": _fetch_google,
}


def collect(batches: list[dict[str, Any]], *, enabled: bool = True) -> dict[str, Any]:
    """Fetch the raw provider objects for every batch in ``batches``.

    ``batches`` is the ``batches`` list from ``batch_lifecycle.collect`` —
    the collector reads only ``provider``, ``provider_batch_id`` and the
    three file ids, so a caller can hand it anything with those keys.
    """

    out: dict[str, Any] = {"fetched_at": None, "objects": []}
    try:
        from datetime import datetime, timezone  # noqa: PLC0415

        out["fetched_at"] = datetime.now(timezone.utc).isoformat()
    except Exception:  # pragma: no cover - defensive
        pass
    if not enabled:
        out["note"] = "provider object capture disabled (PYTHIA_BUNDLE_FETCH_PROVIDER_OBJECTS=0)"
        return out
    if not batches:
        out["note"] = "no batches to fetch"
        return out

    for batch in batches:
        provider = str(batch.get("provider") or "").lower()
        entry: dict[str, Any] = {
            "batch_id": batch.get("batch_id"),
            "provider": provider,
            "provider_batch_id": batch.get("provider_batch_id"),
            "phase": batch.get("phase"),
            "model_id": batch.get("model_id"),
        }
        fetcher = _FETCHERS.get(provider)
        if fetcher is None:
            entry["error"] = f"no provider object fetcher for provider {provider!r}"
        elif not batch.get("provider_batch_id"):
            # A batch with no provider id never reached the provider — its
            # own row is the whole story, and saying so beats a 404.
            entry["error"] = "no provider_batch_id recorded (batch never reached the provider)"
        else:
            try:
                entry.update(fetcher(batch))
            except Exception as exc:  # noqa: BLE001
                entry["error"] = redact_text(f"{type(exc).__name__}: {exc}")
        out["objects"].append(entry)
    return out
