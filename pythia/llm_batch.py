# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Provider Batch-API execution layer (OpenAI / Anthropic / Gemini).

Batch APIs process independent requests asynchronously at 50% of sync
prices (input AND output tokens). This module is the provider-agnostic
submit/poll/fetch layer used by the staged pipeline:

1. A *submit* stage builds prompts exactly as the sync path would (via the
   shared body builders in ``forecaster.providers`` — byte-identical
   payloads), enqueues one ``llm_batch_requests`` row per call, and calls
   :func:`submit_pending`, which chunks rows into provider batches and
   records ``llm_batches`` rows. The DuckDB file then travels to the next
   stage inside the pipeline artifact — the DB *is* the batch state.
2. The poller workflow calls :func:`poll_batch` per pending batch until the
   provider reports completion (most <1h; hard max 24h).
3. A *collect* stage calls :func:`collect_batch` to fetch results into
   ``llm_batch_requests`` (idempotent), then consumers replay them via
   :func:`get_result`. A missing/errored/expired item returns None and the
   consumer falls through to the existing synchronous ``call_chat_ms`` path
   (which is the per-item fallback, inheriting retries/cooldowns).

Every collected usage dict is stamped ``service_tier="batch"`` plus batch
ids, so the shared cost helper (``compute_cost_split_usd``) applies the 50%
batch multiplier and the ledger reflects real spend.

Sequential work (Sibyl belief updating, JSON-repair retries) is NOT
batchable and never goes through this module.

Env flags:
- ``PYTHIA_BATCH_API_ENABLED`` (default 0): master switch.
- ``PYTHIA_BATCH_PROVIDERS`` (default "openai,anthropic,google"): which
  providers may batch; others stay on the sync path.
- ``PYTHIA_BATCH_MAX_WAIT_H`` (default 24): poller gives up past this age
  and the collect stage falls back to sync for unfinished items.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests

LOGGER = logging.getLogger(__name__)

# Batch families. Forecaster families key rows by (question_id, model_key);
# HS families key rows by (iso3, hazard_code, pass_idx).
FAMILIES = ("spd_v2", "binary_v2", "track2_spd", "scenario", "hs_rc", "hs_triage")

# Compact family slugs for custom ids (Anthropic: ^[A-Za-z0-9_-]{1,64}$).
_FAMILY_SLUGS = {
    "spd_v2": "spd",
    "binary_v2": "bin",
    "track2_spd": "t2",
    "scenario": "scn",
    "hs_rc": "hsrc",
    "hs_triage": "hstr",
}
_SLUG_TO_FAMILY = {v: k for k, v in _FAMILY_SLUGS.items()}

_CUSTOM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Conservative per-batch chunking limits (provider caps are far higher:
# Anthropic 100k requests / 256MB, OpenAI 50k lines / ~200MB JSONL).
_MAX_REQUESTS_PER_BATCH = int(os.getenv("PYTHIA_BATCH_MAX_REQUESTS", "5000") or 5000)
_MAX_BYTES_PER_BATCH = int(os.getenv("PYTHIA_BATCH_MAX_BYTES", str(100 * 1024 * 1024)) or (100 * 1024 * 1024))

_HTTP_TIMEOUT = float(os.getenv("PYTHIA_BATCH_HTTP_TIMEOUT_SEC", "120") or 120)


def batch_api_enabled() -> bool:
    return os.getenv("PYTHIA_BATCH_API_ENABLED", "0").strip().lower() in ("1", "true", "yes")


def batch_providers() -> set[str]:
    raw = os.getenv("PYTHIA_BATCH_PROVIDERS", "openai,anthropic,google")
    return {p.strip().lower() for p in raw.split(",") if p.strip()}


def provider_batchable(provider: str) -> bool:
    return batch_api_enabled() and (provider or "").lower() in batch_providers()


def max_wait_hours() -> float:
    try:
        return float(os.getenv("PYTHIA_BATCH_MAX_WAIT_H", "24") or 24)
    except ValueError:
        return 24.0


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _is_test() -> bool:
    try:
        from pythia.test_mode import is_test_mode

        return is_test_mode()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Custom ids
# ---------------------------------------------------------------------------


def _sanitize_token(token: str, max_len: int = 24) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", str(token or ""))
    return cleaned[:max_len] or "x"


def make_custom_id(
    family: str,
    *,
    question_id: Optional[str] = None,
    model_key: Optional[str] = None,
    iso3: Optional[str] = None,
    hazard_code: Optional[str] = None,
    pass_idx: Optional[int] = None,
) -> str:
    """Encode a compact, Anthropic-safe custom id.

    The id is a lookup key only — the authoritative reassembly metadata is
    the ``llm_batch_requests`` row (question_id, iso3, hazard, model_key,
    ...), which always travels with the batch in the DB artifact.
    """

    slug = _FAMILY_SLUGS.get(family)
    if slug is None:
        raise ValueError(f"unknown batch family: {family}")
    if family in ("spd_v2", "binary_v2", "track2_spd", "scenario"):
        if not question_id:
            raise ValueError(f"{family} custom id requires question_id")
        qhash = hashlib.sha1(str(question_id).encode("utf-8")).hexdigest()[:10]
        custom_id = f"{slug}-{qhash}-{_sanitize_token(model_key or 'primary')}"
    else:
        if not (iso3 and hazard_code):
            raise ValueError(f"{family} custom id requires iso3 + hazard_code")
        custom_id = (
            f"{slug}-{_sanitize_token(iso3, 3)}-{_sanitize_token(hazard_code, 8)}"
            f"-p{int(pass_idx or 1)}"
        )
    if not _CUSTOM_ID_RE.match(custom_id):
        raise ValueError(f"custom id fails provider constraints: {custom_id!r}")
    return custom_id


# ---------------------------------------------------------------------------
# Enqueue / state
# ---------------------------------------------------------------------------


def enqueue_request(
    con,
    *,
    family: str,
    provider: str,
    model_id: str,
    request_body: Dict[str, Any],
    prompt_text: str,
    question_id: Optional[str] = None,
    model_key: Optional[str] = None,
    iso3: Optional[str] = None,
    hazard_code: Optional[str] = None,
    metric: Optional[str] = None,
    pass_idx: Optional[int] = None,
    anchor_month: Optional[str] = None,
) -> str:
    """Insert (or refresh) one pending batch request row; returns custom_id.

    Re-enqueueing an id that already holds a terminal result (succeeded /
    fallback_sync) is a no-op so a re-run submit stage cannot clobber
    collected results.
    """

    custom_id = make_custom_id(
        family,
        question_id=question_id,
        model_key=model_key,
        iso3=iso3,
        hazard_code=hazard_code,
        pass_idx=pass_idx,
    )
    existing = con.execute(
        "SELECT status FROM llm_batch_requests WHERE custom_id = ?", [custom_id]
    ).fetchone()
    if existing and existing[0] in ("succeeded", "fallback_sync"):
        return custom_id

    con.execute("DELETE FROM llm_batch_requests WHERE custom_id = ?", [custom_id])
    con.execute(
        """
        INSERT INTO llm_batch_requests (
            custom_id, batch_id, provider, model_id, family, question_id,
            iso3, hazard_code, metric, model_key, pass_idx, anchor_month,
            prompt_sha256, request_body_json, status, created_at, is_test
        ) VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
        """,
        [
            custom_id,
            (provider or "").lower(),
            model_id,
            family,
            question_id,
            (iso3 or None),
            (hazard_code or None),
            (metric or None),
            model_key,
            pass_idx,
            anchor_month,
            hashlib.sha256((prompt_text or "").encode("utf-8")).hexdigest(),
            json.dumps(request_body, ensure_ascii=False),
            _now(),
            _is_test(),
        ],
    )
    return custom_id


@dataclass
class BatchStatus:
    provider_batch_id: str
    state: str  # submitted | in_progress | ended | failed | expired | canceled
    counts: Dict[str, int]
    detail: str = ""

    @property
    def terminal(self) -> bool:
        return self.state in ("ended", "failed", "expired", "canceled")


# ---------------------------------------------------------------------------
# Provider adapters (raw requests, consistent with forecaster/providers.py)
# ---------------------------------------------------------------------------


class _OpenAIBatch:
    provider = "openai"

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def submit(self, rows: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """rows: [(custom_id, request_body)] -> {provider_batch_id, input_file_id}."""

        jsonl = "\n".join(
            json.dumps(
                {
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                },
                ensure_ascii=False,
            )
            for cid, body in rows
        )
        up = requests.post(
            f"{self.base}/files",
            headers=self._headers(),
            files={"file": ("batch.jsonl", jsonl.encode("utf-8"), "application/jsonl")},
            data={"purpose": "batch"},
            timeout=_HTTP_TIMEOUT,
        )
        up.raise_for_status()
        input_file_id = up.json()["id"]
        created = requests.post(
            f"{self.base}/batches",
            headers={**self._headers(), "Content-Type": "application/json"},
            json={
                "input_file_id": input_file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
            timeout=_HTTP_TIMEOUT,
        )
        created.raise_for_status()
        payload = created.json()
        return {"provider_batch_id": payload["id"], "input_file_id": input_file_id}

    _STATE_MAP = {
        "validating": "in_progress",
        "in_progress": "in_progress",
        "finalizing": "in_progress",
        "completed": "ended",
        "failed": "failed",
        "expired": "expired",
        "cancelling": "in_progress",
        "cancelled": "canceled",
    }

    def poll(self, provider_batch_id: str) -> BatchStatus:
        resp = requests.get(
            f"{self.base}/batches/{provider_batch_id}",
            headers=self._headers(),
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        counts = payload.get("request_counts") or {}
        state = self._STATE_MAP.get(str(payload.get("status")), "in_progress")
        detail = json.dumps(
            {
                "output_file_id": payload.get("output_file_id"),
                "error_file_id": payload.get("error_file_id"),
            }
        )
        return BatchStatus(provider_batch_id, state, dict(counts), detail)

    def fetch(self, provider_batch_id: str) -> Iterator[Tuple[str, bool, str, Dict[str, Any], str]]:
        """Yield (custom_id, ok, text, usage, error) per result line."""

        from forecaster.providers import _openai_usage_from_payload  # noqa: PLC0415

        resp = requests.get(
            f"{self.base}/batches/{provider_batch_id}",
            headers=self._headers(),
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        for file_id in (payload.get("output_file_id"), payload.get("error_file_id")):
            if not file_id:
                continue
            content = requests.get(
                f"{self.base}/files/{file_id}/content",
                headers=self._headers(),
                timeout=_HTTP_TIMEOUT,
            )
            content.raise_for_status()
            for line in content.text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = str(item.get("custom_id") or "")
                err_obj = item.get("error")
                response = item.get("response") or {}
                body = response.get("body") or {}
                if err_obj or int(response.get("status_code") or 0) >= 400:
                    message = ""
                    if isinstance(err_obj, dict):
                        message = str(err_obj.get("message") or "")
                    if not message and isinstance(body, dict):
                        be = body.get("error")
                        if isinstance(be, dict):
                            message = str(be.get("message") or "")
                    yield cid, False, "", {}, f"OpenAI batch item error: {message or 'unknown'}"
                    continue
                text = ""
                choices = body.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message") or {}
                    if isinstance(msg, dict):
                        text = str(msg.get("content", "")).strip()
                yield cid, True, text, _openai_usage_from_payload(body), ""

    def cancel(self, provider_batch_id: str) -> None:
        try:
            requests.post(
                f"{self.base}/batches/{provider_batch_id}/cancel",
                headers=self._headers(),
                timeout=_HTTP_TIMEOUT,
            )
        except Exception:  # noqa: BLE001
            pass


class _AnthropicBatch:
    provider = "anthropic"

    def __init__(self) -> None:
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.base = "https://api.anthropic.com/v1"
        self.version = os.getenv("ANTHROPIC_API_VERSION", "2023-06-01")

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
            "content-type": "application/json",
        }

    def submit(self, rows: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        resp = requests.post(
            f"{self.base}/messages/batches",
            headers=self._headers(),
            json={"requests": [{"custom_id": cid, "params": body} for cid, body in rows]},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        return {"provider_batch_id": resp.json()["id"], "input_file_id": None}

    def poll(self, provider_batch_id: str) -> BatchStatus:
        resp = requests.get(
            f"{self.base}/messages/batches/{provider_batch_id}",
            headers=self._headers(),
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        processing = str(payload.get("processing_status") or "")
        state = "ended" if processing == "ended" else "in_progress"
        counts = payload.get("request_counts") or {}
        return BatchStatus(
            provider_batch_id,
            state,
            dict(counts),
            json.dumps({"results_url": payload.get("results_url")}),
        )

    def fetch(self, provider_batch_id: str) -> Iterator[Tuple[str, bool, str, Dict[str, Any], str]]:
        from forecaster.providers import (  # noqa: PLC0415
            _anthropic_stop_reason_error,
            _anthropic_usage_from_payload,
        )

        resp = requests.get(
            f"{self.base}/messages/batches/{provider_batch_id}",
            headers=self._headers(),
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        results_url = resp.json().get("results_url")
        if not results_url:
            return
        content = requests.get(results_url, headers=self._headers(), timeout=_HTTP_TIMEOUT)
        content.raise_for_status()
        for line in content.text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = str(item.get("custom_id") or "")
            result = item.get("result") or {}
            rtype = str(result.get("type") or "")
            if rtype != "succeeded":
                err = result.get("error") or {}
                message = ""
                if isinstance(err, dict):
                    inner = err.get("error")
                    if isinstance(inner, dict):
                        message = str(inner.get("message") or "")
                    if not message:
                        message = str(err.get("message") or "")
                yield cid, False, "", {}, f"Anthropic batch item {rtype}: {message}".strip()
                continue
            message_payload = result.get("message") or {}
            text = ""
            content_blocks = message_payload.get("content")
            if isinstance(content_blocks, list):
                parts = [
                    str(part.get("text", ""))
                    for part in content_blocks
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                text = "".join(parts).strip()
            usage = _anthropic_usage_from_payload(message_payload)
            stop_error = _anthropic_stop_reason_error(message_payload)
            if stop_error:
                yield cid, False, "", usage, stop_error
                continue
            yield cid, True, text, usage, ""

    def cancel(self, provider_batch_id: str) -> None:
        try:
            requests.post(
                f"{self.base}/messages/batches/{provider_batch_id}/cancel",
                headers=self._headers(),
                timeout=_HTTP_TIMEOUT,
            )
        except Exception:  # noqa: BLE001
            pass


class _GoogleBatch:
    """Gemini Batch API (batchGenerateContent long-running operation).

    NOTE (verification item V1 from the cost-optimization plan): the exact
    request/operation/result shapes below follow the documented public API
    but have not been exercised against the live service from this repo.
    The fetch parser is deliberately tolerant (both nested and flat inlined
    response shapes); a mismatch surfaces as a whole-batch failure, which
    the collect stage degrades to the synchronous path — never silent loss.
    Gemini batches are PER MODEL (the model is in the URL), so submit_pending
    groups Google rows by model_id.
    """

    provider = "google"

    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.base = "https://generativelanguage.googleapis.com/v1beta"

    def submit(self, rows: List[Tuple[str, Dict[str, Any]]], *, model_id: str) -> Dict[str, Any]:
        api_model = model_id.split("/", 1)[-1] if "/" in model_id else model_id
        body = {
            "batch": {
                "display_name": f"pythia-{int(time.time())}",
                "input_config": {
                    "requests": {
                        "requests": [
                            {"request": req_body, "metadata": {"key": cid}}
                            for cid, req_body in rows
                        ]
                    }
                },
            }
        }
        resp = requests.post(
            f"{self.base}/models/{api_model}:batchGenerateContent?key={self.api_key}",
            json=body,
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        return {"provider_batch_id": str(payload.get("name") or ""), "input_file_id": None}

    _STATE_MAP = {
        "BATCH_STATE_UNSPECIFIED": "in_progress",
        "BATCH_STATE_PENDING": "in_progress",
        "BATCH_STATE_RUNNING": "in_progress",
        "BATCH_STATE_SUCCEEDED": "ended",
        "BATCH_STATE_FAILED": "failed",
        "BATCH_STATE_CANCELLED": "canceled",
        "BATCH_STATE_EXPIRED": "expired",
    }

    def _get(self, provider_batch_id: str) -> Dict[str, Any]:
        name = provider_batch_id.lstrip("/")
        resp = requests.get(
            f"{self.base}/{name}?key={self.api_key}",
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def poll(self, provider_batch_id: str) -> BatchStatus:
        payload = self._get(provider_batch_id)
        meta = payload.get("metadata") or {}
        state_raw = str(meta.get("state") or "")
        state = self._STATE_MAP.get(state_raw, "in_progress")
        if payload.get("done") and state == "in_progress":
            state = "failed" if payload.get("error") else "ended"
        detail = ""
        if payload.get("error"):
            detail = json.dumps(payload["error"])[:500]
        return BatchStatus(provider_batch_id, state, {}, detail)

    @staticmethod
    def _iter_inlined(payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        response = payload.get("response") or {}
        inlined = response.get("inlinedResponses") or {}
        if isinstance(inlined, dict):
            inlined = inlined.get("inlinedResponses") or []
        if isinstance(inlined, list):
            for entry in inlined:
                if isinstance(entry, dict):
                    yield entry

    def fetch(self, provider_batch_id: str) -> Iterator[Tuple[str, bool, str, Dict[str, Any], str]]:
        from forecaster.providers import _google_usage_from_payload  # noqa: PLC0415

        payload = self._get(provider_batch_id)
        for entry in self._iter_inlined(payload):
            cid = str(((entry.get("metadata") or {}).get("key")) or "")
            if entry.get("error"):
                yield cid, False, "", {}, f"Gemini batch item error: {json.dumps(entry['error'])[:300]}"
                continue
            item_response = entry.get("response") or {}
            text = ""
            try:
                text = (
                    item_response["candidates"][0]["content"]["parts"][0].get("text", "").strip()
                )
            except Exception:  # noqa: BLE001
                text = str(item_response.get("text", "") or "")
            yield cid, True, text, _google_usage_from_payload(item_response), ""

    def cancel(self, provider_batch_id: str) -> None:
        try:
            name = provider_batch_id.lstrip("/")
            requests.post(
                f"{self.base}/{name}:cancel?key={self.api_key}",
                timeout=_HTTP_TIMEOUT,
            )
        except Exception:  # noqa: BLE001
            pass


_ADAPTERS = {
    "openai": _OpenAIBatch,
    "anthropic": _AnthropicBatch,
    "google": _GoogleBatch,
}


def _adapter(provider: str):
    cls = _ADAPTERS.get((provider or "").lower())
    if cls is None:
        raise ValueError(f"no batch adapter for provider {provider!r}")
    return cls()


# ---------------------------------------------------------------------------
# Submit / poll / collect
# ---------------------------------------------------------------------------


def _chunk_rows(
    rows: List[Tuple[str, str]],
) -> Iterator[List[Tuple[str, Dict[str, Any]]]]:
    """Chunk (custom_id, body_json) rows under request-count and byte caps."""

    chunk: List[Tuple[str, Dict[str, Any]]] = []
    chunk_bytes = 0
    for cid, body_json in rows:
        size = len(body_json.encode("utf-8"))
        if chunk and (
            len(chunk) >= _MAX_REQUESTS_PER_BATCH
            or chunk_bytes + size > _MAX_BYTES_PER_BATCH
        ):
            yield chunk
            chunk, chunk_bytes = [], 0
        chunk.append((cid, json.loads(body_json)))
        chunk_bytes += size
    if chunk:
        yield chunk


def submit_pending(
    con,
    *,
    family: str,
    pipeline_id: str,
    stage: str,
    run_id: Optional[str] = None,
    hs_run_id: Optional[str] = None,
) -> List[str]:
    """Submit all pending rows of *family* as provider batches.

    Returns the list of created ``llm_batches.batch_id``. Providers outside
    ``PYTHIA_BATCH_PROVIDERS`` keep their rows pending — the collect stage's
    get_result miss then routes those calls down the sync path.
    """

    created: List[str] = []
    providers_rows = con.execute(
        """
        SELECT provider, model_id, custom_id, request_body_json
        FROM llm_batch_requests
        WHERE family = ? AND status = 'pending'
        ORDER BY provider, model_id, custom_id
        """,
        [family],
    ).fetchall()
    if not providers_rows:
        return created

    grouped: Dict[Tuple[str, Optional[str]], List[Tuple[str, str]]] = {}
    for provider, model_id, custom_id, body_json in providers_rows:
        provider = (provider or "").lower()
        if provider not in batch_providers():
            continue
        # Gemini batches are per model (model id is in the endpoint URL);
        # OpenAI/Anthropic batches carry the model per item.
        group_key = (provider, model_id if provider == "google" else None)
        grouped.setdefault(group_key, []).append((custom_id, body_json))

    for (provider, group_model), rows in grouped.items():
        adapter = _adapter(provider)
        for chunk_idx, chunk in enumerate(_chunk_rows(rows)):
            batch_id = f"b_{pipeline_id}_{_FAMILY_SLUGS[family]}_{provider}" + (
                f"_{_sanitize_token(group_model, 20)}" if group_model else ""
            ) + f"_{chunk_idx}_{uuid.uuid4().hex[:6]}"
            try:
                if provider == "google":
                    submit_info = adapter.submit(chunk, model_id=group_model)
                else:
                    submit_info = adapter.submit(chunk)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "llm_batch: submit failed family=%s provider=%s chunk=%d: %s",
                    family, provider, chunk_idx, exc,
                )
                # Rows stay 'pending' → collect stage falls back to sync.
                continue
            con.execute(
                """
                INSERT INTO llm_batches (
                    batch_id, provider, provider_batch_id, family, run_id,
                    hs_run_id, pipeline_id, stage, model_id, status,
                    n_requests, input_file_id, submitted_at, is_test
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'submitted', ?, ?, ?, ?)
                """,
                [
                    batch_id,
                    provider,
                    submit_info["provider_batch_id"],
                    family,
                    run_id,
                    hs_run_id,
                    pipeline_id,
                    stage,
                    group_model,
                    len(chunk),
                    submit_info.get("input_file_id"),
                    _now(),
                    _is_test(),
                ],
            )
            custom_ids = [cid for cid, _ in chunk]
            placeholders = ", ".join(["?"] * len(custom_ids))
            con.execute(
                f"""
                UPDATE llm_batch_requests
                SET batch_id = ?, status = 'submitted'
                WHERE custom_id IN ({placeholders})
                """,
                [batch_id, *custom_ids],
            )
            created.append(batch_id)
            LOGGER.info(
                "llm_batch: submitted %s (%s, %d requests) as %s",
                batch_id, provider, len(chunk), submit_info["provider_batch_id"],
            )
    return created


def poll_batch(con, batch_id: str) -> Optional[BatchStatus]:
    """Poll one batch; update llm_batches.status; return the provider status."""

    row = con.execute(
        "SELECT provider, provider_batch_id, status FROM llm_batches WHERE batch_id = ?",
        [batch_id],
    ).fetchone()
    if not row:
        return None
    provider, provider_batch_id, status = row
    if status in ("ended", "collected", "failed", "expired", "canceled"):
        return BatchStatus(provider_batch_id or "", status, {})
    if not provider_batch_id:
        return BatchStatus("", status or "building", {})
    try:
        result = _adapter(provider).poll(provider_batch_id)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("llm_batch: poll failed for %s: %s", batch_id, exc)
        return None
    new_status = result.state if result.terminal else "in_progress"
    con.execute(
        "UPDATE llm_batches SET status = ?, ended_at = COALESCE(ended_at, ?) WHERE batch_id = ?",
        [new_status, _now() if result.terminal else None, batch_id],
    )
    return result


def pending_batches(con, pipeline_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Batches not yet collected — the poller's work list."""

    sql = """
        SELECT batch_id, provider, provider_batch_id, family, pipeline_id,
               stage, status, n_requests, submitted_at
        FROM llm_batches
        WHERE status IN ('submitted', 'in_progress', 'ended')
    """
    params: List[Any] = []
    if pipeline_id:
        sql += " AND pipeline_id = ?"
        params.append(pipeline_id)
    rows = con.execute(sql + " ORDER BY submitted_at", params).fetchall()
    cols = [
        "batch_id", "provider", "provider_batch_id", "family", "pipeline_id",
        "stage", "status", "n_requests", "submitted_at",
    ]
    return [dict(zip(cols, r)) for r in rows]


def collect_batch(con, batch_id: str) -> Dict[str, int]:
    """Fetch results for one batch into llm_batch_requests (idempotent).

    Items already terminal (succeeded/fallback_sync) are never overwritten,
    so re-running a collect stage is a no-op for completed work. Usage dicts
    are stamped service_tier=batch + batch ids for 50% pricing in the ledger.
    """

    row = con.execute(
        "SELECT provider, provider_batch_id, status FROM llm_batches WHERE batch_id = ?",
        [batch_id],
    ).fetchone()
    if not row:
        return {}
    provider, provider_batch_id, status = row
    counts = {"succeeded": 0, "errored": 0, "skipped_terminal": 0}
    if status in ("failed", "expired", "canceled") or not provider_batch_id:
        # Whole-batch failure: flip still-submitted items to expired so the
        # consumers' get_result miss routes them to the sync fallback.
        con.execute(
            """
            UPDATE llm_batch_requests SET status = 'expired', completed_at = ?
            WHERE batch_id = ? AND status IN ('pending', 'submitted')
            """,
            [_now(), batch_id],
        )
        return counts

    try:
        results = list(_adapter(provider).fetch(provider_batch_id))
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("llm_batch: fetch failed for %s: %s", batch_id, exc)
        return counts

    for cid, ok, text, usage, error in results:
        if not cid:
            continue
        existing = con.execute(
            "SELECT status FROM llm_batch_requests WHERE custom_id = ?", [cid]
        ).fetchone()
        if not existing:
            continue
        if existing[0] in ("succeeded", "fallback_sync"):
            counts["skipped_terminal"] += 1
            continue
        usage = dict(usage or {})
        usage["service_tier"] = "batch"
        usage["batch_id"] = batch_id
        usage["provider_batch_id"] = provider_batch_id
        con.execute(
            """
            UPDATE llm_batch_requests
            SET status = ?, response_text = ?, usage_json = ?, error_text = ?,
                completed_at = ?
            WHERE custom_id = ?
            """,
            [
                "succeeded" if ok else "errored",
                text or "",
                json.dumps(usage, ensure_ascii=False),
                error or None,
                _now(),
                cid,
            ],
        )
        counts["succeeded" if ok else "errored"] += 1

    con.execute(
        """
        UPDATE llm_batches
        SET status = 'collected', collected_at = ?, n_succeeded = ?, n_errored = ?
        WHERE batch_id = ?
        """,
        [_now(), counts["succeeded"], counts["errored"], batch_id],
    )
    return counts


def cancel_batch(con, batch_id: str) -> None:
    """Cancel a provider batch and mark it canceled locally.

    Used when a batch exceeds PYTHIA_BATCH_MAX_WAIT_H: canceling before the
    sync fallback runs avoids paying for both the late batch results AND the
    fallback calls. collect_batch afterwards flips the outstanding items to
    'expired' so consumers take the sync path.
    """

    row = con.execute(
        "SELECT provider, provider_batch_id FROM llm_batches WHERE batch_id = ?",
        [batch_id],
    ).fetchone()
    if not row:
        return
    provider, provider_batch_id = row
    if provider_batch_id:
        try:
            _adapter(provider).cancel(provider_batch_id)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("llm_batch: cancel failed for %s: %s", batch_id, exc)
    con.execute(
        "UPDATE llm_batches SET status = 'canceled', ended_at = COALESCE(ended_at, ?) WHERE batch_id = ?",
        [_now(), batch_id],
    )


def get_result(
    con,
    family: str,
    *,
    question_id: Optional[str] = None,
    model_key: Optional[str] = None,
    iso3: Optional[str] = None,
    hazard_code: Optional[str] = None,
    pass_idx: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Look up a collected batch result; None = caller should run sync.

    Returns a dict with keys text / usage / error / custom_id when the item
    reached a terminal batch outcome (succeeded OR errored — an errored item
    is returned so the caller can decide to fall back and mark it).
    """

    custom_id = make_custom_id(
        family,
        question_id=question_id,
        model_key=model_key,
        iso3=iso3,
        hazard_code=hazard_code,
        pass_idx=pass_idx,
    )
    row = con.execute(
        """
        SELECT status, response_text, usage_json, error_text
        FROM llm_batch_requests WHERE custom_id = ?
        """,
        [custom_id],
    ).fetchone()
    if not row:
        return None
    status, text, usage_json, error = row
    if status != "succeeded":
        return None
    try:
        usage = json.loads(usage_json) if usage_json else {}
    except json.JSONDecodeError:
        usage = {}
    return {"custom_id": custom_id, "text": text or "", "usage": usage, "error": error or ""}


def mark_fallback_sync(con, family: str, **key_kwargs: Any) -> None:
    """Record that a batch item was re-executed on the sync path."""

    try:
        custom_id = make_custom_id(family, **key_kwargs)
    except ValueError:
        return
    con.execute(
        """
        UPDATE llm_batch_requests SET status = 'fallback_sync', completed_at = ?
        WHERE custom_id = ? AND status <> 'succeeded'
        """,
        [_now(), custom_id],
    )


def clear_request_bodies(con, batch_id: str) -> None:
    """Null out request_body_json after a successful collect (artifact size)."""

    con.execute(
        """
        UPDATE llm_batch_requests SET request_body_json = NULL
        WHERE batch_id = ? AND status IN ('succeeded', 'fallback_sync')
        """,
        [batch_id],
    )
