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
- ``PYTHIA_OPENAI_BATCH_SUBMIT_BUDGET_MIN`` (default 20): how long the OpenAI
  submit guard keeps retrying a validation-rejected input file, shared across
  the process (see ``_OpenAIBatch.submit``).
- ``PYTHIA_BATCH_RESUBMIT_AT_COLLECT`` (default 1) / ``PYTHIA_BATCH_RESUBMIT_WAIT_MIN``
  (default 90): a collect stage re-batches every request that never got a
  batch result and waits, bounded, before the sync fallback (see
  ``resubmit_unserved``). The sync path is the fallback of last resort.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import signal
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import requests

LOGGER = logging.getLogger(__name__)

# Batch families. Forecaster families key rows by (question_id, model_key);
# HS families key rows by (iso3, hazard_code, pass_idx).
# NOTE: no "scenario" family — scenarios run sequentially after the SPD
# aggregation and are never batched; a declared-but-unused family here
# would read as coverage that does not exist.
FAMILIES = ("spd_v2", "binary_v2", "track2_spd", "hs_rc", "hs_triage")

# Compact family slugs for custom ids (Anthropic: ^[A-Za-z0-9_-]{1,64}$).
_FAMILY_SLUGS = {
    "spd_v2": "spd",
    "binary_v2": "bin",
    "track2_spd": "t2",
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


def resubmit_at_collect_enabled() -> bool:
    """``PYTHIA_BATCH_RESUBMIT_AT_COLLECT`` (default on): re-batch before the sync fallback."""

    return os.getenv("PYTHIA_BATCH_RESUBMIT_AT_COLLECT", "1").strip().lower() in ("1", "true", "yes")


def resubmit_wait_minutes() -> float:
    """``PYTHIA_BATCH_RESUBMIT_WAIT_MIN`` (default 90): how long a collect stage waits on re-batches."""

    try:
        return max(0.0, float(os.getenv("PYTHIA_BATCH_RESUBMIT_WAIT_MIN", "90") or 90))
    except ValueError:
        return 90.0


def resubmit_poll_sec() -> float:
    """``PYTHIA_BATCH_RESUBMIT_POLL_SEC`` (default 60): poll interval while waiting on re-batches."""

    try:
        return max(1.0, float(os.getenv("PYTHIA_BATCH_RESUBMIT_POLL_SEC", "60") or 60))
    except ValueError:
        return 60.0


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
    pipeline_id: Optional[str] = None,
) -> str:
    """Encode a compact, Anthropic-safe custom id.

    The id is a lookup key only — the authoritative reassembly metadata is
    the ``llm_batch_requests`` row (question_id, iso3, hazard, model_key,
    ...), which always travels with the batch in the DB artifact.

    ``pipeline_id`` scopes the id to one staged-pipeline execution: the DB
    artifact carries llm_batch_requests forward month to month, and custom_id
    is the sole replay key — without the scope segment, month 2's enqueue
    would find month 1's succeeded row and silently replay last month's
    model output (and a same-epoch forecaster re-run would do the same).
    """

    slug = _FAMILY_SLUGS.get(family)
    if slug is None:
        raise ValueError(f"unknown batch family: {family}")
    if family in ("spd_v2", "binary_v2", "track2_spd"):
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
    if pipeline_id:
        phash = hashlib.sha1(str(pipeline_id).encode("utf-8")).hexdigest()[:8]
        custom_id = f"{custom_id}-{phash}"
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
    pipeline_id: Optional[str] = None,
) -> str:
    """Insert (or refresh) one pending batch request row; returns custom_id.

    Re-enqueueing an id that already holds a terminal result (succeeded /
    fallback_sync) is a no-op so a re-run submit stage cannot clobber
    collected results. The no-clobber rule is safe ONLY because custom ids
    are pipeline-scoped (see make_custom_id) — a different pipeline gets a
    different id and therefore a fresh row.
    """

    custom_id = make_custom_id(
        family,
        question_id=question_id,
        model_key=model_key,
        iso3=iso3,
        hazard_code=hazard_code,
        pass_idx=pass_idx,
        pipeline_id=pipeline_id,
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
            pipeline_id, prompt_sha256, request_body_json, status, created_at,
            is_test
        ) VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
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
            (pipeline_id or None),
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


# OpenAI submit-time guards (see _OpenAIBatch.submit). Read per call so
# tests and operators can tune them without an import-order trap; `_sleep`
# and `_clock` are module attributes so tests can stub the waits and drive a
# fake clock.
_sleep = time.sleep
_clock = time.monotonic
_OPENAI_FILE_POLL_SEC = 2.0
_OPENAI_VALIDATE_POLL_SEC = 3.0
# Backoff between validation retries; the last rung repeats. Seconds.
_OPENAI_SUBMIT_BACKOFF_LADDER_SEC: Tuple[float, ...] = (30.0, 60.0, 120.0, 300.0)
_OPENAI_FILE_ACCESS_MARKERS = ("cannot find file", "does not have access")
# Process-shared retry budget, armed at the FIRST rejection anywhere in the
# process: however many OpenAI groups a submit stage carries, the extra wall
# time the guard can add is bounded once.
_openai_submit_budget_deadline: Optional[float] = None


def _openai_file_ready_wait_sec() -> float:
    return float(os.getenv("PYTHIA_OPENAI_BATCH_FILE_WAIT_SEC", "60") or 60)


def _openai_validate_wait_sec() -> float:
    return float(os.getenv("PYTHIA_OPENAI_BATCH_VALIDATE_WAIT_SEC", "120") or 120)


def _openai_submit_budget_sec() -> float:
    """``PYTHIA_OPENAI_BATCH_SUBMIT_BUDGET_MIN`` (default 20) in seconds."""

    try:
        return max(0.0, float(os.getenv("PYTHIA_OPENAI_BATCH_SUBMIT_BUDGET_MIN", "20") or 20)) * 60.0
    except ValueError:
        return 20.0 * 60.0


def _openai_submit_backoff_sec(retry_idx: int) -> float:
    """Rung ``retry_idx`` (1-based) of the backoff ladder; past the end, the last rung."""

    ladder = _OPENAI_SUBMIT_BACKOFF_LADDER_SEC
    idx = max(1, int(retry_idx)) - 1
    return float(ladder[min(idx, len(ladder) - 1)])


def _openai_submit_budget_remaining() -> float:
    """Seconds left in the process-wide retry budget; arms it on first call."""

    global _openai_submit_budget_deadline
    if _openai_submit_budget_deadline is None:
        _openai_submit_budget_deadline = _clock() + _openai_submit_budget_sec()
    return max(0.0, _openai_submit_budget_deadline - _clock())


def _reset_openai_submit_budget() -> None:
    """Disarm the shared budget (tests, and the canary between providers)."""

    global _openai_submit_budget_deadline
    _openai_submit_budget_deadline = None


def gh_annotation(level: str, title: str, message: str) -> None:
    """Print a GitHub Actions annotation (one line; a person reads these first).

    ``level`` is ``warning`` or ``error``. An ``::error::`` annotation does NOT
    fail the job by itself — it is a statement about what happened, and the
    stage's exit code stays a statement about whether it wrote its output.
    """

    flat = " ".join(str(message).split())
    print(f"::{level} title={title}::{flat}", flush=True)


def _openai_errors_are_file_access(errors: Any) -> bool:
    """True when OpenAI's validation errors are the input-file access rejection.

    ``errors`` is the batch object's ``errors.data`` list (or the whole
    ``errors`` dict). Only that failure is worth retrying; every other
    validation failure is deterministic.
    """

    if isinstance(errors, dict):
        errors = errors.get("data")
    if not isinstance(errors, list) or not errors:
        return False
    for item in errors:
        if not isinstance(item, dict):
            return False
        message = str(item.get("message") or "").lower()
        param = str(item.get("param") or "")
        if param != "file_id" and not any(m in message for m in _OPENAI_FILE_ACCESS_MARKERS):
            return False
    return True


class OpenAIBatchValidationError(RuntimeError):
    """OpenAI's asynchronous validator rejected a batch we had just created.

    ``file_access`` says which of the two kinds this is: the provider-side
    file-access rejection (worth retrying, and the collect stage re-batches
    it) or a deterministic failure such as ``mismatched_model`` (never
    retried). ``same_file_retry_failed`` records whether re-creating the batch
    against the SAME file was refused too — the difference between "the file
    was not visible yet" and "the organisation is being refused".
    """

    def __init__(
        self,
        message: str,
        *,
        provider_batch_id: str = "",
        input_file_id: str = "",
        errors: Any = None,
        file_access: bool = False,
        attempts: int = 1,
        same_file_retry_failed: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider_batch_id = provider_batch_id
        self.input_file_id = input_file_id
        self.errors = errors
        self.file_access = file_access
        self.attempts = attempts
        self.same_file_retry_failed = same_file_retry_failed


class _OpenAIBatch:
    provider = "openai"

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def submit(self, rows: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """rows: [(custom_id, request_body)] -> {provider_batch_id, input_file_id}.

        Upload the JSONL, wait for the file to be readable, create the batch,
        and then WATCH ITS VALIDATION. On 2026-09-01 every one of the four
        OpenAI batches was accepted at creation and rejected seconds later by
        OpenAI's asynchronous validator with ``invalid_request: Cannot find
        file file-..., or organization ... does not have access to it`` — the
        file we had just uploaded, under the same key, already reporting
        ``status=processed``. Nothing in this adapter saw that: the batch was
        persisted as ``submitted``, the poller found it ``failed`` with zero
        counts two minutes later, and all 210 requests expired into
        synchronous full-price calls (~$10.80, a fifth of the run).

        This is NOT eventual consistency that a re-upload cures in seconds.
        It is a provider-side incident that hit many organisations from
        19 August 2026 and spiked on 1 September: batches.create fails
        validation for HOURS at a time while files.retrieve says processed,
        on /v1/chat/completions as well as /v1/responses, and OpenAI applies
        a per-organisation mitigation on request. So the guard is patient
        rather than quick: a shared time budget (``PYTHIA_OPENAI_BATCH_SUBMIT_
        BUDGET_MIN``, default 20) and a backoff ladder, with the FIRST retry
        re-creating the batch against the SAME file (which tells "not visible
        yet" from "refused") and later retries re-uploading. Every rejection
        is a ``::warning`` annotation and giving up is a ``::error``; the rows
        then stay pending, which the collect stage re-batches before it falls
        back to sync (see ``resubmit_unserved``).
        """

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
        input_file_id = self._upload_jsonl(jsonl)
        attempt = 0
        same_file_retry_failed = False
        budget_min = _openai_submit_budget_sec() / 60.0
        while True:
            attempt += 1
            provider_batch_id = self._create_batch(input_file_id)
            verdict, errors = self._confirm_validation(provider_batch_id)
            if verdict != "failed":
                return {"provider_batch_id": provider_batch_id, "input_file_id": input_file_id}
            if not _openai_errors_are_file_access(errors):
                # A validation failure that is NOT the file-access rejection
                # (mismatched_model, a malformed line, ...) will fail the same
                # way on every retry: surface it instead of retrying.
                raise OpenAIBatchValidationError(
                    f"OpenAI batch {provider_batch_id} failed validation: "
                    f"{json.dumps(errors)[:600]}",
                    provider_batch_id=provider_batch_id,
                    input_file_id=input_file_id,
                    errors=errors,
                    file_access=False,
                    attempts=attempt,
                )
            if attempt == 2:
                same_file_retry_failed = True
            wait = _openai_submit_backoff_sec(attempt)
            remaining = _openai_submit_budget_remaining()
            how = "first upload" if attempt == 1 else ("same file" if attempt == 2 else "fresh upload")
            LOGGER.warning(
                "llm_batch: OpenAI batch %s rejected input file %s at validation "
                "(attempt %d, %s): %s",
                provider_batch_id, input_file_id, attempt, how, json.dumps(errors)[:400],
            )
            gh_annotation(
                "warning",
                "OpenAI batch validation rejected input file",
                f"{provider_batch_id} rejected {input_file_id} (attempt {attempt}, {how}); "
                f"{remaining / 60.0:.1f} of {budget_min:.0f} min submit budget left; "
                f"retry in {wait:.0f}s: {json.dumps(errors)[:300]}",
            )
            if remaining < wait:
                why = (
                    "same-file retry also failed (org-level rejection, not a file race)"
                    if same_file_retry_failed
                    else "same-file retry not reached"
                )
                gh_annotation(
                    "error",
                    "OpenAI batch submit gave up",
                    f"validation kept rejecting the input file after {attempt} attempt(s) "
                    f"over the {budget_min:.0f} min budget; {why}; rows stay pending — the "
                    f"collect stage re-batches them, then falls back to sync at full price. "
                    f"Last: {json.dumps(errors)[:300]}",
                )
                raise OpenAIBatchValidationError(
                    f"OpenAI batch validation kept rejecting the uploaded input file after "
                    f"{attempt} attempt(s): {json.dumps(errors)[:600]}",
                    provider_batch_id=provider_batch_id,
                    input_file_id=input_file_id,
                    errors=errors,
                    file_access=True,
                    attempts=attempt,
                    same_file_retry_failed=same_file_retry_failed,
                )
            _sleep(wait)
            if attempt >= 2:
                input_file_id = self._upload_jsonl(jsonl)

    def _upload_jsonl(self, jsonl: str) -> str:
        """POST the JSONL to /files (purpose=batch) and wait until it is readable."""

        up = requests.post(
            f"{self.base}/files",
            headers=self._headers(),
            files={"file": ("batch.jsonl", jsonl.encode("utf-8"), "application/jsonl")},
            data={"purpose": "batch"},
            timeout=_HTTP_TIMEOUT,
        )
        up.raise_for_status()
        input_file_id = str(up.json()["id"])
        self._wait_for_file(input_file_id)
        return input_file_id

    def _create_batch(self, input_file_id: str) -> str:
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
        return str(created.json()["id"])

    def _wait_for_file(self, file_id: str) -> None:
        """Block until the uploaded file reports ``status=processed`` (bounded).

        The Files API returns an id before the file is readable by the batch
        validator; the ``status`` field (``uploaded`` -> ``processed``) is
        the documented way to know. Missing field or timeout: proceed —
        ``_confirm_validation`` is the second line of defence.
        """

        deadline = time.monotonic() + _openai_file_ready_wait_sec()
        while True:
            try:
                meta = requests.get(
                    f"{self.base}/files/{file_id}",
                    headers=self._headers(),
                    timeout=_HTTP_TIMEOUT,
                )
                if meta.ok:
                    status = str((meta.json() or {}).get("status") or "")
                    if not status or status == "processed":
                        return
                    if status == "error":
                        LOGGER.warning("llm_batch: OpenAI file %s reports status=error", file_id)
                        return
            except Exception as exc:  # noqa: BLE001 - the validator is the real check
                LOGGER.debug("llm_batch: file status probe failed for %s: %s", file_id, exc)
                return
            if time.monotonic() >= deadline:
                LOGGER.warning(
                    "llm_batch: OpenAI file %s not 'processed' after %.0fs; creating the batch anyway",
                    file_id, _openai_file_ready_wait_sec(),
                )
                return
            _sleep(_OPENAI_FILE_POLL_SEC)

    def _confirm_validation(self, provider_batch_id: str) -> Tuple[str, Any]:
        """Poll a freshly created batch until it leaves ``validating`` (bounded).

        Returns ``(verdict, errors)`` where verdict is ``failed`` (validation
        rejected the batch — ``errors`` carries OpenAI's ``errors.data``),
        ``accepted`` (it moved on to in_progress/finalizing/completed) or
        ``unknown`` (still validating at the deadline, or the probe failed;
        the poller takes it from here exactly as before this guard existed).
        """

        deadline = time.monotonic() + _openai_validate_wait_sec()
        while True:
            try:
                resp = requests.get(
                    f"{self.base}/batches/{provider_batch_id}",
                    headers=self._headers(),
                    timeout=_HTTP_TIMEOUT,
                )
                resp.raise_for_status()
                payload = resp.json() or {}
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("llm_batch: validation probe failed for %s: %s", provider_batch_id, exc)
                return "unknown", None
            status = str(payload.get("status") or "")
            if status == "failed":
                errors = payload.get("errors")
                if isinstance(errors, dict):
                    errors = errors.get("data")
                return "failed", errors
            if status and status != "validating":
                return "accepted", None
            if time.monotonic() >= deadline:
                return "unknown", None
            _sleep(_OPENAI_VALIDATE_POLL_SEC)

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
        # `errors` is the ONLY place an input-validation failure is reported:
        # when OpenAI ingests zero requests there is no error_file_id and
        # request_counts is all zeros, so recording just the two file ids —
        # as this did on 2026-07-29 and 2026-07-30 — throws away the reason
        # and leaves "0 of 8 results" permanently unexplained.
        detail_obj: Dict[str, Any] = {
            "output_file_id": payload.get("output_file_id"),
            "error_file_id": payload.get("error_file_id"),
        }
        errors = payload.get("errors")
        if errors:
            detail_obj["errors"] = errors
        detail = json.dumps(detail_obj)
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


def expire_and_purge_stale(
    con,
    *,
    current_pipeline_id: str,
    keep_days: int = 45,
) -> Dict[str, int]:
    """Neutralize batch-request rows left behind by OTHER pipelines.

    1. Stale in-flight rows (pending/submitted, not this pipeline, older
       than the max batch wait) are flipped to 'expired' so nothing can
       ever submit or replay them.
    2. Terminal rows older than *keep_days* from other pipelines are
       deleted outright — the canonical DB artifact travels between runs
       and this table would otherwise grow without bound.

    Rows with pipeline_id NULL are pre-scoping legacy rows and are treated
    as foreign (their in-flight collection is still honored via
    get_result's llm_batches JOIN fallback, which reads terminal rows).
    """

    counts = {"expired": 0, "purged": 0}
    stale_cutoff = _now() - timedelta(hours=max_wait_hours())
    purge_cutoff = _now() - timedelta(days=keep_days)
    cur = con.execute(
        """
        UPDATE llm_batch_requests SET status = 'expired', completed_at = ?
        WHERE status IN ('pending', 'submitted')
          AND (pipeline_id IS NULL OR pipeline_id <> ?)
          AND created_at < ?
        """,
        [_now(), current_pipeline_id, stale_cutoff],
    )
    try:
        counts["expired"] = int(cur.fetchall()[0][0]) if cur else 0
    except Exception:
        pass
    cur = con.execute(
        """
        DELETE FROM llm_batch_requests
        WHERE (pipeline_id IS NULL OR pipeline_id <> ?)
          AND created_at < ?
        """,
        [current_pipeline_id, purge_cutoff],
    )
    try:
        counts["purged"] = int(cur.fetchall()[0][0]) if cur else 0
    except Exception:
        pass
    if counts["expired"] or counts["purged"]:
        LOGGER.info(
            "llm_batch: stale-row sweep expired=%d purged=%d (pipeline=%s)",
            counts["expired"], counts["purged"], current_pipeline_id,
        )
    return counts


@dataclass
class SubmitReport:
    """What one ``submit_pending`` call did, group by group.

    ``n_groups`` is the number of provider batches the pending rows WANTED
    (one per (provider, model_id) group, times chunks); ``created`` names the
    ones that exist. A stage that prints only the second number reads
    "submitted 3 provider batch(es)" as success when it wanted 5.
    """

    family: str
    created: List[str] = field(default_factory=list)
    n_groups: int = 0
    failed: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def n_created(self) -> int:
        return len(self.created)

    def summary_line(self) -> str:
        line = f"[batch] {self.family}: {self.n_created} of {self.n_groups} provider batch(es) submitted"
        if self.failed:
            names = ", ".join(
                f"{f['provider']}/{f.get('model_id') or '?'} ({f.get('error_class')})" for f in self.failed
            )
            line += f"; {len(self.failed)} FAILED — {names} — rows stay pending -> sync fallback"
        return line


def _submit_error_class(exc: BaseException) -> str:
    if isinstance(exc, OpenAIBatchValidationError):
        return "validation:file_access" if exc.file_access else "validation:other"
    return type(exc).__name__


def submit_pending_report(
    con,
    *,
    family: str,
    pipeline_id: str,
    stage: str,
    run_id: Optional[str] = None,
    hs_run_id: Optional[str] = None,
) -> SubmitReport:
    """Submit all pending rows of *family* for THIS pipeline as provider batches.

    Returns a :class:`SubmitReport`. Providers outside ``PYTHIA_BATCH_PROVIDERS``
    keep their rows pending — the collect stage's get_result miss then routes
    those calls down the sync path.

    Only rows enqueued under *pipeline_id* are submitted: the DB artifact
    carries llm_batch_requests forward between pipelines, and sweeping a
    stale pipeline's leftover pending rows into a fresh batch would bill
    last month's prompts again.

    A group whose submit raises is recorded in ``report.failed``, announced
    with a ``::warning`` annotation, and its rows stay ``pending``: the stage
    never depends on batch-submission success (the collect stage re-batches
    what is still pending, then falls back to sync).
    """

    report = SubmitReport(family=family)
    expire_and_purge_stale(con, current_pipeline_id=pipeline_id)
    providers_rows = con.execute(
        """
        SELECT provider, model_id, custom_id, request_body_json
        FROM llm_batch_requests
        WHERE family = ? AND status = 'pending' AND pipeline_id = ?
        ORDER BY provider, model_id, custom_id
        """,
        [family, pipeline_id],
    ).fetchall()
    if not providers_rows:
        return report

    grouped: Dict[Tuple[str, Optional[str]], List[Tuple[str, str]]] = {}
    for provider, model_id, custom_id, body_json in providers_rows:
        provider = (provider or "").lower()
        if provider not in batch_providers():
            continue
        # ALWAYS group per model. Gemini needs it because the model id is in
        # the endpoint URL, and OpenAI *requires* it: a batch containing more
        # than one model is rejected wholesale at validation with
        # `mismatched_model`, which surfaces as request_counts={total: 0} and no
        # error file. That is exactly what happened on 2026-07-29 and
        # 2026-07-30 — the ensemble's two OpenAI members (gpt-5.6-sol and
        # gpt-5.6-luna) landed in one spd_v2 batch, every request fell back to
        # synchronous full price, and it cost ~2/3 of forecaster spend twice.
        # Anthropic does permit mixed models, but grouping it per model too is
        # a no-op while there is one Anthropic member and removes the trap for
        # whoever adds a second.
        group_key = (provider, model_id)
        grouped.setdefault(group_key, []).append((custom_id, body_json))

    for (provider, group_model), rows in grouped.items():
        chunks = list(_chunk_rows(rows))
        report.n_groups += len(chunks)
        try:
            adapter = _adapter(provider)
        except Exception as exc:  # noqa: BLE001 - an unknown provider is not a reason to abort the family
            for chunk_idx, chunk in enumerate(chunks):
                _record_submit_failure(report, provider, group_model, chunk_idx, chunk, exc)
            continue
        for chunk_idx, chunk in enumerate(chunks):
            batch_id = f"b_{pipeline_id}_{_FAMILY_SLUGS[family]}_{provider}" + (
                f"_{_sanitize_token(group_model, 20)}" if group_model else ""
            ) + f"_{chunk_idx}_{uuid.uuid4().hex[:6]}"
            try:
                if provider == "google":
                    submit_info = adapter.submit(chunk, model_id=group_model)
                else:
                    submit_info = adapter.submit(chunk)
            except Exception as exc:  # noqa: BLE001
                # Rows stay 'pending' → collect stage re-batches, then falls back to sync.
                _record_submit_failure(report, provider, group_model, chunk_idx, chunk, exc)
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
            report.created.append(batch_id)
            LOGGER.info(
                "llm_batch: submitted %s (%s, %d requests) as %s",
                batch_id, provider, len(chunk), submit_info["provider_batch_id"],
            )
    return report


def _record_submit_failure(
    report: SubmitReport,
    provider: str,
    group_model: Optional[str],
    chunk_idx: int,
    chunk: Sequence[Tuple[str, Any]],
    exc: BaseException,
) -> None:
    error_class = _submit_error_class(exc)
    LOGGER.error(
        "llm_batch: submit failed family=%s provider=%s model=%s chunk=%d (%s): %s",
        report.family, provider, group_model, chunk_idx, error_class, exc,
    )
    report.failed.append(
        {
            "provider": provider,
            "model_id": group_model,
            "chunk_idx": chunk_idx,
            "n_requests": len(chunk),
            "error": str(exc)[:600],
            "error_class": error_class,
        }
    )
    gh_annotation(
        "warning",
        "Batch submit failed",
        f"{report.family} {provider}/{group_model or '?'} chunk {chunk_idx} "
        f"({len(chunk)} requests) [{error_class}]: {str(exc)[:300]} — rows stay pending; "
        "the collect stage re-batches them, then falls back to sync at full price",
    )


def submit_pending(
    con,
    *,
    family: str,
    pipeline_id: str,
    stage: str,
    run_id: Optional[str] = None,
    hs_run_id: Optional[str] = None,
) -> List[str]:
    """Compatibility wrapper over :func:`submit_pending_report`: the created batch ids."""

    return submit_pending_report(
        con,
        family=family,
        pipeline_id=pipeline_id,
        stage=stage,
        run_id=run_id,
        hs_run_id=hs_run_id,
    ).created


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
    # first_polled_at is COALESCEd, so it records the FIRST time anything
    # heard back about this batch. The gap from submitted_at is the poller's
    # ignition latency, which is what turned a 5-minute hand-off into hours
    # in August 2026 — and it was measurable nowhere.
    con.execute(
        """
        UPDATE llm_batches
        SET status = ?,
            first_polled_at = COALESCE(first_polled_at, ?),
            ended_at = COALESCE(ended_at, ?)
        WHERE batch_id = ?
        """,
        [new_status, _now(), _now() if result.terminal else None, batch_id],
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


def _record_empty_batch(
    con, batch_id: str, provider: str, provider_batch_id: str, n_expired: int
) -> None:
    """Persist why a batch yielded no results, and say so loudly.

    DIAGNOSTIC ONLY — every failure path here is swallowed. The caller is
    mid-collect and the sync fallback still has to run; losing the explanation
    is bad, but losing the forecasts would be worse.
    """

    detail: Dict[str, Any] = {"provider_batch_id": provider_batch_id, "n_expired": n_expired}
    try:
        status = _adapter(provider).poll(provider_batch_id)
        detail["provider_state"] = status.state
        detail["counts"] = status.counts
        detail["detail"] = status.detail
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        detail["poll_error"] = f"{type(exc).__name__}: {exc}"

    text = json.dumps(detail, default=str)[:4000]
    try:
        con.execute(
            "UPDATE llm_batches SET error_text = ? WHERE batch_id = ?", [text, batch_id]
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        LOGGER.warning("llm_batch: could not persist empty-batch detail for %s: %s", batch_id, exc)

    LOGGER.error("llm_batch: batch %s returned NO results — %s", batch_id, text)
    print(
        f"::warning title=Batch returned no results::{provider} batch {batch_id} yielded 0 "
        f"results; {n_expired} request(s) fall back to synchronous full-price calls. {text}"
    )


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
    salvage = status in ("failed", "expired", "canceled")
    if not provider_batch_id:
        con.execute(
            """
            UPDATE llm_batch_requests SET status = 'expired', completed_at = ?
            WHERE batch_id = ? AND status IN ('pending', 'submitted')
            """,
            [_now(), batch_id],
        )
        return counts

    # Even for failed/expired/canceled batches, attempt the fetch first:
    # OpenAI and Anthropic both preserve the items completed BEFORE the
    # terminal event, and those results were paid for — discarding them
    # means paying a second time on the sync fallback (the exact double
    # charge cancel_batch exists to avoid).
    try:
        results = list(_adapter(provider).fetch(provider_batch_id))
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("llm_batch: fetch failed for %s: %s", batch_id, exc)
        if salvage:
            con.execute(
                """
                UPDATE llm_batch_requests SET status = 'expired', completed_at = ?
                WHERE batch_id = ? AND status IN ('pending', 'submitted')
                """,
                [_now(), batch_id],
            )
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

    # Items the provider did not return (partial output, or the untouched
    # remainder of a salvaged batch) flip to expired so every row reaches a
    # terminal state and the consumer takes the sync fallback.
    cur = con.execute(
        """
        UPDATE llm_batch_requests SET status = 'expired', completed_at = ?
        WHERE batch_id = ? AND status IN ('pending', 'submitted')
        """,
        [_now(), batch_id],
    )
    n_expired = 0
    try:
        n_expired = int(cur.fetchall()[0][0]) if cur else 0
    except Exception:  # noqa: BLE001
        pass
    counts["expired"] = n_expired

    if not (counts["succeeded"] or counts["errored"] or n_expired) and counts["skipped_terminal"]:
        # Pure re-collect of an already-collected batch (poller double-fire):
        # keep the original telemetry counters instead of zeroing them.
        return counts

    # A batch that produced nothing is the expensive silent failure: every
    # item falls back to sync at full price and the run still looks healthy.
    # It happened on 2026-07-29 — both OpenAI batches returned zero results
    # (succeeded=0, errored=0, expired=12/4), which cost the discount on 67%
    # of forecaster spend, and the provider's own terminal status and error
    # payload had been thrown away, so the cause was unrecoverable. Ask the
    # provider once more and keep the answer.
    if counts["succeeded"] == 0 and n_expired > 0:
        _record_empty_batch(con, batch_id, provider, provider_batch_id, n_expired)

    if salvage:
        # Keep the terminal status (canceled/failed/expired) — record what
        # was salvaged without pretending the batch completed normally.
        con.execute(
            """
            UPDATE llm_batches
            SET collected_at = ?, n_succeeded = ?, n_errored = ?, n_expired = ?
            WHERE batch_id = ?
            """,
            [_now(), counts["succeeded"], counts["errored"], n_expired, batch_id],
        )
    else:
        con.execute(
            """
            UPDATE llm_batches
            SET status = 'collected', collected_at = ?, n_succeeded = ?, n_errored = ?, n_expired = ?
            WHERE batch_id = ?
            """,
            [_now(), counts["succeeded"], counts["errored"], n_expired, batch_id],
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


# ---------------------------------------------------------------------------
# Collect-stage re-batch (before the per-item sync fallback)
# ---------------------------------------------------------------------------

# Failure classes a collect stage is allowed to re-batch. A `validation_other`
# failure (mismatched_model, a malformed line) is deterministic and would fail
# again; it is left to the sync path.
_RESUBMIT_CLASSES = ("file_access", "failed_empty")


def _batch_failure_class(error_text: Optional[str]) -> Optional[str]:
    """Classify the JSON ``_record_empty_batch`` writes into ``llm_batches.error_text``.

    Returns ``file_access`` (OpenAI's input-file rejection), ``validation_other``
    (any other stated validation error), ``failed_empty`` (the provider says
    failed and names no error), or None when the text does not describe a
    failed batch at all.
    """

    if not error_text:
        return None
    try:
        info = json.loads(error_text)
    except (TypeError, ValueError):
        return None
    if not isinstance(info, dict):
        return None
    if str(info.get("provider_state") or "") != "failed":
        return None
    detail_raw = info.get("detail")
    detail: Any = detail_raw
    if isinstance(detail_raw, str):
        try:
            detail = json.loads(detail_raw)
        except (TypeError, ValueError):
            detail = {}
    errors = detail.get("errors") if isinstance(detail, dict) else None
    if isinstance(errors, dict):
        errors = errors.get("data")
    if not errors:
        return "failed_empty"
    if _openai_errors_are_file_access(errors):
        return "file_access"
    return "validation_other"


def unserved_requests(
    con, *, pipeline_id: str, families: Sequence[str]
) -> List[Dict[str, Any]]:
    """Rows of THIS pipeline that never got a batch result and can be re-batched.

    Two kinds: rows still ``pending`` (their submit failed at the submit
    stage, or the provider was excluded then and is included now) and rows
    ``expired`` under a batch that yielded nothing and whose recorded
    provider state (``error_text``, written by ``_record_empty_batch``) is a
    validation failure of a re-batchable class. The recorded state decides,
    not ``llm_batches.status``: whether the poller or the collect stage was
    the first to see the failure changes the latter and not the fact. Rows whose provider is outside ``PYTHIA_BATCH_PROVIDERS`` and rows
    whose request body has already been cleared are never candidates.
    """

    if not families:
        return []
    placeholders = ", ".join(["?"] * len(families))
    rows = con.execute(
        f"""
        SELECT r.custom_id, r.provider, r.model_id, r.family, r.status, r.batch_id,
               r.created_at, b.status, b.error_text, b.submitted_at, b.n_succeeded
        FROM llm_batch_requests r
        LEFT JOIN llm_batches b ON r.batch_id = b.batch_id
        WHERE r.pipeline_id = ?
          AND r.family IN ({placeholders})
          AND r.request_body_json IS NOT NULL
          AND (
                r.status = 'pending'
             OR (r.status = 'expired' AND COALESCE(b.n_succeeded, 0) = 0 AND b.error_text IS NOT NULL)
          )
        ORDER BY r.provider, r.model_id, r.custom_id
        """,
        [pipeline_id, *families],
    ).fetchall()
    allowed = batch_providers()
    out: List[Dict[str, Any]] = []
    for (cid, provider, model_id, family, status, batch_id, created_at,
         b_status, b_error, b_submitted, _n_ok) in rows:
        provider = (provider or "").lower()
        if provider not in allowed:
            continue
        if status == "pending":
            reason = "never_submitted"
        else:
            klass = _batch_failure_class(b_error)
            if klass not in _RESUBMIT_CLASSES:
                continue
            reason = f"batch_failed:{klass}"
        out.append(
            {
                "custom_id": cid,
                "provider": provider,
                "model_id": model_id,
                "family": family,
                "status": status,
                "batch_id": batch_id,
                "reason": reason,
                "original_at": b_submitted or created_at,
            }
        )
    return out


def reset_for_resubmit(con, custom_ids: Sequence[str]) -> int:
    """Return pending/expired rows to ``pending`` with no batch, keeping their bodies."""

    n = 0
    ids = list(custom_ids)
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        placeholders = ", ".join(["?"] * len(chunk))
        cur = con.execute(
            f"""
            UPDATE llm_batch_requests
            SET status = 'pending', batch_id = NULL, completed_at = NULL, error_text = NULL
            WHERE custom_id IN ({placeholders})
              AND status IN ('pending', 'expired')
              AND request_body_json IS NOT NULL
            """,
            chunk,
        )
        try:
            n += int(cur.fetchall()[0][0]) if cur else 0
        except Exception:  # noqa: BLE001
            pass
    return n


def _annotate_resubmitted(con, old_batch_id: str, new_batch_ids: Sequence[str]) -> None:
    """Stamp ``resubmitted_as`` into the old failed batch's error_text JSON (no schema change)."""

    if not old_batch_id or not new_batch_ids:
        return
    try:
        row = con.execute(
            "SELECT error_text FROM llm_batches WHERE batch_id = ?", [old_batch_id]
        ).fetchone()
        info: Dict[str, Any] = {}
        if row and row[0]:
            try:
                parsed = json.loads(row[0])
                if isinstance(parsed, dict):
                    info = parsed
            except (TypeError, ValueError):
                info = {"raw_error_text": str(row[0])[:2000]}
        existing = info.get("resubmitted_as")
        merged = sorted(set((existing if isinstance(existing, list) else []) + list(new_batch_ids)))
        info["resubmitted_as"] = merged
        con.execute(
            "UPDATE llm_batches SET error_text = ? WHERE batch_id = ?",
            [json.dumps(info, default=str)[:4000], old_batch_id],
        )
    except Exception as exc:  # noqa: BLE001 - bookkeeping only
        LOGGER.warning("llm_batch: could not annotate %s as resubmitted: %s", old_batch_id, exc)


def wait_for_batches(
    con, batch_ids: Sequence[str], *, deadline: float, poll_sec: float
) -> Dict[str, str]:
    """Poll each batch until every one is terminal or ``_clock()`` reaches ``deadline``.

    Returns ``{batch_id: last state}``; a failed poll reads ``poll_error`` and
    counts as non-terminal.
    """

    states: Dict[str, str] = {bid: "submitted" for bid in batch_ids}
    while True:
        for bid in batch_ids:
            if states[bid] in ("ended", "failed", "expired", "canceled"):
                continue
            status = poll_batch(con, bid)
            states[bid] = status.state if status else "poll_error"
        if all(s in ("ended", "failed", "expired", "canceled") for s in states.values()):
            return states
        if _clock() >= deadline:
            return states
        _sleep(max(1.0, min(poll_sec, deadline - _clock())))


def resubmit_unserved(
    con,
    *,
    pipeline_id: str,
    families: Sequence[str],
    stage: str,
    run_id: Optional[str] = None,
    hs_run_id: Optional[str] = None,
    wait_min: Optional[float] = None,
    poll_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Re-batch what never got a batch result, and wait for it, before the sync fallback.

    On 2026-09-01 the collect stage found four OpenAI batches rejected at
    validation and had exactly one move for their 210 requests: the
    synchronous call at full price, ~$10.80 of discount gone. The request
    bodies were still in the table and the wait ceiling had 23 hours left.
    This is the other move: reset the unserved rows to pending, submit them
    again (through the patient guard), wait in-process up to
    ``PYTHIA_BATCH_RESUBMIT_WAIT_MIN`` — never past ``PYTHIA_BATCH_MAX_WAIT_H``
    from the ORIGINAL submit — collect what finished, and cancel-and-salvage
    the rest so the sync path below never pays twice for one item.

    NEVER raises: every failure path is annotated and returned, because the
    caller is the collect stage and losing a forecast is worse than losing a
    discount. A SIGTERM/SIGINT while waiting cancels the batches this call
    created — a re-dispatched collect stage starts from the ORIGINAL staged
    DB and would otherwise re-batch on top of paid-for, forgotten work.
    """

    report: Dict[str, Any] = {
        "enabled": resubmit_at_collect_enabled(),
        "stage": stage,
        "n_candidates": 0,
        "n_reset": 0,
        "created": [],
        "states": {},
        "counts": {},
    }
    if not report["enabled"]:
        return report
    created: List[str] = []
    try:
        candidates = unserved_requests(con, pipeline_id=pipeline_id, families=families)
        report["n_candidates"] = len(candidates)
        if not candidates:
            return report
        reasons: Dict[str, int] = {}
        for c in candidates:
            reasons[c["reason"]] = reasons.get(c["reason"], 0) + 1
        report["reasons"] = reasons

        wait_sec = (resubmit_wait_minutes() if wait_min is None else float(wait_min)) * 60.0
        originals = [c["original_at"] for c in candidates if c.get("original_at")]
        if originals:
            oldest = min(originals)
            try:
                age_sec = (_now() - oldest).total_seconds()
            except Exception:  # noqa: BLE001
                age_sec = 0.0
            wait_sec = min(wait_sec, max_wait_hours() * 3600.0 - age_sec)
        if wait_sec <= 0:
            gh_annotation(
                "warning",
                "Collect re-batch skipped",
                f"{len(candidates)} unserved request(s) in {pipeline_id} are past "
                f"PYTHIA_BATCH_MAX_WAIT_H ({max_wait_hours():g}h) from their original submit; "
                "they take the sync fallback",
            )
            report["skipped"] = "past_max_wait"
            return report

        old_batches: Dict[str, List[str]] = {}
        for c in candidates:
            if c.get("batch_id"):
                old_batches.setdefault(c["batch_id"], []).append(c["custom_id"])
        report["n_reset"] = reset_for_resubmit(con, [c["custom_id"] for c in candidates])

        for family in families:
            sub = submit_pending_report(
                con,
                family=family,
                pipeline_id=pipeline_id,
                stage=stage,
                run_id=run_id,
                hs_run_id=hs_run_id,
            )
            if sub.n_groups:
                print(sub.summary_line(), flush=True)
            created.extend(sub.created)
        report["created"] = list(created)

        if old_batches:
            for old_id, cids in old_batches.items():
                placeholders = ", ".join(["?"] * len(cids))
                new_ids = [
                    r[0]
                    for r in con.execute(
                        f"SELECT DISTINCT batch_id FROM llm_batch_requests "
                        f"WHERE custom_id IN ({placeholders}) AND batch_id IS NOT NULL",
                        cids,
                    ).fetchall()
                ]
                _annotate_resubmitted(con, old_id, new_ids)

        if not created:
            gh_annotation(
                "warning",
                "Collect re-batch created nothing",
                f"{len(candidates)} unserved request(s) in {pipeline_id} could not be "
                "re-batched; they take the sync fallback at full price",
            )
            return report

        deadline = _clock() + wait_sec
        print(
            f"[batch] collect re-batch: {report['n_reset']} request(s) re-submitted as "
            f"{len(created)} batch(es); waiting up to {wait_sec / 60.0:.0f} min",
            flush=True,
        )
        with _cancel_on_signal(con, created):
            states = wait_for_batches(
                con, created, deadline=deadline,
                poll_sec=resubmit_poll_sec() if poll_sec is None else float(poll_sec),
            )
        report["states"] = dict(states)
        totals = {"succeeded": 0, "errored": 0, "expired": 0}
        for bid in created:
            if states.get(bid) not in ("ended", "failed", "expired", "canceled"):
                gh_annotation(
                    "warning",
                    "Re-batched batch still running at deadline",
                    f"{bid} is {states.get(bid)} after {wait_sec / 60.0:.0f} min; "
                    "canceling and salvaging completed items; the rest take the sync fallback",
                )
                cancel_batch(con, bid)
            counts = collect_batch(con, bid) or {}
            for k in totals:
                totals[k] += int(counts.get(k, 0) or 0)
        report["counts"] = totals
        print(
            f"[batch] collect re-batch: succeeded={totals['succeeded']} "
            f"errored={totals['errored']} expired={totals['expired']} (expired -> sync fallback)",
            flush=True,
        )
        return report
    except Exception as exc:  # noqa: BLE001 - the collect stage must go on
        LOGGER.exception("llm_batch: collect re-batch failed: %s", exc)
        gh_annotation(
            "warning",
            "Collect re-batch failed",
            f"{type(exc).__name__}: {str(exc)[:300]} — unserved requests take the sync fallback",
        )
        report["error"] = f"{type(exc).__name__}: {exc}"
        _cancel_and_collect_quietly(con, created)
        return report


def _cancel_and_collect_quietly(con, batch_ids: Sequence[str]) -> None:
    for bid in batch_ids:
        try:
            status = poll_batch(con, bid)
            if not (status and status.terminal):
                cancel_batch(con, bid)
            collect_batch(con, bid)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("llm_batch: cleanup of re-batched %s failed: %s", bid, exc)


class _cancel_on_signal:
    """Context manager: on SIGTERM/SIGINT cancel the given batches, then re-raise.

    Installed only in the main thread (signal handlers cannot be set
    elsewhere); anywhere else it is a no-op.
    """

    def __init__(self, con, batch_ids: Sequence[str]) -> None:
        self._con = con
        self._ids = list(batch_ids)
        self._previous: Dict[int, Any] = {}

    def _handler(self, signum, frame):  # noqa: ANN001
        gh_annotation(
            "warning",
            "Collect re-batch interrupted",
            f"signal {signum} while waiting on {len(self._ids)} re-batched batch(es); canceling them",
        )
        for bid in self._ids:
            try:
                cancel_batch(self._con, bid)
            except Exception:  # noqa: BLE001
                pass
        previous = self._previous.get(signum)
        if callable(previous) and previous not in (signal.SIG_IGN, signal.SIG_DFL):
            previous(signum, frame)
        else:
            raise KeyboardInterrupt(f"signal {signum}")

    def __enter__(self):
        try:
            import threading  # noqa: PLC0415

            if threading.current_thread() is not threading.main_thread():
                return self
            for sig in (signal.SIGTERM, signal.SIGINT):
                self._previous[sig] = signal.signal(sig, self._handler)
        except Exception:  # noqa: BLE001 - never let the guard break the wait
            self._previous = {}
        return self

    def __exit__(self, exc_type, exc, tb):
        for sig, prev in self._previous.items():
            try:
                signal.signal(sig, prev)
            except Exception:  # noqa: BLE001
                pass
        return False


def get_result(
    con,
    family: str,
    *,
    question_id: Optional[str] = None,
    model_key: Optional[str] = None,
    iso3: Optional[str] = None,
    hazard_code: Optional[str] = None,
    pass_idx: Optional[int] = None,
    pipeline_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Look up a collected batch result; None = caller should run sync.

    Returns a dict with keys status / text / usage / error / custom_id for
    an item that reached a terminal batch outcome (``status`` is
    "succeeded" or "errored"). An errored item is returned WITH its stored
    usage so the caller can cost the burned tokens (e.g. an Anthropic item
    truncated at max_tokens still consumed thinking+output tokens) before
    taking the sync fallback. Pending/expired items return None.

    When ``pipeline_id`` is given, the lookup uses the scoped id. A legacy
    (unscoped) row is accepted ONLY when its batch provably belongs to this
    pipeline (JOIN on llm_batches.pipeline_id) — deploy-time continuity for
    a pipeline that submitted under pre-scoping code, without ever replaying
    another pipeline's result.
    """

    custom_id = make_custom_id(
        family,
        question_id=question_id,
        model_key=model_key,
        iso3=iso3,
        hazard_code=hazard_code,
        pass_idx=pass_idx,
        pipeline_id=pipeline_id,
    )
    row = con.execute(
        """
        SELECT status, response_text, usage_json, error_text, request_body_json
        FROM llm_batch_requests WHERE custom_id = ?
        """,
        [custom_id],
    ).fetchone()
    if not row and pipeline_id:
        legacy_id = make_custom_id(
            family,
            question_id=question_id,
            model_key=model_key,
            iso3=iso3,
            hazard_code=hazard_code,
            pass_idx=pass_idx,
        )
        row = con.execute(
            """
            SELECT r.status, r.response_text, r.usage_json, r.error_text,
                   r.request_body_json
            FROM llm_batch_requests r
            JOIN llm_batches b ON r.batch_id = b.batch_id
            WHERE r.custom_id = ? AND b.pipeline_id = ?
            """,
            [legacy_id, pipeline_id],
        ).fetchone()
        if row:
            custom_id = legacy_id
    if not row:
        return None
    status, text, usage_json, error, body_json = row
    if status not in ("succeeded", "errored"):
        return None
    try:
        usage = json.loads(usage_json) if usage_json else {}
    except json.JSONDecodeError:
        usage = {}
    return {
        "custom_id": custom_id,
        "status": status,
        "text": text or "",
        "usage": usage,
        "error": error or "",
        # The prompt the provider ACTUALLY received (extracted from the
        # stored request body — still present at replay time; bodies are
        # cleared only after the collect stage). The collect stage runs up
        # to 24h after submit and prompts embed today's date, so the
        # rebuilt prompt can differ from what was sent.
        "sent_prompt": _prompt_from_body_json(body_json),
    }


def _prompt_from_body_json(body_json: Optional[str]) -> str:
    """Extract the user prompt text from a stored provider request body."""

    if not body_json:
        return ""
    try:
        body = json.loads(body_json)
    except json.JSONDecodeError:
        return ""
    try:
        if isinstance(body, dict) and body.get("messages"):
            content = (body["messages"][0] or {}).get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # Anthropic cache_control blocks
                return "".join(
                    str(b.get("text") or "") for b in content if isinstance(b, dict)
                )
        if isinstance(body, dict) and body.get("contents"):
            parts = (body["contents"][0] or {}).get("parts") or []
            return "".join(
                str(p.get("text") or "") for p in parts if isinstance(p, dict)
            )
    except (KeyError, IndexError, TypeError):
        pass
    return ""


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
