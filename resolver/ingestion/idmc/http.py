"""HTTP helpers for the IDMC connector."""
from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

from .rate_limit import TokenBucket, parse_retry_after

__all__ = ["HttpRequestError", "http_get"]


@dataclass
class HttpRequestError(Exception):
    """Raised when an HTTP request exhausts all retries."""

    message: str
    diagnostics: Dict[str, object]

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def __str__(self) -> str:  # pragma: no cover - defensive string repr
        return self.message


def _apply_token(headers: MutableMapping[str, str]) -> None:
    token = os.getenv("IDMC_API_TOKEN")
    if token:
        headers.setdefault("Authorization", f"Bearer {token}")


def _skip_sleep() -> bool:
    raw = os.getenv("IDMC_TEST_NO_SLEEP", "0").strip().lower()
    return raw not in {"", "0", "false", "no"}


def _plan_sleep(duration: float, *, sleep_fn: Callable[[float], None], bucket: list[float]) -> float:
    duration = max(0.0, float(duration))
    if duration <= 0.0:
        return 0.0
    bucket.append(duration)
    if not _skip_sleep():
        sleep_fn(duration)
    return duration


def http_get(
    url: str,
    *,
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = 10.0,
    retries: int = 2,
    backoff_s: float = 0.5,
    rate_limiter: TokenBucket | None = None,
    max_bytes: Optional[int] = None,
    stream_path: Optional[str] = None,
    chunk_size: int = 64 * 1024,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Tuple[int, Dict[str, str], bytes | None, Dict[str, object]]:
    """Perform a GET request with retries, rate limiting, and diagnostics."""

    attempts: Iterable[int] = range(1, retries + 2)
    request_headers: Dict[str, str] = {"User-Agent": "pythia-idmc/1.0"}
    if headers:
        request_headers.update(headers)
    _apply_token(request_headers)

    exceptions: list[Dict[str, object]] = []
    total_elapsed = 0.0
    total_backoff = 0.0
    attempt_durations: list[float] = []
    planned_waits: list[float] = []
    retry_after_waits: list[float] = []
    rate_waits: list[float] = []

    last_status: Optional[int] = None
    last_headers: Dict[str, str] = {}

    for attempt in attempts:
        if rate_limiter is not None:
            waited = rate_limiter.acquire()
            if waited > 0:
                rate_waits.append(waited)
        started = time.monotonic()
        request = urllib.request.Request(url, headers=request_headers, method="GET")
        stream_used = False
        stream_handle = None
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", None) or response.getcode()
                last_status = int(status)
                last_headers = dict(response.headers.items())

                wire_bytes = 0
                body_bytes = 0
                body_chunks: list[bytes] = []
                try:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        wire_bytes += len(chunk)
                        body_bytes += len(chunk)
                        if (
                            not stream_used
                            and stream_path
                            and max_bytes is not None
                            and max_bytes > 0
                            and body_bytes > max_bytes
                        ):
                            stream_dir = os.path.dirname(stream_path) or "."
                            os.makedirs(stream_dir, exist_ok=True)
                            stream_handle = open(stream_path, "wb")
                            stream_used = True
                            for piece in body_chunks:
                                stream_handle.write(piece)
                            body_chunks.clear()
                        if stream_used:
                            assert stream_handle is not None
                            stream_handle.write(chunk)
                        else:
                            body_chunks.append(chunk)
                finally:
                    if stream_handle is not None:
                        stream_handle.close()

                elapsed = time.monotonic() - started
                total_elapsed += elapsed
                attempt_durations.append(elapsed)

                body: bytes | None
                streamed_to: Optional[str]
                if stream_used:
                    body = None
                    streamed_to = stream_path
                else:
                    body = b"".join(body_chunks)
                    streamed_to = None

                diagnostics = {
                    "attempts": attempt,
                    "retries": attempt - 1,
                    "duration_s": round(total_elapsed, 6),
                    "backoff_s": round(total_backoff, 6),
                    "exceptions": exceptions,
                    "status": last_status,
                    "attempt_durations_s": [round(value, 6) for value in attempt_durations],
                    "wire_bytes": wire_bytes,
                    "body_bytes": body_bytes,
                    "streamed_to": streamed_to,
                    "retry_after_s": retry_after_waits,
                    "rate_limit_wait_s": rate_waits,
                    "planned_sleep_s": planned_waits,
                }
                return last_status, last_headers, body, diagnostics
        except urllib.error.HTTPError as err:
            elapsed = time.monotonic() - started
            total_elapsed += elapsed
            attempt_durations.append(elapsed)
            status = getattr(err, "code", None)
            last_status = int(status) if status is not None else last_status
            headers = dict(getattr(err, "headers", {}) or {})
            last_headers = headers or last_headers
            retry_after = parse_retry_after(headers.get("Retry-After")) if headers else None
            if retry_after is not None:
                waited = _plan_sleep(retry_after, sleep_fn=sleep_fn, bucket=planned_waits)
                if waited > 0:
                    retry_after_waits.append(waited)
            exceptions.append(
                {
                    "attempt": attempt,
                    "type": err.__class__.__name__,
                    "status": getattr(err, "code", None),
                    "message": str(err),
                }
            )
            if stream_used and stream_path and os.path.exists(stream_path):
                try:
                    os.remove(stream_path)
                except OSError:  # pragma: no cover - defensive cleanup
                    pass
        except urllib.error.URLError as err:  # pragma: no cover - network dependent
            elapsed = time.monotonic() - started
            total_elapsed += elapsed
            attempt_durations.append(elapsed)
            reason = getattr(err, "reason", None)
            if reason is not None:
                try:
                    reason_text = str(reason)
                except Exception:  # pragma: no cover - defensive
                    reason_text = repr(reason)
            else:
                reason_text = None
            exceptions.append(
                {
                    "attempt": attempt,
                    "type": err.__class__.__name__,
                    "reason": reason_text,
                    "message": str(err),
                }
            )
            if stream_used and stream_path and os.path.exists(stream_path):
                try:
                    os.remove(stream_path)
                except OSError:  # pragma: no cover - defensive cleanup
                    pass
        except Exception as err:  # pragma: no cover - defensive
            elapsed = time.monotonic() - started
            total_elapsed += elapsed
            attempt_durations.append(elapsed)
            exceptions.append(
                {
                    "attempt": attempt,
                    "type": err.__class__.__name__,
                    "message": str(err),
                }
            )
            if stream_used and stream_path and os.path.exists(stream_path):
                try:
                    os.remove(stream_path)
                except OSError:  # pragma: no cover - defensive cleanup
                    pass

        if attempt > retries:
            break
        sleep_for = backoff_s * (2 ** (attempt - 1))
        waited = _plan_sleep(sleep_for, sleep_fn=sleep_fn, bucket=planned_waits)
        total_backoff += waited

    diagnostics = {
        "attempts": len(exceptions),
        "retries": max(len(exceptions) - 1, 0),
        "duration_s": round(total_elapsed, 6),
        "backoff_s": round(total_backoff, 6),
        "exceptions": exceptions,
        "status": last_status,
        "attempt_durations_s": [round(value, 6) for value in attempt_durations],
        "wire_bytes": 0,
        "body_bytes": 0,
        "streamed_to": None,
        "retry_after_s": retry_after_waits,
        "rate_limit_wait_s": rate_waits,
        "planned_sleep_s": planned_waits,
    }
    raise HttpRequestError(f"GET {url} failed", diagnostics)
