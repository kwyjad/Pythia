"""HTTP helpers for the IDMC connector."""
from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import requests
from requests import Response
from requests.exceptions import RequestException, Timeout

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


def _resolve_user_agent() -> str:
    for name in ("IDMC_USER_AGENT", "RELIEFWEB_USER_AGENT"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return "Pythia-IDMC/1.0 (+contact)"


def _normalise_timeout(value: float | Tuple[float, float]) -> Tuple[float, float]:
    if isinstance(value, (tuple, list)):
        if len(value) == 0:
            return (10.0, 10.0)
        if len(value) == 1:
            connect = max(float(value[0]), 0.0)
            return (connect, connect)
        connect = max(float(value[0]), 0.0)
        read = max(float(value[1]), 0.0)
        return (connect, read if read > 0 else connect)
    timeout = max(float(value), 0.0)
    return (timeout, timeout)


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
    timeout: float | Tuple[float, float] = 10.0,
    retries: int = 2,
    backoff_s: float = 0.5,
    rate_limiter: TokenBucket | None = None,
    max_bytes: Optional[int] = None,
    stream_path: Optional[str] = None,
    chunk_size: int = 64 * 1024,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Tuple[int | str, Dict[str, str], bytes | None, Dict[str, object]]:
    """Perform a GET request with retries, rate limiting, and diagnostics."""

    attempts: Iterable[int] = range(1, retries + 2)
    request_headers: Dict[str, str] = {
        "User-Agent": _resolve_user_agent(),
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
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

    last_status: int | str | None = None
    last_headers: Dict[str, str] = {}
    last_exception: Optional[str] = None
    timeout_hit = False

    timeout_tuple = _normalise_timeout(timeout)

    session = requests.Session()
    try:
        for attempt in attempts:
            if rate_limiter is not None:
                waited = rate_limiter.acquire()
                if waited > 0:
                    rate_waits.append(waited)
            started = time.monotonic()
            stream_used = False
            stream_handle = None
            wire_bytes = 0
            body_bytes = 0
            try:
                response: Response = session.get(
                    url,
                    headers=request_headers,
                    timeout=timeout_tuple,
                    stream=True,
                )
                status = int(response.status_code)
                last_status = status
                last_headers = dict(response.headers.items())
                if status >= 400:
                    retry_after = parse_retry_after(response.headers.get("Retry-After"))
                    if retry_after is not None:
                        waited = _plan_sleep(
                            retry_after, sleep_fn=sleep_fn, bucket=planned_waits
                        )
                        if waited > 0:
                            retry_after_waits.append(waited)
                    message = response.reason or "HTTP error"
                    try:
                        preview = response.text[:256]
                        if preview:
                            message = f"{message}: {preview}"
                    except Exception:  # pragma: no cover - defensive
                        preview = None
                    exceptions.append(
                        {
                            "attempt": attempt,
                            "type": "HTTPError",
                            "status": status,
                            "message": message,
                        }
                    )
                    last_exception = "HTTPError"
                    elapsed = time.monotonic() - started
                    total_elapsed += elapsed
                    attempt_durations.append(elapsed)
                    response.close()
                else:
                    body_chunks: list[bytes] = []
                    try:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if not chunk:
                                continue
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
                        response.close()
                        if stream_handle is not None:
                            stream_handle.close()

                    elapsed = time.monotonic() - started
                    total_elapsed += elapsed
                    attempt_durations.append(elapsed)

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
                        "status": status,
                        "attempt_durations_s": [
                            round(value, 6) for value in attempt_durations
                        ],
                        "wire_bytes": wire_bytes,
                        "body_bytes": body_bytes,
                        "streamed_to": streamed_to,
                        "retry_after_s": retry_after_waits,
                        "rate_limit_wait_s": rate_waits,
                        "planned_sleep_s": planned_waits,
                        "timeout": False,
                        "exception_type": None,
                    }
                    return status, last_headers, body, diagnostics
            except Timeout as err:  # pragma: no cover - network dependent
                elapsed = time.monotonic() - started
                total_elapsed += elapsed
                attempt_durations.append(elapsed)
                message = str(err)
                timeout_hit = True
                last_status = "timeout"
                last_exception = err.__class__.__name__
                exceptions.append(
                    {
                        "attempt": attempt,
                        "type": err.__class__.__name__,
                        "message": message,
                        "timeout": True,
                    }
                )
            except RequestException as err:  # pragma: no cover - network dependent
                elapsed = time.monotonic() - started
                total_elapsed += elapsed
                attempt_durations.append(elapsed)
                response = err.response  # type: ignore[assignment]
                status = getattr(response, "status_code", None)
                if status is not None:
                    last_status = int(status)
                    last_headers = (
                        dict(response.headers.items()) if hasattr(response, "headers") else {}
                    )
                    retry_after = None
                    if hasattr(response, "headers"):
                        retry_after = parse_retry_after(response.headers.get("Retry-After"))
                    if retry_after is not None:
                        waited = _plan_sleep(
                            retry_after, sleep_fn=sleep_fn, bucket=planned_waits
                        )
                        if waited > 0:
                            retry_after_waits.append(waited)
                reason = getattr(err, "__cause__", None)
                if reason is None:
                    reason = getattr(err, "args", [None])[0]
                try:
                    reason_text = str(reason) if reason is not None else None
                except Exception:  # pragma: no cover - defensive
                    reason_text = repr(reason)
                exceptions.append(
                    {
                        "attempt": attempt,
                        "type": err.__class__.__name__,
                        "status": status,
                        "message": str(err),
                        "reason": reason_text,
                    }
                )
                last_exception = err.__class__.__name__
            except socket.timeout as err:  # pragma: no cover - network dependent
                elapsed = time.monotonic() - started
                total_elapsed += elapsed
                attempt_durations.append(elapsed)
                timeout_hit = True
                last_status = "timeout"
                last_exception = err.__class__.__name__
                exceptions.append(
                    {
                        "attempt": attempt,
                        "type": err.__class__.__name__,
                        "message": str(err),
                        "timeout": True,
                    }
                )
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
                last_exception = err.__class__.__name__
            finally:
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
    finally:
        session.close()

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
        "timeout": timeout_hit,
        "exception_type": last_exception,
    }
    raise HttpRequestError(f"GET {url} failed", diagnostics)
