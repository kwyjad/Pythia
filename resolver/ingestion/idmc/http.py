"""HTTP helpers for the IDMC connector."""
from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

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


def http_get(
    url: str,
    *,
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = 10.0,
    retries: int = 2,
    backoff_s: float = 0.5,
) -> Tuple[int, Dict[str, str], bytes, Dict[str, object]]:
    """Perform a GET request with basic retries and diagnostics."""

    attempts: Iterable[int] = range(1, retries + 2)
    request_headers: Dict[str, str] = {"User-Agent": "pythia-idmc/1.0"}
    if headers:
        request_headers.update(headers)
    _apply_token(request_headers)

    exceptions: list[Dict[str, object]] = []
    total_elapsed = 0.0
    total_backoff = 0.0

    last_status: Optional[int] = None
    last_headers: Dict[str, str] = {}

    for attempt in attempts:
        started = time.monotonic()
        request = urllib.request.Request(url, headers=request_headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                status = getattr(response, "status", None) or response.getcode()
                last_status = int(status)
                last_headers = dict(response.headers.items())
                total_elapsed += time.monotonic() - started
                diagnostics = {
                    "attempts": attempt,
                    "retries": attempt - 1,
                    "duration_s": round(total_elapsed, 3),
                    "backoff_s": round(total_backoff, 3),
                    "exceptions": exceptions,
                    "status": last_status,
                }
                return last_status, last_headers, body, diagnostics
        except urllib.error.URLError as err:  # pragma: no cover - network dependent
            total_elapsed += time.monotonic() - started
            exceptions.append({
                "attempt": attempt,
                "type": err.__class__.__name__,
                "reason": getattr(err, "reason", None),
                "message": str(err),
            })
        except Exception as err:  # pragma: no cover - defensive
            total_elapsed += time.monotonic() - started
            exceptions.append({
                "attempt": attempt,
                "type": err.__class__.__name__,
                "message": str(err),
            })

        if attempt > retries:
            break
        sleep_for = backoff_s * (2 ** (attempt - 1))
        total_backoff += sleep_for
        time.sleep(sleep_for)

    diagnostics = {
        "attempts": len(exceptions),
        "retries": max(len(exceptions) - 1, 0),
        "duration_s": round(total_elapsed, 3),
        "backoff_s": round(total_backoff, 3),
        "exceptions": exceptions,
        "status": last_status,
    }
    raise HttpRequestError(f"GET {url} failed", diagnostics)
