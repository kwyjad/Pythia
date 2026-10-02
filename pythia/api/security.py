# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Request-level protections for the public API: rate limits and headers.

The API is read-only and public, so the threats worth guarding against are
load (one client keeping the 2 GB instance busy with heavy queries until it
is OOM-killed) and disclosure (a browser treating a response as something
other than what it is). Both are handled here, in process, with no new
dependency.

The limiter is a token bucket per client and per route group. Render sits in
front of the API, so the client is the FIRST hop of ``X-Forwarded-For``; the
socket peer is Render's proxy and would put every visitor in one bucket. A
client can forge the header, which lets it pick a new bucket, so this limits
a careless or ordinary client rather than a determined one; an edge proxy is
the answer to the second.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


@dataclass(frozen=True)
class Bucket:
    """Capacity in requests, refilled at ``per_minute`` requests a minute."""

    name: str
    capacity: float
    per_minute: float


# Heavier routes get smaller buckets. Order matters: the first prefix that
# matches wins, so the specific ones come before "/v1/".
_GROUPS: Tuple[Tuple[str, Bucket], ...] = (
    ("/v1/downloads/", Bucket("downloads", 10, 10)),
    ("/v1/question_bundle", Bucket("bundle", 30, 30)),
    ("/v1/forecasts/", Bucket("forecasts", 30, 30)),
    ("/v1/diagnostics/", Bucket("diagnostics", 60, 60)),
    ("/v1/debug/", Bucket("debug", 30, 30)),
    ("/v1/admin/", Bucket("admin", 5, 5)),
    ("/v1/", Bucket("default", 240, 240)),
)

# Never limited: Render's health check polls this, and a 429 there would get
# the instance restarted.
_EXEMPT = ("/v1/health",)


def _scale() -> float:
    """``PYTHIA_RATE_LIMIT_SCALE`` multiplies every bucket; 0 disables."""

    raw = os.getenv("PYTHIA_RATE_LIMIT_SCALE")
    if raw is None and "PYTEST_CURRENT_TEST" in os.environ:
        # The API test suites drive hundreds of requests through one shared
        # limiter; a test of the limiter sets the scale explicitly.
        return 0.0
    try:
        return max(0.0, float(raw if raw is not None else "1"))
    except ValueError:
        return 1.0


def bucket_for(path: str) -> Optional[Bucket]:
    if path in _EXEMPT:
        return None
    for prefix, bucket in _GROUPS:
        if path.startswith(prefix):
            return bucket
    return None


def client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    first = forwarded.split(",", 1)[0].strip()
    if first:
        return first
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """Token buckets keyed by (client, group). Thread-safe, bounded memory."""

    MAX_KEYS = 50_000

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._state: Dict[Tuple[str, str], Tuple[float, float]] = {}

    def reset(self) -> None:
        with self._lock:
            self._state.clear()

    def take(self, client: str, bucket: Bucket, scale: float = 1.0) -> Optional[float]:
        """Spend one token. Returns None when allowed, else seconds to wait."""

        capacity = bucket.capacity * scale
        rate = bucket.per_minute * scale / 60.0
        now = self._clock()
        key = (client, bucket.name)
        with self._lock:
            if len(self._state) > self.MAX_KEYS:
                # A flood of forged client keys must not grow memory without
                # bound; dropping every bucket is crude and correct.
                self._state.clear()
            tokens, last = self._state.get(key, (capacity, now))
            tokens = min(capacity, tokens + (now - last) * rate)
            if tokens >= 1.0:
                self._state[key] = (tokens - 1.0, now)
                return None
            self._state[key] = (tokens, now)
            return (1.0 - tokens) / rate if rate > 0 else 60.0


LIMITER = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        scale = _scale()
        if scale > 0 and request.method != "OPTIONS":
            bucket = bucket_for(request.url.path)
            if bucket is not None:
                wait = LIMITER.take(client_key(request), bucket, scale)
                if wait is not None:
                    return JSONResponse(
                        {"detail": "Too many requests"},
                        status_code=429,
                        headers={"Retry-After": str(int(wait) + 1)},
                    )
        return await call_next(request)


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-Frame-Options": "DENY",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    # The API serves JSON, CSV and files; none of it is a page to render.
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response


def docs_enabled() -> bool:
    """Interactive docs and the OpenAPI schema are off unless asked for."""

    return os.getenv("PYTHIA_API_DOCS", "0").strip() in ("1", "true", "yes")
