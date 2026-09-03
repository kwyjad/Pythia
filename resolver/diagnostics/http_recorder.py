# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Record every outbound HTTP call a connector makes. One place, not twenty.

The single most valuable thing missing from a resolver diagnostic was the
transport: which URL was called, what status came back, how big the answer
was, and — for a JSON API — what the response ENVELOPE said. ACLED CAST
returns ``status``, ``success``, ``last_update``, ``count``, ``messages``
and ``data_query_restrictions``; the connector reads ``data`` and discards
the rest, so a stalled vintage could not be told apart from a spent
account quota without a bespoke probe.

Rather than thread a logger through ten connectors, this patches
``requests.Session.request`` once. Every connector in the repo goes through
it — ``session.get``, ``session.post`` and the module-level ``requests.get``
alike, since that builds a Session internally.

Three rules:

* **Opt-in.** :func:`maybe_install_from_env` does nothing unless
  ``PYTHIA_RUN_LOG_DIR`` is set, so local runs and tests are untouched.
* **Never raises, never delays.** The wrapper does its bookkeeping inside a
  try/except and returns the real response either way; a recorder that can
  break an ingest is worse than no recorder.
* **Redacted at the point of capture.** URLs, headers and envelopes are
  redacted before they reach disk, so no later stage can forget to.

Attribution is derived from the call stack rather than a per-connector tag:
the first frame belonging to a repo module names the connector. That keeps
this genuinely one place — a new connector is recorded the day it is
written, with no edit here and none there.
"""

from __future__ import annotations

import inspect
import logging
import os
import time
from typing import Any

from resolver.diagnostics import run_log
from resolver.diagnostics.redaction import redact_obj, redact_text, redact_url, secret_values

LOG = logging.getLogger(__name__)

_INSTALLED = False
_ORIGINAL = None

#: How much of a response body to look at when deriving the envelope.
MAX_ENVELOPE_BYTES = 2_000_000

#: How many data rows to keep as a sample per connector.
ENVELOPE_SAMPLE_ROWS = 5

#: The environment's secrets, resolved once. This runs on every request, and
#: re-scanning os.environ tens of thousands of times a run buys nothing: the
#: credentials a process holds are fixed at start.
_SECRETS: list[str] | None = None

#: One envelope per (connector, host, path) — the first response wins, so a
#: paginated pull records page 1 rather than 400 identical shapes.
_ENVELOPES_SEEN: set[tuple[str, str, str]] = set()

#: Repo packages a frame must belong to for it to name the caller.
_REPO_PACKAGES = ("resolver", "pythia", "horizon_scanner", "forecaster", "sibyl")

#: Frames that are plumbing rather than a caller.
_SKIP_MODULES = (
    "resolver.diagnostics.http_recorder",
    "resolver.diagnostics.run_log",
)


def _caller() -> str:
    """The repo module that made this request, e.g. ``resolver.connectors.gdacs``."""

    try:
        frame = inspect.currentframe()
        depth = 0
        while frame is not None and depth < 40:
            frame = frame.f_back
            depth += 1
            if frame is None:
                break
            module = frame.f_globals.get("__name__", "")
            if not module or module in _SKIP_MODULES:
                continue
            root = module.split(".")[0]
            if root in _REPO_PACKAGES:
                return module
    except Exception:  # pragma: no cover - frame walking is best effort
        pass
    return "unknown"


def _body_shape(body: Any) -> dict[str, Any] | None:
    """A description of the request body, never the body itself.

    A POST body can carry credentials (the ACLED password grant does), so
    what goes in the bundle is its SHAPE: the field names and their sizes.
    """

    if body is None:
        return None
    if isinstance(body, dict):
        return {
            "kind": "form_or_json",
            "fields": sorted(str(k) for k in body),
            "n_fields": len(body),
        }
    if isinstance(body, (bytes, str)):
        return {"kind": "raw", "bytes": len(body)}
    return {"kind": type(body).__name__}


def _envelope(response: Any, secrets: list[str]) -> dict[str, Any] | None:
    """Every top-level field of a JSON response EXCEPT the bulk payload.

    Plus the payload's column names and its first few rows — enough to see
    the shape a connector was handed without carrying the whole download.
    """

    content_type = str(response.headers.get("Content-Type", "")).lower()
    if "json" not in content_type:
        return {"content_type": content_type or "(none)", "json": False}
    try:
        if len(response.content or b"") > MAX_ENVELOPE_BYTES:
            return {
                "content_type": content_type,
                "json": True,
                "note": f"response larger than {MAX_ENVELOPE_BYTES} bytes; envelope not parsed",
            }
        payload = response.json()
    except Exception as exc:
        return {"content_type": content_type, "json": False, "parse_error": str(exc)[:200]}

    envelope: dict[str, Any] = {"content_type": content_type, "json": True}
    if isinstance(payload, list):
        rows = payload
        envelope["top_level"] = {"(root)": f"list[{len(rows)}]"}
    elif isinstance(payload, dict):
        scalars: dict[str, Any] = {}
        rows: list[Any] = []
        for key, value in payload.items():
            if isinstance(value, list):
                scalars[key] = f"list[{len(value)}]"
                if len(value) > len(rows):
                    rows = value
            elif isinstance(value, dict):
                scalars[key] = redact_obj(value, secrets)
            else:
                scalars[key] = value
        envelope["top_level"] = redact_obj(scalars, secrets)
    else:
        return {"content_type": content_type, "json": True, "top_level": {"(root)": type(payload).__name__}}

    sample = rows[:ENVELOPE_SAMPLE_ROWS]
    envelope["n_rows"] = len(rows)
    envelope["columns"] = sorted(
        {str(k) for row in rows[:200] if isinstance(row, dict) for k in row}
    )
    envelope["sample_rows"] = redact_obj(sample, secrets)
    # The arithmetic behind "this data is N days old": every date-shaped
    # column's maximum, so a reader can check the connector's conclusion
    # rather than take its word.
    envelope["max_by_date_column"] = _max_dates(rows)
    return envelope


_DATE_HINTS = ("date", "time", "updated", "issued", "year", "month", "stamp", "period")


def _max_dates(rows: list[Any]) -> dict[str, str]:
    """Max value of every date-looking column, as strings."""

    maxima: dict[str, str] = {}
    for row in rows[:5000]:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            name = str(key).lower()
            if not any(hint in name for hint in _DATE_HINTS):
                continue
            if value is None or isinstance(value, (dict, list)):
                continue
            text = str(value)
            if not text:
                continue
            current = maxima.get(str(key))
            if current is None or text > current:
                maxima[str(key)] = text
    return maxima


def _cached_secrets() -> list[str]:
    global _SECRETS
    if _SECRETS is None:
        _SECRETS = secret_values()
    return _SECRETS


def _record(
    *,
    method: str,
    url: str,
    body: Any,
    response: Any,
    elapsed_ms: float,
    error: str | None,
) -> None:
    secrets = _cached_secrets()
    connector = _caller()
    safe_url = redact_url(str(url), secrets)

    status = getattr(response, "status_code", None)
    history = getattr(response, "history", None) or []
    content = getattr(response, "content", None)
    from_cache = bool(getattr(response, "from_cache", False))

    run_log.record(
        run_log.STREAM_HTTP,
        {
            "connector": connector,
            "method": str(method).upper(),
            "url": safe_url,
            "request_body": _body_shape(body),
            "status": status,
            "elapsed_ms": round(elapsed_ms, 1),
            "response_bytes": len(content) if content is not None else None,
            "redirects": len(history),
            "from_cache": from_cache,
            "error": redact_text(error, secrets) if error else None,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )

    if response is None or status is None:
        return
    try:
        from urllib.parse import urlsplit

        parts = urlsplit(safe_url)
        key = (connector, parts.netloc, parts.path)
    except Exception:  # pragma: no cover
        key = (connector, "", str(safe_url))
    if key in _ENVELOPES_SEEN:
        return
    _ENVELOPES_SEEN.add(key)
    envelope = _envelope(response, secrets)
    if envelope is None:
        return
    run_log.record(
        run_log.STREAM_ENVELOPE,
        {
            "connector": connector,
            "host": key[1],
            "path": key[2],
            "url": safe_url,
            "status": status,
            "envelope": envelope,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def install() -> bool:
    """Patch ``requests.Session.request``. Idempotent; returns True if active."""

    global _INSTALLED, _ORIGINAL
    if _INSTALLED:
        return True
    try:
        import requests
    except Exception as exc:  # pragma: no cover - requests is a hard dependency
        LOG.warning("[http_recorder] requests unavailable (%s); not recording", exc)
        return False

    _ORIGINAL = requests.Session.request

    def request(self, method, url, *args, **kwargs):  # type: ignore[no-untyped-def]
        started = time.monotonic()
        response = None
        error = None
        try:
            response = _ORIGINAL(self, method, url, *args, **kwargs)
            return response
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                _record(
                    method=method,
                    url=url,
                    body=kwargs.get("json", kwargs.get("data")),
                    response=response,
                    elapsed_ms=(time.monotonic() - started) * 1000.0,
                    error=error,
                )
            except Exception:  # noqa: BLE001 - never break the call being recorded
                pass

    requests.Session.request = request  # type: ignore[method-assign]
    _INSTALLED = True
    LOG.info(
        "[http_recorder] recording outbound HTTP to %s",
        run_log.stream_path(run_log.STREAM_HTTP),
    )
    return True


def uninstall() -> None:
    """Restore the original ``Session.request`` (used by tests)."""

    global _INSTALLED, _ORIGINAL, _SECRETS
    if not _INSTALLED:
        return
    import requests

    requests.Session.request = _ORIGINAL  # type: ignore[method-assign]
    _INSTALLED = False
    _ORIGINAL = None
    _ENVELOPES_SEEN.clear()
    _SECRETS = None


def maybe_install_from_env() -> bool:
    """Install the recorder iff this run is collecting evidence.

    Called from the entry points the ingest workflow invokes. Off unless
    ``PYTHIA_RUN_LOG_DIR`` is set, which is what keeps the default path
    byte-identical to before.
    """

    if not run_log.enabled():
        return False
    if os.environ.get("PYTHIA_HTTP_RECORD", "1").strip().lower() in {"0", "false", "no"}:
        return False
    return install()
