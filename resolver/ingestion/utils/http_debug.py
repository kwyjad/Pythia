from __future__ import annotations

from typing import Iterable, Mapping, Optional

SENSITIVE_DEFAULTS = ("authorization", "x-api-key", "apikey", "api-key", "x_api_key")


def redact(headers: Optional[Mapping[str, str]], keys: Iterable[str] = SENSITIVE_DEFAULTS) -> dict[str, str]:
    """Return a copy of ``headers`` with sensitive keys redacted."""

    if not headers:
        return {}
    lower_map = {str(k).lower(): str(k) for k in headers.keys()}
    redacted: dict[str, str] = {}
    sensitive = {str(key).lower() for key in keys}
    for lower_key, original_key in lower_map.items():
        value = headers[original_key]
        if lower_key in sensitive:
            redacted[original_key] = "***"
        else:
            redacted[original_key] = value
    return redacted


def _sample_body(body: Optional[bytes], limit: int = 1024) -> str:
    if not body:
        return ""
    snippet = body[:limit]
    try:
        return snippet.decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - extremely defensive
        return repr(snippet)


def log_request(logger, method: str, url: str, *, headers: Optional[Mapping[str, str]] = None, params: Optional[Mapping[str, object]] = None) -> None:
    safe_headers = redact(headers)
    logger.info("dtm: request %s %s headers=%s params=%s", method.upper(), url, safe_headers, dict(params or {}))


def log_response(logger, status: int, *, headers: Optional[Mapping[str, str]] = None, sample_body: Optional[bytes] = None) -> None:
    safe_headers = redact(headers)
    body_snippet = _sample_body(sample_body)
    logger.info("dtm: response status=%s headers=%s body~%s", status, safe_headers, body_snippet)


__all__ = ["log_request", "log_response", "redact"]
