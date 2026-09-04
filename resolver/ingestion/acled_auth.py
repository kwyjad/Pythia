# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Helper utilities for authenticating against the ACLED API."""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import requests

OAUTH_TOKEN_URL = "https://acleddata.com/oauth/token"
OAUTH_CLIENT_ID = "acled"
_MIN_TTL = 300  # seconds

# ACLED's token endpoint sits behind a WAF, and on 2026-09-03 both grants were
# answered HTTP 200 with a body that was not JSON: the status check passed and
# resp.json() then raised, so the whole monthly ingest died on a bare
# JSONDecodeError naming neither the status, the body, nor the URL actually
# reached. A bare python-requests User-Agent with no Accept header is the
# request shape most likely to be handed a website instead of an API; this repo
# has hit the same wall at BoM (403 on a generic UA) and at crisisgroup.org.
# Asking for JSON explicitly costs nothing and removes one of the candidates.
_DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; Pythia/1.0; +https://github.com/kwyjad/Pythia)"


def _user_agent() -> str:
    return (os.environ.get("ACLED_USER_AGENT") or "").strip() or _DEFAULT_USER_AGENT


def _token_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "User-Agent": _user_agent(),
    }

_LOG = logging.getLogger("resolver.ingestion.acled.auth")

_CACHE: Dict[str, Optional[str | int]] = {
    "access_token": None,
    "refresh_token": None,
    "expiry": None,
}


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _jwt_exp(token: str) -> Optional[int]:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        payload = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
    except Exception:
        return None
    exp = payload.get("exp")
    try:
        return int(exp) if exp is not None else None
    except Exception:
        return None


def _jwt_is_valid(token: str, *, min_ttl: int = _MIN_TTL) -> bool:
    exp = _jwt_exp(token)
    if not exp:
        return False
    return (exp - int(time.time())) > min_ttl


def _jwt_is_usable(token: str, *, min_ttl: int = _MIN_TTL) -> bool:
    """Return True unless the token is a JWT with an expired ``exp`` claim.

    Opaque tokens (non-JWT) are assumed usable — the caller set them
    explicitly.  Only JWTs whose exp is within *min_ttl* of now are rejected.
    """
    exp = _jwt_exp(token)
    if exp is None:
        # Not a JWT or no exp claim — trust the caller
        return True
    return (exp - int(time.time())) > min_ttl


def _describe_token(token: Optional[str], expiry: Optional[int]) -> Dict[str, Optional[str]]:
    return {
        "token_length": len(token) if token else 0,
        "expires_at": datetime.fromtimestamp(expiry, tz=timezone.utc).isoformat()
        if expiry
        else None,
    }


def _redact(text: str) -> str:
    """Strip anything matching a credential in this process's environment.

    An OAuth refusal routinely quotes the username back, and these strings go
    to a public build log and into the diagnostics artifact.
    """

    try:
        from resolver.diagnostics.redaction import redact_text

        return redact_text(text)
    except Exception:  # pragma: no cover - redaction must never break auth
        return text


def _body_snippet(resp: requests.Response, *, limit: int = 300) -> str:
    """Return a short, single-line, credential-free snippet of the response body."""
    try:
        text = (resp.text or "").strip()
    except Exception:  # pragma: no cover - defensive
        return "<unable to read body>"
    return _redact(" ".join(text.split()))[:limit]


def _looks_like_html(text: str, content_type: str) -> bool:
    if "html" in content_type.lower():
        return True
    head = text.lstrip()[:200].lower()
    return head.startswith("<!doctype html") or head.startswith("<html") or "<title" in head


def describe_response(resp: requests.Response) -> str:
    """Say what actually came back, in one line, with no credentials in it.

    A token endpoint has three ways to disappoint us and they want three
    different repairs: a refusal that names the account, a WAF page where the
    API should be, and a moved route. Only the status, the content type, the
    URL reached after redirects and the first of the body can tell them apart,
    so all four are named. The body is redacted against the environment's own
    secrets, because an OAuth error routinely quotes the username back and
    this string is printed to a public build log.
    """

    content_type = ""
    try:
        content_type = resp.headers.get("Content-Type", "") or ""
    except Exception:  # pragma: no cover - defensive
        pass

    parts = [f"status={resp.status_code}", f"content_type={content_type!r}"]

    final_url = ""
    try:
        final_url = resp.url or ""
    except Exception:  # pragma: no cover - defensive
        pass
    if final_url and final_url != OAUTH_TOKEN_URL:
        parts.append(f"final_url={_redact(final_url)!r}")

    try:
        hops = [str(h.status_code) for h in (resp.history or [])]
    except Exception:  # pragma: no cover - defensive
        hops = []
    if hops:
        parts.append(f"redirects={'->'.join(hops)}")

    body = _body_snippet(resp)
    if _looks_like_html(body, content_type):
        parts.append("body_shape=html")
    parts.append(f"body={body!r}")
    return " ".join(parts)


def _log_token_http(resp: requests.Response, *, flow: str) -> None:
    # Deliberately not ``extra=``: those fields render in no default formatter,
    # which is why the 2026-09-03 failure left the body recorded nowhere at
    # all. The description goes in the message itself.
    _LOG.debug("ACLED OAuth %s grant response | %s", flow, describe_response(resp))


def _token_request(data: Dict[str, str], *, flow: str) -> Dict[str, str]:
    """POST the token endpoint and return its JSON object, or raise saying why not."""

    resp = requests.post(
        OAUTH_TOKEN_URL,
        data=data,
        headers=_token_headers(),
        timeout=30,
    )
    _log_token_http(resp, flow=flow)

    if resp.status_code != 200:
        raise RuntimeError(f"ACLED OAuth {flow} grant failed: {describe_response(resp)}")

    # A 200 is not an answer. Until 2026-09-03 this branch was a bare
    # resp.json(), so a WAF page served with a 200 surfaced as "Expecting
    # value: line 1 column 1 (char 0)" and killed the monthly ingest with
    # nothing recorded about what had actually been served.
    try:
        payload = resp.json()
    except ValueError:
        raise RuntimeError(
            f"ACLED OAuth {flow} grant returned HTTP 200 with a body that is not JSON: "
            f"{describe_response(resp)}"
        ) from None

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"ACLED OAuth {flow} grant returned a JSON {type(payload).__name__}, not an object: "
            f"{describe_response(resp)}"
        )
    return payload


def _password_grant(username: str, password: str) -> Dict[str, str]:
    # ``scope=authenticated`` is required by ACLED's OAuth token endpoint (see the
    # documented password-grant example at acleddata.com/api-documentation/getting-started).
    # Omitting it makes the gateway reject the request with a non-standard HTTP 415.
    data = {
        "username": username,
        "password": password,
        "grant_type": "password",
        "client_id": OAUTH_CLIENT_ID,
        "scope": "authenticated",
    }
    _LOG.debug("ACLED password grant for username=%s", data["username"])
    return _token_request(data, flow="password")


def _refresh_grant(refresh_token: str) -> Dict[str, str]:
    data = {
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
        "client_id": OAUTH_CLIENT_ID,
    }
    return _token_request(data, flow="refresh")


def _set_cache(token: str, refresh_token: Optional[str]) -> None:
    _CACHE["access_token"] = token
    _CACHE["expiry"] = _jwt_exp(token)
    if refresh_token:
        _CACHE["refresh_token"] = refresh_token


def _resolve_refresh_token() -> Optional[str]:
    cached = _CACHE.get("refresh_token")
    if cached:
        return cached
    refresh_from_env = (os.environ.get("ACLED_REFRESH_TOKEN") or "").strip()
    if refresh_from_env:
        _CACHE["refresh_token"] = refresh_from_env
        return refresh_from_env
    return None


def _resolve_password_creds() -> Optional[Dict[str, str]]:
    username = (os.environ.get("ACLED_USERNAME") or "").strip()
    password = (os.environ.get("ACLED_PASSWORD") or "").strip()
    if username and password:
        return {"username": username, "password": password}
    return None


def _resolve_existing_token() -> Optional[str]:
    for name in ("ACLED_ACCESS_TOKEN", "ACLED_TOKEN"):
        raw = os.environ.get(name)
        if not raw:
            continue
        token = raw.strip()
        if not token:
            continue
        if name == "ACLED_TOKEN":
            os.environ.setdefault("ACLED_ACCESS_TOKEN", token)
        return token
    return None


def get_access_token() -> str:
    """Return a valid ACLED access token, refreshing credentials when required."""

    now = int(time.time())
    cached_token = _CACHE.get("access_token")
    cached_expiry = _CACHE.get("expiry")
    if cached_token and isinstance(cached_expiry, int) and (cached_expiry - now) > _MIN_TTL:
        _LOG.debug(
            "Using cached ACLED access token",
            extra=_describe_token(cached_token, cached_expiry),
        )
        return cached_token

    existing = _resolve_existing_token()
    if existing and _jwt_is_usable(existing):
        expiry = _jwt_exp(existing)
        _LOG.debug(
            "Using environment-provided ACLED token",
            extra=_describe_token(existing, expiry),
        )
        print("[acled_auth] Using existing environment token")
        _set_cache(existing, os.environ.get("ACLED_REFRESH_TOKEN"))
        return existing
    elif existing:
        _LOG.debug("Environment-provided ACLED token is an expired JWT; falling through to refresh/password grant")

    refresh_token = _resolve_refresh_token()
    if refresh_token:
        _LOG.debug(
            "Attempting ACLED refresh grant",
            extra={"token_length": len(refresh_token)},
        )
        print("[acled_auth] Attempting refresh grant...")
        try:
            tokens = _refresh_grant(refresh_token)
        except Exception as exc:  # pragma: no cover - network stack errors
            _LOG.debug("ACLED refresh grant failed", extra={"error": str(exc)})
            print(f"[acled_auth] Refresh grant failed: {exc}; trying password grant")
        else:
            access_token = tokens.get("access_token")
            if not access_token:
                raise RuntimeError("ACLED refresh grant response missing access_token")
            new_refresh = tokens.get("refresh_token") or refresh_token
            os.environ["ACLED_ACCESS_TOKEN"] = access_token
            os.environ["ACLED_REFRESH_TOKEN"] = new_refresh
            _set_cache(access_token, new_refresh)
            expiry = _CACHE.get("expiry") if isinstance(_CACHE.get("expiry"), int) else _jwt_exp(access_token)
            _LOG.debug(
                "Obtained ACLED access token via refresh",
                extra=_describe_token(access_token, expiry if isinstance(expiry, int) else None),
            )
            print("[acled_auth] Refresh grant succeeded")
            return access_token

    password_creds = _resolve_password_creds()
    if password_creds:
        _LOG.debug("Attempting ACLED password grant")
        print("[acled_auth] Attempting password grant...")
        tokens = _password_grant(password_creds["username"], password_creds["password"])
        access_token = tokens.get("access_token")
        if not access_token:
            raise RuntimeError("ACLED password grant response missing access_token")
        refresh = tokens.get("refresh_token")
        if refresh:
            os.environ["ACLED_REFRESH_TOKEN"] = refresh
        os.environ["ACLED_ACCESS_TOKEN"] = access_token
        _set_cache(access_token, refresh)
        expiry = _CACHE.get("expiry") if isinstance(_CACHE.get("expiry"), int) else _jwt_exp(access_token)
        _LOG.debug(
            "Obtained ACLED access token via password grant",
            extra=_describe_token(access_token, expiry if isinstance(expiry, int) else None),
        )
        print("[acled_auth] Password grant succeeded")
        return access_token

    print("[acled_auth] All auth methods exhausted — no valid credentials found")
    raise RuntimeError(
        "ACLED authentication failed: set ACLED_ACCESS_TOKEN/ACLED_TOKEN or "
        "ACLED_REFRESH_TOKEN or ACLED_USERNAME/ACLED_PASSWORD."
    )


def get_auth_header() -> Dict[str, str]:
    """Return an Authorization header for ACLED requests."""

    token = get_access_token()
    return {"Authorization": f"Bearer {token}"}
