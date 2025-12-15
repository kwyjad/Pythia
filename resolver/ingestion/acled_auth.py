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


def _describe_token(token: Optional[str], expiry: Optional[int]) -> Dict[str, Optional[str]]:
    return {
        "token_length": len(token) if token else 0,
        "expires_at": datetime.fromtimestamp(expiry, tz=timezone.utc).isoformat()
        if expiry
        else None,
    }


def _log_token_http(resp: requests.Response, *, flow: str) -> None:
    try:
        body = resp.text[:400]
    except Exception:  # pragma: no cover - defensive logging guard
        body = "<unable to read body>"

    _LOG.debug(
        "ACLED OAuth HTTP response",
        extra={
            "flow": flow,
            "status": resp.status_code,
            "url": OAUTH_TOKEN_URL,
            "body_snippet": body,
        },
    )


def _password_grant(username: str, password: str) -> Dict[str, str]:
    data = {
        "username": username,
        "password": password,
        "grant_type": "password",
        "client_id": OAUTH_CLIENT_ID,
    }
    resp = requests.post(
        OAUTH_TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    _log_token_http(resp, flow="password")
    if resp.status_code != 200:
        raise RuntimeError(f"ACLED OAuth password grant failed: status={resp.status_code}")
    return resp.json()


def _refresh_grant(refresh_token: str) -> Dict[str, str]:
    data = {
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
        "client_id": OAUTH_CLIENT_ID,
    }
    resp = requests.post(
        OAUTH_TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    _log_token_http(resp, flow="refresh")
    if resp.status_code != 200:
        raise RuntimeError(f"ACLED OAuth refresh grant failed: status={resp.status_code}")
    return resp.json()


def _set_cache(token: str, refresh_token: Optional[str]) -> None:
    _CACHE["access_token"] = token
    _CACHE["expiry"] = _jwt_exp(token)
    if refresh_token:
        _CACHE["refresh_token"] = refresh_token


def _resolve_refresh_token() -> Optional[str]:
    cached = _CACHE.get("refresh_token")
    if cached:
        return cached
    refresh_from_env = os.environ.get("ACLED_REFRESH_TOKEN")
    if refresh_from_env:
        _CACHE["refresh_token"] = refresh_from_env
        return refresh_from_env
    return None


def _resolve_password_creds() -> Optional[Dict[str, str]]:
    username = os.environ.get("ACLED_USERNAME")
    password = os.environ.get("ACLED_PASSWORD")
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
    if existing:
        expiry = _jwt_exp(existing)
        _LOG.debug(
            "Using environment-provided ACLED token",
            extra=_describe_token(existing, expiry),
        )
        _set_cache(existing, os.environ.get("ACLED_REFRESH_TOKEN"))
        return existing

    refresh_token = _resolve_refresh_token()
    if refresh_token:
        _LOG.debug(
            "Attempting ACLED refresh grant",
            extra={"token_length": len(refresh_token)},
        )
        try:
            tokens = _refresh_grant(refresh_token)
        except Exception as exc:  # pragma: no cover - network stack errors
            _LOG.debug("ACLED refresh grant failed", extra={"error": str(exc)})
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
            return access_token

    password_creds = _resolve_password_creds()
    if password_creds:
        _LOG.debug("Attempting ACLED password grant")
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
        return access_token

    raise RuntimeError(
        "ACLED authentication failed: set ACLED_ACCESS_TOKEN/ACLED_TOKEN or "
        "ACLED_REFRESH_TOKEN or ACLED_USERNAME/ACLED_PASSWORD."
    )


def get_auth_header() -> Dict[str, str]:
    """Return an Authorization header for ACLED requests."""

    token = get_access_token()
    return {"Authorization": f"Bearer {token}"}
