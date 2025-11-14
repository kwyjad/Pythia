"""Helper utilities for authenticating against the ACLED API."""
from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Dict, Optional

_TOKEN_URL = "https://acleddata.com/oauth/token"
_HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}
_CLIENT_ID = "acled"
_MIN_TTL = 300  # seconds

_LOG = logging.getLogger("resolver.ingestion.acled.auth")

_cached_token: Optional[str] = None
_cached_expiry: Optional[int] = None
_cached_refresh_token: Optional[str] = None


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


def _post(data: Dict[str, str]) -> Dict[str, str]:
    body = urllib.parse.urlencode(data).encode("utf-8")
    request = urllib.request.Request(_TOKEN_URL, data=body, headers=_HEADERS, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _exchange_refresh(refresh_token: str) -> Dict[str, str]:
    return _post(
        {
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
            "client_id": _CLIENT_ID,
        }
    )


def _password_grant(username: str, password: str) -> Dict[str, str]:
    return _post(
        {
            "username": username,
            "password": password,
            "grant_type": "password",
            "client_id": _CLIENT_ID,
        }
    )


def _set_cache(token: str, refresh_token: Optional[str]) -> None:
    global _cached_token, _cached_expiry, _cached_refresh_token

    _cached_token = token
    _cached_expiry = _jwt_exp(token)
    if refresh_token:
        _cached_refresh_token = refresh_token


def _resolve_refresh_token() -> Optional[str]:
    global _cached_refresh_token

    if _cached_refresh_token:
        return _cached_refresh_token
    refresh_from_env = os.environ.get("ACLED_REFRESH_TOKEN")
    if refresh_from_env:
        _cached_refresh_token = refresh_from_env
    return _cached_refresh_token


def _resolve_password_creds() -> Optional[Dict[str, str]]:
    username = os.environ.get("ACLED_USERNAME")
    password = os.environ.get("ACLED_PASSWORD")
    if username and password:
        return {"username": username, "password": password}
    return None


def _resolve_existing_token() -> Optional[str]:
    token = os.environ.get("ACLED_ACCESS_TOKEN")
    if token and _jwt_is_valid(token):
        return token
    legacy = os.environ.get("ACLED_TOKEN")
    if legacy and _jwt_is_valid(legacy):
        os.environ.setdefault("ACLED_ACCESS_TOKEN", legacy)
        return legacy
    return None


def get_access_token() -> str:
    """Return a valid ACLED access token, refreshing credentials when required."""

    global _cached_token, _cached_expiry

    now = int(time.time())
    if _cached_token and _cached_expiry and (_cached_expiry - now) > _MIN_TTL:
        _LOG.debug(
            "Using cached ACLED access token",
            extra=_describe_token(_cached_token, _cached_expiry),
        )
        return _cached_token

    existing = _resolve_existing_token()
    if existing:
        _LOG.debug(
            "Using environment-provided ACLED token",
            extra=_describe_token(existing, _jwt_exp(existing)),
        )
        _set_cache(existing, os.environ.get("ACLED_REFRESH_TOKEN"))
        return existing

    refresh_token = _resolve_refresh_token()
    if refresh_token:
        _LOG.debug("Attempting ACLED refresh grant", extra={"token_length": len(refresh_token)})
        try:
            tokens = _exchange_refresh(refresh_token)
        except Exception as exc:  # pragma: no cover - network stack errors
            _LOG.debug("ACLED refresh grant failed", extra={"error": str(exc)})
        else:
            access_token = tokens.get("access_token")
            if not access_token:
                raise RuntimeError("ACLED refresh grant response missing access_token")
            new_refresh = tokens.get("refresh_token")
            if new_refresh:
                os.environ["ACLED_REFRESH_TOKEN"] = new_refresh
            os.environ["ACLED_ACCESS_TOKEN"] = access_token
            _set_cache(access_token, tokens.get("refresh_token") or refresh_token)
            _LOG.debug(
                "Obtained ACLED access token via refresh",
                extra=_describe_token(access_token, _cached_expiry),
            )
            return access_token

    password_creds = _resolve_password_creds()
    if password_creds:
        _LOG.debug("Attempting ACLED password grant")
        tokens = _password_grant(password_creds["username"], password_creds["password"])
        access_token = tokens.get("access_token")
        if not access_token:
            raise RuntimeError("ACLED password grant response missing access_token")
        new_refresh = tokens.get("refresh_token")
        if new_refresh:
            os.environ["ACLED_REFRESH_TOKEN"] = new_refresh
        os.environ["ACLED_ACCESS_TOKEN"] = access_token
        _set_cache(access_token, new_refresh)
        _LOG.debug(
            "Obtained ACLED access token via password grant",
            extra=_describe_token(access_token, _cached_expiry),
        )
        return access_token

    raise RuntimeError(
        "ACLED authentication failed: provide ACLED_REFRESH_TOKEN or "
        "ACLED_USERNAME/ACLED_PASSWORD credentials."
    )


def get_auth_header() -> Dict[str, str]:
    """Return an Authorization header for ACLED requests."""

    token = get_access_token()
    return {"Authorization": f"Bearer {token}"}
