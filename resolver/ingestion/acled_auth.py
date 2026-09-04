# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Helper utilities for authenticating against the ACLED API.

Three things this module settles for every ACLED caller (Sept 2026, from
run 33841370196):

**One token per run.** Tokens are valid for 86,400 seconds, and the run made
separate token requests for ``acled_client``, ``acled_cast`` and
``pythia.acled_political`` plus retries, because each runs in its own
process and the cache was a module dict. The cache is now also a small
file (``ACLED_TOKEN_CACHE_PATH``, default under the system temp directory,
never inside the workspace an artifact upload could sweep up) that every
process reads before asking for a token and writes after obtaining one.

**The password grant comes first.** The stored refresh token has a ~14-day
TTL and the ingest runs monthly, so the refresh grant's normal, expected
outcome was ``invalid_grant`` — seven ERROR lines a run that trained the
reader to skip them. When password credentials exist they are used
directly; the refresh grant is tried only when they do not.

**An HTML body is a failure, never an empty result.** ACLED's gateway
answers an unauthenticated API call with a Drupal page titled
"Unauthorized" — with HTTP 200, and even when the request sets ``Accept:
application/json`` — so a 401, a WAF interstitial and a session expiry are
indistinguishable by status alone. :func:`parse_json_response` is the one
place every ACLED response is turned into JSON: it checks the content
type and the leading character before parsing and raises
:class:`AcledHtmlResponse` (status, URL, first 200 characters) on a page.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests

OAUTH_TOKEN_URL = "https://acleddata.com/oauth/token"
OAUTH_CLIENT_ID = "acled"
_MIN_TTL = 300  # seconds

#: Where the cross-process token cache lives. Empty string disables it.
#: Deliberately outside the repository checkout: ``resolver/staging`` and
#: ``diagnostics/`` are uploaded as artifacts, and a bearer token must never
#: travel in one.
_TOKEN_CACHE_ENV = "ACLED_TOKEN_CACHE_PATH"
_TOKEN_CACHE_DEFAULT = str(Path(tempfile.gettempdir()) / "pythia_acled_token.json")


class AcledResponseError(RuntimeError):
    """An ACLED response that cannot be used, described so the cause is readable."""


class AcledHtmlResponse(AcledResponseError):
    """ACLED served a web page where JSON was expected.

    This is the shape of an unauthenticated call, a WAF challenge and a
    session expiry alike, and it has already killed one monthly ingest. It
    is never zero records.
    """

    def __init__(self, *, what: str, status: int, url: str, snippet: str,
                 description: str = "") -> None:
        self.status = status
        self.url = url
        self.snippet = snippet
        message = (
            f"ACLED {what} returned an HTML page, not JSON "
            f"(status={status}, url={url!r}): {snippet!r}"
        )
        if description:
            message += f" | {description}"
        super().__init__(message)

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


def body_shape(resp: Any) -> str:
    """'html', 'json' or 'other', from the content type and the body's first character."""

    content_type = ""
    try:
        raw_type = resp.headers.get("Content-Type", "")
        content_type = raw_type if isinstance(raw_type, str) else ""
    except Exception:  # pragma: no cover - defensive
        pass
    head = ""
    try:
        text = resp.text
        head = text.lstrip()[:200] if isinstance(text, str) else ""
    except Exception:  # pragma: no cover - defensive
        head = ""
    if _looks_like_html(head, content_type):
        return "html"
    if "json" in content_type.lower() or head[:1] in ("{", "["):
        return "json"
    return "other"


def _status_of(resp: Any) -> Optional[int]:
    status = getattr(resp, "status_code", None)
    if isinstance(status, bool) or not isinstance(status, int):
        return None
    return status


def parse_json_response(resp: Any, *, what: str) -> Any:
    """The single path from an ACLED HTTP response to its JSON payload.

    Raises :class:`AcledHtmlResponse` when the body is a web page — whatever
    the status code says, because the gateway serves its "Unauthorized" page
    with a 200 — and :class:`AcledResponseError` for a non-200 status or a
    body that will not parse. Every message carries
    :func:`describe_response`, so the status, the content type, the URL
    reached after redirects and a redacted body snippet are in the log, and
    the four causes of "not JSON" can be told apart.
    """

    shape = body_shape(resp)
    if shape == "html":
        status = _status_of(resp) or 0
        url = ""
        try:
            url = _redact(str(resp.url or ""))
        except Exception:  # pragma: no cover - defensive
            pass
        snippet = _body_snippet(resp, limit=200)
        description = describe_response(resp)
        _LOG.error("ACLED %s served HTML | %s", what, description)
        raise AcledHtmlResponse(
            what=what, status=status, url=url, snippet=snippet, description=description,
        )

    status = _status_of(resp)
    if status is not None and status != 200:
        raise AcledResponseError(f"ACLED {what} failed: {describe_response(resp)}")

    try:
        return resp.json()
    except ValueError:
        raise AcledResponseError(
            f"ACLED {what} returned HTTP 200 with a body that is not JSON: "
            f"{describe_response(resp)}"
        ) from None


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

    # A 200 is not an answer. Until 2026-09-03 this branch was a bare
    # resp.json(), so a WAF page served with a 200 surfaced as "Expecting
    # value: line 1 column 1 (char 0)" and killed the monthly ingest with
    # nothing recorded about what had actually been served.
    payload = parse_json_response(resp, what=f"OAuth {flow} grant")

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
    _save_file_cache(token, refresh_token)


# ---------------------------------------------------------------------------
# The cross-process cache: one token per run
# ---------------------------------------------------------------------------

def _token_cache_path() -> Optional[Path]:
    raw = os.environ.get(_TOKEN_CACHE_ENV)
    if raw is None:
        raw = _TOKEN_CACHE_DEFAULT
    raw = raw.strip()
    return Path(raw) if raw else None


def _credential_fingerprint() -> str:
    """Which account the cached token belongs to, without naming it."""

    username = (os.environ.get("ACLED_USERNAME") or "").strip()
    return hashlib.sha256(username.encode("utf-8")).hexdigest()[:12] if username else ""


def _load_file_cache() -> Optional[Dict[str, Any]]:
    path = _token_cache_path()
    if path is None:
        return None
    try:
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a corrupt cache costs one grant
        _LOG.debug("ACLED token cache unreadable (%s); ignoring", exc)
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("fingerprint") != _credential_fingerprint():
        return None
    return payload


def _save_file_cache(token: str, refresh_token: Optional[str]) -> None:
    path = _token_cache_path()
    if path is None:
        return
    expiry = _jwt_exp(token)
    if not expiry:
        # An opaque token has no expiry to judge it by; caching it could
        # serve a dead token to every later process.
        return
    previous = _load_file_cache() or {}
    payload = {
        "access_token": token,
        "refresh_token": refresh_token or previous.get("refresh_token"),
        "expiry": expiry,
        "fingerprint": _credential_fingerprint(),
        "saved_at": int(time.time()),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 - caching is a saving, never a requirement
        _LOG.debug("ACLED token cache not written (%s)", exc)


def _resolve_file_cached_token() -> Optional[str]:
    cached = _load_file_cache()
    if not cached:
        return None
    token = cached.get("access_token")
    expiry = cached.get("expiry")
    if not token or not isinstance(expiry, int):
        return None
    if (expiry - int(time.time())) <= _MIN_TTL:
        return None
    _CACHE["access_token"] = str(token)
    _CACHE["expiry"] = expiry
    if cached.get("refresh_token"):
        _CACHE["refresh_token"] = str(cached["refresh_token"])
    return str(token)


def clear_token_cache() -> None:
    """Forget every cached token (tests, and a deliberate re-authentication)."""

    _CACHE.update({"access_token": None, "refresh_token": None, "expiry": None})
    path = _token_cache_path()
    if path is not None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except Exception:  # pragma: no cover - defensive
            pass


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
    """Return a valid ACLED access token, obtaining one only when none is cached.

    Order: the in-process cache, the cross-process file cache (the token
    the previous process of this run obtained), an environment-provided
    token, the password grant, and only then the refresh grant. The
    password grant precedes the refresh grant because the stored refresh
    token expires in ~14 days and the ingest runs monthly, so a refresh
    attempt's expected outcome was a failure line on every run.
    """

    now = int(time.time())
    cached_token = _CACHE.get("access_token")
    cached_expiry = _CACHE.get("expiry")
    if cached_token and isinstance(cached_expiry, int) and (cached_expiry - now) > _MIN_TTL:
        _LOG.debug(
            "Using cached ACLED access token",
            extra=_describe_token(cached_token, cached_expiry),
        )
        return cached_token

    file_cached = _resolve_file_cached_token()
    if file_cached:
        _LOG.info("[acled_auth] using the access token cached by an earlier process of this run")
        print("[acled_auth] Using the token cached by an earlier process of this run")
        return file_cached

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
        _LOG.debug("Environment-provided ACLED token is an expired JWT; falling through to the grants")

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

    refresh_token = _resolve_refresh_token()
    if refresh_token:
        _LOG.debug(
            "Attempting ACLED refresh grant (no password credentials configured)",
            extra={"token_length": len(refresh_token)},
        )
        print("[acled_auth] No password credentials; attempting refresh grant...")
        tokens = _refresh_grant(refresh_token)
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

    print("[acled_auth] All auth methods exhausted — no valid credentials found")
    raise RuntimeError(
        "ACLED authentication failed: set ACLED_ACCESS_TOKEN/ACLED_TOKEN or "
        "ACLED_USERNAME/ACLED_PASSWORD (or ACLED_REFRESH_TOKEN)."
    )


def get_auth_header() -> Dict[str, str]:
    """Return an Authorization header for ACLED requests."""

    token = get_access_token()
    return {"Authorization": f"Bearer {token}"}
