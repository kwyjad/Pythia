# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Why does ACLED's token endpoint answer HTTP 200 with something that is not JSON?

On 2026-09-03 the monthly Resolver Update died in Phase 1. Both OAuth grants
were answered ``200`` with a body ``json.loads`` refused, so the ACLED
connector wrote zero rows and exited 0, and the monthly-fatalities CLI then
crashed on a bare ``JSONDecodeError``. Nothing in the run recorded the status,
the content type, the URL reached after redirects, or one character of the
body, so the cause could be settled nowhere.

Four explanations fit, and they call for four different repairs:

**A WAF page.** ACLED sits behind Cloudflare and GitHub runner addresses are
routinely challenged. The body is HTML and names the challenge. Fix: none in
this repo. Ask ACLED to allow the runner, or move the pull off CI.

**A moved route.** ``acleddata.com/oauth/token`` now serves the website, so any
POST to it returns the marketing page with a cheerful 200. Fix: the URL in
``resolver/ingestion/acled_auth.py``.

**Content negotiation.** The gateway serves HTML to a request that did not ask
for JSON. Fix: already applied. The grants now send ``Accept:
application/json`` and a real User-Agent, and this script's ``bare_headers``
probe is what proves whether that was the cause.

**The credentials.** An expired token or a closed account. The body says so in
words, usually with a 4xx. Fix: rotate the secret.

Read-only. Never writes to the database, always exits 0; "the endpoint serves
a login page" is a finding, not a build failure. Response bodies are redacted
against the environment's own secrets before they are printed or stored: an
OAuth refusal quotes the username back, and this report lands in a public
artifact.

    python -m scripts.ci.diagnose_acled_auth
    python -m scripts.ci.diagnose_acled_auth --out diagnostics/acled_auth_diagnostic.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from resolver.diagnostics.redaction import redact_text
from resolver.ingestion import acled_auth

TIMEOUT = 30
BODY_CHARS = 1200

#: Candidate token routes, tried only to tell "moved" from "challenged". The
#: first is what the code uses; the rest are guesses, and the report says so.
CANDIDATE_URLS = (
    acled_auth.OAUTH_TOKEN_URL,
    "https://api.acleddata.com/oauth/token",
    "https://acleddata.com/api/oauth/token",
)

VERDICT_OK = "ok"
VERDICT_WAF = "waf_challenge"
VERDICT_MOVED = "endpoint_moved"
VERDICT_HTML = "html_not_json"
VERDICT_CREDENTIALS = "credentials_rejected"
VERDICT_UPSTREAM = "upstream_error"
VERDICT_INCONCLUSIVE = "inconclusive"

_WAF_WORDS = (
    "cloudflare",
    "just a moment",
    "attention required",
    "captcha",
    "checking your browser",
    "ray id",
    "access denied",
    "request blocked",
)
_CREDENTIAL_WORDS = (
    "invalid_grant",
    "invalid_client",
    "invalid credentials",
    "incorrect password",
    "unauthorized",
    "no user",
    "not registered",
    "expired",
    "subscription",
)

RequestFn = Callable[..., Any]


def _default_request(method: str, url: str, **kwargs: Any) -> Any:
    import requests

    return requests.request(method, url, timeout=TIMEOUT, **kwargs)


def _summarise(resp: Any) -> Dict[str, Any]:
    content_type = ""
    try:
        content_type = resp.headers.get("Content-Type", "") or ""
    except Exception:
        pass
    try:
        text = " ".join((resp.text or "").split())
    except Exception:
        text = ""
    body = redact_text(text)[:BODY_CHARS]

    parsed: Optional[str] = None
    has_access_token = False
    try:
        payload = resp.json()
    except Exception:
        parsed = None
    else:
        parsed = type(payload).__name__
        if isinstance(payload, dict):
            has_access_token = bool(payload.get("access_token"))

    try:
        final_url = redact_text(resp.url or "")
    except Exception:
        final_url = ""
    try:
        redirects = [h.status_code for h in (resp.history or [])]
    except Exception:
        redirects = []

    return {
        "status": getattr(resp, "status_code", None),
        "content_type": content_type,
        "final_url": final_url,
        "redirects": redirects,
        "json_type": parsed,
        "is_json": parsed is not None,
        "has_access_token": has_access_token,
        "looks_like_html": acled_auth._looks_like_html(body, content_type),
        "body_head": body,
    }


def _probe(request: RequestFn, name: str, method: str, url: str, **kwargs: Any) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"probe": name, "method": method, "url": url}
    try:
        resp = request(method, url, **kwargs)
    except Exception as exc:
        entry["error"] = redact_text(f"{type(exc).__name__}: {exc}")
        entry["status"] = None
        entry["is_json"] = False
        return entry
    entry.update(_summarise(resp))
    return entry


def decide(probes: Dict[str, Dict[str, Any]], *, have_credentials: bool) -> tuple[str, str]:
    """Name the cause from the probe results. Pure; no IO, no network."""

    if not have_credentials:
        return VERDICT_INCONCLUSIVE, (
            "neither ACLED_USERNAME/ACLED_PASSWORD nor ACLED_REFRESH_TOKEN is set; "
            "no grant was attempted"
        )

    grants = [p for name, p in probes.items() if name in ("password", "refresh", "bare_headers")]
    if any(p.get("has_access_token") for p in grants):
        which = next(n for n, p in probes.items() if p.get("has_access_token"))
        if which == "bare_headers":
            return VERDICT_OK, "the endpoint answers, and it answered the header-less request too"
        return VERDICT_OK, f"the {which} grant returned an access token, so authentication is working"

    bodies = " ".join(str(p.get("body_head") or "").lower() for p in probes.values())

    if any(word in bodies for word in _WAF_WORDS):
        return VERDICT_WAF, (
            "the body carries a WAF challenge, so the request never reached the API. "
            "ask ACLED to allow the runner, or move the pull off shared CI addresses"
        )

    json_refusals = [p for p in grants if p.get("is_json") and not p.get("has_access_token")]
    if json_refusals or any(p.get("status") in (400, 401, 403) for p in grants):
        return VERDICT_CREDENTIALS, (
            "the endpoint answered as an API and refused the credentials. Rotate "
            "ACLED_REFRESH_TOKEN / ACLED_USERNAME / ACLED_PASSWORD"
        )

    alternates = {
        name: p for name, p in probes.items() if name.startswith("candidate:") and p.get("is_json")
    }
    if alternates:
        return VERDICT_MOVED, (
            "a different route answers with JSON while the configured one does not: "
            f"{sorted(alternates)}. Update OAUTH_TOKEN_URL in resolver/ingestion/acled_auth.py"
        )

    configured_get = probes.get("get_token_url", {})
    if configured_get.get("looks_like_html") and configured_get.get("status") == 200:
        if configured_get.get("redirects"):
            return VERDICT_MOVED, (
                "the token route redirects to a page that is not an API "
                f"(hops {configured_get['redirects']}, final {configured_get.get('final_url')!r})"
            )
        return VERDICT_HTML, (
            "the token route serves HTML with a 200, no challenge and no redirect: "
            "either the route is gone or the gateway is content-negotiating; compare the "
            "bare_headers probe against the password probe"
        )

    statuses = {p.get("status") for p in probes.values() if p.get("status") is not None}
    if statuses and statuses <= {500, 502, 503, 504}:
        return VERDICT_UPSTREAM, f"every probe failed the same way (statuses {sorted(statuses)})"

    if any(p.get("looks_like_html") for p in grants):
        return VERDICT_HTML, "the grants were answered with HTML rather than JSON"

    return VERDICT_INCONCLUSIVE, f"mixed answers: statuses {sorted(str(s) for s in statuses)}"


def run(request: RequestFn | None = None) -> Dict[str, Any]:
    request = request or _default_request

    username = (os.environ.get("ACLED_USERNAME") or "").strip()
    password = (os.environ.get("ACLED_PASSWORD") or "").strip()
    refresh = (os.environ.get("ACLED_REFRESH_TOKEN") or "").strip()
    have_credentials = bool((username and password) or refresh)

    url = acled_auth.OAUTH_TOKEN_URL
    headers = acled_auth._token_headers()
    probes: Dict[str, Dict[str, Any]] = {}

    # What does the route serve at all? Answers "moved" before any credential
    # is spent, and is the only probe that runs without credentials.
    probes["get_token_url"] = _probe(request, "get_token_url", "GET", url, headers=headers)

    if username and password:
        form = {
            "username": username,
            "password": password,
            "grant_type": "password",
            "client_id": acled_auth.OAUTH_CLIENT_ID,
            "scope": "authenticated",
        }
        probes["password"] = _probe(request, "password", "POST", url, data=form, headers=headers)
        # The pre-fix request shape: no Accept, no User-Agent. If this one is
        # served HTML and the probe above is served JSON, content negotiation
        # was the whole story.
        probes["bare_headers"] = _probe(
            request,
            "bare_headers",
            "POST",
            url,
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if refresh:
        probes["refresh"] = _probe(
            request,
            "refresh",
            "POST",
            url,
            data={
                "refresh_token": refresh,
                "grant_type": "refresh_token",
                "client_id": acled_auth.OAUTH_CLIENT_ID,
            },
            headers=headers,
        )

    # Only ask the alternates when the configured route has already failed ;
    # a working endpoint needs no search, and these URLs are guesses.
    if not any(p.get("has_access_token") for p in probes.values()):
        for candidate in CANDIDATE_URLS[1:]:
            probes[f"candidate:{candidate}"] = _probe(
                request, f"candidate:{candidate}", "GET", candidate, headers=headers
            )

    verdict, reason = decide(probes, have_credentials=have_credentials)
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "token_url": url,
        "user_agent": headers.get("User-Agent"),
        "have_credentials": have_credentials,
        "credentials_present": {
            "username_password": bool(username and password),
            "refresh_token": bool(refresh),
        },
        "verdict": verdict,
        "reason": reason,
        "probes": probes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose ACLED OAuth from CI")
    parser.add_argument("--out", default=None, help="Write the JSON report here")
    args = parser.parse_args(argv)

    report = run()
    print(f"[acled_auth] verdict: {report['verdict']}; {report['reason']}")
    print(f"[acled_auth] token_url: {report['token_url']}")
    for name, probe in report["probes"].items():
        print(
            f"[acled_auth] probe {name}: status={probe.get('status')} "
            f"json={probe.get('json_type')} token={probe.get('has_access_token')} "
            f"html={probe.get('looks_like_html')} ct={probe.get('content_type')!r} "
            f"redirects={probe.get('redirects')}"
        )
        head = str(probe.get("body_head") or probe.get("error") or "")[:300]
        if head:
            print(f"[acled_auth]   body: {head}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[acled_auth] report written to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim
    sys.exit(main())
