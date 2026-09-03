# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Redaction for anything that leaves the runner in a diagnostic bundle.

Two independent rules, because either one alone leaks:

* **By name** — a mapping key, a query parameter or an environment variable
  whose NAME matches :data:`SECRET_NAME_PATTERN` has its value replaced,
  whether or not that value looks like a secret.
* **By value** — every secret present in the environment at build time is
  replaced wherever it appears, including inside captured logs, response
  bodies and URLs that no name rule would have caught.

Secrets are **fingerprinted, never blanked**. ``<redacted:sha256:1f4c9a02>``
still answers "is this run's key the one the last good run used?", which a
constant mask cannot, and it is not reversible. That is the same convention
the debug bundle's own redaction uses; keep the two in step.

A value shorter than :data:`MIN_SECRET_LEN` is not searched for by value —
substring-replacing a four-character token would corrupt half the bundle for
no security gain. The name rule still covers it.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

#: Names that mark a value as a credential. Deliberately broad: a false
#: positive costs one unreadable field, a false negative publishes a key.
SECRET_NAME_PATTERN = re.compile(
    r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|EMAIL|CLIENT_ID|AUTH|BEARER|COOKIE",
    re.IGNORECASE,
)

#: Environment variable names that MATCH the pattern but are not credentials.
#: Without this the bundle would redact its own configuration — and a reader
#: who cannot see which appname was sent cannot diagnose a ReliefWeb refusal.
SECRET_NAME_ALLOWLIST = frozenset(
    {
        "RELIEFWEB_APPNAME",
        "IDMC_HELIX_ENV",
        "PYTHIA_LLM_PROFILE",
        "MODEL_COSTS_JSON",
    }
)

#: Below this length a secret is redacted by NAME only. Substring-replacing a
#: short value would mangle unrelated text.
MIN_SECRET_LEN = 8


def fingerprint(value: str) -> str:
    """``<redacted:sha256:xxxxxxxx>`` for a secret value."""

    digest = hashlib.sha256(str(value).encode("utf-8", "replace")).hexdigest()[:8]
    return f"<redacted:sha256:{digest}>"


def is_secret_name(name: str) -> bool:
    """Does this variable / parameter / key name mark a credential?"""

    text = str(name or "")
    if text.upper() in SECRET_NAME_ALLOWLIST:
        return False
    return bool(SECRET_NAME_PATTERN.search(text))


def secret_values(environ: Mapping[str, str] | None = None) -> list[str]:
    """Every credential value in the environment, longest first.

    Longest first matters: a key that contains a shorter key as a prefix
    must be replaced whole, or the tail of it survives in the output.
    """

    env = os.environ if environ is None else environ
    found = {
        str(value)
        for name, value in env.items()
        if is_secret_name(name) and value and len(str(value)) >= MIN_SECRET_LEN
    }
    return sorted(found, key=len, reverse=True)


def redact_text(text: str, values: Iterable[str] | None = None) -> str:
    """Replace every known secret value in ``text`` with its fingerprint."""

    if not text:
        return text
    out = str(text)
    for value in values if values is not None else secret_values():
        if value and value in out:
            out = out.replace(value, fingerprint(value))
    return out


def redact_url(url: str, values: Iterable[str] | None = None) -> str:
    """Redact a URL by query-parameter name AND by secret value.

    The path and host survive: "which URL did the connector call" is the
    question the bundle exists to answer, and the answer is worthless with
    the route removed.
    """

    if not url:
        return url
    values = list(values) if values is not None else secret_values()
    try:
        parts = urlsplit(str(url))
    except Exception:  # pragma: no cover - urlsplit is total in practice
        return redact_text(str(url), values)

    query = parts.query
    if query:
        pairs = parse_qsl(query, keep_blank_values=True)
        query = urlencode(
            [
                (key, fingerprint(val) if (is_secret_name(key) and val) else val)
                for key, val in pairs
            ],
            # Keep the fingerprint legible: percent-encoding "<redacted:...>"
            # makes the one field a reader most wants to compare unreadable.
            safe="<>:",
        )

    netloc = parts.netloc
    if "@" in netloc:  # user:password@host
        netloc = "<redacted:userinfo>@" + netloc.rsplit("@", 1)[1]

    rebuilt = urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))
    return redact_text(rebuilt, values)


def redact_obj(obj: Any, values: Iterable[str] | None = None) -> Any:
    """Recursively redact a JSON-shaped structure, by key name and by value."""

    values = list(values) if values is not None else secret_values()
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for key, val in obj.items():
            if is_secret_name(str(key)) and isinstance(val, (str, int, float)) and val != "":
                out[str(key)] = fingerprint(str(val))
            else:
                out[str(key)] = redact_obj(val, values)
        return out
    if isinstance(obj, (list, tuple)):
        return [redact_obj(item, values) for item in obj]
    if isinstance(obj, str):
        return redact_text(obj, values)
    return obj


def find_secrets(text: str, values: Iterable[str] | None = None) -> list[str]:
    """Which known secret values appear verbatim in ``text``.

    The post-assembly scan calls this over every file in the bundle. It
    returns the FINGERPRINTS of the offending values, never the values —
    a leak report that quotes the leak is a second leak.
    """

    if not text:
        return []
    values = list(values) if values is not None else secret_values()
    return sorted({fingerprint(v) for v in values if v and v in text})
