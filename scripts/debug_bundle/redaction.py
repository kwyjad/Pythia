# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Credential redaction for everything the bundle writes.

Two jobs, and they are not the same job.

``redact_env_value`` handles a value we hold because its NAME says it is a
secret. It is replaced by ``<redacted:sha256:xxxxxxxx>`` rather than by a
constant, so a reader can still answer the question that actually comes up
in an incident: is the key this run used the same key the last good run
used? A constant ``***`` cannot answer it.

``redact_text`` handles free text — captured workflow logs above all —
where a secret arrives inside a sentence nobody chose. It matches the
shapes providers actually issue, and it is deliberately greedy: a false
positive costs a reader one unreadable token, a false negative publishes a
key into an artifact anyone with repo read access can download.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

# Env var NAMES whose value is a secret. Substring match, case-insensitive:
# the point is to catch OPENAI_API_KEY, GH_TOKEN, ACAPS_PASSWORD and
# whatever is added next without anyone remembering to extend a list.
SECRET_NAME_PARTS: tuple[str, ...] = (
    "key",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "auth",
)

# Names that CONTAIN a secret-looking word but are not secrets. Without
# this, PYTHIA_PROMPT_CACHE_KEY (a cache-partition label) and the batch
# custom-id knobs come back redacted and the config table stops being
# readable.
SECRET_NAME_ALLOWLIST: frozenset[str] = frozenset(
    {
        "PYTHIA_API_KEY_HEADER",
        "PYTHIA_PROMPT_CACHE_KEY",
        "PYTHIA_KEY_TABLES",
        "CANONICAL_DB_ARTIFACT_NAME",
    }
)

_TOKEN_SHAPES: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),            # OpenAI
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"),        # Anthropic
    re.compile(r"AIza[A-Za-z0-9_\-]{16,}"),           # Google
    re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}"),        # GitHub
    re.compile(r"github_pat_[A-Za-z0-9_]{16,}"),      # GitHub fine-grained
    re.compile(r"BSA[A-Za-z0-9_\-]{16,}"),            # Brave Search
    re.compile(r"xox[abposr]-[A-Za-z0-9\-]{10,}"),    # Slack
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWT
)

# key=value / "key": "value" / Authorization: Bearer xxx, where the key half
# names a secret. Runs after the shape patterns so a recognised token is
# already gone.
_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Za-z0-9_\-]*(?:api[_\-]?key|token|secret|password|passwd|credential)"
    r"[A-Za-z0-9_\-]*)\b(\s*[:=]\s*)(\"?)([^\s\"',;&]{8,})",
)
_BEARER = re.compile(r"(?i)\b(bearer|basic)\s+([A-Za-z0-9_\-\.=+/]{12,})")

REDACTED_PREFIX = "<redacted:sha256:"


def fingerprint(value: str) -> str:
    """First eight hex of sha256 — enough to tell two secrets apart."""

    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:8]


def redaction_marker(value: str) -> str:
    return f"{REDACTED_PREFIX}{fingerprint(value)}>"


def is_secret_name(name: str) -> bool:
    """True when an env var NAME says its value is a credential."""

    if not name:
        return False
    if name.upper() in SECRET_NAME_ALLOWLIST:
        return False
    lowered = name.lower()
    return any(part in lowered for part in SECRET_NAME_PARTS)


def redact_env_value(name: str, value: str | None) -> str | None:
    """Redact by NAME, keeping a stable fingerprint of the value.

    An unset variable stays ``None`` and an empty one stays empty: "the key
    was missing" and "the key was wrong" are different failures and the
    bundle must not blur them.
    """

    if value is None:
        return None
    if not value:
        return ""
    if is_secret_name(name):
        return redaction_marker(value)
    return redact_text(value)


def redact_text(text: str | None) -> str:
    """Redact credential SHAPES inside free text (logs, error payloads)."""

    if not text:
        return text or ""
    out = text
    for pattern in _TOKEN_SHAPES:
        out = pattern.sub(lambda m: redaction_marker(m.group(0)), out)
    out = _ASSIGNMENT.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{redaction_marker(m.group(4))}",
        out,
    )
    out = _BEARER.sub(lambda m: f"{m.group(1)} {redaction_marker(m.group(2))}", out)
    return out


def redact_mapping(mapping: dict[str, object], *, name_keys: Iterable[str] = ()) -> dict[str, object]:
    """Redact a JSON-ish mapping in place-safe fashion.

    Keys named like secrets are fingerprinted; every string value is passed
    through the shape matcher, because a provider error payload routinely
    quotes the request it rejected.
    """

    extra = {k.lower() for k in name_keys}
    out: dict[str, object] = {}
    for key, value in mapping.items():
        if is_secret_name(str(key)) or str(key).lower() in extra:
            out[key] = redact_env_value(str(key), str(value)) if value is not None else None
        elif isinstance(value, str):
            out[key] = redact_text(value)
        elif isinstance(value, dict):
            out[key] = redact_mapping(value, name_keys=name_keys)
        elif isinstance(value, list):
            out[key] = [
                redact_mapping(v, name_keys=name_keys)
                if isinstance(v, dict)
                else (redact_text(v) if isinstance(v, str) else v)
                for v in value
            ]
        else:
            out[key] = value
    return out
