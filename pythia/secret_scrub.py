# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Remove credentials from text before it is stored or published.

One list of credential shapes serves three readers: the writers that store
error and debug text, the release builder that scrubs the published DB, and
the debug bundle. Two lists would drift, and the drift would be a leak.

The patterns are written once as plain strings so the same text compiles in
Python's ``re`` and in DuckDB's RE2 (``regexp_replace``). Both engines read
``\\b``, character classes and ``{n,}`` identically; nothing here uses a
feature only one of them has.

Every shape is anchored on a word boundary. Without it ``sk-`` matches inside
"risk-assessment-framework", and the release scrub runs over model prompts
and answers, where a false positive would quietly rewrite published text.
"""

from __future__ import annotations

import re
from typing import Iterable

REDACTED = "<redacted>"

# Provider credential shapes. Order matters only for readability: the
# replacement is the same marker for all of them.
TOKEN_SHAPE_PATTERNS: tuple[str, ...] = (
    r"\bsk-ant-[A-Za-z0-9_\-]{16,}",          # Anthropic
    r"\bsk-[A-Za-z0-9_\-]{16,}",              # OpenAI
    r"\bAIza[A-Za-z0-9_\-]{16,}",             # Google
    r"\bgh[pousr]_[A-Za-z0-9]{16,}",          # GitHub
    r"\bgithub_pat_[A-Za-z0-9_]{16,}",        # GitHub fine-grained
    r"\bBSA[A-Za-z0-9_\-]{16,}",              # Brave Search
    r"\bxox[abposr]-[A-Za-z0-9\-]{10,}",      # Slack
    r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",  # JWT
)

# A credential carried as a query parameter: the IPC key has no fixed shape,
# so it is caught by its parameter name instead. Group 1 keeps the name, so
# a reader can still see WHICH parameter was redacted.
#
# The value must be at least 12 characters. Public URLs quoted in model
# research carry short parameters that share the name: the live release held
# an ADRC disaster-database link ending ``&Key=28xx`` in a Sibyl trace, and a
# looser pattern rewrote it. A real API key or client id is far longer.
QUERY_SECRET_PATTERN = (
    r"([?&](?:key|api_key|apikey|client_id|access_token|token)=)[^&\s\"'<>]{12,}"
)

TOKEN_SHAPES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p) for p in TOKEN_SHAPE_PATTERNS
)
_QUERY_SECRET = re.compile(QUERY_SECRET_PATTERN, re.IGNORECASE)

# (pattern, replacement) pairs for DuckDB's regexp_replace. RE2 writes a
# back-reference as \\1, the same as Python.
SQL_PATTERNS: tuple[tuple[str, str], ...] = tuple(
    (p, REDACTED) for p in TOKEN_SHAPE_PATTERNS
) + (("(?i)" + QUERY_SECRET_PATTERN, "\\1" + REDACTED),)

# One alternation of everything above, for a single cheap existence scan per
# column before any per-pattern rewrite runs.
COMBINED_PATTERN = "|".join(
    [f"(?:{p})" for p in TOKEN_SHAPE_PATTERNS] + [f"(?i:{QUERY_SECRET_PATTERN})"]
)


def _env_secret_values() -> list[str]:
    try:
        from resolver.diagnostics.redaction import secret_values

        return secret_values()
    except Exception:  # noqa: BLE001 - scrubbing must never raise
        return []


def scrub_text(text, values: Iterable[str] | None = None):
    """Return ``text`` with every credential replaced by ``<redacted>``.

    Shapes first, then query parameters, then the exact secret values held
    in the environment (which catches a key whose shape nobody listed).
    Anything that is not a non-empty string comes back unchanged, and the
    function never raises: it sits on write paths that must not fail.
    """

    if not isinstance(text, str) or not text:
        return text
    try:
        out = text
        for pattern in TOKEN_SHAPES:
            out = pattern.sub(REDACTED, out)
        out = _QUERY_SECRET.sub(lambda m: m.group(1) + REDACTED, out)
        for value in values if values is not None else _env_secret_values():
            if value and value in out:
                out = out.replace(value, REDACTED)
        return out
    except Exception:  # noqa: BLE001
        return text


def contains_secret(text, values: Iterable[str] | None = None) -> bool:
    """True when ``scrub_text`` would change ``text``."""

    return isinstance(text, str) and bool(text) and scrub_text(text, values) != text
