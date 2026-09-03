# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""What the run had to retry, and what it waited on.

``attempts_used`` and ``backoffs_sec`` have been written into every call's
usage since the retry loop was built, and nothing has ever read them. So a
run in which a third of the calls to one model needed two attempts, and
which spent quarter of an hour asleep in backoff, was indistinguishable
from a clean one — the latency table showed the last attempt's duration and
the cost table showed a bill nobody could account for.

The reason classes are the ones that call for different responses: a rate
limit means slow down, a server error means the provider is unwell, a
timeout means the ceiling is too low for the model, a parse failure means
the output is wrong rather than the transport, and a billing error means
the account is out of credit and every later call will fail the same way.

Circuit breaker trips are counted here too. The Brave breaker's trip is the
one that silently changes what the forecast SEES: a tripped breaker blocks
ungrounded hazards from forecasting at all, and the only trace it leaves in
the database is a sentinel in ``model_id``.
"""

from __future__ import annotations

import json
import re
from typing import Any

FIELDNAMES = [
    "phase", "provider", "model_id", "n_calls", "n_attempts",
    "n_calls_needing_retry", "n_extra_attempts", "total_backoff_sec",
    "n_rate_limit", "n_server_error", "n_timeout", "n_parse_failure",
    "n_auth", "n_billing", "n_other_error", "n_credit_retries",
    "credit_retry_pause_sec", "n_breaker_short_circuits",
]

# Sentinels that mean the call never reached any backend. A breaker trip is
# a run-shaping event, not a slow call.
_BREAKER_SENTINELS = {"grounding-breaker-tripped"}
_NO_BACKEND_SENTINELS = {
    "grounding-failed", "grounding-breaker-tripped", "grounding-unavailable",
    "grounding-disabled", "grounding-budget-exceeded",
}

_RATE_LIMIT = re.compile(r"429|rate.?limit|RESOURCE_EXHAUSTED|too many requests", re.I)
_SERVER = re.compile(r"\b5\d\d\b|internal server|service unavailable|overloaded|bad gateway", re.I)
_TIMEOUT = re.compile(r"timeout|timed.?out|DEADLINE_EXCEEDED|read timed", re.I)
_PARSE = re.compile(
    r"json|parse|decode|unmarshal|malformed|no spds|expecting value|"
    r"expecting property|invalid literal|unterminated string",
    re.I,
)
_AUTH = re.compile(r"401|403|unauthorized|forbidden|invalid.?api.?key|authentication", re.I)
_BILLING = re.compile(r"quota|billing|insufficient|credit|payment", re.I)


def classify(error_text: str | None) -> str:
    """One error class per row. Order matters: a 429 that also says quota is
    a billing error, and treating it as a rate limit invites a retry that
    cannot succeed."""

    text = (error_text or "").strip()
    if not text:
        return "none"
    if _BILLING.search(text):
        return "billing"
    if _AUTH.search(text):
        return "auth"
    if _RATE_LIMIT.search(text):
        return "rate_limit"
    if _TIMEOUT.search(text):
        return "timeout"
    if _SERVER.search(text):
        return "server_error"
    if _PARSE.search(text):
        return "parse_failure"
    return "other_error"


def _usage(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def collect(con, *, predicate: str | None, params: list[Any]) -> list[dict[str, Any]]:
    """Per (phase, provider, model) retry rows. Never raises."""

    if not predicate:
        return []
    try:
        cur = con.execute(
            f"""
            SELECT COALESCE(phase,'') AS phase, COALESCE(provider,'') AS provider,
                   COALESCE(model_id,'') AS model_id,
                   COALESCE(error_text,'') AS error_text,
                   usage_json
            FROM llm_calls
            WHERE {predicate}
            """,
            params,
        )
    except Exception:
        return []
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    agg: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["phase"]), str(row["provider"]), str(row["model_id"]))
        entry = agg.get(key)
        if entry is None:
            entry = {name: 0 for name in FIELDNAMES}
            entry.update({"phase": key[0], "provider": key[1], "model_id": key[2]})
            entry["total_backoff_sec"] = 0.0
            entry["credit_retry_pause_sec"] = 0.0
            agg[key] = entry

        usage = _usage(row.get("usage_json"))
        # attempts_used is 1 on a call that succeeded first time, so
        # attempts minus calls is the number of retries actually spent.
        attempts = int(usage.get("attempts_used") or 1)
        entry["n_calls"] += 1
        entry["n_attempts"] += attempts
        if attempts > 1:
            entry["n_calls_needing_retry"] += 1
            entry["n_extra_attempts"] += attempts - 1
        backoffs = usage.get("backoffs_sec")
        if isinstance(backoffs, list):
            entry["total_backoff_sec"] += float(sum(float(b or 0.0) for b in backoffs))
        entry["n_credit_retries"] += int(usage.get("credit_retries_used") or 0)
        entry["credit_retry_pause_sec"] += float(usage.get("credit_retry_pauses_sec") or 0.0)

        cls = classify(row.get("error_text"))
        if cls != "none":
            entry[f"n_{cls}"] = int(entry.get(f"n_{cls}") or 0) + 1
        if str(row["model_id"]).strip() in _BREAKER_SENTINELS:
            entry["n_breaker_short_circuits"] += 1

    out = list(agg.values())
    for entry in out:
        entry["total_backoff_sec"] = round(float(entry["total_backoff_sec"]), 1)
        entry["credit_retry_pause_sec"] = round(float(entry["credit_retry_pause_sec"]), 1)
    out.sort(key=lambda r: (-int(r["n_extra_attempts"]), -int(r["n_calls"]), r["phase"]))
    return out


def breaker_summary(con, *, predicate: str | None, params: list[Any]) -> dict[str, Any]:
    """Circuit breaker evidence, from the sentinels the run left behind.

    The Brave breaker resets per HS run and its stats live in process
    memory, so the durable trace is the sentinel model ids the grounding
    log sites write when no backend was reached.
    """

    out: dict[str, Any] = {"brave_breaker_short_circuits": 0, "no_backend_calls": 0, "by_sentinel": {}}
    if not predicate:
        return out
    try:
        cur = con.execute(
            f"""
            SELECT COALESCE(model_id,'') AS model_id, COUNT(*) AS n
            FROM llm_calls WHERE {predicate} GROUP BY 1
            """,
            params,
        )
    except Exception:
        return out
    for model_id, n in cur.fetchall():
        name = str(model_id).strip()
        if name in _NO_BACKEND_SENTINELS:
            out["by_sentinel"][name] = int(n)
            out["no_backend_calls"] += int(n)
            if name in _BREAKER_SENTINELS:
                out["brave_breaker_short_circuits"] += int(n)
    return out
