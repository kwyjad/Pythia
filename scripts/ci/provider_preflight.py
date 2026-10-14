# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Ask each LLM provider one minimal question before a run spends anything.

On 13 October 2026 Anthropic answered every call with "You have reached your
specified API usage limits. You will regain access on 2026-11-01". Nothing
asked before the run started, so the ensemble lost its Claude member call by
call and the interpreter failed at the end. This step asks first.

One call per provider, on the cheapest model the registry names for it
(``PREFLIGHT_ROLES``; ``PYTHIA_PREFLIGHT_MODELS`` overrides, e.g.
``anthropic:claude-haiku-4-5-20251001,openai:gpt-6-luna``). A usage limit or
an account restriction is per account, so the cheapest model answers for
the dearest. Each answer is classified by ``forecaster.providers.
classify_provider_error``:

* ``ok``           the provider answered (a truncated answer still counts);
* ``usage_limit``  the account's spending limit is reached; terminal;
* ``invalid_key``  the key is refused; terminal;
* ``billing``      credit or quota exhausted; may recover (auto-recharge);
* ``missing_key``  no key in this job's environment;
* ``error``        anything else (a timeout, a 5xx).

A provider whose status is terminal is BLOCKED: with ``--apply-block`` its
name is appended to ``PYTHIA_BLOCK_PROVIDERS`` in ``$GITHUB_ENV``, so later
steps drop its ensemble member, and the stage carries the list to the next
stage through the batch-state artifact. Each blocked provider gets an
``::error::`` annotation. Step outputs: ``blocked_providers`` (comma list),
``anthropic_ok``, ``openai_ok``, ``google_ok`` (``true``/``false``), and
``reason`` (one line for the step summary).

Writes ``--out`` (default ``diagnostics/provider_preflight.json``). Never
fails the step: it always exits 0. The decision is in the outputs.

Usage:
    python -m scripts.ci.provider_preflight --providers openai,anthropic,google \
        --out diagnostics/provider_preflight.json --apply-block
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PREFLIGHT_PROMPT = "Reply with the single word OK."
DEFAULT_PROVIDERS = ("openai", "anthropic", "google")
#: The cheapest model the registry names per provider.
PREFLIGHT_ROLES = {
    "anthropic": "grounding_claude",
    "openai": "grounding_openai_fallback",
    "google": "grounding_gemini",
}
PREFLIGHT_FALLBACK_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-6-luna",
    "google": "gemini-2.5-flash",
}
#: Statuses that block a provider for the run.
BLOCKING_STATUSES = frozenset({"usage_limit", "invalid_key"})
STATUS_OK = "ok"

_REGAIN_RE = re.compile(r"regain access on\s+(\d{4}-\d{2}-\d{2})(?:\s+at\s+(\d{2}:\d{2}))?", re.I)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def regain_at(error_text: str) -> Optional[str]:
    """The date a usage limit lifts, when the provider states it."""
    m = _REGAIN_RE.search(error_text or "")
    if not m:
        return None
    return m.group(1) + (f"T{m.group(2)}Z" if m.group(2) else "")


def preflight_models(providers: List[str]) -> Dict[str, str]:
    """provider -> model id for the preflight call."""
    override = (os.getenv("PYTHIA_PREFLIGHT_MODELS") or "").strip()
    chosen: Dict[str, str] = {}
    for part in override.split(","):
        if ":" in part:
            prov, model = part.split(":", 1)
            chosen[prov.strip().lower()] = model.strip()
    out: Dict[str, str] = {}
    for prov in providers:
        if prov in chosen:
            out[prov] = chosen[prov]
            continue
        model = PREFLIGHT_FALLBACK_MODELS.get(prov, "")
        try:
            from pythia.llm_profiles import get_role_model  # noqa: PLC0415

            resolved = get_role_model(PREFLIGHT_ROLES.get(prov, "")) or ""
            r_prov, _, r_model = resolved.partition(":")
            if r_prov.strip().lower() == prov and r_model.strip():
                model = r_model.strip()
        except Exception:  # noqa: BLE001
            pass
        out[prov] = model
    return out


def _default_call(provider: str, model: str) -> Tuple[str, Optional[str]]:
    """(text, error) from one minimal call."""
    from forecaster import providers as p  # noqa: PLC0415

    if provider == "anthropic":
        r = p.call_anthropic(PREFLIGHT_PROMPT, model, 0.0, max_tokens_override=16,
                             timeout_sec=_env_float("PYTHIA_PREFLIGHT_TIMEOUT_SEC", 60.0))
    elif provider == "openai":
        r = p.call_openai(PREFLIGHT_PROMPT, model, 0.0)
    elif provider in ("google", "gemini"):
        r = p.call_google(PREFLIGHT_PROMPT, model, 0.0,
                          timeout_sec=_env_float("PYTHIA_PREFLIGHT_TIMEOUT_SEC", 60.0))
    else:
        return "", f"unknown provider {provider}"
    return r.text or "", r.error


def classify(provider: str, text: str, error: Optional[str]) -> str:
    from forecaster.providers import classify_provider_error  # noqa: PLC0415

    if not error:
        return STATUS_OK
    lower = error.lower()
    if lower.startswith("missing ") and "api_key" in lower:
        return "missing_key"
    kind = classify_provider_error(provider, error)
    if kind:
        return kind
    # A truncated answer means the account answered.
    if "max_tokens" in lower or "truncat" in lower:
        return STATUS_OK
    return "error"


def run(
    providers: List[str],
    *,
    call: Callable[[str, str], Tuple[str, Optional[str]]] = _default_call,
) -> Dict[str, Any]:
    models = preflight_models(providers)
    results: List[Dict[str, Any]] = []
    for prov in providers:
        model = models.get(prov, "")
        t0 = time.time()
        try:
            text, error = call(prov, model)
        except Exception as exc:  # noqa: BLE001
            text, error = "", f"{type(exc).__name__}: {exc}"
        status = classify(prov, text, error)
        try:
            from pythia.secret_scrub import scrub_text  # noqa: PLC0415

            shown = scrub_text(error or "")[:500]
        except Exception:  # noqa: BLE001
            shown = (error or "")[:500]
        results.append({
            "provider": prov,
            "model": model,
            "status": status,
            "blocking": status in BLOCKING_STATUSES,
            "error": shown or None,
            "regain_at": regain_at(error or ""),
            "elapsed_sec": round(time.time() - t0, 2),
        })
    blocked = sorted(r["provider"] for r in results if r["blocking"])
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "providers": results,
        "blocked_providers": blocked,
        "ok": not blocked,
    }


def reason_line(report: Dict[str, Any]) -> str:
    parts = []
    for r in report.get("providers") or []:
        if r["status"] == STATUS_OK:
            continue
        line = f"{r['provider']} {r['status']}"
        if r.get("regain_at"):
            line += f" until {r['regain_at']}"
        parts.append(line)
    return "; ".join(parts) or "all providers answered"


def merged_block_list(existing: str, blocked: List[str]) -> str:
    names = [p.strip().lower() for p in (existing or "").split(",") if p.strip()]
    for b in blocked:
        if b not in names:
            names.append(b)
    return ",".join(names)


def _write_kv(path: Optional[str], pairs: Dict[str, str]) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        for k, v in pairs.items():
            fh.write(f"{k}={v}\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--providers", default=",".join(DEFAULT_PROVIDERS))
    ap.add_argument("--out", default="diagnostics/provider_preflight.json")
    ap.add_argument("--apply-block", action="store_true",
                    help="append blocked providers to PYTHIA_BLOCK_PROVIDERS in $GITHUB_ENV")
    args = ap.parse_args(argv)
    providers = [p.strip().lower() for p in args.providers.split(",") if p.strip()]
    try:
        report = run(providers)
    except Exception as exc:  # noqa: BLE001
        report = {"checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "providers": [], "blocked_providers": [], "ok": True,
                  "error": f"{type(exc).__name__}: {exc}"}
    reason = reason_line(report)
    report["reason"] = reason
    try:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"::warning title=Provider preflight::could not write {args.out}: {exc}")

    for r in report.get("providers") or []:
        print(f"[preflight] {r['provider']:<9} {r['model']:<28} {r['status']}"
              + (f"  {r['error'][:160]}" if r.get("error") else ""))
        if r["blocking"]:
            print(f"::error title=Provider preflight::{r['provider']} is {r['status']}"
                  + (f" until {r['regain_at']}" if r.get("regain_at") else "")
                  + ": its ensemble member is dropped and steps that need it do not start.")
        elif r["status"] not in (STATUS_OK,):
            print(f"::warning title=Provider preflight::{r['provider']} answered {r['status']}")

    status_by = {r["provider"]: r["status"] for r in report.get("providers") or []}
    outputs = {
        "blocked_providers": ",".join(report.get("blocked_providers") or []),
        "reason": reason.replace("\n", " "),
    }
    for prov in DEFAULT_PROVIDERS:
        st = status_by.get(prov)
        outputs[f"{prov}_ok"] = "false" if st in BLOCKING_STATUSES else "true"
    _write_kv(os.getenv("GITHUB_OUTPUT"), outputs)
    if args.apply_block and report.get("blocked_providers"):
        merged = merged_block_list(os.getenv("PYTHIA_BLOCK_PROVIDERS", ""),
                                   report["blocked_providers"])
        _write_kv(os.getenv("GITHUB_ENV"), {"PYTHIA_BLOCK_PROVIDERS": merged})
        print(f"[preflight] PYTHIA_BLOCK_PROVIDERS={merged}")
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("## Provider preflight\n\n| provider | model | status | regain at |\n|---|---|---|---|\n")
            for r in report.get("providers") or []:
                fh.write(f"| {r['provider']} | {r['model']} | {r['status']} | {r.get('regain_at') or ''} |\n")
            fh.write(f"\n{reason}\n\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
