# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Pre-run batch canary: does each provider's Batch API accept OUR batches today?

Why this exists: on 2026-09-01 every OpenAI batch the forecast pipeline
submitted was accepted at creation and rejected by OpenAI's asynchronous
validator two minutes later (``Cannot find file ..., or organization ... does
not have access to it``), a provider-side incident that hit many organisations
for hours at a time from late August. 210 requests then ran synchronously at
full price. Nothing in the pipeline could have said beforehand that the batch
route was closed, because nothing asked.

This asks. For each provider in ``PYTHIA_BATCH_PROVIDERS`` and each ensemble
member of that provider (``pythia/config.yaml``, resolved by
``get_ensemble_resolved``), it builds ONE trivial request through the same body
builder the pipeline uses, submits it through the same adapter, watches it
through validation and completion for a bounded time, fetches the answer, and
cancels anything still running. No database is touched; the cost is one
tiny request per model at batch price.

Verdicts: ``ok`` (a result came back), ``slow_but_accepted`` (validation
passed but the batch did not finish inside ``--wait-min``; cancelled),
``validation_failed:file_access`` (the 2026-09-01 rejection — the batch route
is closed for this organisation right now), ``validation_failed:other`` (a
deterministic rejection, which is a bug in our request), ``submit_error``
(the HTTP layer refused), ``skipped:no_api_key``.

Exit code 1 when any provider failed validation or submit — this is a
dispatch-only diagnostic, so red is the point. Run via the ``canary`` input
on ``poll_llm_batches.yml`` (only CI holds the keys), then read the table
before dispatching the pipeline.

    python -m scripts.ci.batch_canary --wait-min 10 --out diagnostics/batch_canary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from pythia import llm_batch

CANARY_PROMPT = "Reply with the single word OK."
CANARY_MAX_TOKENS = 512  # reasoning models spend some of this before the word

VERDICT_OK = "ok"
VERDICT_SLOW = "slow_but_accepted"
VERDICT_FILE_ACCESS = "validation_failed:file_access"
VERDICT_OTHER = "validation_failed:other"
VERDICT_SUBMIT_ERROR = "submit_error"
VERDICT_NO_KEY = "skipped:no_api_key"

FAILING_VERDICTS = (VERDICT_FILE_ACCESS, VERDICT_OTHER, VERDICT_SUBMIT_ERROR)

_KEY_ENV = {"openai": ("OPENAI_API_KEY",), "anthropic": ("ANTHROPIC_API_KEY",),
            "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY")}

# Patchable for tests.
_sleep = time.sleep
_clock = time.monotonic


def _has_key(provider: str) -> bool:
    return any(os.getenv(k, "").strip() for k in _KEY_ENV.get(provider, ()))


def members(providers: set[str], override: Optional[str] = None) -> List[Dict[str, Any]]:
    """The (provider, model_id, params) list to probe, deduplicated."""

    entries: List[Dict[str, Any]] = []
    if override:
        for token in override.split(","):
            token = token.strip()
            if not token or ":" not in token:
                continue
            provider, model_id = token.split(":", 1)
            entries.append({"provider": provider.strip().lower(), "model_id": model_id.strip()})
    else:
        from pythia.llm_profiles import get_ensemble_resolved  # noqa: PLC0415

        entries = list(get_ensemble_resolved())
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for e in entries:
        provider = str(e.get("provider") or "").lower()
        model_id = str(e.get("model_id") or "")
        if not provider or not model_id or provider not in providers:
            continue
        key = (provider, model_id)
        if key in seen:
            continue
        seen.add(key)
        out.append({"provider": provider, "model_id": model_id,
                    "temperature": e.get("temperature"), "thinking": e.get("thinking")})
    return out


def tiny_body(provider: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Cap the answer so the canary costs cents whatever the model is."""

    body = dict(body)
    if provider == "openai":
        if "max_completion_tokens" in body or "max_tokens" not in body:
            body["max_completion_tokens"] = CANARY_MAX_TOKENS
            body.pop("max_tokens", None)
        else:
            body["max_tokens"] = CANARY_MAX_TOKENS
    elif provider == "anthropic":
        body["max_tokens"] = CANARY_MAX_TOKENS
    elif provider == "google":
        gen = dict(body.get("generationConfig") or {})
        gen["maxOutputTokens"] = CANARY_MAX_TOKENS
        body["generationConfig"] = gen
    return body


def build_body(entry: Dict[str, Any]) -> Dict[str, Any]:
    from forecaster.providers import ModelSpec, build_body_for_spec  # noqa: PLC0415

    ms = ModelSpec(
        name=entry["model_id"],
        provider=entry["provider"],
        model_id=entry["model_id"],
        active=True,
        purpose=None,
        temperature=entry.get("temperature"),
        thinking=entry.get("thinking"),
    )
    return tiny_body(entry["provider"], build_body_for_spec(ms, CANARY_PROMPT))


def _classify_failed(detail: str) -> str:
    try:
        info = json.loads(detail or "{}")
    except (TypeError, ValueError):
        info = {}
    errors = info.get("errors") if isinstance(info, dict) else None
    if errors and llm_batch._openai_errors_are_file_access(errors):
        return VERDICT_FILE_ACCESS
    return VERDICT_OTHER


def _new_result(provider: str, model_id: str) -> Dict[str, Any]:
    return {
        "provider": provider,
        "model_id": model_id,
        "verdict": None,
        "provider_batch_id": None,
        "input_file_id": None,
        "submit_sec": None,
        "elapsed_sec": None,
        "detail": "",
        "item_ok": None,
        "item_error": "",
    }


def submit_one(provider: str, model_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Submit one request as a batch. Returns a result dict; ``verdict`` is set
    only when the submit itself settled the question (no key, refused,
    rejected at validation) — otherwise ``_pending`` carries what to follow.
    Never raises."""

    result = _new_result(provider, model_id)
    if not _has_key(provider):
        result["verdict"] = VERDICT_NO_KEY
        return result
    cid = f"canary-{provider}-{llm_batch._sanitize_token(model_id, 20)}-{int(time.time())}"
    rows = [(cid, body)]
    t0 = _clock()
    try:
        adapter = llm_batch._adapter(provider)
        if provider == "google":
            info = adapter.submit(rows, model_id=model_id)
        else:
            info = adapter.submit(rows)
    except llm_batch.OpenAIBatchValidationError as exc:
        result["verdict"] = VERDICT_FILE_ACCESS if exc.file_access else VERDICT_OTHER
        result["provider_batch_id"] = exc.provider_batch_id
        result["input_file_id"] = exc.input_file_id
        result["submit_sec"] = round(_clock() - t0, 1)
        result["detail"] = (
            f"attempts={exc.attempts} same_file_retry_failed={exc.same_file_retry_failed}: "
            f"{json.dumps(exc.errors)[:600]}"
        )
        return result
    except Exception as exc:  # noqa: BLE001
        result["verdict"] = VERDICT_SUBMIT_ERROR
        result["submit_sec"] = round(_clock() - t0, 1)
        result["detail"] = f"{type(exc).__name__}: {str(exc)[:600]}"
        return result
    result["submit_sec"] = round(_clock() - t0, 1)
    pbid = str(info.get("provider_batch_id") or "")
    result["provider_batch_id"] = pbid
    result["input_file_id"] = info.get("input_file_id")
    if not pbid:
        result["verdict"] = VERDICT_SUBMIT_ERROR
        result["detail"] = "adapter returned no provider batch id"
        return result
    result["_pending"] = {"adapter": adapter, "pbid": pbid, "t0": t0}
    return result


def _settle(result: Dict[str, Any], state: str, detail: str) -> None:
    """Turn a terminal (or deadline) poll state into a verdict."""

    pending = result.pop("_pending")
    adapter, pbid, t0 = pending["adapter"], pending["pbid"], pending["t0"]
    result["elapsed_sec"] = round(_clock() - t0, 1)
    result["detail"] = detail[:1000] if detail else ""
    if state in ("ended", "collected"):
        try:
            items = list(adapter.fetch(pbid))
        except Exception as exc:  # noqa: BLE001
            items = []
            result["detail"] = f"fetch failed: {type(exc).__name__}: {str(exc)[:300]}"
        if items:
            _cid, ok, text, _usage, error = items[0]
            result["item_ok"] = bool(ok)
            result["item_error"] = error or ""
            result["detail"] = (text or "")[:120] if ok else (error or "")[:300]
        result["verdict"] = VERDICT_OK
        return
    if state == "failed":
        result["verdict"] = _classify_failed(detail)
        return
    if state in ("expired", "canceled"):
        result["verdict"] = VERDICT_OTHER
        return
    # Still validating / in progress at the deadline: accepted, just slow.
    try:
        adapter.cancel(pbid)
    except Exception:  # noqa: BLE001
        pass
    result["verdict"] = VERDICT_SLOW


def follow(results: List[Dict[str, Any]], *, wait_sec: float, poll_sec: float) -> None:
    """Poll every submitted batch together until each is terminal or ONE shared
    deadline passes — the deadline is for the whole set, so the job's wall
    time is bounded whatever the number of models."""

    deadline = _clock() + wait_sec
    while True:
        open_ = [r for r in results if "_pending" in r]
        if not open_:
            return
        for r in open_:
            pending = r["_pending"]
            try:
                status = pending["adapter"].poll(pending["pbid"])
                state, detail = status.state, status.detail
            except Exception as exc:  # noqa: BLE001
                state, detail = "poll_error", f"{type(exc).__name__}: {str(exc)[:300]}"
            if state in ("ended", "collected", "failed", "expired", "canceled"):
                _settle(r, state, detail)
        open_ = [r for r in results if "_pending" in r]
        if not open_:
            return
        if _clock() >= deadline:
            for r in open_:
                _settle(r, "deadline", "")
            return
        _sleep(max(1.0, min(poll_sec, deadline - _clock())))


def run_one(
    provider: str,
    model_id: str,
    body: Dict[str, Any],
    *,
    wait_sec: float,
    poll_sec: float,
) -> Dict[str, Any]:
    """Submit one request as a batch and follow it to a verdict. Never raises."""

    result = submit_one(provider, model_id, body)
    if "_pending" in result:
        follow([result], wait_sec=wait_sec, poll_sec=poll_sec)
    return result


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_provider: Dict[str, str] = {}
    for r in results:
        prov = r["provider"]
        v = r["verdict"] or "unknown"
        prior = by_provider.get(prov)
        # The worst verdict a provider produced is the provider's verdict.
        rank = {VERDICT_FILE_ACCESS: 4, VERDICT_OTHER: 4, VERDICT_SUBMIT_ERROR: 3,
                VERDICT_SLOW: 2, VERDICT_OK: 1, VERDICT_NO_KEY: 0}
        if prior is None or rank.get(v, 5) > rank.get(prior, 5):
            by_provider[prov] = v
    failed = [r for r in results if (r["verdict"] or "") in FAILING_VERDICTS]
    return {
        "n_models": len(results),
        "n_failed": len(failed),
        "by_provider": by_provider,
        "verdicts": {r["model_id"]: r["verdict"] for r in results},
    }


def _table(results: List[Dict[str, Any]]) -> str:
    lines = ["| provider | model | verdict | submit s | total s | detail |", "|---|---|---|--:|--:|---|"]
    for r in results:
        d = (r.get("detail") or "").replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {r['provider']} | `{r['model_id']}` | **{r['verdict']}** | "
            f"{r.get('submit_sec') if r.get('submit_sec') is not None else ''} | "
            f"{r.get('elapsed_sec') if r.get('elapsed_sec') is not None else ''} | {d[:160]} |"
        )
    return "\n".join(lines)


def _emit_step_summary(md: str) -> None:
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(md + "\n")
    except OSError:
        pass


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wait-min", type=float, default=10.0,
                   help="How long to follow the submitted batches, all together, before cancelling the rest (default 10)")
    p.add_argument("--poll-sec", type=float, default=15.0)
    p.add_argument("--submit-budget-min", type=float, default=3.0,
                   help="PYTHIA_OPENAI_BATCH_SUBMIT_BUDGET_MIN for the canary's own guard (default 3)")
    p.add_argument("--providers", default="",
                   help="Comma list; default PYTHIA_BATCH_PROVIDERS (openai,anthropic,google)")
    p.add_argument("--models", default="",
                   help="Comma list of provider:model_id to probe instead of the ensemble")
    p.add_argument("--out", default="diagnostics/batch_canary.json")
    args = p.parse_args(argv)

    os.environ["PYTHIA_OPENAI_BATCH_SUBMIT_BUDGET_MIN"] = str(args.submit_budget_min)
    providers = (
        {x.strip().lower() for x in args.providers.split(",") if x.strip()}
        if args.providers else llm_batch.batch_providers()
    )
    entries = members(providers, args.models or None)
    if not entries:
        print("::error title=Batch canary::no ensemble members for the requested providers")
        return 1

    # Submit everything first, then follow the whole set against one deadline:
    # a sequential submit-and-wait per model multiplies the wait by the model
    # count and outran the job's timeout on the canary's own first run.
    llm_batch._reset_openai_submit_budget()
    results: List[Dict[str, Any]] = []
    for entry in entries:
        provider, model_id = entry["provider"], entry["model_id"]
        print(f"[canary] {provider}/{model_id}: submitting ...", flush=True)
        try:
            body = build_body(entry)
        except Exception as exc:  # noqa: BLE001
            r = _new_result(provider, model_id)
            r["verdict"] = VERDICT_SUBMIT_ERROR
            r["detail"] = f"body build failed: {type(exc).__name__}: {exc}"
            results.append(r)
            continue
        r = submit_one(provider, model_id, body)
        if "_pending" in r:
            print(f"[canary] {provider}/{model_id}: submitted {r['provider_batch_id']} in {r['submit_sec']}s", flush=True)
        else:
            print(f"[canary] {provider}/{model_id}: {r['verdict']} ({r['detail'][:200]})", flush=True)
        results.append(r)
    follow(results, wait_sec=args.wait_min * 60.0, poll_sec=args.poll_sec)
    for r in results:
        print(f"[canary] {r['provider']}/{r['model_id']}: {r['verdict']} ({(r['detail'] or '')[:200]})", flush=True)

    summary = summarize(results)
    md = "### Batch canary\n\n" + _table(results) + "\n"
    if summary["n_failed"]:
        md += (
            f"\n**{summary['n_failed']} of {summary['n_models']} model(s) cannot be batched right now.** "
            "A `validation_failed:file_access` verdict is the 2026-09-01 rejection: OpenAI's batch "
            "route is closed for this organisation at the moment; the pipeline's patient submit guard "
            "and the collect-stage re-batch will keep trying, and the sync fallback pays full price "
            "if they cannot. Contact OpenAI support for the per-organisation mitigation.\n"
        )
    print(md)
    _emit_step_summary(md)
    for r in results:
        if r["verdict"] in FAILING_VERDICTS:
            print(
                f"::error title=Batch canary failed::{r['provider']}/{r['model_id']}: "
                f"{r['verdict']} — {r['detail'][:300]}"
            )
        elif r["verdict"] == VERDICT_SLOW:
            print(
                f"::warning title=Batch canary slow::{r['provider']}/{r['model_id']}: accepted "
                f"but not finished within {args.wait_min:g} min (cancelled)"
            )

    payload = {
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wait_min": args.wait_min,
        "providers": sorted(providers),
        "results": results,
        "summary": summary,
        "exit_code": 1 if summary["n_failed"] else 0,
    }
    try:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"wrote {args.out}")
    except OSError as exc:
        print(f"[warn] could not write {args.out}: {exc}")
    return payload["exit_code"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
