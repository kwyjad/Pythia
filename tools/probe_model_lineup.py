# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Live probe of the December 2026 lineup before it is merged.

The build sandbox cannot reach the vendor APIs, so the lineup swap is checked
here, in CI, where the keys are. Three questions:

1. Model metadata: do ``gemini-4-argon`` and ``gemini-3.8-flash`` list
   ``batchGenerateContent``? (The staged pipeline batches every Gemini SPD
   call; a model without it would fall back to full-price sync on every call.)
2. One real SPD prompt through the real sync path (``call_chat_ms`` with
   ``log_call=False``) for each new model: status, finish reason, thinking
   and output tokens, whether the SPD parses, and the priced cost.
3. A two-item Gemini batch for ``gemini-4-argon`` through ``pythia.llm_batch``,
   polled to completion (bounded), priced at the batch tier.

It writes nothing to any database. Exit 1 when Argon cannot be batched or a
sync call failed, because a red probe is the signal not to merge the swap.

    python -m tools.probe_model_lineup --out diagnostics/model_lineup_probe.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests

NEW_MODELS = [
    ("google", "gemini-4-argon"),
    ("google", "gemini-3.8-flash"),
    ("openai", "gpt-6.1-sol"),
]
METADATA_MODELS = ["gemini-4-argon", "gemini-3.8-flash"]

# One small question, the shape forecaster/tests/test_prompt_v3_order.py uses.
_QUESTION = {
    "question_id": "SOM_ACE_FATALITIES_PROBE",
    "iso3": "SOM",
    "hazard_code": "ACE",
    "metric": "FATALITIES",
    "wording": "How many conflict fatalities will Somalia record?",
    "window_start_date": "2027-01-01",
    "target_month": "2027-06",
}
_HISTORY = {"source": "acled", "summary": "24 months of data; monthly deaths 150-400"}
_TRIAGE = {"regime_change_level": 1, "triage_score": 0.6}


def spd_prompt() -> str:
    from forecaster.prompts import build_spd_prompt_v2

    return build_spd_prompt_v2(_QUESTION, _HISTORY, _TRIAGE, {})


def spd_valid(text: str, n_buckets: int = 7) -> Dict[str, Any]:
    """Whether the answer holds six month vectors of the right length."""

    try:
        from forecaster.cli import _safe_json_loads

        obj = _safe_json_loads(text)
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "reason": f"parse: {type(exc).__name__}"}
    spds = obj.get("spds") if isinstance(obj, dict) else None
    if not isinstance(spds, dict) or not spds:
        return {"valid": False, "reason": "no spds object"}
    months_ok = 0
    for _key, value in spds.items():
        probs = value.get("probs") if isinstance(value, dict) else value
        if isinstance(probs, list) and len(probs) == n_buckets:
            try:
                if abs(sum(float(p) for p in probs) - 1.0) < 0.05:
                    months_ok += 1
            except (TypeError, ValueError):
                pass
    return {"valid": months_ok >= 6, "months_ok": months_ok, "n_months": len(spds)}


def model_metadata(model_id: str) -> Dict[str, Any]:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    out: Dict[str, Any] = {"model_id": model_id}
    if not key:
        out["error"] = "no GEMINI_API_KEY"
        return out
    try:
        resp = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}",
            headers={"x-goog-api-key": key},  # never in the URL
            timeout=30,
        )
        out["status"] = resp.status_code
        data = resp.json() if resp.content else {}
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}"
        return out
    out["supportedGenerationMethods"] = data.get("supportedGenerationMethods")
    out["inputTokenLimit"] = data.get("inputTokenLimit")
    out["outputTokenLimit"] = data.get("outputTokenLimit")
    out["batch_supported"] = "batchGenerateContent" in (data.get("supportedGenerationMethods") or [])
    if resp.status_code != 200:
        out["error"] = str((data.get("error") or {}).get("message", ""))[:300]
    return out


def ensemble_spec(provider: str, model_id: str):
    """The config ensemble's spec for this model (its own thinking/temperature)."""

    import dataclasses

    from forecaster.providers import ModelSpec, _load_ensemble_from_config

    for ms in _load_ensemble_from_config():
        if ms.model_id == model_id:
            return dataclasses.replace(ms, purpose="spd_v2", active=True)
    return ModelSpec(name=model_id, provider=provider, model_id=model_id,
                     active=True, purpose="spd_v2")


def sync_call(provider: str, model_id: str, prompt: str) -> Dict[str, Any]:
    from forecaster.providers import call_chat_ms, estimate_cost_usd

    ms = ensemble_spec(provider, model_id)
    t0 = time.monotonic()
    text, usage, error = asyncio.run(call_chat_ms(ms, prompt, 0.2, log_call=False))
    usage = dict(usage or {})
    out = {
        "model_id": model_id,
        "thinking": ms.thinking,
        "ok": not error,
        "error": (error or "")[:300],
        "seconds": round(time.monotonic() - t0, 1),
        "finish_reason": usage.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "thoughts_tokens": usage.get("thoughts_tokens") or usage.get("reasoning_tokens"),
        "cost_usd": round(float(usage.get("cost_usd") or estimate_cost_usd(model_id, usage)), 4),
    }
    out.update(spd_valid(text or ""))
    return out


def argon_batch(prompt: str, *, wait_sec: float, poll_sec: float = 30.0) -> Dict[str, Any]:
    from forecaster.providers import build_body_for_spec, estimate_cost_usd
    from pythia import llm_batch

    out: Dict[str, Any] = {"model_id": "gemini-4-argon", "verdict": None}
    if not (os.getenv("GEMINI_API_KEY") or "").strip():
        out["verdict"] = "skipped:no_api_key"
        return out
    ms = ensemble_spec("google", "gemini-4-argon")
    body = build_body_for_spec(ms, prompt, 0.2)
    stamp = int(time.time())
    rows = [(f"probe-argon-{stamp}-{i}", body) for i in (1, 2)]
    adapter = llm_batch._adapter("google")
    t0 = time.monotonic()
    try:
        info = adapter.submit(rows, model_id="gemini-4-argon")
    except Exception as exc:  # noqa: BLE001
        out["verdict"] = "submit_error"
        out["detail"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        return out
    pbid = str(info.get("provider_batch_id") or "")
    out["provider_batch_id"] = pbid
    state = "in_progress"
    detail = ""
    deadline = t0 + wait_sec
    while time.monotonic() < deadline:
        try:
            status = adapter.poll(pbid)
            state, detail = status.state, status.detail
        except Exception as exc:  # noqa: BLE001
            state, detail = "poll_error", f"{type(exc).__name__}"
        if state in ("ended", "failed", "expired", "canceled"):
            break
        time.sleep(poll_sec)
    out["state"] = state
    out["seconds"] = round(time.monotonic() - t0, 1)
    out["detail"] = (detail or "")[:400]
    if state != "ended":
        try:
            adapter.cancel(pbid)
        except Exception:  # noqa: BLE001
            pass
        out["verdict"] = "not_finished" if state == "in_progress" else state
        return out
    items: List[Dict[str, Any]] = []
    total = 0.0
    for cid, ok, text, usage, error in adapter.fetch(pbid):
        usage = dict(usage or {})
        usage["service_tier"] = "batch"
        cost = estimate_cost_usd("gemini-4-argon", usage)
        total += cost
        item = {"custom_id": cid, "ok": ok, "error": (error or "")[:200],
                "completion_tokens": usage.get("completion_tokens"),
                "thoughts_tokens": usage.get("thoughts_tokens"),
                "finish_reason": usage.get("finish_reason"),
                "cost_usd_batch": round(cost, 4)}
        item.update(spd_valid(text or ""))
        items.append(item)
    out["items"] = items
    out["cost_usd_batch_total"] = round(total, 4)
    out["verdict"] = "ok" if items and all(i["ok"] for i in items) else "items_failed"
    return out


def _table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="diagnostics/model_lineup_probe.json")
    ap.add_argument("--batch-wait-min", type=float, default=30.0)
    ap.add_argument("--skip-batch", action="store_true")
    args = ap.parse_args(argv)

    # Nothing here may write to a database or count as production data.
    os.environ["PYTHIA_TEST_MODE"] = "1"
    os.environ.pop("PYTHIA_DB_URL", None)

    report: Dict[str, Any] = {"metadata": [], "sync": [], "batch": None}
    for mid in METADATA_MODELS:
        report["metadata"].append(model_metadata(mid))
    prompt = spd_prompt()
    report["prompt_chars"] = len(prompt)
    for provider, mid in NEW_MODELS:
        try:
            report["sync"].append(sync_call(provider, mid, prompt))
        except Exception as exc:  # noqa: BLE001
            report["sync"].append({"model_id": mid, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]})
    if not args.skip_batch:
        report["batch"] = argon_batch(prompt, wait_sec=args.batch_wait_min * 60)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    md = ["## Model lineup probe", "", "### Gemini model metadata",
          _table(report["metadata"], ["model_id", "status", "batch_supported", "inputTokenLimit",
                                      "outputTokenLimit", "supportedGenerationMethods", "error"]),
          "", "### One SPD call each (sync)",
          _table(report["sync"], ["model_id", "thinking", "ok", "finish_reason", "prompt_tokens",
                                  "completion_tokens", "thoughts_tokens", "valid", "cost_usd",
                                  "seconds", "error"]),
          ""]
    if report["batch"] is not None:
        b = report["batch"]
        md += ["### Argon two-item batch",
               f"verdict **{b.get('verdict')}**, state {b.get('state')}, {b.get('seconds')} s, "
               f"batch cost ${b.get('cost_usd_batch_total')}", "",
               _table(b.get("items") or [], ["custom_id", "ok", "finish_reason", "completion_tokens",
                                             "thoughts_tokens", "valid", "cost_usd_batch", "error"])]
    text = "\n".join(md)
    print(text)
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")

    failed = any(not m.get("batch_supported") for m in report["metadata"])
    failed = failed or any(not s.get("ok") for s in report["sync"])
    if report["batch"] is not None and report["batch"].get("verdict") != "ok":
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
