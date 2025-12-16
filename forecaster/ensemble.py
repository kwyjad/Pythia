# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations
"""
ensemble.py — async per-model calls with robust parsing + usage/cost capture,
using providers.call_chat_ms(...) which supports OpenRouter, Gemini-direct, Grok-direct.
"""

import asyncio, time, re, json, os
from dataclasses import dataclass
from typing import Any, List, Optional
import numpy as np

from .providers import ModelSpec, llm_semaphore, call_chat_ms, estimate_cost_usd
from .llm_logging import log_forecaster_llm_call
from pythia.db.schema import connect as pythia_connect

@dataclass
class MemberOutput:
    name: str
    ok: bool
    parsed: Any
    raw_text: str
    elapsed_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    error: str = ""  # non-empty if provider errored quickly
    model_spec: ModelSpec | None = None
    usage: dict | None = None

@dataclass
class EnsembleResult:
    members: List[MemberOutput]

# PA bucket centroids: conditional expected PA given each bucket, in people.
# Bucket 1 centroid is 0.0 because there is no explicit "0" bucket and much of
# the probability mass in this bucket will be on zero.
SPD_BUCKET_CENTROIDS_PA: list[float] = [
    0.0,  # Bucket 1: <10k (treated as no major emergency)
    30_000.0,  # Bucket 2: 10k–<50k
    150_000.0,  # Bucket 3: 50k–<250k
    375_000.0,  # Bucket 4: 250k–<500k
    700_000.0,  # Bucket 5: >=500k
]

# Backwards-compatible alias used by debug tools and any generic callers.
# "Default" SPD centroids are PA-style centroids aligned with SPD_BUCKET_TEXT_PA.
SPD_BUCKET_CENTROIDS_DEFAULT: list[float] = SPD_BUCKET_CENTROIDS_PA

# Fatalities bucket centroids: expected deaths conditional on each bucket.
# Bucket 1 centroid is 0.0 to reflect heavy mass on zero within the "<5" range.
SPD_BUCKET_CENTROIDS_FATALITIES: list[float] = [
    0.0,  # Bucket 1: <5
    15.0,  # Bucket 2: 5–<25
    62.0,  # Bucket 3: 25–<100
    300.0,  # Bucket 4: 100–<500
    700.0,  # Bucket 5: >=500
]

# ---------- helpers ----------
def sanitize_mcq_vector(vec: List[float], n_options: Optional[int] = None) -> List[float]:
    try:
        v = np.array([float(x) for x in (vec or [])], dtype=float)
    except Exception:
        v = np.array([], dtype=float)
    if n_options is not None:
        if v.size < n_options:
            v = np.pad(v, (0, n_options - v.size), mode="constant", constant_values=0.0)
        elif v.size > n_options:
            v = v[:n_options]
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.clip(v, 0.0, 1.0)
    s = float(v.sum())
    if s <= 0.0:
        n = v.size if v.size > 0 else (n_options if n_options else 1)
        return (np.ones(n) / float(n)).tolist()
    return (v / s).tolist()


def _normalize_spd_keys(
    data: dict,
    *,
    n_months: int = 6,
    n_buckets: int = 5,
) -> dict:
    """
    Normalize SPD dict keys to canonical 'month_1'..'month_n'.

    Accepts dicts that might have weird keys like '\n     "month_1"',
    'Month 1', or 'month-1'. Unrecognized keys are ignored. Missing months
    are filled with uniform distributions across buckets.
    """
    if not isinstance(data, dict):
        return {}

    month_map: dict[str, list[float]] = {}
    for raw_key, raw_vec in data.items():
        if not isinstance(raw_key, str):
            continue
        k = raw_key.strip()
        if k.startswith("\"") and k.endswith("\"") and len(k) > 2:
            k = k[1:-1]
        kl = k.lower()

        clean_key = None
        for i in range(1, n_months + 1):
            canonical = f"month_{i}"
            if kl == canonical:
                clean_key = canonical
                break
            if kl.replace(" ", "_") == canonical:
                clean_key = canonical
                break
            if kl.replace("-", "_") == canonical:
                clean_key = canonical
                break
        if clean_key is None:
            continue

        if isinstance(raw_vec, list):
            month_map[clean_key] = raw_vec

    out: dict[str, list[float]] = {}
    for i in range(1, n_months + 1):
        key = f"month_{i}"
        raw_vec = month_map.get(key, [])
        vec = sanitize_mcq_vector(raw_vec if isinstance(raw_vec, list) else [], n_options=n_buckets)
        out[key] = vec

    if os.getenv("PYTHIA_DEBUG_SPD", "0") == "1":
        try:
            raw_keys = sorted(list(data.keys()))
        except Exception:
            raw_keys = ["<unprintable>"]
        print(f"[spd] normalize keys: raw_keys={raw_keys!r} -> canonical_keys={sorted(out.keys())!r}")

    return out

def _parse_spd_json(
    text: str,
    n_months: int = 6,
    n_buckets: int = 5,
) -> Optional[dict]:
    """
    Parse an SPD-style JSON object of the form:
      {"month_1": [...5 numbers...], ..., "month_6": [...5 numbers...]}

    Returns a dict mapping month keys to sanitized lists of length n_buckets,
    or None if parsing fails.
    """
    if not text:
        return None

    orig = text.strip()
    candidates: List[str] = []

    # 1) If the whole thing looks like pure JSON, try it first.
    if orig.startswith("{") and orig.endswith("}"):
        candidates.append(orig)

    # 2) Extract any ```json ... ``` or ``` ... ``` fenced blocks.
    fences = re.findall(r"```(?:json)?\s*([\s\S]*?)```", orig, flags=re.I)
    for block in fences:
        block = block.strip()
        if block.startswith("{") and block.endswith("}"):
            candidates.append(block)

    # 3) As a fallback, look for JSON-looking objects and try the last ones first.
    all_objs = list(re.finditer(r"\{[\s\S]*?\}", orig))
    for m in reversed(all_objs):
        block = m.group(0).strip()
        if len(block) >= 2:
            candidates.append(block)

    tried = set()
    data: Optional[dict] = None
    for cand in candidates:
        if cand in tried:
            continue
        tried.add(cand)
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict):
            data = obj
            break

    if not isinstance(data, dict):
        if os.getenv("PYTHIA_DEBUG_SPD", "0") == "1":
            head = orig[:200].replace("\n", " ")
            print(f"[spd] parse failed: no dict JSON found; head={head!r}")
        return None

    out: dict = {}
    any_nonzero_raw = False
    for m_idx in range(1, n_months + 1):
        key = f"month_{m_idx}"
        raw_vec = data.get(key, [])
        raw_vals: list[float] = []

        if isinstance(raw_vec, list):
            for x in raw_vec:
                try:
                    raw_vals.append(float(x))
                except Exception:
                    continue

        if sum(abs(v) for v in raw_vals) > 1e-8:
            any_nonzero_raw = True

        vec = sanitize_mcq_vector(raw_vec if isinstance(raw_vec, list) else [], n_options=n_buckets)
        out[key] = vec

    if not any_nonzero_raw:
        if os.getenv("PYTHIA_DEBUG_SPD", "0") == "1":
            head = orig[:200].replace("\n", " ")
            print(f"[spd] parse produced all-zero SPD; treating as failure; head={head!r}")
        return None
    return _normalize_spd_keys(out, n_months=n_months, n_buckets=n_buckets)


def _load_bucket_centroids_db(
    hazard_code: str,
    metric: str,
    class_bins: List[str] | List[int],
) -> List[float] | None:
    """
    Load bucket centroids for a hazard/metric from DuckDB.

    Prefers hazard-specific entries, then wildcard ('*'), and returns a list of
    centroids matching the provided class_bins. Returns None if no complete set
    is available.
    """

    hz = (hazard_code or "").upper()
    mt = (metric or "").upper()
    if not mt or not class_bins:
        return None

    con = None
    try:
        con = pythia_connect(read_only=True)
    except Exception:
        return None

    try:
        rows = con.execute(
            """
            SELECT bucket_index, centroid
            FROM bucket_centroids
            WHERE upper(metric) = ?
              AND upper(hazard_code) = ?
            ORDER BY bucket_index
            """,
            [mt, hz],
        ).fetchall()

        if not rows:
            rows = con.execute(
                """
                SELECT bucket_index, centroid
                FROM bucket_centroids
                WHERE upper(metric) = ?
                  AND hazard_code = '*'
                ORDER BY bucket_index
                """,
                [mt],
            ).fetchall()
    except Exception:
        rows = []
    finally:
        if con is not None:
            con.close()

    if not rows:
        return None

    n_bins = len(class_bins)
    centroids: List[float] = [0.0] * n_bins
    found: set[int] = set()
    for bucket_idx, centroid in rows:
        try:
            b = int(bucket_idx)
        except Exception:
            continue
        if 1 <= b <= n_bins:
            centroids[b - 1] = float(centroid or 0.0)
            found.add(b)

    if len(found) != n_bins:
        return None

    return centroids

def _parse_binary_probability(text: str) -> Optional[float]:
    if not text:
        return None
    t = re.sub(r"^```.*?\n|\n```$", "", text.strip(), flags=re.S)
    # percents
    percents = re.findall(r"(?<!\d)(\d{1,2}(?:\.\d+)?)\s*%", t)
    for p in reversed(percents):
        try:
            v = float(p)/100.0
            if 0.0 <= v <= 1.0: return v
        except: pass
    # probability terms
    prob_blocks = re.findall(
        r"(?:prob(?:ability)?|likelihood|chance|p\s*=?|pr\s*\()\s*[:=]?\s*(\d{1,2}(?:\.\d+)?\s*%|0?\.\d+|1(?:\.0+)?)",
        t, flags=re.I)
    for tok in reversed(prob_blocks):
        tok = tok.strip()
        if tok.endswith('%'):
            try:
                v = float(tok[:-1].strip())/100.0
                if 0.0 <= v <= 1.0: return v
            except: pass
        else:
            try:
                v = float(tok)
                if 0.0 <= v <= 1.0: return v
            except: pass
    # naked decimals
    decimals = re.findall(r"(?<!\d)(0?\.\d+|1(?:\.0+)?)\b", t)
    for d in reversed(decimals):
        try:
            v = float(d)
            if 0.0 <= v <= 1.0: return v
        except: pass
    return None

# ---------- BINARY ----------
async def run_ensemble_binary(prompt: str, specs: List[ModelSpec]) -> EnsembleResult:
    tasks = []
    for ms in specs:
        async def _one(ms=ms):
            t0 = time.time()
            text, usage, err = await call_chat_ms(ms, prompt, temperature=0.1)
            p = _parse_binary_probability(text)
            ok = p is not None
            if not ok: p = 0.0
            cost = estimate_cost_usd(ms.model_id, usage)
            return MemberOutput(
                name=ms.name, ok=ok, parsed=p, raw_text=text, error=err,
                elapsed_ms=int((time.time()-t0)*1000),
                prompt_tokens=usage.get("prompt_tokens",0),
                completion_tokens=usage.get("completion_tokens",0),
                total_tokens=usage.get("total_tokens",0),
                cost_usd=cost
            )
        tasks.append(_one())
    return EnsembleResult(members=await asyncio.gather(*tasks))

# ---------- MCQ ----------
async def run_ensemble_mcq(prompt: str, n_options: int, specs: List[ModelSpec]) -> EnsembleResult:
    tasks = []
    for ms in specs:
        async def _one(ms=ms):
            t0 = time.time()
            text, usage, err = await call_chat_ms(ms, prompt, temperature=0.2)
            ok = False
            try:
                jmatch = re.search(r"\[[^\]]+\]", text, flags=re.S)
                if jmatch:
                    vec = [float(x) for x in json.loads(jmatch.group(0))]
                else:
                    nums = [float(x) for x in re.findall(r"[0-9]*\.?[0-9]+", text)]
                    vec = nums[:n_options]
                vec = sanitize_mcq_vector(vec, n_options)
                ok = len(vec) == n_options
            except Exception:
                vec = sanitize_mcq_vector([], n_options)
                ok = False
            cost = estimate_cost_usd(ms.model_id, usage)
            return MemberOutput(
                name=ms.name, ok=ok, parsed=vec, raw_text=text, error=err,
                elapsed_ms=int((time.time()-t0)*1000),
                prompt_tokens=usage.get("prompt_tokens",0),
                completion_tokens=usage.get("completion_tokens",0),
                total_tokens=usage.get("total_tokens",0),
                cost_usd=cost
            )
        tasks.append(_one())
    return EnsembleResult(members=await asyncio.gather(*tasks))

# ---------- NUMERIC ----------
async def run_ensemble_numeric(prompt: str, specs: List[ModelSpec]) -> EnsembleResult:
    tasks = []
    for ms in specs:
        async def _one(ms=ms):
            t0 = time.time()
            text, usage, err = await call_chat_ms(ms, prompt, temperature=0.2)
            out = {"P10": None, "P50": None, "P90": None}
            ok = False
            try:
                nums = [float(x) for x in re.findall(r"[0-9]*\.?[0-9]+", text)]
                if len(nums) >= 3:
                    out["P10"], out["P50"], out["P90"] = nums[0], nums[1], nums[2]
                    ok = True
            except Exception:
                ok = False
            cost = estimate_cost_usd(ms.model_id, usage)
            return MemberOutput(
                name=ms.name, ok=ok, parsed=out, raw_text=text, error=err,
                elapsed_ms=int((time.time()-t0)*1000),
                prompt_tokens=usage.get("prompt_tokens",0),
                completion_tokens=usage.get("completion_tokens",0),
                total_tokens=usage.get("total_tokens",0),
                cost_usd=cost
            )
        tasks.append(_one())
    return EnsembleResult(members=await asyncio.gather(*tasks))

# ---------- SPD (5 buckets × 6 months) ----------
async def run_ensemble_spd(
    prompt: str,
    specs: List[ModelSpec],
    *,
    run_id: str,
    question_id: str,
    hs_run_id: str | None = None,
) -> EnsembleResult:
    """
    Run the ensemble for SPD questions.

    Each member's parsed output is a dict:
      { "month_1": [...5...], ..., "month_6": [...5...] }.
    """
    tasks = []

    for ms in specs:
        async def _one(ms=ms):
            # Low-level async call: wraps call_chat_ms(ms, prompt, temperature=0.2)
            async def _low_level_call(prompt_text: str, **kwargs):
                text, usage, err = await call_chat_ms(
                    ms,
                    prompt_text,
                    temperature=kwargs.get("temperature", 0.2),
                )
                # Attach provider error text into usage so we can surface it downstream
                if err:
                    usage = dict(usage or {})
                    usage.setdefault("error_text", err)
                return text, usage

            # Log the forecast call and get (text, usage) back
            text, usage = await log_forecaster_llm_call(
                call_type="forecast",
                run_id=run_id,
                question_id=str(question_id),
                model_name=ms.name,
                provider=ms.provider,
                model_id=ms.model_id,
                prompt_text=prompt,
                low_level_call=_low_level_call,
                low_level_kwargs={"temperature": 0.2},
                hs_run_id=hs_run_id,
                parsed_json=None,  # we could add SPD here later if desired
            )

            # Parse SPD JSON
            parsed = _parse_spd_json(text, n_months=6, n_buckets=5)
            ok = parsed is not None
            if parsed is None:
                parsed = {}

            # Compute/attach cost if not present
            try:
                usage = dict(usage or {})
            except Exception:
                usage = {}
            if "cost_usd" not in usage:
                usage["cost_usd"] = estimate_cost_usd(ms.model_id, usage)

            elapsed_ms = int(usage.get("elapsed_ms", 0) or 0)
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            total_tokens = int(
                usage.get("total_tokens", prompt_tokens + completion_tokens) or 0
            )
            cost_usd = float(usage.get("cost_usd", 0.0) or 0.0)
            err_text = usage.get("error_text", "")

            return MemberOutput(
                name=ms.name,
                ok=ok,
                parsed=parsed,
                raw_text=text,
                elapsed_ms=elapsed_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                error=err_text,
                model_spec=ms,
                usage=usage,
            )

        tasks.append(_one())

    members = await asyncio.gather(*tasks)
    return EnsembleResult(members=members)
