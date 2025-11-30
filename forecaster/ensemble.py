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
from forecaster.llm_logging import log_forecaster_llm_call
from resolver.db import duckdb_io

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

@dataclass
class EnsembleResult:
    members: List[MemberOutput]

# PA bucket centroids: conditional expected PA given each bucket, in people
SPD_BUCKET_CENTROIDS_PA: list[float] = [
    0.0,  # Bucket 1: <10k (treated as no major emergency)
    30_000.0,  # Bucket 2: 10k–<50k
    150_000.0,  # Bucket 3: 50k–<250k
    375_000.0,  # Bucket 4: 250k–<500k
    1_000_000.0,  # Bucket 5: >=500k
]

# Fatalities bucket centroids: expected deaths conditional on each bucket
SPD_BUCKET_CENTROIDS_FATALITIES: list[float] = [
    0.0,  # Bucket 1: <10
    30.0,  # Bucket 2: 10–<50
    150.0,  # Bucket 3: 50–<250
    625.0,  # Bucket 4: 250–<1000
    2_000.0,  # Bucket 5: >=1000
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
    t = text.strip()
    # Strip surrounding ``` fences if present
    t = re.sub(r"^```.*?\n|\n```$", "", t, flags=re.S).strip()

    data = None
    # First try full text as JSON
    try:
        data = json.loads(t)
    except Exception:
        # Fallback: find the first JSON-looking object
        m = re.search(r"\{.*\}", t, flags=re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None

    if not isinstance(data, dict):
        if os.getenv("PYTHIA_DEBUG_SPD", "0") == "1":
            head = t[:200].replace("\n", " ")
            print(f"[spd] parse failed: top-level JSON not dict; head={head!r}")
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
            head = t[:200].replace("\n", " ")
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
        con = duckdb_io.get_db(duckdb_io.DEFAULT_DB_URL)
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
            duckdb_io.close_db(con)

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
    run_id: str | None = "",
    question_id: str | None = "",
) -> EnsembleResult:
    """
    Run the ensemble for SPD questions.
    Each member's parsed output is a dict: { "month_1": [..5..], ..., "month_6": [..5..] }.
    """

    tasks = []
    for ms in specs:
        async def _one(ms=ms):
            parsed_for_log: Optional[dict] = None
            err_text: str = ""

            async def _call_llm(p: str, **kwargs):
                nonlocal parsed_for_log, err_text
                text, usage, err = await call_chat_ms(ms, p, **kwargs)
                err_text = err or ""
                try:
                    usage = dict(usage or {})
                except Exception:
                    usage = {}
                try:
                    usage["cost_usd"] = estimate_cost_usd(ms.model_id, usage)
                except Exception:
                    pass

                try:
                    parsed = _parse_spd_json(text, n_months=6, n_buckets=5)
                    if parsed is not None:
                        parsed = _normalize_spd_keys(parsed, n_months=6, n_buckets=5)
                    parsed_for_log = parsed
                except Exception as exc:
                    parsed_for_log = None
                    if not err_text:
                        err_text = f"{type(exc).__name__}: {exc}"
                return text, usage

            t0 = time.time()
            response_text, usage = await log_forecaster_llm_call(
                call_type="forecast",
                run_id=run_id or "",
                question_id=question_id or "",
                model_name=ms.name,
                provider=ms.provider,
                model_id=ms.model_id,
                prompt_text=prompt,
                low_level_call=_call_llm,
                low_level_kwargs={"temperature": 0.2},
                parsed_json=lambda: parsed_for_log,
            )

            usage = usage or {}

            try:
                parsed = parsed_for_log if isinstance(parsed_for_log, dict) else _parse_spd_json(
                    response_text, n_months=6, n_buckets=5
                )
                if parsed is not None:
                    parsed = _normalize_spd_keys(parsed, n_months=6, n_buckets=5)
                ok = isinstance(parsed, dict) and bool(parsed)
                if not ok:
                    parsed = {} if not isinstance(parsed, dict) else parsed or {}
            except Exception as exc:
                parsed = {}
                ok = False
                if not err_text:
                    err_text = f"{type(exc).__name__}: {exc}"

            err_field = err_text if err_text else ("" if ok else "parse_error")

            elapsed_ms = int((usage.get("elapsed_ms") or 0))
            if elapsed_ms <= 0:
                elapsed_ms = int((time.time() - t0) * 1000)

            prompt_tokens = int((usage.get("prompt_tokens") or 0))
            completion_tokens = int((usage.get("completion_tokens") or 0))
            total_tokens = int((usage.get("total_tokens") or (prompt_tokens + completion_tokens) or 0))
            cost = float((usage.get("cost_usd") or 0.0))
            if cost == 0.0:
                try:
                    cost = float(estimate_cost_usd(ms.model_id, usage))
                except Exception:
                    cost = 0.0
            return MemberOutput(
                name=ms.name,
                ok=ok,
                parsed=parsed,
                raw_text=response_text,
                error=err_field,
                elapsed_ms=elapsed_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
            )

        tasks.append(_one())
    return EnsembleResult(members=await asyncio.gather(*tasks))
