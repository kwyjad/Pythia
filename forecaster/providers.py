# forecaster/providers.py
from __future__ import annotations
"""
providers.py — unified access to OpenRouter, direct Gemini, and direct Grok.

- HONORS .env flags:
    USE_OPENROUTER_DEFAULT, USE_ANTHROPIC, USE_GEMINI, USE_GROK
- KNOWN_MODELS gives stable names used by CSV columns & weighting.
- DEFAULT_ENSEMBLE lists only active models this run.
- call_chat_ms(ModelSpec, prompt, temperature) → (text, usage_dict, error)
- estimate_cost_usd(...) uses MODEL_COSTS_JSON ($ per 1k tokens).
"""

import os, asyncio, json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from .config import (
    GPT5_CALL_TIMEOUT_SEC,
    GEMINI_CALL_TIMEOUT_SEC,
    GROK_CALL_TIMEOUT_SEC,
)

# ---------------- stable names for CSV schema ----------------
KNOWN_MODELS = [
    "OpenRouter-Default",
    "Claude-3.7-Sonnet (OR)",
    "Gemini",
    "Grok",
]

@dataclass
class ModelSpec:
    name: str         # stable display name (must match KNOWN_MODELS)
    provider: str     # "openrouter" | "gemini" | "grok"
    model_id: str     # provider-specific model identifier
    weight: float = 1.0
    active: bool = True

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, str(int(default))).strip().lower()
    return v in ("1","true","yes","y","on")

# ----------- provider toggles -----------
USE_OPENROUTER_DEFAULT = _env_bool("USE_OPENROUTER_DEFAULT", True)
USE_ANTHROPIC         = _env_bool("USE_ANTHROPIC", True)
USE_GEMINI            = _env_bool("USE_GEMINI", True)    # DIRECT Gemini
USE_GROK              = _env_bool("USE_GROK", True)      # DIRECT Grok

# ----------- OpenRouter (AsyncOpenAI-compatible) -----------
OR_API_KEY  = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("OR_API_KEY", "")
OR_BASE_URL = os.getenv("OPENROUTER_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1"
OPENROUTER_FALLBACK_ID = os.getenv("OPENROUTER_FALLBACK_ID", "openai/gpt-4o")
CLAUDE_ON_OR_ID        = os.getenv("OPENROUTER_CLAUDE37_ID", "anthropic/claude-3.7-sonnet")

try:
    from openai import AsyncOpenAI  # works for OpenRouter when base_url is set
except Exception:
    AsyncOpenAI = None  # type: ignore

llm_semaphore = asyncio.Semaphore(int(os.getenv("LLM_MAX_CONCURRENCY","4")))
_async_client_singleton = None

_OR_TIMEOUT = max(1.0, float(GPT5_CALL_TIMEOUT_SEC or 0))
_GEMINI_TIMEOUT = max(1.0, float(GEMINI_CALL_TIMEOUT_SEC or 0))
_GROK_TIMEOUT = max(1.0, float(GROK_CALL_TIMEOUT_SEC or 0))

def _get_or_client():
    """Return AsyncOpenAI client (OpenRouter) or None."""
    global _async_client_singleton
    if not OR_API_KEY or AsyncOpenAI is None:
        return None
    if _async_client_singleton is None:
        _async_client_singleton = AsyncOpenAI(
            api_key=OR_API_KEY,
            base_url=OR_BASE_URL,
            timeout=_OR_TIMEOUT,
        )
    return _async_client_singleton

# ----------- Gemini (DIRECT) -----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY","").strip()
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID","gemini-2.5-pro").strip()

# ----------- Grok (DIRECT) -----------
XAI_API_KEY = os.getenv("XAI_API_KEY","").strip()
GROK_MODEL_ID = os.getenv("GROK_MODEL_ID","grok-4").strip()
XAI_BASE_URL = os.getenv("XAI_BASE_URL","https://api.x.ai/v1/chat/completions").strip()

# ----------- Build the potential ensemble -----------
_POTENTIAL_ENSEMBLE: List[ModelSpec] = [
    ModelSpec("OpenRouter-Default",     "openrouter", OPENROUTER_FALLBACK_ID, active=USE_OPENROUTER_DEFAULT),
    ModelSpec("Claude-3.7-Sonnet (OR)", "openrouter", CLAUDE_ON_OR_ID,        active=USE_ANTHROPIC),
    ModelSpec("Gemini",                 "gemini",     GEMINI_MODEL_ID,        active=USE_GEMINI),
    ModelSpec("Grok",                   "grok",       GROK_MODEL_ID,          active=USE_GROK),
]
DEFAULT_ENSEMBLE: List[ModelSpec] = [m for m in _POTENTIAL_ENSEMBLE if m.active]

# ---------------- pricing helpers ----------------
_MODEL_PRICES: Optional[Dict[str, Dict[str, float]]] = None
def _load_model_prices() -> Dict[str, Dict[str, float]]:
    global _MODEL_PRICES
    if _MODEL_PRICES is not None:
        return _MODEL_PRICES
    s = os.getenv("MODEL_COSTS_JSON","").strip()
    if s:
        try: _MODEL_PRICES = json.loads(s)
        except Exception: _MODEL_PRICES = {}
    else:
        _MODEL_PRICES = {}
    return _MODEL_PRICES

def usage_to_dict(usage_obj: Any) -> Dict[str,int]:
    """
    Map provider-specific usage to a common dict.
    For OpenRouter/OpenAI v1: usage has .prompt_tokens/.completion_tokens.
    """
    d = {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
    try:
        if usage_obj is None:
            return d
        pt = getattr(usage_obj, "prompt_tokens", None)
        ct = getattr(usage_obj, "completion_tokens", None)
        tt = getattr(usage_obj, "total_tokens", None)
        d["prompt_tokens"] = int(pt or 0)
        d["completion_tokens"] = int(ct or 0)
        d["total_tokens"] = int(tt or (d["prompt_tokens"] + d["completion_tokens"]))
        return d
    except Exception:
        return d

def estimate_cost_usd(model_id: str, usage: Dict[str,int]) -> float:
    """
    usage: {"prompt_tokens": int, "completion_tokens": int}
    MODEL_COSTS_JSON: {"model_id":{"prompt":$/1k,"completion":$/1k}, ...}
    """
    prices = _load_model_prices()
    if not usage or not isinstance(usage, dict):
        return 0.0
    p = prices.get(model_id) or prices.get(model_id.split("/",1)[-1]) or {}
    try:
        rp = float(p.get("prompt", 0.0))
        rc = float(p.get("completion", 0.0))
        return (usage.get("prompt_tokens",0)/1000.0)*rp + (usage.get("completion_tokens",0)/1000.0)*rc
    except Exception:
        return 0.0

# ---------------- provider-specific calls ----------------
import requests

async def _call_openrouter(model_id: str, prompt: str, temperature: float) -> tuple[str, Dict[str,int], str]:
    client = _get_or_client()
    if client is None:
        return "", {}, "no OpenRouter client"
    try:
        async with llm_semaphore:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_id,
                    messages=[{"role":"user","content":prompt}],
                    temperature=temperature,
                    timeout=_OR_TIMEOUT,
                ),
                timeout=_OR_TIMEOUT,
            )
        content = getattr(resp.choices[0].message, "content", "")
        if isinstance(content, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            ).strip()
        else:
            text = (content or "").strip()
        usage = usage_to_dict(getattr(resp, "usage", None))
        return text, usage, ""
    except asyncio.TimeoutError:
        return "", {}, f"TimeoutError: OpenRouter call exceeded {_OR_TIMEOUT:.0f}s"
    except Exception as e:
        return "", {}, f"{type(e).__name__}: {str(e)[:200]}"

async def _call_gemini_direct(model_id: str, prompt: str, temperature: float) -> tuple[str, Dict[str,int], str]:
    if not GEMINI_API_KEY:
        return "", {}, "no GEMINI_API_KEY"
    def _do():
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GEMINI_API_KEY}"
        body = {
            "contents":[{"role":"user","parts":[{"text": prompt}]}],
            "generationConfig":{"temperature":float(temperature)}
        }
        try:
            r = requests.post(url, json=body, timeout=_GEMINI_TIMEOUT)
            j = r.json()
        except Exception as e:
            return "", {}, f"Gemini request error: {e!r}"
        if r.status_code != 200:
            msg = j.get("error", {}).get("message","")
            return "", {}, f"Gemini HTTP {r.status_code}: {msg[:160]}"
        # Extract text
        text = ""
        try:
            text = j["candidates"][0]["content"]["parts"][0].get("text","").strip()
        except Exception:
            text = j.get("text","").strip()
        # usageMetadata: promptTokenCount / candidatesTokenCount / totalTokenCount
        um = j.get("usageMetadata") or {}
        usage = {
            "prompt_tokens": int(um.get("promptTokenCount", 0)),
            "completion_tokens": int(um.get("candidatesTokenCount", 0)),
            "total_tokens": int(um.get("totalTokenCount", 0)),
        }
        return text, usage, ""
    return await asyncio.to_thread(_do)

async def _call_grok_direct(model_id: str, prompt: str, temperature: float) -> tuple[str, Dict[str,int], str]:
    if not XAI_API_KEY:
        return "", {}, "no XAI_API_KEY"

    def _dig(d: Any, path: list[str], default: str = "") -> str:
        cur: Any = d
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return default
        return cur if isinstance(cur, str) else default

    def _to_json(obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, (bytes, str)):
            try:
                parsed = json.loads(obj)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def _err_msg_from_json(j: Dict[str, Any]) -> str:
        err = j.get("error")
        if isinstance(err, dict):
            for key in ("message", "error"):
                val = err.get(key)
                if isinstance(val, str) and val:
                    return val[:500]
        if isinstance(err, str) and err:
            return err[:500]
        for path in (["message"], ["detail"], ["error_message"]):
            val = _dig(j, path, "")
            if val:
                return val[:500]
        return ""

    def _do():
        headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type":"application/json"}
        body = {"model": model_id, "messages":[{"role":"user","content":prompt}], "temperature": float(temperature)}
        try:
            r = requests.post(XAI_BASE_URL, json=body, headers=headers, timeout=_GROK_TIMEOUT)
        except Exception as e:
            return "", {}, f"RequestError: {type(e).__name__}: {str(e)[:300]}"

        try:
            payload = r.json()
        except Exception:
            payload = r.text
        j = _to_json(payload)

        if not r.ok:
            msg = _err_msg_from_json(j)
            if not msg and isinstance(r.text, str):
                msg = r.text[:500]
            return "", {}, f"HTTP {r.status_code}: {msg}"

        text = ""
        choices = j.get("choices")
        if isinstance(choices, list) and choices:
            choice0 = choices[0] if isinstance(choices[0], dict) else {}
            message = choice0.get("message") if isinstance(choice0, dict) else {}
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    text = content.strip()
        if not text and isinstance(j.get("message"), str):
            text = j["message"].strip()

        usage_obj = j.get("usage") if isinstance(j.get("usage"), dict) else {}
        usage = {
            "prompt_tokens": int(usage_obj.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage_obj.get("completion_tokens", 0) or 0),
            "total_tokens": int(
                usage_obj.get("total_tokens")
                or usage_obj.get("totalTokens", 0)
                or (usage_obj.get("prompt_tokens", 0) or 0)
                + (usage_obj.get("completion_tokens", 0) or 0)
            ),
        }
        return text, usage, ""

    return await asyncio.to_thread(_do)

# -------------- public: one call to rule them all --------------
async def call_chat_ms(ms: ModelSpec, prompt: str, temperature: float = 0.2) -> tuple[str, Dict[str,int], str]:
    """
    Return (text, usage_dict, error_message).
    error_message non-empty means provider call failed quickly or was unauthorized.
    """
    if ms.provider == "openrouter":
        return await _call_openrouter(ms.model_id, prompt, temperature)
    if ms.provider == "gemini":
        return await _call_gemini_direct(ms.model_id, prompt, temperature)
    if ms.provider == "grok":
        text, usage, err = await _call_grok_direct(ms.model_id, prompt, temperature)
        if not isinstance(usage, dict):
            usage = {}
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        if not isinstance(err, str):
            err = "" if err is None else str(err)
        return text, usage, err
    return "", {}, f"unsupported provider {ms.provider}"

# -------- Gemini helper used by research.py fallback ----------
async def _call_google(prompt_text: str, model: str = None, timeout: float = 120.0, temperature: float = 0.3) -> str:
    """
    Lightweight wrapper for research composition. Uses DIRECT Gemini when enabled,
    otherwise returns "".
    """
    if not USE_GEMINI:
        return ""
    model_id = (model or GEMINI_MODEL_ID).strip()
    text, _, err = await _call_gemini_direct(model_id, prompt_text, temperature)
    return text if not err else ""
