# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Pull what the agent asked for out of a long document (Oct 2026).

A situation report runs to tens of thousands of characters; putting it whole
into an Opus prompt every step is dear, and cutting it at 6,000 characters
(as fetch_url did) loses the table on page nine. So the agent sends an
``extraction_request`` with each fetch, and a cheaper model (role
``sibyl_extraction``, Haiku-class) returns at most 500 words: figures copied
word for word with units, the period each refers to, the publication date,
and who said it. It must say "nothing relevant" when that is the case and
must never estimate.

Documents under ``SIBYL_EXTRACTION_SKIP_CHARS`` are shown whole and cost
nothing. If extraction fails the first ``SIBYL_EXTRACTION_SKIP_CHARS``
characters are shown instead, so a failed call never costs the read.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from sibyl import config as _cfg

logger = logging.getLogger(__name__)

ROLE = "sibyl_extraction"

EXTRACTION_PROMPT = """You extract facts from one document for a forecaster. Read the document below and answer the request.

Question being forecast: {question}
Country: {country}
What the forecaster asked for: {request}

Return at most {max_words} words, as short lines:
- every relevant figure copied WORD FOR WORD with its unit, the period it refers to, and who stated it;
- the document's publication date if it states one;
- one line on anything else that bears directly on the request.
Do not estimate, convert, add up or infer figures. If the document says nothing relevant, answer exactly: nothing relevant.

=== DOCUMENT ({url}) ===
{text}
=== END DOCUMENT ==="""

ModelCall = Callable[[str, str], Tuple[str, Dict[str, Any], str]]


@dataclass
class Extraction:
    text: str
    extracted: bool  # True when the extraction model answered
    cost_usd: float = 0.0
    error: Optional[str] = None


def extraction_model() -> str:
    """The model id for the role (provider prefix stripped)."""
    from pythia.llm_profiles import get_role_model, split_model_ref  # noqa: PLC0415

    ref = get_role_model(ROLE)
    _provider, model_id = split_model_ref(ref)
    return model_id


def _default_call(prompt: str, model_id: str) -> Tuple[str, Dict[str, Any], str]:
    from forecaster.providers import call_anthropic, estimate_cost_usd  # noqa: PLC0415

    result = call_anthropic(prompt, model_id, 0.0, purpose="sibyl_extraction",
                            max_tokens_override=1500)
    usage = dict(result.usage or {})
    if not usage.get("cost_usd"):
        usage["cost_usd"] = estimate_cost_usd(model_id, usage)
    return result.text or "", usage, result.error or ""


def extract(
    text: str,
    request: str,
    *,
    url: str,
    question: str,
    country: str,
    call: Optional[ModelCall] = None,
    log: Optional[Callable[..., None]] = None,
) -> Extraction:
    """The text the agent sees for one document. Never raises."""
    text = text or ""
    if len(text) < _cfg.EXTRACTION_SKIP_CHARS:
        return Extraction(text=text, extracted=False)
    head = text[: _cfg.EXTRACTION_SKIP_CHARS]
    try:
        model_id = extraction_model()
        prompt = EXTRACTION_PROMPT.format(
            question=question or "(not stated)", country=country or "",
            request=(request or "the figures and dated facts relevant to the question").strip(),
            max_words=_cfg.EXTRACTION_MAX_WORDS, url=url, text=text,
        )
        answer, usage, error = (call or _default_call)(prompt, model_id)
        cost = float(usage.get("cost_usd") or 0.0)
        if log is not None:
            try:
                log(prompt=prompt, response=answer, usage=usage, model_id=model_id, error=error)
            except Exception:  # noqa: BLE001
                pass
        if error or not (answer or "").strip():
            return Extraction(text=head, extracted=False, cost_usd=cost,
                              error=error or "empty extraction")
        return Extraction(text=answer.strip(), extracted=True, cost_usd=cost)
    except Exception as exc:  # noqa: BLE001 - a failed extraction never costs the read
        logger.warning("sibyl.extract: extraction failed for %s: %s", url, exc)
        return Extraction(text=head, extracted=False, error=str(exc))
