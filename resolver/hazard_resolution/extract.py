# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""LLM extraction of STATED figures from ReliefWeb documents (ladder rung 2).

**Hard rule 3 lives here.** The model's only job is transcription: find
every people-affected figure a document explicitly states, and copy it out
with the sentence it came from. It is forbidden to estimate, to convert
units, to sum across areas, or to carry a figure over from its own
knowledge. Everything a number could be *turned into* — households into
people, many figures into one — happens deterministically afterwards, in
:mod:`resolver.hazard_resolution.figures`.

Three mechanisms enforce that, and none of them trust the prompt alone:

* **Quote verification.** Every figure must arrive with a quote, the
  quote must appear in the document text that was actually sent, AND the
  figure's value must appear in the quote (as digits, with any grouping,
  or as an "N million/thousand" phrasing). A figure whose quote cannot be
  found — or whose quote does not state the number being claimed — is
  dropped with a reason. A model that invents a number would have to find
  a sentence in the document that already states it.
* **A closed schema.** Values must be numbers and units must be one of
  ``people`` / ``households``. Anything else is dropped, not coerced.
* **No arithmetic.** The parser never adds, scales or reconciles. It
  returns what was said, and says what it discarded.

**Cost.** This is the machine's only paid step, so it is guarded at three
levels: documents are capped per cell (``documents.max_docs_per_cell``),
calls are capped per calendar month across all runs
(``extraction.max_calls_per_month``), and a cell whose higher ladder rung
already has a figure is skipped entirely when
``extraction.skip_when_higher_rung_populated`` is on — reliefweb_extracted
sits below EM-DAT and cannot win those cells.

Extractions are cached in ``haz_doc_extractions`` keyed by (document,
model, prompt version, **cell**), which makes a re-run free and makes the
monthly cap count real spend rather than repeated work. The cell — iso3,
hazard, target month — is part of the key because it is part of the
PROMPT: a typhoon-plus-flooding sitrep is returned by both hazards'
document queries, and a cached answer to "people affected by flooding"
must never be served as the answer to "people affected by a tropical
cyclone". Only ``status='ok'`` rows are cache hits — an error row is
overwritten by the next attempt, so a transient outage never permanently
blinds a document.

The model is addressed by ROLE (``extraction.model_role``), resolved
through the repo's model registry — the rulebook owns the policy,
``pythia/config.yaml`` owns which model backs it.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from resolver.hazard_resolution.rulebook import Rulebook
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.hazard_resolution.sources import parse_number, utcnow_iso

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

#: The ladder rung these figures land on (``rulebook.ladder``).
SOURCE = "reliefweb_extracted"

UNIT_PEOPLE = "people"
UNIT_HOUSEHOLDS = "households"
KNOWN_UNITS = (UNIT_PEOPLE, UNIT_HOUSEHOLDS)

#: Injectable model seam for tests: (model_id, prompt, rulebook) ->
#: (text, usage, error). Mirrors ``forecaster.providers.call_chat_ms``'s
#: return shape so the default implementation is a thin adapter.
CallFn = Callable[[str, str, Rulebook], "tuple[str, dict, str]"]

_CUMULATIVE_VALUES = ("cumulative", "new", "unstated")

_PROMPT_TEMPLATE = """\
You are a transcription tool. You extract figures that a document states. \
You never estimate, infer, calculate, or use knowledge from outside the \
document.

TASK
Read the document below and find every explicitly stated figure for the \
number of PEOPLE AFFECTED by {hazard_label} in {country}. The impact of \
interest occurred in {period}. A document published later may restate \
figures for that event; those still count.

WHAT COUNTS
Count a figure only if the document states it as people (or persons, or \
individuals) affected, or as households/families affected. Related but \
DIFFERENT quantities — displaced, evacuated, killed, injured, missing, \
houses damaged, people in need, people targeted or reached by assistance \
— are NOT people affected. Do not include them.

RULES (all mandatory)
1. Transcribe only. If a number is not written in the document, it does \
not exist. Never estimate or infer one.
2. Never convert units. If the document says households or families, \
report the number of households and set "unit" to "households". Do not \
multiply by an assumed household size.
3. Never sum, total, or combine figures. If the document gives figures for \
three provinces, report three figures, each with its own area. Do not add \
them.
4. Every figure must carry the exact sentence it came from, copied \
character for character from the document. A figure without a verbatim \
quote will be discarded.
5. Report figures for {country} and {hazard_label} only. Ignore figures \
about other countries, other hazards, or events outside {period} (a later \
report restating a {period} event's figures still counts).
6. If the document states no qualifying figure, return an empty list. An \
empty list is a correct and expected answer.

OUTPUT
Return JSON only — no preamble, no explanation, no markdown fences:

{{"figures": [
  {{
    "value": <number, digits only, no separators>,
    "unit": "people" | "households",
    "quote": "<the exact sentence from the document stating this figure>",
    "stated_by": "<who the document attributes the figure to, verbatim; \
empty string if the document does not say>",
    "area": "<the area the figure covers, verbatim; empty string if not \
stated>",
    "date": "<the date the figure refers to, YYYY-MM-DD if stated, else \
empty string>",
    "cumulative_or_new": "cumulative" | "new" | "unstated"
  }}
]}}

DOCUMENT
Title: {title}
Published: {published}
Source: {source}
URL: {url}

{body}
"""

#: Hazard codes -> the wording used in the prompt. Plain language, because
#: the model is reading humanitarian prose, not repo taxonomy.
_HAZARD_LABEL = {
    "FL": "flooding (including flash floods and landslides caused by rain)",
    "TC": "a tropical cyclone (including hurricanes and typhoons)",
    "DR": "drought",
}


@dataclass
class ExtractedFigure:
    """One figure a document STATES, as transcribed and verified."""

    value: float
    unit: str
    quote: str
    stated_by: str
    area: str
    date: str
    cumulative_or_new: str
    doc_id: str
    doc_url: str
    doc_title: str
    doc_date: str
    doc_source_rank: int
    model: str
    #: When the SOURCE published the document (ReliefWeb ``date.original``),
    #: which is the reporting window's best proxy when the figure itself
    #: carries no date. Empty for a cached document written before Sept 2026.
    doc_date_original: str = ""
    #: The country the document is ABOUT (``primary_country.iso3``). Empty
    #: for a cached document written before Sept 2026.
    doc_primary_country: str = ""
    doc_country_iso3s: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "quote": self.quote,
            "stated_by": self.stated_by,
            "area": self.area,
            "date": self.date,
            "cumulative_or_new": self.cumulative_or_new,
            "doc_id": self.doc_id,
            "doc_url": self.doc_url,
            "doc_title": self.doc_title,
            "doc_date": self.doc_date,
            "doc_date_original": self.doc_date_original,
            "doc_primary_country": self.doc_primary_country,
            "doc_country_iso3s": list(self.doc_country_iso3s),
            "doc_source_rank": self.doc_source_rank,
            "model": self.model,
        }


@dataclass
class ExtractionBudget:
    """The cost guard, shared across every cell in one run.

    ``used_this_month`` is read from ``haz_doc_extractions`` at
    construction, so the cap spans runs: three CLI invocations in one month
    cannot each spend the monthly allowance.

    A BACKCAST run is bounded twice over, so the live trailing-window run's
    rung 2 cannot be starved (it was: the backcast spent all of August 2026's
    allowance by mid-month, and cyclone August then ran out at Jamaica with
    documents fetched and never read):

    ``backcast_max_calls_per_month``
        the backcast's own share, a carve-out of the monthly total and never
        an addition;
    ``live_reserve_calls``
        calls the backcast may never take, whatever its share says. Until
        Sept 2026 the reserve was implicit — whatever the total minus the
        share happened to be — so raising or dropping the share silently
        removed it. Stated outright, it cannot be lost by accident.

    A LIVE run checks only the total: the reserve exists for its benefit, so
    it does not also apply to it.
    """

    max_calls_per_month: int
    used_this_month: int = 0
    calls_this_run: int = 0
    cost_this_run_usd: float = 0.0
    cached_hits: int = 0
    capped_cells: list[str] = field(default_factory=list)
    run_type: str = "live"
    backcast_max_calls_per_month: int | None = None
    backcast_used_this_month: int = 0
    live_reserve_calls: int = 0
    #: Optional ceiling on ONE run's calls (``extraction.max_calls_per_run``),
    #: for a live pass that must fit a workflow step: the budget, not a step
    #: timeout, is then what stops it, and a budget stop is recorded per cell
    #: while a timeout records nothing.
    max_calls_per_run: int | None = None

    def _limits(self) -> list[tuple[str, int]]:
        """Every limit in force, named, so the binding one can be reported."""

        limits = [(
            "monthly total",
            max(0, self.max_calls_per_month - self.used_this_month - self.calls_this_run),
        )]
        if self.max_calls_per_run is not None:
            limits.append(("per-run cap", max(0, self.max_calls_per_run - self.calls_this_run)))
        if self.run_type != "backcast":
            return limits
        if self.backcast_max_calls_per_month is not None:
            limits.append((
                "backcast share",
                max(
                    0,
                    self.backcast_max_calls_per_month
                    - self.backcast_used_this_month
                    - self.calls_this_run,
                ),
            ))
        if self.live_reserve_calls:
            # What the backcast may spend before it starts eating the live
            # pass's reserve. Counted against the WHOLE month's usage, so a
            # backcast that has already run tonight cannot ignore it.
            limits.append((
                "live reserve",
                max(
                    0,
                    self.max_calls_per_month
                    - self.live_reserve_calls
                    - self.used_this_month
                    - self.calls_this_run,
                ),
            ))
        return limits

    @property
    def remaining(self) -> int:
        return min(value for _, value in self._limits())

    @property
    def binding_limit(self) -> str:
        """Which limit is the tightest, e.g. ``backcast share (2000)``.

        Last night's log said "monthly cap of 3000 reached" when the 2,000
        backcast share was what had bound; a message that names the wrong
        limit sends the reader to the wrong knob.
        """

        name, _ = min(self._limits(), key=lambda pair: pair[1])
        values = {
            "monthly total": self.max_calls_per_month,
            "per-run cap": self.max_calls_per_run,
            "backcast share": self.backcast_max_calls_per_month,
            "live reserve": (
                self.max_calls_per_month - self.live_reserve_calls
                if self.live_reserve_calls else None
            ),
        }
        return f"{name} ({values.get(name)})"

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def note_capped(self, cell: str) -> None:
        if cell not in self.capped_cells:
            self.capped_cells.append(cell)

    def as_provenance(self) -> dict[str, Any]:
        out = {
            "max_calls_per_month": self.max_calls_per_month,
            "used_this_month_before_run": self.used_this_month,
            "calls_this_run": self.calls_this_run,
            "cached_hits": self.cached_hits,
            "cost_this_run_usd": round(self.cost_this_run_usd, 4),
            "remaining": self.remaining,
            "capped": self.exhausted,
            "binding_limit": self.binding_limit,
            "run_type": self.run_type,
        }
        if self.max_calls_per_run is not None:
            out["max_calls_per_run"] = self.max_calls_per_run
        if self.run_type == "backcast" and self.backcast_max_calls_per_month is not None:
            out["backcast_max_calls_per_month"] = self.backcast_max_calls_per_month
            out["backcast_used_this_month_before_run"] = self.backcast_used_this_month
        return out


@dataclass
class CellExtraction:
    """What extraction did for one country-month-hazard."""

    iso3: str
    ym: str
    hazard: str
    ok: bool = True
    skipped_reason: str | None = None
    docs_read: int = 0
    docs_cached: int = 0
    docs_failed: int = 0
    calls_made: int = 0
    cost_usd: float = 0.0
    budget_capped: bool = False
    figures: list[ExtractedFigure] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def as_provenance(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "skipped_reason": self.skipped_reason,
            "docs_read": self.docs_read,
            "docs_cached": self.docs_cached,
            "docs_failed": self.docs_failed,
            "calls_made": self.calls_made,
            "cost_usd": round(self.cost_usd, 4),
            "budget_capped": self.budget_capped,
            "figures_extracted": len(self.figures),
            "figures_rejected": self.rejected,
        }


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


def resolve_extraction_model(rulebook: Rulebook) -> str:
    """The ``provider:model_id`` backing ``extraction.model_role``.

    Imported lazily: the resolver package must stay importable without the
    forecaster's provider stack, and a role lookup is the only thing this
    module needs from ``pythia``.
    """

    from pythia.llm_profiles import get_role_model

    role = str(rulebook.get("extraction.model_role"))
    ref = get_role_model(role)
    if not ref:
        raise RuntimeError(
            f"extraction.model_role {role!r} does not resolve to a model; add it to "
            "pythia/config.yaml's roles block and pythia/llm_profiles._ROLE_FALLBACKS"
        )
    return ref


def _default_call(model_ref: str, prompt: str, rulebook: Rulebook) -> tuple[str, dict, str]:
    """Call the extraction model once. Never raises.

    Goes through ``forecaster.providers.call_chat_ms`` — the repo's single
    provider path, so retries, circuit breakers and cost accounting are the
    ones every other call site gets. ``log_call=False`` because this module
    keeps its own ledger in ``haz_doc_extractions`` and the generic
    ``llm_calls`` row would double-count the spend.
    """

    import asyncio

    from forecaster.providers import ModelSpec, call_chat_ms
    from pythia.llm_profiles import split_model_ref

    provider, model_id = split_model_ref(model_ref)
    spec = ModelSpec(
        name=model_id,
        provider=provider,
        model_id=model_id,
        purpose="hazard_extraction",
        temperature=float(rulebook.get("extraction.temperature")),
    )
    try:
        return asyncio.run(
            call_chat_ms(
                spec,
                prompt,
                temperature=float(rulebook.get("extraction.temperature")),
                prompt_key="resolver.hazard_extraction",
                component="HazardResolution",
                log_call=False,
                # The rulebook's declared budgets, actually enforced: the
                # timeout at both the async layer and the Anthropic
                # transport, the token ceiling in the request body.
                timeout_sec=float(rulebook.get("extraction.request_timeout_sec")),
                max_output_tokens=int(rulebook.get("extraction.max_output_tokens")),
            )
        )
    except Exception as exc:  # pragma: no cover - network/runtime failures
        LOG.warning("[extract] model call failed: %s", exc)
        return "", {}, str(exc)


def _log_llm_call(
    *,
    model_ref: str,
    prompt: str,
    response: str,
    usage: dict[str, Any],
    error: str,
    iso3: str,
    hazard: str,
    ym: str,
) -> None:
    """One rich ``llm_calls`` telemetry row per REAL extraction call.

    ``haz_doc_extractions`` is the machine's cache and per-document ledger;
    ``llm_calls`` is the repo-wide cost telemetry the Costs page, the debug
    bundle and every diagnostic read. A paid call that writes no
    ``llm_calls`` row is invisible spend — this is the repo's
    ``log_call=False`` contract: the generic row is suppressed at
    ``call_chat_ms`` and a rich one (phase + iso3 + hazard, ``is_test``
    inherited) is written here instead. Never raises; a telemetry failure
    must not cost a resolution.
    """

    try:
        import asyncio

        from forecaster.llm_logging import log_forecaster_llm_call
        from pythia.llm_profiles import split_model_ref

        provider, model_id = split_model_ref(model_ref)
        asyncio.run(
            log_forecaster_llm_call(
                run_id=f"hazres_{ym}",
                question_id="",
                prompt_text=prompt,
                model_name=model_id,
                provider=provider,
                model_id=model_id,
                phase="hazard_extraction",
                call_type="hazard_extraction",
                iso3=iso3.upper(),
                hazard_code=hazard,
                metric="PA",
                response_text=response,
                usage=usage or {},
                error_text=(error or None),
            )
        )
    except Exception as exc:  # noqa: BLE001 - telemetry must never break a run
        LOG.warning("[extract] llm_calls telemetry row not written: %s", exc)


#: Memoised (pricer, splitter) or the sentinel False once pricing has been
#: found unavailable. Resolved once per process: the import is lazy, and a
#: sparse environment must not produce one warning per document read.
_PRICER: Any = None


def _pricer():
    global _PRICER
    if _PRICER is not None:
        return _PRICER
    try:
        from forecaster.providers import estimate_cost_usd
        from pythia.llm_profiles import split_model_ref

        _PRICER = (estimate_cost_usd, split_model_ref)
    except Exception as exc:  # pragma: no cover - sparse environments
        LOG.warning(
            "[extract] cost pricing unavailable (%s) — calls will still be made "
            "and counted against the monthly cap, but the run's spend line will "
            "read $0.00",
            exc,
        )
        _PRICER = False
    return _PRICER


def _cost_usd(model_ref: str, usage: dict[str, Any]) -> float:
    """Spend for one call, from the provider's own token counts.

    Best-effort: pricing that cannot be resolved must not stop a run, and
    the CALL CAP rather than the dollar figure is what actually bounds
    spend — so a $0.00 spend line degrades the report, not the guard.
    """

    pricer = _pricer()
    if not pricer:
        return 0.0
    estimate_cost_usd, split_model_ref = pricer
    try:
        _, model_id = split_model_ref(model_ref)
        return float(estimate_cost_usd(model_id, usage or {}))
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("[extract] could not price a call on %s: %s", model_ref, exc)
        return 0.0


# ---------------------------------------------------------------------------
# Prompt + strict parsing
# ---------------------------------------------------------------------------


_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _period_label(ym: str | None) -> str:
    """``2026-06`` -> ``June 2026``; anything unparseable -> a safe fallback."""

    try:
        year, month = str(ym).split("-")
        return f"{_MONTH_NAMES[int(month) - 1]} {int(year)}"
    except Exception:
        return "the period this document reports on"


def build_prompt(
    document: dict[str, Any],
    iso3: str,
    hazard: str,
    country: str | None = None,
    ym: str | None = None,
) -> str:
    """The extraction prompt for one document. Pure function.

    ``ym`` names the target month IN the prompt — the model is told to
    ignore other time periods, and an instruction to filter by period is
    meaningless unless the period is stated. ``country`` should be the
    human-readable name; the model is reading humanitarian prose that says
    "the Philippines", not "PHL".
    """

    return _PROMPT_TEMPLATE.format(
        hazard_label=_HAZARD_LABEL.get(hazard, hazard),
        country=country or iso3,
        period=_period_label(ym),
        title=document.get("title") or "(untitled)",
        published=document.get("date_created") or "(undated)",
        source=", ".join(document.get("sources") or []) or "(unattributed)",
        url=document.get("url") or "(no url)",
        body=document.get("body") or "",
    )


def _normalise_for_match(text: str) -> str:
    """Collapse whitespace and case so a quote can be found in a body.

    Deliberately conservative: it normalises only whitespace, case and the
    typographic characters that differ between a JSON string and the HTML
    a document was rendered from. It never removes words, so a quote that
    does not appear in the document still fails to match.
    """

    text = str(text)
    for fancy, plain in (
        ("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'),
        ("–", "-"), ("—", "-"), (" ", " "),
    ):
        text = text.replace(fancy, plain)
    return re.sub(r"\s+", " ", text).strip().lower()


#: Number tokens a quote can state a figure with: a grouped integer
#: ("12,345", "5 000", "12.345"), or a plain integer/decimal ("8300",
#: "1.2"), optionally scaled by a word multiplier ("1.2 million").
_QUOTE_NUMBER_RE = re.compile(
    r"(\d{1,3}(?:[,. ]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:\s*(million|mn|thousand)\b)?",
    re.IGNORECASE,
)

_NUMBER_SUFFIX_MULTIPLIERS = {"million": 1_000_000.0, "mn": 1_000_000.0, "thousand": 1_000.0}


def _numbers_in_text(text: str) -> set[float]:
    """Every number the text states, under each plausible reading.

    Grouping is ambiguous in the wild — "12.345" is twelve-point-three in
    one report and twelve thousand in another — so both readings are
    admitted. Permissive on AMBIGUITY, never on ABSENCE: a quote with no
    digits yields the empty set, and a stated "1.2 million" yields
    1,200,000, not 1.2.
    """

    out: set[float] = set()
    for match in _QUOTE_NUMBER_RE.finditer(text):
        raw, suffix = match.group(1), (match.group(2) or "").lower()
        multiplier = _NUMBER_SUFFIX_MULTIPLIERS.get(suffix, 1.0)
        compact = raw.replace(",", "").replace(" ", "")
        readings: set[float] = set()
        try:
            readings.add(float(compact))
        except ValueError:  # pragma: no cover - regex guarantees digits
            continue
        if "." in compact:
            try:
                readings.add(float(compact.replace(".", "")))
            except ValueError:  # pragma: no cover - defensive
                pass
        out.update(value * multiplier for value in readings)
    return out


def _value_stated_in_quote(value: float, quote: str) -> bool:
    """Does the quote actually state this figure?

    The quote-in-body check alone lets a model pair a fabricated value with
    any real sentence — verification must tie the NUMBER to the sentence,
    not just the sentence to the document.
    """

    return any(abs(n - float(value)) < 0.5 for n in _numbers_in_text(_normalise_for_match(quote)))


def _strip_fences(text: str) -> str:
    """Remove a ```json fence if the model wrapped its JSON in one."""

    stripped = str(text).strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
    if stripped.endswith("```"):
        stripped = stripped[: -3]
    return stripped.strip()


def _loads(text: str) -> Any:
    """Parse the model's JSON, tolerating a fence or surrounding prose."""

    candidate = _strip_fences(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("response contains no JSON object")
    return json.loads(candidate[start : end + 1])


def parse_response(
    text: str, document: dict[str, Any], model: str
) -> tuple[list[ExtractedFigure], list[dict[str, Any]]]:
    """Parse and VERIFY one model response. Returns (figures, rejected).

    Strict by construction — every rejection carries a reason, because a
    silent drop and a document that genuinely stated nothing look identical
    downstream, and only one of them is a bug.
    """

    rejected: list[dict[str, Any]] = []
    figures: list[ExtractedFigure] = []

    try:
        payload = _loads(text)
    except (ValueError, json.JSONDecodeError) as exc:
        rejected.append({"reason": "unparseable_response", "detail": str(exc)})
        return figures, rejected

    if not isinstance(payload, dict) or not isinstance(payload.get("figures"), list):
        rejected.append(
            {"reason": "unexpected_shape", "detail": "no 'figures' list in the response"}
        )
        return figures, rejected

    body = _normalise_for_match(document.get("body") or "")

    for index, raw in enumerate(payload["figures"]):
        where = {"index": index, "raw": raw}
        if not isinstance(raw, dict):
            rejected.append({**where, "reason": "not_an_object"})
            continue

        value = parse_number(raw.get("value"))
        if value is None:
            rejected.append({**where, "reason": "value_not_a_number"})
            continue

        unit = str(raw.get("unit") or "").strip().lower()
        if unit not in KNOWN_UNITS:
            # Never coerced to "people": a households figure read as people
            # understates by roughly five, and guessing which was meant is
            # exactly the inference this module is not allowed to make.
            rejected.append({**where, "reason": "unknown_unit", "detail": unit})
            continue

        quote = str(raw.get("quote") or "").strip()
        if not quote:
            rejected.append({**where, "reason": "missing_quote"})
            continue
        if _normalise_for_match(quote) not in body:
            rejected.append({**where, "reason": "quote_not_found_in_document"})
            continue
        if not _value_stated_in_quote(float(value), quote):
            rejected.append({**where, "reason": "value_not_found_in_quote"})
            continue

        cumulative = str(raw.get("cumulative_or_new") or "unstated").strip().lower()
        if cumulative not in _CUMULATIVE_VALUES:
            cumulative = "unstated"

        figures.append(
            ExtractedFigure(
                value=float(value),
                unit=unit,
                quote=quote,
                stated_by=str(raw.get("stated_by") or "").strip(),
                area=str(raw.get("area") or "").strip(),
                date=str(raw.get("date") or "").strip(),
                cumulative_or_new=cumulative,
                doc_id=str(document.get("doc_id") or ""),
                doc_url=str(document.get("url") or ""),
                doc_title=str(document.get("title") or ""),
                doc_date=str(document.get("date_created") or ""),
                doc_source_rank=int(document.get("source_rank") or 0),
                model=model,
                doc_date_original=str(document.get("date_original") or ""),
                doc_primary_country=str(document.get("primary_country_iso3") or "").upper(),
                doc_country_iso3s=tuple(
                    str(c).upper() for c in (document.get("country_iso3s") or [])
                ),
            )
        )

    return figures, rejected


# ---------------------------------------------------------------------------
# The extraction cache / cost ledger
# ---------------------------------------------------------------------------


def _year_month(ym: str) -> tuple[int, int]:
    year, month = ym.split("-")
    return int(year), int(month)


def calls_this_calendar_month(
    con: "duckdb.DuckDBPyConnection",
    today: dt.date | None = None,
    *,
    backcast_only: bool = False,
) -> int:
    """Extraction calls already billed in the current calendar month.

    Counts ``haz_doc_extractions`` rows by creation time, not by target
    month: the cap guards SPEND, and spend happens when the call is made.
    Error rows count only when the provider actually billed tokens — a
    call that never left the process (missing API key, circuit-breaker
    cooldown) spent nothing and must not consume the allowance.

    With ``backcast_only``, counts only the backcast's share — rows with
    ``run_type = 'backcast'`` OR a NULL run_type: every pre-split row was
    backcast-created (the live path first runs 2026-08-28), and counting
    the legacy rows against the live reserve instead would let one more
    night of backcast eat exactly the budget the split exists to protect.
    """

    ensure_haz_schema(con)
    reference = today or dt.date.today()
    run_type_clause = (
        "AND (run_type = 'backcast' OR run_type IS NULL)" if backcast_only else ""
    )
    row = con.execute(
        f"""
        SELECT COUNT(*) FROM haz_doc_extractions
        WHERE CAST(strftime(created_at, '%Y-%m') AS VARCHAR) = ?
          AND (status = 'ok'
               OR COALESCE(prompt_tokens, 0) + COALESCE(completion_tokens, 0) > 0)
          {run_type_clause}
        """,
        [f"{reference.year:04d}-{reference.month:02d}"],
    ).fetchone()
    return int(row[0] or 0)


def load_budget(
    con: "duckdb.DuckDBPyConnection",
    rulebook: Rulebook,
    today: dt.date | None = None,
    *,
    run_type: str = "live",
) -> ExtractionBudget:
    """The run's cost guard, seeded from what this month already cost."""

    backcast_cap: int | None = None
    backcast_used = 0
    if run_type == "backcast":
        raw_cap = rulebook.get("extraction.backcast_max_calls_per_month", None)
        backcast_cap = int(raw_cap) if raw_cap is not None else None
        if backcast_cap is not None:
            backcast_used = calls_this_calendar_month(
                con, today=today, backcast_only=True
            )
    try:
        reserve = int(rulebook.get("extraction.live_reserve_calls"))
    except Exception:  # noqa: BLE001 - older rulebooks have no such key
        reserve = 0
    raw_run_cap = rulebook.get("extraction.max_calls_per_run", None)
    run_cap = int(raw_run_cap) if raw_run_cap is not None else None
    budget = ExtractionBudget(
        max_calls_per_month=int(rulebook.get("extraction.max_calls_per_month")),
        used_this_month=calls_this_calendar_month(con, today=today),
        run_type=run_type,
        backcast_max_calls_per_month=backcast_cap,
        backcast_used_this_month=backcast_used,
        live_reserve_calls=reserve,
        max_calls_per_run=run_cap,
    )
    LOG.info(
        "[extract] budget: %d of %d extraction calls already made this calendar "
        "month, %d remaining%s",
        budget.used_this_month,
        budget.max_calls_per_month,
        budget.remaining,
        (
            f" (backcast share: {backcast_used} of {backcast_cap}; "
            f"live reserve: {reserve})"
            if backcast_cap is not None
            else ""
        ),
    )
    return budget


def load_cached_extraction(
    con: "duckdb.DuckDBPyConnection",
    doc_id: str,
    model: str,
    prompt_version: str,
    *,
    iso3: str,
    hazard: str,
    ym: str,
) -> dict[str, Any] | None:
    """A previous USABLE extraction of this document for this cell, or None.

    Scoped to the cell because the prompt is: it names the country, the
    hazard and the target month, so an answer produced for one cell is not
    an answer for another even when the document is shared. Only
    ``status='ok'`` rows are hits — an error row means the call failed, and
    the next run should try again rather than inherit the failure forever.
    """

    ensure_haz_schema(con)
    year, month = _year_month(ym)
    row = con.execute(
        """
        SELECT status, figures_json, n_rejected, cost_usd
        FROM haz_doc_extractions
        WHERE doc_id = ? AND model = ? AND prompt_version = ?
          AND iso3 = ? AND hazard = ? AND year = ? AND month = ?
          AND status = 'ok'
        """,
        [str(doc_id), str(model), str(prompt_version), iso3.upper(), hazard, year, month],
    ).fetchone()
    if row is None:
        return None
    return {
        "status": row[0],
        "payload": json.loads(row[1] or "{}"),
        "n_rejected": int(row[2] or 0),
        "cost_usd": float(row[3] or 0.0),
    }


def write_extraction(
    con: "duckdb.DuckDBPyConnection",
    *,
    document: dict[str, Any],
    iso3: str,
    ym: str,
    hazard: str,
    model: str,
    prompt_version: str,
    status: str,
    figures: list[ExtractedFigure],
    rejected: list[dict[str, Any]],
    usage: dict[str, Any],
    cost_usd: float,
    error: str | None,
    run_type: str = "live",
) -> None:
    """Record one extraction call — the cache entry AND the ledger line."""

    ensure_haz_schema(con)
    year, month = _year_month(ym)
    if status != "ok":
        # INSERT OR REPLACE would let a transient failure overwrite an ok
        # row for the same key, and only load_cached_extraction's short
        # circuit stood between that and a paid answer being lost. Keep the
        # ok row; the failed attempt is logged and costs nothing.
        existing = con.execute(
            """
            SELECT 1 FROM haz_doc_extractions
            WHERE doc_id = ? AND model = ? AND prompt_version = ?
              AND iso3 = ? AND hazard = ? AND year = ? AND month = ?
              AND status = 'ok'
            """,
            [str(document.get("doc_id") or ""), model, prompt_version,
             iso3.upper(), hazard, year, month],
        ).fetchone()
        if existing:
            LOG.debug(
                "[extract] keeping the cached ok extraction for %s %s %s doc %s "
                "over a failed re-read (%s)",
                iso3, hazard, ym, document.get("doc_id"), error,
            )
            return
    payload = {
        "figures": [f.as_dict() for f in figures],
        "rejected": rejected,
        "extracted_at": utcnow_iso(),
    }
    con.execute(
        """
        INSERT OR REPLACE INTO haz_doc_extractions
            (doc_id, iso3, year, month, hazard, model, prompt_version, status,
             figures_json, n_figures, n_rejected, prompt_tokens,
             completion_tokens, cost_usd, doc_url, error, run_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            str(document.get("doc_id") or ""),
            iso3.upper(),
            year,
            month,
            hazard,
            model,
            prompt_version,
            status,
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
            len(figures),
            len(rejected),
            int((usage or {}).get("prompt_tokens") or 0),
            int((usage or {}).get("completion_tokens") or 0),
            float(cost_usd),
            str(document.get("url") or ""),
            error,
            run_type,
        ],
    )


def _figures_from_cache(cached: dict[str, Any], model: str) -> list[ExtractedFigure]:
    out: list[ExtractedFigure] = []
    for entry in cached["payload"].get("figures") or []:
        fields = {k: v for k, v in entry.items() if k in ExtractedFigure.__annotations__}
        fields.setdefault("model", model)
        try:
            out.append(ExtractedFigure(**fields))
        except TypeError:  # pragma: no cover - a cache row from an older shape
            LOG.warning("[extract] skipping an unreadable cached figure for %s", model)
    return out


# ---------------------------------------------------------------------------
# Per-cell orchestration
# ---------------------------------------------------------------------------


def extract_for_cell(
    con: "duckdb.DuckDBPyConnection",
    *,
    iso3: str,
    ym: str,
    hazard: str,
    rulebook: Rulebook,
    budget: ExtractionBudget,
    documents: list[dict[str, Any]] | None = None,
    call: CallFn | None = None,
    model_ref: str | None = None,
    country_name: str | None = None,
) -> CellExtraction:
    """Read this cell's cached documents and transcribe their figures.

    Callers must only reach this for TRIGGERED country-months — the
    acceptance criterion is that no extraction call is ever made for a cell
    that resolved without one. :mod:`impact` enforces that by walking only
    the triggered cells.
    """

    from resolver.hazard_resolution import reliefweb_docs as docs_mod

    iso3 = iso3.upper()
    result = CellExtraction(iso3=iso3, ym=ym, hazard=hazard)

    if not bool(rulebook.get("extraction.enabled")):
        result.skipped_reason = "extraction_disabled"
        return result

    if documents is None:
        documents = docs_mod.documents_for_country_month(
            con, iso3, ym, hazard, rulebook=rulebook
        )
    if not documents:
        result.skipped_reason = "no_documents"
        return result

    prompt_version = str(rulebook.get("extraction.prompt_version"))
    try:
        model_ref = model_ref or resolve_extraction_model(rulebook)
    except Exception as exc:
        LOG.error("[extract] %s %s %s: %s", iso3, hazard, ym, exc)
        result.ok = False
        result.skipped_reason = f"model_unresolved: {exc}"
        return result

    # The rich llm_calls telemetry row is written only for the REAL model
    # seam: an injected test seam makes no billable call and must not write
    # telemetry (network-free tests would otherwise reach for the Pythia DB).
    using_default_call = call is None
    call = call or _default_call

    for document in documents:
        doc_id = str(document.get("doc_id") or "")
        if not doc_id:
            continue

        cached = load_cached_extraction(
            con, doc_id, model_ref, prompt_version, iso3=iso3, hazard=hazard, ym=ym
        )
        if cached is not None:
            result.docs_cached += 1
            result.figures.extend(_figures_from_cache(cached, model_ref))
            budget.cached_hits += 1
            continue

        if budget.exhausted:
            # The cap is a hard stop, and a stop is not a finding: the
            # remaining UNCACHED documents are UNREAD, which the cell's
            # provenance records so nobody later mistakes it for "ReliefWeb
            # was silent". Cached documents are already paid for, so the
            # loop keeps walking to collect them — only fresh reads stop.
            if not result.budget_capped:
                result.budget_capped = True
                budget.note_capped(f"{iso3}/{hazard}/{ym}")
                LOG.warning(
                    "[extract] extraction budget exhausted at its %s — %s %s %s has "
                    "unread documents; the rung is UNREAD, not empty",
                    budget.binding_limit, iso3, hazard, ym,
                )
            continue

        prompt = build_prompt(document, iso3, hazard, country_name, ym=ym)
        text, usage, error = call(model_ref, prompt, rulebook)
        if using_default_call:
            _log_llm_call(
                model_ref=model_ref, prompt=prompt, response=text,
                usage=usage, error=error, iso3=iso3, hazard=hazard, ym=ym,
            )
        # A call that never reached the provider — missing API key, circuit
        # breaker cooldown, a local exception — billed nothing and must not
        # consume the monthly allowance. Anything that billed tokens, or
        # succeeded, counts.
        billed_tokens = int((usage or {}).get("prompt_tokens") or 0) + int(
            (usage or {}).get("completion_tokens") or 0
        )
        attempted = bool(billed_tokens) or not error
        if attempted:
            budget.calls_this_run += 1
            result.calls_made += 1

        cost = _cost_usd(model_ref, usage)
        budget.cost_this_run_usd += cost
        result.cost_usd += cost

        if error or not str(text).strip():
            result.docs_failed += 1
            detail = error or "empty response"
            LOG.warning(
                "[extract] %s %s %s doc %s: call failed (%s)",
                iso3, hazard, ym, doc_id, detail,
            )
            write_extraction(
                con, document=document, iso3=iso3, ym=ym, hazard=hazard,
                model=model_ref, prompt_version=prompt_version, status="error",
                figures=[], rejected=[], usage=usage, cost_usd=cost, error=detail,
                run_type=budget.run_type,
            )
            continue

        figures, rejected = parse_response(text, document, model_ref)
        result.docs_read += 1
        result.figures.extend(figures)
        result.rejected.extend(
            {**entry, "doc_id": doc_id} for entry in rejected
        )
        if rejected:
            LOG.info(
                "[extract] %s %s %s doc %s: %d figure(s) kept, %d rejected (%s)",
                iso3, hazard, ym, doc_id, len(figures), len(rejected),
                ",".join(sorted({str(r.get("reason")) for r in rejected})),
            )
        write_extraction(
            con, document=document, iso3=iso3, ym=ym, hazard=hazard,
            model=model_ref, prompt_version=prompt_version, status="ok",
            figures=figures, rejected=rejected, usage=usage, cost_usd=cost,
            error=None, run_type=budget.run_type,
        )

    LOG.info(
        "[extract] %s %s %s: %d docs read (%d cached, %d failed), %d figures, "
        "$%.4f this cell",
        iso3, hazard, ym, result.docs_read, result.docs_cached,
        result.docs_failed, len(result.figures), result.cost_usd,
    )
    return result
