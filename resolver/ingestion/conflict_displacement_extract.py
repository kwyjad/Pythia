# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""People newly displaced by conflict, transcribed from ReliefWeb documents.

ACE/PA questions resolve on IDMC conflict displacement, and IDMC reports the
biggest crises as figures spanning several months, which the IDMC ingest
holds out of every month rather than apportion (COD 3,059,379 and SDN
3,606,786 people held since 2025-01, with no monthly row). This module is a
second source for those months: per (country, month) it reads what OCHA,
UN agencies and governments wrote on ReliefWeb and keeps a figure only when
the document states, verbatim, that people were NEWLY displaced by conflict
or violence within that month.

Part A (Oct 2026): this module records and reports. Nothing reads its
tables to resolve a question yet; Part B will, through
:func:`extracted_figure` / :func:`extracted_figures`.

Everything that could be reused from the PA resolution machine is imported,
not copied:

* document selection and the raw cache: ``reliefweb_docs.select_documents``
  (ranking by ``reliefweb.documents.source_priority``) and the
  ``haz_raw_reliefweb_docs`` cache, stored under hazard ``ACE``;
* the extraction loop: ``extract.extract_for_cell`` with an
  :class:`~resolver.hazard_resolution.extract.ExtractionProfile` that swaps
  only the prompt, the prompt version, the parser and the ledger table. The
  model seam, the truncation retry, the billing rule, the error-row rule and
  the rich ``llm_calls`` row (``phase='hazard_extraction'``,
  ``call_type='conflict_displacement_extraction'``) are the machine's;
* quote verification: ``extract.parse_response`` (the quote must be found
  in the text sent and must state the value);
* households to people, deduplication and authority precedence:
  ``figures.convert_units`` / ``deduplicate`` / ``preference_key``, and the
  primary-country test ``figures.apply_attribution``.

What is this module's own: the prompt (``conflict_v1``), the admissibility
rules that make a figure a monthly FLOW (new, conflict, inside the month),
its tables, and its budget. The budget is counted over
``conflict_doc_extractions`` alone, so it never draws on the PA machine's
monthly cap or its live reserve, which count ``haz_doc_extractions``.

Admissible only if all hold, each rejection counted with its reason:

1. the quote is found in the text sent and states the value
   (``extract.parse_response``);
2. ``cumulative_or_new == "new"`` (a total since an earlier date, or people
   currently displaced, is a stock, not a month's flow);
3. the figure is attributed to conflict or violence
   (``conflict_displacement.admissible_causes``);
4. the figure's date or period lies inside the month. A period longer than
   the month is rejected, never apportioned; a period crossing a month
   boundary is rejected, never split; an undated figure is rejected;
5. the document's primary country is the cell's country.

It never runs in backtest: ``SIBYL_BACKTEST_MODE=1`` disables it, and no
backcast path calls it.
"""

from __future__ import annotations

import argparse
import calendar
import dataclasses
import datetime as dt
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from resolver.hazard_resolution import extract as extract_mod
from resolver.hazard_resolution import figures as figures_mod
from resolver.hazard_resolution import reliefweb_docs as docs_mod
from resolver.hazard_resolution.extract import ExtractedFigure, ExtractionBudget, ExtractionProfile
from resolver.hazard_resolution.rulebook import Rulebook, load_rulebook
from resolver.hazard_resolution.sources import (
    RawRecord,
    month_bounds,
    parse_date,
    store_raw_records,
    utcnow_iso,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

HAZARD = "ACE"
DOC_SOURCE = docs_mod.SOURCE  # the shared haz_raw_reliefweb_docs cache
CALL_TYPE = "conflict_displacement_extraction"
VALUE_TYPE = "new_displacements"
#: The one-dispatch override of the monthly cap (a backfill raises it).
CAP_OVERRIDE_ENV = "PYTHIA_CONFLICT_EXTRACTION_MONTHLY_CAP"
#: Observed price per extraction call (the PA machine's ledger, Sept 2026),
#: used to price a plan before this source has a ledger of its own.
DEFAULT_COST_PER_CALL_USD = 0.0018

# --- Reason codes -----------------------------------------------------------
# Per figure.
REJECT_NOT_NEW = "not_new_displacement"
REJECT_NOT_CONFLICT = "not_attributed_to_conflict"
REJECT_UNDATED = "undated"
REJECT_REVERSED = "period_reversed"
REJECT_LONGER_THAN_MONTH = "period_longer_than_month"
REJECT_CROSSES_MONTH = "period_crosses_month_boundary"
REJECT_OUTSIDE_MONTH = "period_outside_month"
# Per cell.
CELL_ADMITTED = "admitted_figure"
CELL_NO_DOCUMENTS = "no_documents"
CELL_FETCH_FAILED = "fetch_failed"
CELL_BUDGET_CAPPED = "budget_capped"
CELL_NO_FIGURE_STATED = "no_figure_stated"
CELL_NO_ADMISSIBLE = "no_admissible_figure"
CELL_EXTRACTION_FAILED = "extraction_failed"
CELL_DISABLED = "extraction_disabled"
CELL_RAISED = "cell_raised"

_DDL = (
    # One extraction call per (document, model, prompt version, cell): the
    # cache AND the cost ledger, the same shape as haz_doc_extractions but its
    # own table, so its calls never count against the PA machine's cap.
    """
    CREATE TABLE IF NOT EXISTS conflict_doc_extractions (
        doc_id TEXT NOT NULL,
        iso3 TEXT NOT NULL,
        year INTEGER NOT NULL,
        month INTEGER NOT NULL CHECK (month BETWEEN 1 AND 12),
        hazard TEXT NOT NULL,
        model TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('ok','error')),
        figures_json TEXT NOT NULL,
        n_figures INTEGER NOT NULL DEFAULT 0,
        n_rejected INTEGER NOT NULL DEFAULT 0,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        cost_usd DOUBLE,
        doc_url TEXT,
        error TEXT,
        run_type TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (doc_id, model, prompt_version, iso3, hazard, year, month)
    )
    """,
    # Every ADMITTED figure for a cell after deduplication, ranked: rank 0 is
    # the figure the source offers for the month. Rewritten whole per cell.
    """
    CREATE TABLE IF NOT EXISTS conflict_displacement_candidates (
        iso3 TEXT NOT NULL,
        ym TEXT NOT NULL,
        source_ref TEXT NOT NULL,
        preference_rank INTEGER NOT NULL,
        value_type TEXT NOT NULL,
        value_persons DOUBLE NOT NULL,
        stated_value DOUBLE,
        stated_unit TEXT,
        conversion_factor DOUBLE,
        conversion_origin TEXT,
        quote TEXT NOT NULL,
        stated_by TEXT,
        attribution_source TEXT,
        authority_tier TEXT,
        area TEXT,
        figure_date TEXT,
        period_start TEXT,
        period_end TEXT,
        period_end_source TEXT,
        cumulative_or_new TEXT,
        cause TEXT,
        doc_id TEXT,
        doc_url TEXT,
        doc_title TEXT,
        doc_date TEXT,
        doc_date_original TEXT,
        doc_publisher TEXT,
        doc_primary_country TEXT,
        model TEXT,
        prompt_version TEXT,
        run_id TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (iso3, ym, source_ref)
    )
    """,
    # One row per (country, month) this source assessed, with the reason it
    # has or has no figure, like the hazard cell ledger.
    """
    CREATE TABLE IF NOT EXISTS conflict_displacement_cell_ledger (
        iso3 TEXT NOT NULL,
        ym TEXT NOT NULL,
        reason TEXT NOT NULL,
        best_value DOUBLE,
        best_source_ref TEXT,
        docs_selected INTEGER,
        docs_read INTEGER,
        docs_cached INTEGER,
        docs_failed INTEGER,
        calls_made INTEGER,
        cost_usd DOUBLE,
        figures_extracted INTEGER,
        figures_admitted INTEGER,
        rejected_counts_json TEXT,
        rejected_json TEXT,
        budget_capped BOOLEAN,
        window_closed BOOLEAN,
        detail_json TEXT,
        run_id TEXT,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (iso3, ym)
    )
    """,
)


def ensure_schema(con: "duckdb.DuckDBPyConnection") -> None:
    for ddl in _DDL:
        con.execute(ddl)


# ---------------------------------------------------------------------------
# The figure and the prompt
# ---------------------------------------------------------------------------


@dataclass
class ConflictFigure(ExtractedFigure):
    """An extracted figure plus the three fields the conflict prompt adds."""

    period_start: str = ""
    period_end: str = ""
    cause: str = ""

    def as_dict(self) -> dict[str, Any]:
        out = super().as_dict()
        out.update(
            period_start=self.period_start, period_end=self.period_end, cause=self.cause
        )
        return out


_EXTRAS = ("period_start", "period_end", "cause")

_PROMPT_TEMPLATE = """\
You are a transcription tool. You extract figures that a document states. \
You never estimate, infer, calculate, or use knowledge from outside the \
document.

TASK
Read the document below and find every explicitly stated figure for the \
number of PEOPLE NEWLY DISPLACED by conflict or violence in {country} \
during {period}.

WHAT COUNTS
Count a figure only if the document states it as people (or persons, \
individuals, households or families) who were displaced, fled, or were \
forced from their homes. Related but DIFFERENT quantities are not wanted: \
people killed or injured, people in need, people targeted or reached by \
assistance, refugees counted in another country, returns.

RULES (all mandatory)
1. Transcribe only. If a number is not written in the document, it does \
not exist. Never estimate or infer one.
2. Never convert units. If the document says households or families, \
report the number of households and set "unit" to "households".
3. Never sum, total, or combine figures. Report each stated figure \
separately with its own area.
4. Every figure must carry the exact sentence it came from, copied \
character for character from the document.
5. Say whether the figure counts people NEWLY displaced in a stated period \
("new"), or a running total since an earlier date or people currently \
displaced ("cumulative"). If the document does not make this clear, use \
"unstated".
6. Give the date or period the figure refers to exactly as the document \
states it, as YYYY-MM-DD. For a period give its first and last day. Leave \
a field empty if the document does not state it. Never guess a date.
7. Give the cause the document attributes the displacement to: "conflict" \
(armed conflict, fighting, clashes, attacks, military operations), \
"violence" (communal, criminal or gang violence, insecurity), "disaster", \
"other", or "unstated".
8. Report figures for {country} only. If the document states no qualifying \
figure, return an empty list. An empty list is a correct and expected answer.

OUTPUT
Return JSON only, no preamble, no explanation, no markdown fences:

{{"figures": [
  {{
    "value": <number, digits only, no separators>,
    "unit": "people" | "households",
    "quote": "<the exact sentence from the document stating this figure>",
    "stated_by": "<who the document attributes the figure to, verbatim; \
empty string if the document does not say>",
    "area": "<the area the figure covers, verbatim; empty string if not \
stated>",
    "date": "<the single date the figure refers to, YYYY-MM-DD, else empty>",
    "period_start": "<first day of the period the figure covers, \
YYYY-MM-DD, else empty>",
    "period_end": "<last day of the period the figure covers, YYYY-MM-DD, \
else empty>",
    "cumulative_or_new": "new" | "cumulative" | "unstated",
    "cause": "conflict" | "violence" | "disaster" | "other" | "unstated"
  }}
]}}

DOCUMENT
Title: {title}
Published: {published}
Source: {source}
URL: {url}

{body}
"""


def build_prompt(
    document: dict[str, Any],
    iso3: str,
    hazard: str,
    country: str | None = None,
    ym: str | None = None,
) -> str:
    """The conflict extraction prompt for one document. Pure function."""

    return _PROMPT_TEMPLATE.format(
        country=country or iso3,
        period=extract_mod._period_label(ym),
        title=document.get("title") or "(untitled)",
        published=document.get("date_created") or "(undated)",
        source=", ".join(document.get("sources") or []) or "(unattributed)",
        url=document.get("url") or "(no url)",
        body=document.get("body") or "",
    )


def _with_extras(figure: ExtractedFigure, raw: Any) -> ConflictFigure:
    raw = raw if isinstance(raw, dict) else {}
    base = {f.name: getattr(figure, f.name) for f in dataclasses.fields(ExtractedFigure)}
    return ConflictFigure(
        **base,
        period_start=str(raw.get("period_start") or "").strip(),
        period_end=str(raw.get("period_end") or "").strip(),
        cause=str(raw.get("cause") or "").strip().lower(),
    )


def parse_response(
    text: str, document: dict[str, Any], model: str
) -> tuple[list[ConflictFigure], list[dict[str, Any]]]:
    """Verify with the machine's parser, then carry the conflict fields across.

    ``extract.parse_response`` rejects every entry it drops WITH its index
    and keeps the rest in order, so the accepted figures line up with the
    raw entries whose index was not rejected.
    """

    figures, rejected = extract_mod.parse_response(text, document, model)
    if not figures:
        return [], rejected
    try:
        raw_list = list(extract_mod._loads(text).get("figures") or [])
    except Exception:  # noqa: BLE001 - the machine parser already accepted it
        raw_list = []
    rejected_idx = {r.get("index") for r in rejected if "index" in r}
    accepted_raw = [raw for i, raw in enumerate(raw_list) if i not in rejected_idx]
    if len(accepted_raw) != len(figures):
        accepted_raw = [{}] * len(figures)
    return [_with_extras(f, raw) for f, raw in zip(figures, accepted_raw)], rejected


# ---------------------------------------------------------------------------
# The extraction cache / ledger (conflict_doc_extractions)
# ---------------------------------------------------------------------------


def _year_month(ym: str) -> tuple[int, int]:
    year, month = ym.split("-")
    return int(year), int(month)


def load_cached(
    con: "duckdb.DuckDBPyConnection",
    doc_id: str,
    model: str,
    prompt_version: str,
    *,
    iso3: str,
    hazard: str,
    ym: str,
) -> dict[str, Any] | None:
    """A previous ok extraction of this document for this cell, or None."""

    ensure_schema(con)
    year, month = _year_month(ym)
    row = con.execute(
        """
        SELECT status, figures_json, n_rejected, cost_usd
        FROM conflict_doc_extractions
        WHERE doc_id = ? AND model = ? AND prompt_version = ?
          AND iso3 = ? AND hazard = ? AND year = ? AND month = ? AND status = 'ok'
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
    """Record one call. A failed re-read never overwrites a paid ok answer."""

    ensure_schema(con)
    year, month = _year_month(ym)
    doc_id = str(document.get("doc_id") or "")
    if status != "ok":
        existing = con.execute(
            """
            SELECT 1 FROM conflict_doc_extractions
            WHERE doc_id = ? AND model = ? AND prompt_version = ?
              AND iso3 = ? AND hazard = ? AND year = ? AND month = ? AND status = 'ok'
            """,
            [doc_id, model, prompt_version, iso3.upper(), hazard, year, month],
        ).fetchone()
        if existing:
            return
    payload = {
        "figures": [f.as_dict() for f in figures],
        "rejected": rejected,
        "extracted_at": utcnow_iso(),
    }
    con.execute(
        """
        INSERT OR REPLACE INTO conflict_doc_extractions
            (doc_id, iso3, year, month, hazard, model, prompt_version, status,
             figures_json, n_figures, n_rejected, prompt_tokens,
             completion_tokens, cost_usd, doc_url, error, run_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            doc_id, iso3.upper(), year, month, hazard, model, prompt_version, status,
            json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str),
            len(figures), len(rejected),
            int((usage or {}).get("prompt_tokens") or 0),
            int((usage or {}).get("completion_tokens") or 0),
            float(cost_usd), str(document.get("url") or ""), error, run_type,
        ],
    )


def figures_from_cache(
    cached: dict[str, Any], model: str, document: dict[str, Any] | None = None
) -> list[ConflictFigure]:
    """The machine's replay (document fields re-derived), plus the extras."""

    base = extract_mod._figures_from_cache(cached, model, document)
    entries = list(cached.get("payload", {}).get("figures") or [])
    if len(entries) != len(base):
        entries = [{}] * len(base)
    return [_with_extras(f, entry) for f, entry in zip(base, entries)]


def _enabled(rulebook: Rulebook) -> bool:
    if str(os.getenv("SIBYL_BACKTEST_MODE", "")).strip().lower() in {"1", "true", "yes"}:
        return False
    return bool(rulebook.get("conflict_displacement.extraction.enabled", False))


def _prompt_version(rulebook: Rulebook) -> str:
    return str(rulebook.get("conflict_displacement.extraction.prompt_version"))


def profile(run_id: str | None = None) -> ExtractionProfile:
    """The :class:`ExtractionProfile` that makes the machine's loop this source."""

    return ExtractionProfile(
        enabled=_enabled,
        prompt_version=_prompt_version,
        build_prompt=build_prompt,
        parse=parse_response,
        load_cached=load_cached,
        figures_from_cache=figures_from_cache,
        write=write_extraction,
        log_kwargs={
            "call_type": CALL_TYPE,
            "metric": "PA",
            **({"run_id": run_id} if run_id else {}),
        },
    )


# ---------------------------------------------------------------------------
# Budget: its own monthly cap, never the PA machine's
# ---------------------------------------------------------------------------


def calls_this_calendar_month(
    con: "duckdb.DuckDBPyConnection", today: dt.date | None = None
) -> int:
    """Billed calls in ``conflict_doc_extractions`` this calendar month.

    The PA machine's billing rule: an ok row, or an error row that billed
    tokens. A call that never reached the provider cost nothing.
    """

    ensure_schema(con)
    ref = today or dt.date.today()
    row = con.execute(
        """
        SELECT COUNT(*) FROM conflict_doc_extractions
        WHERE CAST(strftime(created_at, '%Y-%m') AS VARCHAR) = ?
          AND (status = 'ok'
               OR COALESCE(prompt_tokens, 0) + COALESCE(completion_tokens, 0) > 0)
        """,
        [f"{ref.year:04d}-{ref.month:02d}"],
    ).fetchone()
    return int(row[0] or 0)


def monthly_cap(rulebook: Rulebook) -> int:
    """The rulebook cap, or the one-dispatch override when it is set."""

    configured = int(rulebook.get("conflict_displacement.extraction.max_calls_per_month"))
    raw = (os.getenv(CAP_OVERRIDE_ENV) or "").strip()
    if not raw:
        return configured
    try:
        wanted = int(raw)
    except ValueError:
        LOG.warning("[conflict_extract] %s=%r is not an integer; keeping %d",
                    CAP_OVERRIDE_ENV, raw, configured)
        return configured
    if wanted < 0:
        return configured
    LOG.warning(
        "[conflict_extract] monthly cap set for THIS RUN ONLY: %d -> %d "
        "(rulebook default unchanged)", configured, wanted,
    )
    return wanted


def load_budget(
    con: "duckdb.DuckDBPyConnection", rulebook: Rulebook, today: dt.date | None = None
) -> ExtractionBudget:
    """The machine's budget object, seeded from THIS source's ledger only."""

    budget = ExtractionBudget(
        max_calls_per_month=monthly_cap(rulebook),
        used_this_month=calls_this_calendar_month(con, today),
        run_type="live",
    )
    LOG.info(
        "[conflict_extract] budget: %d of %d calls already made this calendar month",
        budget.used_this_month, budget.max_calls_per_month,
    )
    return budget


def observed_cost_per_call(con: "duckdb.DuckDBPyConnection") -> float:
    """Mean billed cost per call over this source's ledger, else the default."""

    try:
        ensure_schema(con)
        row = con.execute(
            "SELECT AVG(cost_usd) FROM conflict_doc_extractions "
            "WHERE status = 'ok' AND cost_usd > 0"
        ).fetchone()
        if row and row[0]:
            return float(row[0])
    except Exception:  # noqa: BLE001
        pass
    return DEFAULT_COST_PER_CALL_USD


# ---------------------------------------------------------------------------
# Document selection
# ---------------------------------------------------------------------------


PostFn = Callable[[str, dict, dict, float], dict]


def _payload(iso3: str, ym: str, rulebook: Rulebook) -> dict[str, Any]:
    start, end = month_bounds(ym)
    pad = int(rulebook.get("conflict_displacement.documents.publication_pad_days"))
    window_end = end + dt.timedelta(days=pad)
    pool = int(rulebook.get("reliefweb.documents.candidate_pool_size"))
    return {
        "filter": {
            "operator": "AND",
            "conditions": [
                {"field": "primary_country.iso3", "value": iso3.upper()},
                {
                    "field": "date.created",
                    "value": {
                        "from": f"{start.isoformat()}T00:00:00+00:00",
                        "to": f"{window_end.isoformat()}T23:59:59+00:00",
                    },
                },
                {"field": "format.name",
                 "value": list(rulebook.get("conflict_displacement.documents.formats"))},
                {"field": "source.type.name",
                 "value": list(rulebook.get("conflict_displacement.documents.source_types"))},
            ],
        },
        "query": {
            "value": str(rulebook.get("conflict_displacement.documents.query")),
            "fields": ["title", "body"],
        },
        "fields": {"include": list(docs_mod._FIELDS)},
        "sort": ["date.created:desc"],
        "limit": pool,
    }


def fetch_documents(
    con: "duckdb.DuckDBPyConnection",
    iso3: str,
    ym: str,
    rulebook: Rulebook,
    *,
    post: PostFn | None = None,
) -> dict[str, Any]:
    """Fetch, rank and cache this cell's documents. Never raises.

    Ranked by the machine (``reliefweb_docs.select_documents``), kept only
    when the document's primary country is the cell's, capped at
    ``conflict_displacement.documents.max_docs_per_cell``, and cached in
    ``haz_raw_reliefweb_docs`` under hazard ``ACE``.
    """

    from resolver.hazard_resolution.reliefweb_sweep import appname, default_post

    iso3 = iso3.upper()
    api_url = str(rulebook.get("reliefweb.api_base_url")).rstrip("/") + "/reports"
    timeout = float(rulebook.get("reliefweb.documents.request_timeout_sec"))
    post = post or default_post
    try:
        data = post(api_url, _payload(iso3, ym, rulebook), {"appname": appname()}, timeout)
    except Exception as exc:  # noqa: BLE001 - requests errors, JSON errors
        LOG.warning("[conflict_extract] fetch failed for %s %s: %s", iso3, ym, exc)
        return {"ok": False, "error": str(exc)[:500]}
    items = list((data or {}).get("data") or [])
    ranked, _over = docs_mod.select_documents(items, iso3, ym, HAZARD, rulebook)
    about_cell = [d for d in ranked if (d.get("primary_country_iso3") or iso3) == iso3]
    cap = int(rulebook.get("conflict_displacement.documents.max_docs_per_cell"))
    selected = about_cell[:cap]
    stored = store_raw_records(
        con,
        DOC_SOURCE,
        [
            RawRecord(
                record_id=f"rw-{doc['doc_id']}", payload=doc, iso3=iso3, ym=ym,
                hazard=HAZARD, source_url=doc.get("url") or None,
            )
            for doc in selected
        ],
    )
    return {
        "ok": True,
        "returned": len(items),
        "total_available": int((data or {}).get("totalCount") or len(items)),
        "not_about_cell": len(ranked) - len(about_cell),
        "selected": len(selected),
        "inserted": stored.get("inserted", 0),
    }


def cell_documents(
    con: "duckdb.DuckDBPyConnection", iso3: str, ym: str, rulebook: Rulebook
) -> list[dict[str, Any]]:
    """The cached documents for the cell, best first, capped."""

    docs = docs_mod.documents_for_country_month(con, iso3.upper(), ym, HAZARD)
    cap = int(rulebook.get("conflict_displacement.documents.max_docs_per_cell"))
    return docs[:cap]


# ---------------------------------------------------------------------------
# Admissibility (pure)
# ---------------------------------------------------------------------------


def figure_period(
    figure: ConflictFigure,
) -> tuple[dt.date | None, dt.date | None, str]:
    """``(start, end, end_source)`` of the period a figure is about.

    The stated period first; a single stated date is a one-day period; a
    period with a start and no end ends on the day the document was
    published ("since 3 March, 12,000 people have fled"). No stated date at
    all gives ``(None, None, "")``: the figure is undated.
    """

    single = parse_date(figure.date)
    start = parse_date(figure.period_start) or (single if not figure.period_end else None)
    end = parse_date(figure.period_end) or (single if not figure.period_start else None)
    end_source = "stated"
    if start is not None and end is None:
        end = parse_date(figure.doc_date_original) or parse_date(figure.doc_date)
        end_source = "document_date"
    if end is not None and start is None:
        start = end
    if start is None and end is None:
        return None, None, ""
    return start, end, end_source


def period_reason(start: dt.date | None, end: dt.date | None, ym: str) -> str | None:
    """None when the period lies inside ``ym``; otherwise the rejection reason."""

    if start is None or end is None:
        return REJECT_UNDATED
    if end < start:
        return REJECT_REVERSED
    m_start, m_end = month_bounds(ym)
    days_in_month = calendar.monthrange(m_start.year, m_start.month)[1]
    if (end - start).days + 1 > days_in_month:
        return REJECT_LONGER_THAN_MONTH
    if end < m_start or start > m_end:
        return REJECT_OUTSIDE_MONTH
    if start < m_start or end > m_end:
        return REJECT_CROSSES_MONTH
    return None


def _rejection(figure: ExtractedFigure, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "reason": reason,
        "doc_id": figure.doc_id,
        "value": figure.value,
        "unit": figure.unit,
        "quote": figure.quote,
        **extra,
    }


def admit_figures(
    figures: Iterable[ExtractedFigure],
    iso3: str,
    ym: str,
    rulebook: Rulebook,
) -> tuple[list[tuple[ConflictFigure, float, dict[str, Any]]], list[dict[str, Any]]]:
    """Apply the admissibility rules. Returns (admitted, rejected).

    ``admitted`` is ``(figure, persons, info)`` in preference order (rank 0
    first), after households are converted and restatements collapse.
    Every rejection carries its reason.
    """

    iso3 = iso3.upper()
    causes = {str(c).strip().lower()
              for c in rulebook.get("conflict_displacement.admissible_causes")}
    rejected: list[dict[str, Any]] = []
    dated: list[tuple[ConflictFigure, dict[str, Any]]] = []
    for raw in figures:
        figure = raw if isinstance(raw, ConflictFigure) else _with_extras(raw, {})
        if str(figure.cumulative_or_new).lower() != "new":
            rejected.append(_rejection(figure, REJECT_NOT_NEW,
                                       cumulative_or_new=figure.cumulative_or_new))
            continue
        if str(figure.cause).lower() not in causes:
            rejected.append(_rejection(figure, REJECT_NOT_CONFLICT, cause=figure.cause))
            continue
        start, end, end_source = figure_period(figure)
        reason = period_reason(start, end, ym)
        if reason is not None:
            rejected.append(_rejection(
                figure, reason,
                period_start=start.isoformat() if start else "",
                period_end=end.isoformat() if end else "",
                cell_ym=ym,
            ))
            continue
        # The date the machine's attribution test reads is the period's end,
        # inside the month by now, so only its country test can still bite.
        dated.append((
            dataclasses.replace(figure, date=end.isoformat()),
            {"period_start": start.isoformat(), "period_end": end.isoformat(),
             "period_end_source": end_source},
        ))
    period_info = {id(f): info for f, info in dated}
    pairs = [(f, float(f.value)) for f, _ in dated]
    kept_pairs, attribution_rejected = figures_mod.apply_attribution(pairs, iso3, ym, rulebook)
    rejected.extend(attribution_rejected)

    converted: list[tuple[ConflictFigure, float]] = []
    infos: dict[int, dict[str, Any]] = {}
    for figure, _value in kept_pairs:
        persons, conversion = figures_mod.convert_units(figure, iso3, rulebook)
        factor, origin = figures_mod.conversion_for_unit(figure.unit, iso3, rulebook)
        info = dict(period_info.get(id(figure), {}))
        info.update(conversion_factor=factor, conversion_origin=origin,
                    conversion=conversion)
        infos[id(figure)] = info
        converted.append((figure, persons))
    kept, dropped = figures_mod.deduplicate(converted, rulebook)
    rejected.extend(dropped)
    return [(f, v, infos.get(id(f), {})) for f, v in kept], rejected


# ---------------------------------------------------------------------------
# Writers and readers
# ---------------------------------------------------------------------------


def write_candidates(
    con: "duckdb.DuckDBPyConnection",
    iso3: str,
    ym: str,
    admitted: list[tuple[ConflictFigure, float, dict[str, Any]]],
    *,
    rulebook: Rulebook,
    run_id: str | None,
) -> int:
    ensure_schema(con)
    iso3 = iso3.upper()
    con.execute(
        "DELETE FROM conflict_displacement_candidates WHERE iso3 = ? AND ym = ?",
        [iso3, ym],
    )
    prompt_version = _prompt_version(rulebook)
    for rank, (figure, persons, info) in enumerate(admitted):
        stated_by, attribution = figures_mod._attribute(figure)
        _tier_rank, tier = figures_mod.authority_tier(figure.stated_by, rulebook)
        con.execute(
            """
            INSERT OR REPLACE INTO conflict_displacement_candidates
                (iso3, ym, source_ref, preference_rank, value_type, value_persons,
                 stated_value, stated_unit, conversion_factor, conversion_origin,
                 quote, stated_by, attribution_source, authority_tier, area,
                 figure_date, period_start, period_end, period_end_source,
                 cumulative_or_new, cause, doc_id, doc_url, doc_title, doc_date,
                 doc_date_original, doc_publisher, doc_primary_country, model,
                 prompt_version, run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                iso3, ym, figures_mod._source_ref(figure), rank, VALUE_TYPE,
                float(persons), float(figure.value), figure.unit,
                info.get("conversion_factor"), info.get("conversion_origin"),
                figure.quote, stated_by, attribution, tier, figure.area,
                figure.date, info.get("period_start"), info.get("period_end"),
                info.get("period_end_source"), figure.cumulative_or_new, figure.cause,
                figure.doc_id, figure.doc_url, figure.doc_title, figure.doc_date,
                figure.doc_date_original, figure.doc_publisher,
                figure.doc_primary_country, figure.model, prompt_version, run_id,
            ],
        )
    return len(admitted)


def write_ledger(con: "duckdb.DuckDBPyConnection", row: dict[str, Any]) -> None:
    ensure_schema(con)
    cols = [
        "iso3", "ym", "reason", "best_value", "best_source_ref", "docs_selected",
        "docs_read", "docs_cached", "docs_failed", "calls_made", "cost_usd",
        "figures_extracted", "figures_admitted", "rejected_counts_json",
        "rejected_json", "budget_capped", "window_closed", "detail_json", "run_id",
    ]
    con.execute(
        f"INSERT OR REPLACE INTO conflict_displacement_cell_ledger ({', '.join(cols)}, updated_at) "
        f"VALUES ({', '.join('?' for _ in cols)}, CURRENT_TIMESTAMP)",
        [row.get(c) for c in cols],
    )


def _row_dict(cursor) -> list[dict[str, Any]]:
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, r)) for r in cursor.fetchall()]


def extracted_figures(
    con: "duckdb.DuckDBPyConnection", iso3: str | None = None
) -> dict[tuple[str, str], dict[str, Any]]:
    """Per (iso3, ym), the figure this source offers: the rank-0 candidate.

    The API Part B reads. Each value is the full candidate row (value in
    ``value_persons``, the verbatim ``quote``, ``doc_url``, the period and
    who stated it). A cell with no admitted figure is absent; why is in
    ``conflict_displacement_cell_ledger``.
    """

    try:
        ensure_schema(con)
        sql = ("SELECT * FROM conflict_displacement_candidates WHERE preference_rank = 0")
        params: list[Any] = []
        if iso3:
            sql += " AND iso3 = ?"
            params.append(iso3.upper())
        rows = _row_dict(con.execute(sql + " ORDER BY iso3, ym", params))
    except Exception as exc:  # noqa: BLE001 - a reader must not raise
        LOG.warning("[conflict_extract] candidates unreadable: %s", exc)
        return {}
    return {(r["iso3"], r["ym"]): r for r in rows}


def extracted_figure(
    con: "duckdb.DuckDBPyConnection", iso3: str, ym: str
) -> dict[str, Any] | None:
    """The rank-0 candidate for one cell, or None."""

    return extracted_figures(con, iso3).get((iso3.upper(), ym))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def window_closed(ym: str, rulebook: Rulebook, today: dt.date) -> bool:
    _start, end = month_bounds(ym)
    pad = int(rulebook.get("conflict_displacement.documents.publication_pad_days"))
    return today > end + dt.timedelta(days=pad)


def run_cell(
    con: "duckdb.DuckDBPyConnection",
    iso3: str,
    ym: str,
    rulebook: Rulebook,
    budget: ExtractionBudget,
    *,
    post: PostFn | None = None,
    call: Any = None,
    model_ref: str | None = None,
    country_name: str | None = None,
    today: dt.date | None = None,
    run_id: str | None = None,
    fetch: bool = True,
) -> dict[str, Any]:
    """Select, extract, admit and record one (country, month). Never raises."""

    iso3 = iso3.upper()
    today = today or dt.date.today()
    row: dict[str, Any] = {
        "iso3": iso3, "ym": ym, "run_id": run_id, "budget_capped": False,
        "window_closed": window_closed(ym, rulebook, today),
    }
    detail: dict[str, Any] = {}
    try:
        if not _enabled(rulebook):
            row["reason"] = CELL_DISABLED
            write_ledger(con, row)
            return row
        if fetch:
            fetched = fetch_documents(con, iso3, ym, rulebook, post=post)
            detail["fetch"] = fetched
        documents = cell_documents(con, iso3, ym, rulebook)
        row["docs_selected"] = len(documents)
        if not documents:
            row["reason"] = (
                CELL_FETCH_FAILED if fetch and not detail["fetch"].get("ok") else CELL_NO_DOCUMENTS
            )
            row["detail_json"] = json.dumps(detail, default=str)
            write_ledger(con, row)
            return row
        result = extract_mod.extract_for_cell(
            con, iso3=iso3, ym=ym, hazard=HAZARD, rulebook=rulebook, budget=budget,
            documents=documents, call=call, model_ref=model_ref,
            country_name=country_name, profile=profile(run_id),
        )
        admitted, rejected = admit_figures(result.figures, iso3, ym, rulebook)
        all_rejected = [*result.rejected, *rejected]
        counts: dict[str, int] = {}
        for r in all_rejected:
            counts[str(r.get("reason"))] = counts.get(str(r.get("reason")), 0) + 1
        write_candidates(con, iso3, ym, admitted, rulebook=rulebook, run_id=run_id)
        row.update(
            docs_read=result.docs_read, docs_cached=result.docs_cached,
            docs_failed=result.docs_failed, calls_made=result.calls_made,
            cost_usd=round(result.cost_usd, 6),
            figures_extracted=len(result.figures), figures_admitted=len(admitted),
            rejected_counts_json=json.dumps(counts, sort_keys=True),
            rejected_json=json.dumps(all_rejected[:50], default=str),
            budget_capped=bool(result.budget_capped),
        )
        if admitted:
            best_fig, best_value, _info = admitted[0]
            row["reason"] = CELL_ADMITTED
            row["best_value"] = float(best_value)
            row["best_source_ref"] = figures_mod._source_ref(best_fig)
        elif result.skipped_reason:
            row["reason"] = CELL_DISABLED if "disabled" in result.skipped_reason else CELL_EXTRACTION_FAILED
            detail["skipped_reason"] = result.skipped_reason
        elif result.budget_capped and result.docs_read + result.docs_cached < len(documents):
            row["reason"] = CELL_BUDGET_CAPPED
        elif result.docs_read + result.docs_cached == 0 and result.docs_failed:
            row["reason"] = CELL_EXTRACTION_FAILED
        elif not result.figures:
            row["reason"] = CELL_NO_FIGURE_STATED
        else:
            row["reason"] = CELL_NO_ADMISSIBLE
    except Exception as exc:  # noqa: BLE001 - one bad cell never ends the run
        LOG.exception("[conflict_extract] %s %s raised", iso3, ym)
        row["reason"] = CELL_RAISED
        detail["error"] = f"{type(exc).__name__}: {exc}"[:500]
    row["detail_json"] = json.dumps(detail, default=str)
    try:
        write_ledger(con, row)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("[conflict_extract] ledger not written for %s %s: %s", iso3, ym, exc)
    return row


# ---------------------------------------------------------------------------
# Country and month planning
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
COUNTRY_LIST = REPO_ROOT / "horizon_scanner" / "hs_country_list.txt"


def listed_countries(path: Path = COUNTRY_LIST) -> list[str]:
    """ISO3s of every country on the Horizon Scanner list (ACE is assessed
    for every listed country)."""

    from resolver.ingestion.utils.iso_normalize import to_iso3

    out: list[str] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            code = name.upper() if len(name) == 3 and name.isalpha() else to_iso3(name)
            if code and code not in out:
                out.append(code)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("[conflict_extract] country list unreadable: %s", exc)
    return out


def ace_question_countries(con: "duckdb.DuckDBPyConnection") -> list[str]:
    """ISO3s with any ACE question in the database (test runs excluded)."""

    try:
        rows = con.execute(
            "SELECT DISTINCT UPPER(iso3) FROM questions WHERE UPPER(hazard_code) = 'ACE' "
            "AND COALESCE(is_test, FALSE) = FALSE ORDER BY 1"
        ).fetchall()
    except Exception:  # noqa: BLE001 - no questions table: a fresh DB
        try:
            rows = con.execute(
                "SELECT DISTINCT UPPER(iso3) FROM questions WHERE UPPER(hazard_code) = 'ACE' ORDER BY 1"
            ).fetchall()
        except Exception:  # noqa: BLE001
            return []
    return [r[0] for r in rows if r[0]]


def plan_countries(con: "duckdb.DuckDBPyConnection", *, all_listed: bool = False) -> list[str]:
    """ACE countries on the list: those with an ACE question, else every
    listed country (or every listed country when ``all_listed``)."""

    listed = listed_countries()
    if all_listed:
        return listed
    asked = set(ace_question_countries(con))
    chosen = [c for c in listed if c in asked]
    return chosen or listed


def complete_months(today: dt.date, n: int) -> list[str]:
    """The ``n`` complete calendar months before ``today``, newest first."""

    out: list[str] = []
    year, month = today.year, today.month
    for _ in range(n):
        month -= 1
        if month == 0:
            month, year = 12, year - 1
        out.append(f"{year:04d}-{month:02d}")
    return out


def plan(
    con: "duckdb.DuckDBPyConnection",
    countries: list[str],
    months: list[str],
    rulebook: Rulebook,
    budget: ExtractionBudget,
) -> dict[str, Any]:
    """What a run will ask for, and what it may cost, before it runs."""

    cap = int(rulebook.get("conflict_displacement.documents.max_docs_per_cell"))
    cells = len(countries) * len(months)
    try:
        ensure_schema(con)
        cached = int(con.execute(
            "SELECT COUNT(*) FROM conflict_doc_extractions WHERE status = 'ok' "
            "AND prompt_version = ?", [_prompt_version(rulebook)],
        ).fetchone()[0])
    except Exception:  # noqa: BLE001
        cached = 0
    per_call = observed_cost_per_call(con)
    upper = cells * cap
    return {
        "countries": len(countries),
        "months": months,
        "cells": cells,
        "max_docs_per_cell": cap,
        "max_calls_upper_bound": upper,
        "cached_extractions_held": cached,
        "cost_per_call_usd": round(per_call, 5),
        "max_cost_upper_bound_usd": round(upper * per_call, 2),
        "monthly_cap": budget.max_calls_per_month,
        "used_this_month": budget.used_this_month,
        "calls_allowed_this_run": budget.remaining,
        "cost_allowed_this_run_usd": round(budget.remaining * per_call, 2),
    }


def run(
    con: "duckdb.DuckDBPyConnection",
    countries: list[str],
    months: list[str],
    rulebook: Rulebook,
    *,
    post: PostFn | None = None,
    call: Any = None,
    model_ref: str | None = None,
    today: dt.date | None = None,
    max_minutes: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Every (country, month), newest month first. Returns the summary."""

    ensure_schema(con)
    today = today or dt.date.today()
    run_id = run_id or f"conflict_extract_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S}"
    budget = load_budget(con, rulebook, today)
    summary: dict[str, Any] = {"run_id": run_id, "plan": plan(con, countries, months, rulebook, budget)}
    deadline = None if not max_minutes else clock() + float(max_minutes) * 60.0
    names = _country_names()
    reasons: dict[str, int] = {}
    not_reached: list[str] = []
    calls = 0
    cost = 0.0
    for ym in sorted(months, reverse=True):
        for iso3 in countries:
            if deadline is not None and clock() >= deadline:
                not_reached.append(f"{iso3}/{ym}")
                continue
            row = run_cell(
                con, iso3, ym, rulebook, budget, post=post, call=call,
                model_ref=model_ref, country_name=names.get(iso3), today=today,
                run_id=run_id,
            )
            reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
            calls += int(row.get("calls_made") or 0)
            cost += float(row.get("cost_usd") or 0.0)
    summary.update(
        cells_by_reason=reasons,
        cells_not_reached_for_time=len(not_reached),
        not_reached=not_reached[:200],
        calls_made=calls,
        cost_usd=round(cost, 4),
        budget=budget.as_provenance(),
        capped_cells=len(budget.capped_cells),
    )
    return summary


def _country_names() -> dict[str, str]:
    from resolver.hazard_resolution.impact import _load_country_names

    try:
        return _load_country_names()
    except Exception:  # noqa: BLE001
        return {}


def _print_plan(info: dict[str, Any]) -> None:
    print(
        "[conflict_extract] plan: {countries} countries x {n} months = {cells} cells; "
        "at most {max_calls_upper_bound} calls ({max_docs_per_cell} documents a cell), "
        "<= USD {max_cost_upper_bound_usd} at USD {cost_per_call_usd} a call; "
        "monthly cap {monthly_cap}, {used_this_month} already used, so this run may make "
        "{calls_allowed_this_run} calls (<= USD {cost_allowed_this_run_usd}); "
        "{cached_extractions_held} documents already extracted are free".format(
            n=len(info["months"]), **info
        ),
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=None, help="DuckDB path or duckdb:/// URL "
                        "(default RESOLVER_DB_URL)")
    parser.add_argument("--months", nargs="*", default=None, help="YYYY-MM months")
    parser.add_argument("--trailing", type=int, default=3,
                        help="complete months before today when --months is not given")
    parser.add_argument("--countries", default="", help="ISO3s (default: ACE countries)")
    parser.add_argument("--all-listed", action="store_true",
                        help="every country on hs_country_list.txt, not only those with an ACE question")
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    from resolver.db.duckdb_io import close_db, get_db

    db = args.db or os.getenv("RESOLVER_DB_URL") or None
    if db and "://" not in db:
        db = f"duckdb:///{Path(db).resolve()}"
    summary: dict[str, Any] = {}
    exit_code = 0
    con = None
    try:
        rulebook = load_rulebook()
        con = get_db(db)
        ensure_schema(con)
        countries = [c.strip().upper() for c in args.countries.replace(",", " ").split() if c.strip()]
        countries = countries or plan_countries(con, all_listed=args.all_listed)
        today = dt.date.today()
        months = args.months or complete_months(today, args.trailing)
        budget = load_budget(con, rulebook, today)
        info = plan(con, countries, months, rulebook, budget)
        _print_plan(info)
        summary = {"plan": info, "countries": countries}
        if not args.plan_only:
            summary = run(con, countries, months, rulebook, today=today,
                          max_minutes=args.max_minutes)
            summary["countries"] = countries
            print(f"[conflict_extract] done: {json.dumps(summary['cells_by_reason'])}; "
                  f"{summary['calls_made']} calls, USD {summary['cost_usd']}", flush=True)
    except Exception as exc:  # noqa: BLE001
        LOG.exception("[conflict_extract] run failed")
        summary["error"] = f"{type(exc).__name__}: {exc}"[:500]
        exit_code = 1
    finally:
        close_db(con)
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
