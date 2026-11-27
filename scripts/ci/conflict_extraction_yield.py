# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Where the ReliefWeb conflict displacement figures are lost.

ACE/PA resolves on a ladder (``pythia.tools.base_rate_spd.resolve_conflict_month_detail``):
IDMC's settled monthly figure, then a figure extracted from ReliefWeb
(``resolver.ingestion.conflict_displacement_extract``, prompt ``conflict_v1``),
then the DTM stock increase as a lower bound. After the 24-month backfill
the ladder revived 4 of 149 ACE/PA questions. This report measures where
the extracted figures fall away before any rule is changed. It changes no
resolution, scoring or eligibility behaviour.

For every ACE country on ``horizon_scanner/hs_country_list.txt`` and every
month from ``--start`` (2024-10) to the latest complete month:

a. the funnel: cells assessed, cells with a document, documents selected,
   documents extracted, figures extracted, figures admitted;
b. rejections by reason, summed from ``rejected_counts_json``; the reason
   strings are the extraction module's own (``REJECT_*``, which are also
   what ``period_reason`` returns) and the machine's, never re-derived;
c. the period rejections split by the stored period: shorter than the
   month and inside another single month, crossing a month boundary,
   longer than the month, reversed, undated;
d. the ``not_new_displacement`` rejections split by the quote: an explicit
   "since <date>", another cumulative marker, or a plain stock;
e. cells where ``max_docs_per_cell`` bound, with how many more documents
   ReliefWeb said matched the query (stored in the ledger's fetch detail);
f. the publication lag of admitted figures, and the documents published
   31 to 90 days after a month ended that the cell never read;
g. per country, the months each rung resolves and the unresolved months
   by status, through ``scripts/ci/ace_pa_rung_table.py``'s readers.

Read-only: the connection is opened ``read_only=True`` and nothing here
creates, alters or writes a table. ``rejected_json`` keeps at most 50
rejections per cell, so (c) and (d) cover the stored rejections and say how
many that is of the counted ones. Always exits 0.

Usage:
    python -m scripts.ci.conflict_extraction_yield --db data/resolver.duckdb \\
        [--start 2024-10] [--end 2026-10] [--out-dir diagnostics/conflict_extraction_yield]
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_START = "2024-10"
#: How many verbatim quotes per reason the ranked list carries.
N_EXAMPLES = 3
QUOTE_MAX_CHARS = 200
#: Days after month end a later document must fall in to count for (f).
LATE_DOC_MIN_DAYS, LATE_DOC_MAX_DAYS = 31, 90

LEDGER = "conflict_displacement_cell_ledger"
EXTRACTIONS = "conflict_doc_extractions"
CANDIDATES = "conflict_displacement_candidates"
DOC_CACHE = "haz_raw_reliefweb_docs"

# Period split categories (section c).
SPLIT_UNDATED = "undated"
SPLIT_REVERSED = "reversed"
SPLIT_LONGER = "longer_than_month"
SPLIT_CROSSES = "crosses_month_boundary"
SPLIT_SUB_MONTH_OTHER = "sub_month_in_another_month"
SPLIT_SUB_MONTH_INSIDE = "sub_month_inside_cell_month"

# not_new split categories (section d).
NOT_NEW_SINCE = "since_date"
NOT_NEW_CUMULATIVE = "cumulative_marker"
NOT_NEW_STOCK = "plain_stock"

_MONTHS = ("jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|"
           "april|june|july|august|september|october|november|december")
_SINCE_RE = re.compile(
    rf"\bsince\s+(?:(?:early|mid|late|the\s+(?:start|beginning|end)\s+of)\s+)?"
    rf"(?:\d{{1,2}}\s+)?(?:{_MONTHS}|\d{{4}}|\d{{1,2}}[/-]\d{{1,2}})",
    re.IGNORECASE,
)
_CUMULATIVE_RE = re.compile(
    r"\b(?:cumulative(?:ly)?|so far|to date|in total|a total of|total of|overall|"
    r"since the (?:start|beginning|onset|outbreak|escalation)|since (?:then|last)|"
    r"as of|over the past|over the last|in the past|in the last)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def month_range(start: str, end: str) -> List[str]:
    y, m = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    out: List[str] = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def latest_complete_month(today: date) -> str:
    y, m = today.year, today.month - 1
    if m == 0:
        y, m = y - 1, 12
    return f"{y:04d}-{m:02d}"


def month_end(ym: str) -> date:
    y, m = int(ym[:4]), int(ym[5:7])
    return date(y, m, calendar.monthrange(y, m)[1])


def _date(text: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(text)[:10])
    except (TypeError, ValueError):
        return None


def _loads(text: Any, default: Any) -> Any:
    if text in (None, ""):
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def _table_exists(con, name: str) -> bool:
    try:
        return bool(con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [name]
        ).fetchone()[0])
    except Exception:  # noqa: BLE001
        return False


def _quote(text: Any) -> str:
    """A quote cut to ``QUOTE_MAX_CHARS`` and scrubbed: scrub first, then cut,
    so a credential cut in half cannot survive the pattern."""
    from pythia.secret_scrub import scrub_text  # noqa: PLC0415

    clean = " ".join(str(scrub_text(str(text or ""))).split())
    return clean if len(clean) <= QUOTE_MAX_CHARS else clean[: QUOTE_MAX_CHARS - 1] + "…"


def open_read_only(path: str):
    """The one way this report opens a database."""
    import duckdb  # noqa: PLC0415

    if path.startswith("duckdb:///"):
        path = path[len("duckdb:///"):]
    return duckdb.connect(path, read_only=True)


def _reason_codes() -> Dict[str, Any]:
    """The extraction module's own reason strings."""
    from resolver.ingestion import conflict_displacement_extract as cde  # noqa: PLC0415

    period = (cde.REJECT_UNDATED, cde.REJECT_REVERSED, cde.REJECT_LONGER_THAN_MONTH,
              cde.REJECT_OUTSIDE_MONTH, cde.REJECT_CROSSES_MONTH)
    conflict_rules = (cde.REJECT_NOT_NEW, cde.REJECT_FLAG_MISSING, cde.REJECT_FLAG_UNCLEAR,
                      cde.REJECT_NOT_CONFLICT, *period)
    return {"module": cde, "period": period, "conflict_rules": conflict_rules,
            "not_new": cde.REJECT_NOT_NEW}


def _rulebook_settings() -> Dict[str, Any]:
    from resolver.hazard_resolution.rulebook import load_rulebook  # noqa: PLC0415

    rb = load_rulebook()
    return {
        "max_docs_per_cell": int(rb.get("conflict_displacement.documents.max_docs_per_cell")),
        "publication_pad_days": int(rb.get("conflict_displacement.documents.publication_pad_days")),
        "prompt_version": str(rb.get("conflict_displacement.extraction.prompt_version")),
    }


# ---------------------------------------------------------------------------
# Readers (SELECT only)
# ---------------------------------------------------------------------------


def ledger_rows(con, countries: Iterable[str], months: List[str]) -> List[Dict[str, Any]]:
    if not _table_exists(con, LEDGER):
        return []
    cs = sorted({c.upper() for c in countries})
    if not cs or not months:
        return []
    cur = con.execute(
        f"SELECT iso3, ym, reason, docs_selected, docs_read, docs_cached, docs_failed, "
        f"calls_made, figures_extracted, figures_admitted, rejected_counts_json, "
        f"rejected_json, budget_capped, window_closed, detail_json FROM {LEDGER} "
        f"WHERE upper(iso3) IN ({','.join('?' for _ in cs)}) AND ym >= ? AND ym <= ? "
        f"ORDER BY iso3, ym",
        [*cs, months[0], months[-1]],
    )
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


def extraction_counts(con, countries: Iterable[str], months: List[str],
                      prompt_version: Optional[str]) -> Dict[Tuple[str, str], Dict[str, int]]:
    """Per (iso3, ym): documents with an ok extraction and with an error."""
    out: Dict[Tuple[str, str], Dict[str, int]] = {}
    if not _table_exists(con, EXTRACTIONS) or not months:
        return out
    cs = sorted({c.upper() for c in countries})
    if not cs:
        return out
    y0, m0 = int(months[0][:4]), int(months[0][5:7])
    y1, m1 = int(months[-1][:4]), int(months[-1][5:7])
    pv = " AND prompt_version = ?" if prompt_version else ""
    rows = con.execute(
        f"SELECT upper(iso3), year, month, status, COUNT(DISTINCT doc_id) FROM {EXTRACTIONS} "
        f"WHERE upper(iso3) IN ({','.join('?' for _ in cs)}) "
        f"AND year * 100 + month BETWEEN ? AND ?{pv} GROUP BY 1, 2, 3, 4",
        [*cs, y0 * 100 + m0, y1 * 100 + m1, *([prompt_version] if prompt_version else [])],
    ).fetchall()
    for iso, y, m, status, n in rows:
        cell = out.setdefault((iso, f"{int(y):04d}-{int(m):02d}"), {"ok": 0, "error": 0})
        cell["ok" if status == "ok" else "error"] += int(n)
    return out


def admitted_candidates(con, countries: Iterable[str], months: List[str]) -> List[Dict[str, Any]]:
    if not _table_exists(con, CANDIDATES) or not months:
        return []
    cs = sorted({c.upper() for c in countries})
    if not cs:
        return []
    cur = con.execute(
        f"SELECT iso3, ym, preference_rank, doc_id, doc_date, doc_date_original FROM {CANDIDATES} "
        f"WHERE upper(iso3) IN ({','.join('?' for _ in cs)}) AND ym >= ? AND ym <= ?",
        [*cs, months[0], months[-1]],
    )
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


def cached_ace_documents(con, countries: Iterable[str]) -> List[Tuple[str, str, str, Optional[date]]]:
    """``(iso3, ym, doc_id, date_created)`` for each ACE document in the cache.

    Read with SQL, not ``sources.load_raw_records``: that helper runs
    ``ensure_haz_schema`` first, and a reader writes nothing.
    """
    if not _table_exists(con, DOC_CACHE):
        return []
    cs = sorted({c.upper() for c in countries})
    if not cs:
        return []
    rows = con.execute(
        f"SELECT DISTINCT upper(iso3), ym, "
        f"json_extract_string(payload_json, '$.doc_id'), "
        f"json_extract_string(payload_json, '$.date_created') "
        f"FROM {DOC_CACHE} WHERE hazard = 'ACE' "
        f"AND upper(iso3) IN ({','.join('?' for _ in cs)})",
        cs,
    ).fetchall()
    return [(r[0], str(r[1] or ""), str(r[2] or ""), _date(r[3])) for r in rows]


# ---------------------------------------------------------------------------
# Classification (pure)
# ---------------------------------------------------------------------------


def classify_period(rejection: Dict[str, Any], cell_ym: str, period_reasons: Dict[str, str]) -> str:
    """The section (c) category of one stored period rejection.

    ``period_reasons`` maps the module's constants to their meaning so the
    stored reason decides undated and reversed; the stored period decides
    the rest, because ``period_outside_month`` alone does not say whether a
    figure is about one other month or spans several.
    """
    reason = str(rejection.get("reason") or "")
    if reason == period_reasons["undated"]:
        return SPLIT_UNDATED
    if reason == period_reasons["reversed"]:
        return SPLIT_REVERSED
    start = _date(rejection.get("period_start"))
    end = _date(rejection.get("period_end"))
    if start is None or end is None:
        return SPLIT_UNDATED
    if end < start:
        return SPLIT_REVERSED
    days = calendar.monthrange(start.year, start.month)[1]
    if (end - start).days + 1 > days:
        return SPLIT_LONGER
    if (start.year, start.month) != (end.year, end.month):
        return SPLIT_CROSSES
    ym = f"{start.year:04d}-{start.month:02d}"
    return SPLIT_SUB_MONTH_INSIDE if ym == cell_ym else SPLIT_SUB_MONTH_OTHER


def month_offset(rejection: Dict[str, Any], cell_ym: str) -> Optional[int]:
    """Months from the cell's month to the month a sub-month period lies in."""
    start = _date(rejection.get("period_start"))
    if start is None:
        return None
    return (start.year - int(cell_ym[:4])) * 12 + (start.month - int(cell_ym[5:7]))


def classify_not_new(quote: Any) -> str:
    text = str(quote or "")
    if _SINCE_RE.search(text):
        return NOT_NEW_SINCE
    if _CUMULATIVE_RE.search(text):
        return NOT_NEW_CUMULATIVE
    return NOT_NEW_STOCK


def _lag_label(days: int) -> str:
    if days <= 0:
        return "<=0 (inside the month)"
    if days <= 7:
        return "1-7"
    if days <= 14:
        return "8-14"
    if days <= 21:
        return "15-21"
    if days <= 30:
        return "22-30"
    return ">30"


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _funnel(rows: List[Dict[str, Any]], extractions: Dict[Tuple[str, str], Dict[str, int]]) -> Dict[str, Any]:
    f = {"cells_assessed": len(rows), "cells_with_documents": 0, "documents_selected": 0,
         "documents_extracted": 0, "documents_extraction_failed": 0,
         "figures_extracted": 0, "figures_admitted": 0, "cells_admitted": 0}
    reasons: Counter = Counter()
    for r in rows:
        sel = int(r.get("docs_selected") or 0)
        f["cells_with_documents"] += 1 if sel > 0 else 0
        f["documents_selected"] += sel
        ex = extractions.get((str(r["iso3"]).upper(), r["ym"]), {})
        f["documents_extracted"] += ex.get("ok", 0)
        f["documents_extraction_failed"] += ex.get("error", 0)
        f["figures_extracted"] += int(r.get("figures_extracted") or 0)
        f["figures_admitted"] += int(r.get("figures_admitted") or 0)
        reasons[str(r.get("reason"))] += 1
    f["cells_admitted"] = reasons.get("admitted_figure", 0)
    f["cell_reasons"] = dict(reasons.most_common())
    return f


def build_report(
    con,
    *,
    start: str = DEFAULT_START,
    end: Optional[str] = None,
    today: Optional[date] = None,
    countries: Optional[List[str]] = None,
    settings: Optional[Dict[str, Any]] = None,
    max_seconds: Optional[float] = None,
    include_rungs: bool = True,
) -> Dict[str, Any]:
    """The whole report as a dict. Never raises; a section that fails or runs
    out of time is a stub naming why."""

    today = today or date.today()
    end = end or latest_complete_month(today)
    months = month_range(start, end)
    deadline = time.monotonic() + max_seconds if max_seconds else None
    if countries is None:
        from resolver.ingestion.conflict_displacement_extract import listed_countries  # noqa: PLC0415

        countries = listed_countries()
    countries = [c.upper() for c in countries]
    settings = settings or _rulebook_settings()
    codes = _reason_codes()
    cde = codes["module"]
    period_reasons = {"undated": cde.REJECT_UNDATED, "reversed": cde.REJECT_REVERSED}
    report: Dict[str, Any] = {
        "window": [start, end], "today": today.isoformat(), "n_countries": len(countries),
        "settings": settings,
        "reason_codes": {"conflict_rules": list(codes["conflict_rules"]),
                         "period": list(codes["period"])},
        "sections_skipped": {},
    }

    def out_of_time(name: str) -> bool:
        if deadline is not None and time.monotonic() > deadline:
            report["sections_skipped"][name] = "time budget spent"
            return True
        return False

    try:
        rows = ledger_rows(con, countries, months)
    except Exception as exc:  # noqa: BLE001
        report["stub"] = True
        report["reason"] = f"ledger unreadable: {type(exc).__name__}: {exc}"
        return report
    if not rows:
        report["stub"] = True
        report["reason"] = ("no extraction rows"
                            + ("" if _table_exists(con, LEDGER) else f" ({LEDGER} absent)"))
        if include_rungs and not out_of_time("rungs"):
            report["rungs"] = _rungs_section(con, countries, months, today)
        return report

    by_country: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_country[str(r["iso3"]).upper()].append(r)

    # (a) funnel -------------------------------------------------------------
    try:
        # Every prompt version: a document read under conflict_v1 and kept under
        # conflict_v2 (REUSABLE_PRIOR_VERSIONS) was still read once.
        ex = extraction_counts(con, countries, months, None)
    except Exception as exc:  # noqa: BLE001
        ex = {}
        report["sections_skipped"]["documents_extracted"] = str(exc)
    report["funnel"] = _funnel(rows, ex)
    report["funnel_by_country"] = {
        c: _funnel(by_country[c], ex) if c in by_country else None for c in countries
    }

    # (b) rejections ---------------------------------------------------------
    total: Counter = Counter()
    per_country: Dict[str, Counter] = defaultdict(Counter)
    stored: List[Tuple[str, str, Dict[str, Any]]] = []
    for r in rows:
        iso = str(r["iso3"]).upper()
        for reason, n in (_loads(r.get("rejected_counts_json"), {}) or {}).items():
            total[str(reason)] += int(n or 0)
            per_country[iso][str(reason)] += int(n or 0)
        for rej in _loads(r.get("rejected_json"), []) or []:
            if isinstance(rej, dict):
                stored.append((iso, r["ym"], rej))
    stored_counts = Counter(str(rej.get("reason")) for _, _, rej in stored)
    report["rejections"] = {
        "counted": dict(total.most_common()),
        "stored": dict(stored_counts.most_common()),
        "counted_total": sum(total.values()),
        "stored_total": len(stored),
        "by_family": {
            "conflict_rule": sum(n for k, n in total.items() if k in codes["conflict_rules"]),
            "machine": sum(n for k, n in total.items() if k not in codes["conflict_rules"]),
        },
        "by_country": {c: dict(per_country[c].most_common()) for c in sorted(per_country)},
    }

    # (c) period split -------------------------------------------------------
    if not out_of_time("period_split"):
        split: Counter = Counter()
        by_reason: Dict[str, Counter] = defaultdict(Counter)
        offsets: Counter = Counter()
        examples: Dict[str, List[str]] = defaultdict(list)
        for _iso, ym, rej in stored:
            reason = str(rej.get("reason"))
            if reason not in codes["period"]:
                continue
            cat = classify_period(rej, str(rej.get("cell_ym") or ym), period_reasons)
            split[cat] += 1
            by_reason[reason][cat] += 1
            if cat == SPLIT_SUB_MONTH_OTHER:
                off = month_offset(rej, str(rej.get("cell_ym") or ym))
                offsets[f"{off:+d}" if off is not None else "?"] += 1
            if len(examples[cat]) < N_EXAMPLES and rej.get("quote"):
                examples[cat].append(_quote(rej.get("quote")))
        report["period_split"] = {
            "stored": sum(split.values()),
            "counted": sum(n for k, n in total.items() if k in codes["period"]),
            "split": dict(split.most_common()),
            "by_reason": {k: dict(v.most_common()) for k, v in by_reason.items()},
            "sub_month_other_month_offset": dict(sorted(offsets.items())),
            "examples": dict(examples),
        }

    # (d) not_new split ------------------------------------------------------
    if not out_of_time("not_new_split"):
        nn: Counter = Counter()
        flags: Counter = Counter()
        nn_examples: Dict[str, List[str]] = defaultdict(list)
        for _iso, _ym, rej in stored:
            if str(rej.get("reason")) != codes["not_new"]:
                continue
            cat = classify_not_new(rej.get("quote"))
            nn[cat] += 1
            flags[str(rej.get("cumulative_or_new") or "(blank)")] += 1
            if len(nn_examples[cat]) < N_EXAMPLES and rej.get("quote"):
                nn_examples[cat].append(_quote(rej.get("quote")))
        report["not_new_split"] = {
            "stored": sum(nn.values()),
            "counted": total.get(codes["not_new"], 0),
            "split": dict(nn.most_common()),
            "cumulative_or_new_as_stated": dict(flags.most_common()),
            "examples": dict(nn_examples),
        }

    # (e) document cap -------------------------------------------------------
    if not out_of_time("cap"):
        cap = int(settings.get("max_docs_per_cell") or 0)
        hit = [r for r in rows if cap and int(r.get("docs_selected") or 0) >= cap]
        more: List[int] = []
        not_about: List[int] = []
        unknown = 0
        for r in hit:
            fetch = (_loads(r.get("detail_json"), {}) or {}).get("fetch") or {}
            if "total_available" not in fetch:
                unknown += 1
                continue
            more.append(max(0, int(fetch.get("total_available") or 0) - int(fetch.get("selected") or 0)))
            not_about.append(int(fetch.get("not_about_cell") or 0))
        report["cap"] = {
            "max_docs_per_cell": cap,
            "cells_at_cap": len(hit),
            "cells_at_cap_admitted": sum(1 for r in hit if r.get("reason") == "admitted_figure"),
            "cells_with_match_count_stored": len(more),
            "cells_with_match_count_not_stored": unknown,
            "more_matched_total": sum(more),
            "more_matched_median": statistics.median(more) if more else None,
            "more_matched_max": max(more) if more else None,
            "not_about_cell_total": sum(not_about),
            "note": ("'more matched' is ReliefWeb's totalCount for the cell's query less the "
                     "documents selected. The query already filters on the cell's primary "
                     "country, so these are documents about this country, in the cell's "
                     "publication window, that the cap left unread."),
        }

    # (f) publication lag ----------------------------------------------------
    if not out_of_time("lag"):
        report["lag"] = _lag_section(con, countries, months, rows, settings)

    # (g) rungs --------------------------------------------------------------
    if include_rungs and not out_of_time("rungs"):
        report["rungs"] = _rungs_section(con, countries, months, today)

    # The ranked list for the PR: reasons that cost the most figures that
    # carried a verified quote. Parse failures carry no figure, so they are
    # left out ("admissible-looking" means a figure with a quote).
    ranked = []
    quoted = {str(rej.get("reason")) for _, _, rej in stored if rej.get("quote")}
    for reason, n in total.most_common():
        if reason not in quoted:
            continue
        ex_q: List[str] = []
        for _iso, _ym, rej in stored:
            if str(rej.get("reason")) == reason and rej.get("quote"):
                q = _quote(rej.get("quote"))
                if q not in ex_q:
                    ex_q.append(q)
            if len(ex_q) >= N_EXAMPLES:
                break
        ranked.append({"reason": reason, "counted": n, "examples": ex_q})
        if len(ranked) >= 5:
            break
    report["top_reasons"] = ranked
    return report


def _lag_section(con, countries, months, rows, settings) -> Dict[str, Any]:
    pad = int(settings.get("publication_pad_days") or 30)
    try:
        cands = admitted_candidates(con, countries, months)
    except Exception as exc:  # noqa: BLE001
        return {"stub": True, "reason": str(exc)}
    lags: List[int] = []
    rank0: List[int] = []
    for c in cands:
        d = _date(c.get("doc_date")) or _date(c.get("doc_date_original"))
        if d is None:
            continue
        lag = (d - month_end(str(c["ym"]))).days
        lags.append(lag)
        if int(c.get("preference_rank") or 0) == 0:
            rank0.append(lag)
    buckets = Counter(_lag_label(x) for x in lags)
    out: Dict[str, Any] = {
        "admitted_figures_dated": len(lags),
        "admitted_figures": len(cands),
        "buckets": {k: buckets.get(k, 0) for k in
                    ["<=0 (inside the month)", "1-7", "8-14", "15-21", "22-30", ">30"]},
        "median_days": statistics.median(lags) if lags else None,
        "p90_days": sorted(lags)[int(0.9 * (len(lags) - 1))] if lags else None,
        "rank0_median_days": statistics.median(rank0) if rank0 else None,
        "in_last_7_days_of_window": sum(1 for x in lags if pad - 6 <= x <= pad),
        "publication_pad_days": pad,
    }
    # Documents published 31-90 days after a month ended. The cell's own query
    # stopped at month end + pad, so it never saw them; what the cache holds
    # are documents fetched for the same country's LATER cells. A lower bound.
    try:
        docs = cached_ace_documents(con, countries)
    except Exception as exc:  # noqa: BLE001
        out["late_documents"] = {"stub": True, "reason": str(exc)}
        return out
    by_country: Dict[str, List[Tuple[str, str, Optional[date]]]] = defaultdict(list)
    for iso, ym, doc_id, created in docs:
        by_country[iso].append((ym, doc_id, created))
    cells_with = 0
    n_docs = 0
    for r in rows:
        iso, ym = str(r["iso3"]).upper(), r["ym"]
        me = month_end(ym)
        lo, hi = me + timedelta(days=LATE_DOC_MIN_DAYS), me + timedelta(days=LATE_DOC_MAX_DAYS)
        found = {doc_id for (oym, doc_id, created) in by_country.get(iso, ())
                 if oym != ym and created is not None and lo <= created <= hi}
        if found:
            cells_with += 1
            n_docs += len(found)
    out["late_documents"] = {
        "cells_assessed": len(rows),
        "cells_with_late_documents": cells_with,
        "late_documents": n_docs,
        "window_days": [LATE_DOC_MIN_DAYS, LATE_DOC_MAX_DAYS],
        "note": ("a lower bound: documents cached for the same country's later months and "
                 "published 31-90 days after this month ended. ReliefWeb was never asked for "
                 "documents past month end + publication_pad_days for this cell, so documents "
                 "no later cell selected are not in the database."),
    }
    return out


def _rungs_section(con, countries: List[str], months: List[str], today: date) -> Dict[str, Any]:
    """Per country, months resolved by each rung and the unresolved months by
    status. Reads the ladder through ``ace_pa_rung_table``'s readers."""
    try:
        from pythia.tools import base_rate_spd as brs  # noqa: PLC0415
        from scripts.ci import ace_pa_rung_table as apr  # noqa: PLC0415

        reported, held = brs.conflict_displacement_series(con)
        lower = brs.conflict_ladder_lower(con)
        next_window = apr._default_window(today)
        admission = {r["iso3"]: r for r in apr.rung_table(con, next_window)}
        try:
            revived = apr.revived_questions(con, today)
        except Exception as exc:  # noqa: BLE001 - no questions table in a bare DB
            revived = {"stub": True, "reason": str(exc)}
        rows: List[Dict[str, Any]] = []
        totals: Counter = Counter()
        for iso in countries:
            n: Counter = Counter()
            for ym in months:
                res = brs.resolve_conflict_month_detail(
                    reported.get(iso, {}), ym, today, held=held.get(iso, set()),
                    lower=lower.get(iso, {}).get(ym, ()))
                if res.status == brs.CONFLICT_STATUS_REPORTED:
                    n[res.rung or "reported"] += 1
                elif res.status == brs.CONFLICT_STATUS_QUIET:
                    n["idmc_zero"] += 1
                else:
                    n[res.status] += 1
            totals.update(n)
            adm = admission.get(iso) or {}
            rows.append({"iso3": iso, **dict(n),
                         "admission_rung": adm.get("admission_rung"),
                         "months_counted": adm.get("months_counted", 0)})
        return {
            "months": len(months),
            "columns": {
                "idmc": brs.CONFLICT_RUNG_IDMC, "extracted": brs.CONFLICT_RUNG_EXTRACTED,
                "dtm": brs.CONFLICT_RUNG_DTM,
                "unresolved": [brs.CONFLICT_STATUS_UNSETTLED, brs.CONFLICT_STATUS_HELD,
                               brs.CONFLICT_STATUS_IRREGULAR, brs.CONFLICT_STATUS_TRAILING],
            },
            "totals": dict(totals.most_common()),
            "rows": rows,
            "admission_window": next_window,
            "revived": revived,
        }
    except Exception as exc:  # noqa: BLE001
        return {"stub": True, "reason": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _table(header: List[str], rows: List[List[Any]]) -> List[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return out


def render_markdown(rep: Dict[str, Any]) -> str:
    start, end = rep.get("window", ["", ""])
    out = [f"# Conflict displacement extraction yield, {start} to {end}", "",
           f"Countries: {rep.get('n_countries', 0)} ACE countries on hs_country_list.txt. "
           f"Settings: {rep.get('settings', {})}.", ""]
    if rep.get("stub"):
        out += [f"**{rep.get('reason', 'no extraction rows')}.** Nothing below is a measurement of "
                "the ReliefWeb rung; no count is shown as zero.", ""]
        if rep.get("rungs"):
            out += _render_rungs(rep["rungs"])
        return "\n".join(out) + "\n"

    f = rep["funnel"]
    out += ["## a. Funnel", ""]
    out += _table(["cells assessed", "cells with a document", "documents selected",
                   "documents extracted", "extraction failed", "figures extracted",
                   "figures admitted", "cells admitted"],
                  [[f["cells_assessed"], f["cells_with_documents"], f["documents_selected"],
                    f["documents_extracted"], f["documents_extraction_failed"],
                    f["figures_extracted"], f["figures_admitted"], f["cells_admitted"]]])
    out += ["", f"Cell reasons: {f['cell_reasons']}", "",
            "Documents extracted can exceed documents selected: the ledger counts the cell's "
            "current selection, while the extraction cache keeps every document a cell was "
            "ever sent, including those an earlier fetch selected.", "", "Per country:", ""]
    prow = []
    for c, cf in rep["funnel_by_country"].items():
        if cf is None:
            prow.append([c, "not assessed", "", "", "", "", ""])
        else:
            prow.append([c, cf["cells_assessed"], cf["cells_with_documents"], cf["documents_selected"],
                         cf["documents_extracted"], cf["figures_extracted"], cf["figures_admitted"]])
    out += _table(["country", "cells", "with docs", "docs selected", "docs extracted",
                   "figures extracted", "figures admitted"], prow)

    rj = rep["rejections"]
    rule = set(rep["reason_codes"]["conflict_rules"])
    out += ["", "## b. Rejections by reason", "",
            f"{rj['counted_total']} rejections counted ({rj['by_family']['conflict_rule']} by the "
            f"conflict rules, {rj['by_family']['machine']} by the machine's parse, attribution and "
            f"dedup rules); {rj['stored_total']} stored in rejected_json (50 a cell at most).", ""]
    out += _table(["reason", "family", "counted", "stored"],
                  [[k, "conflict rule" if k in rule else "machine", v, rj["stored"].get(k, 0)]
                   for k, v in rj["counted"].items()])
    out += ["", "Per country (counted):", ""]
    out += _table(["country", "rejections"],
                  [[c, ", ".join(f"{k} {v}" for k, v in d.items())] for c, d in rj["by_country"].items()])

    ps = rep.get("period_split")
    if ps:
        out += ["", "## c. Period rejections by the stored period", "",
                f"{ps['stored']} stored of {ps['counted']} counted.", ""]
        out += _table(["category", "stored"], [[k, v] for k, v in ps["split"].items()])
        out += ["", f"By reason: {ps['by_reason']}",
                f"Sub-month periods in another month, by offset from the cell month: "
                f"{ps['sub_month_other_month_offset']}"]
    nn = rep.get("not_new_split")
    if nn:
        out += ["", "## d. not_new_displacement rejections by the quote", "",
                f"{nn['stored']} stored of {nn['counted']} counted.", ""]
        out += _table(["category", "stored"], [[k, v] for k, v in nn["split"].items()])
        out += ["", f"cumulative_or_new as the model stated it: {nn['cumulative_or_new_as_stated']}"]
    cp = rep.get("cap")
    if cp:
        out += ["", f"## e. Cells at the document cap ({cp['max_docs_per_cell']})", "",
                f"{cp['cells_at_cap']} cells at the cap ({cp['cells_at_cap_admitted']} admitted a figure). "
                f"Match count stored for {cp['cells_with_match_count_stored']}, not stored for "
                f"{cp['cells_with_match_count_not_stored']}. More documents matched: "
                f"{cp['more_matched_total']} in all, median {cp['more_matched_median']} a cell, "
                f"most {cp['more_matched_max']}.", "", cp["note"]]
    lg = rep.get("lag")
    if lg:
        out += ["", "## f. Publication lag of admitted figures", ""]
        if lg.get("stub"):
            out += [f"Not measured: {lg.get('reason')}"]
        else:
            out += [f"{lg['admitted_figures_dated']} of {lg['admitted_figures']} admitted figures carry a "
                    f"document date. Median {lg['median_days']} days after month end, 90th percentile "
                    f"{lg['p90_days']}, rank-0 median {lg['rank0_median_days']}; "
                    f"{lg['in_last_7_days_of_window']} in the last 7 days of the "
                    f"{lg['publication_pad_days']}-day window.", ""]
            out += _table(["days after month end", "figures"], [[k, v] for k, v in lg["buckets"].items()])
            ld = lg.get("late_documents") or {}
            if ld.get("stub"):
                out += ["", f"Late documents not measured: {ld.get('reason')}"]
            elif ld:
                out += ["", f"{ld['cells_with_late_documents']} of {ld['cells_assessed']} cells have "
                        f"{ld['late_documents']} cached documents published 31-90 days after the month "
                        "ended, which the cell never read.", "", ld["note"]]
    if rep.get("rungs"):
        out += [""] + _render_rungs(rep["rungs"])
    tr = rep.get("top_reasons") or []
    if tr:
        out += ["", "## Ranked: the reasons that cost the most quoted figures", ""]
        for i, t in enumerate(tr, 1):
            out.append(f"{i}. **{t['reason']}** ({t['counted']} figures)")
            for q in t["examples"]:
                out.append(f"   - \"{q}\"")
    if rep.get("sections_skipped"):
        out += ["", f"Sections skipped: {rep['sections_skipped']}"]
    return "\n".join(out) + "\n"


def _render_rungs(rg: Dict[str, Any]) -> List[str]:
    out = ["## g. Months by ladder rung, per country", ""]
    if rg.get("stub"):
        return out + [f"Not measured: {rg.get('reason')}"]
    cols = rg["columns"]
    keys = [cols["idmc"], "idmc_zero", cols["extracted"], cols["dtm"], *cols["unresolved"]]
    out += [f"{rg['months']} months a country. Totals: {rg['totals']}", ""]
    out += _table(["country", "IDMC", "IDMC zero", "extracted", "DTM", "unsettled", "held",
                   "irregular", "trailing", "admitted for " + str(rg.get("admission_window"))],
                  [[r["iso3"], *[r.get(k, 0) for k in keys], r.get("admission_rung") or "no"]
                   for r in rg["rows"]])
    rv = rg.get("revived") or {}
    if rv and not rv.get("stub"):
        out += ["", f"Revived questions: {rv.get('n_revived', 0)} of {rv.get('n_questions', 0)} "
                f"({rv.get('horizon_months_revived', 0)} horizon-months; by rung {rv.get('by_rung')})."]
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default="")
    ap.add_argument("--out-dir", default="diagnostics/conflict_extraction_yield")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    args = ap.parse_args(argv)
    try:
        con = open_read_only(args.db)
        try:
            rep = build_report(con, start=args.start, end=args.end or None,
                               max_seconds=args.max_seconds or None)
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        rep = {"stub": True, "reason": f"{type(exc).__name__}: {exc}", "window": [args.start, args.end]}
        print(f"::warning title=Conflict extraction yield::{rep['reason']}")
    text = render_markdown(rep)
    print(text)
    try:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "conflict_extraction_yield.md").write_text(text, encoding="utf-8")
        (out / "conflict_extraction_yield.json").write_text(
            json.dumps(rep, indent=2, default=str), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"::warning title=Conflict extraction yield::not written: {exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
