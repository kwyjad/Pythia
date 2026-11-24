# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Why flood and cyclone PA cells end NO_DATA, read from the database alone.

    python -m scripts.diagnostics.pa_no_data_breakdown --db PATH \\
        --hazard flood [--hazard cyclone] --window 2025-09:2026-08 --out-dir DIR

A NO_DATA cell is one the detector triggered, the ladder walked, and no rung
answered before the freeze. The acceptance report counts them; this says
why, per cell, under five headings:

1. **Detection.** What triggered the cell: GDACS (and the strongest alert
   colour among the events that touched it), IBTrACS, a ReliefWeb hit the
   ladder confirmed, or a legacy sweep row. A cell no detector would trigger
   today is one where NO_DATA may be the honest answer.
2. **Rungs.** Per rung: was it read, and what did it hold. IDMC IDU and IFRC
   GO are read from their raw caches, for this hazard and for any hazard;
   ReliefWeb from the documents cached for the cell, the documents the
   extractor read, and the figures it transcribed and rejected, by reason.
3. **The nearest miss.** Every rejected figure is replayed from the cache
   through the CURRENT rules (``quote_period_audit.replay_cell``: no network,
   no model call) once with all gates on and once per gate with that gate
   alone turned off. A figure the second replay admits was rejected for that
   reason alone. Rejections made while parsing (unknown unit, a quote not in
   the document, a value not in the quote) never reach the later gates, so
   they are counted under their own reason and marked as such.
4. **Concentration.** NO_DATA by country and by month.
5. **Freeze timing.** How long after month end the figures that DID resolve
   a cell were published, per rung, and which NO_DATA cells now hold an
   IDMC or IFRC record dated after the cell froze — the cells a later
   freeze would have answered.

It measures and changes nothing. The connection is read-only, and the DB
file's SHA-256 is compared before and after; a mismatch exits 2.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import datetime as dt
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

LOG = logging.getLogger(__name__)

HAZARD_CODES = {"flood": "FL", "cyclone": "TC", "fl": "FL", "tc": "TC"}

#: The gates a single switch turns off, keyed by the rejection reason each
#: one gives. The value is (lever name, dotted rulebook key, value that
#: turns the gate off).
GATE_LEVERS: dict[str, tuple[str, str, Any]] = {
    "outside_reporting_window": (
        "window", "reliefweb.documents.enforce_reporting_window", False),
    "document_primary_country_mismatch": (
        "primary_country", "reliefweb.documents.require_primary_country", False),
    "exceeds_gdacs_exposure_ceiling": (
        "ceiling", "sanity.ceiling_multiplier", 1e15),
    "quote_names_earlier_period": (
        "quote_period", "reliefweb.documents.enforce_quote_period", False),
    "document_about_earlier_event": (
        "earlier_event", "reliefweb.documents.reject_undated_in_dated_document", False),
    "cyclone_figure_in_flood_cell": (
        "one_hazard", "reliefweb.documents.one_hazard_per_figure", False),
    "household_with_stated_people": (
        "household", "reliefweb.household_conversion.prefer_stated_people", False),
}

#: Rejections made while the model's answer is parsed. They never reach
#: the later gates, so "rejected for this reason alone" cannot be tested;
#: and the verification ones must never be loosened (a quote not in the
#: document is the fabrication guard).
PARSE_REASONS = {
    "unknown_unit": "unit",
    "missing_quote": "verification",
    "quote_not_found_in_document": "verification",
    "value_not_found_in_quote": "verification",
    "value_not_a_number": "parse",
    "not_an_object": "parse",
    "unparseable_response": "parse",
}

#: Not a miss: the people reading of the same sentence was kept instead.
NOT_A_MISS = {"household_superseded_by_stated_people"}

_ALERT_ORDER = {"green": 1, "orange": 2, "red": 3}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loads(text: Any) -> dict[str, Any]:
    if isinstance(text, dict):
        return text
    try:
        loaded = json.loads(text) if text else {}
    except (TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def parse_window(text: str) -> tuple[str, str]:
    first, _, last = text.partition(":")
    for part in (first, last):
        dt.date(int(part[:4]), int(part[5:7]), 1)
    if (first or "") > (last or ""):
        raise ValueError(f"window {text!r} ends before it starts")
    return first, last


def month_end(ym: str) -> dt.date:
    year, month = (int(p) for p in ym.split("-"))
    return dt.date(year, month, calendar.monthrange(year, month)[1])


def _parse_date(value: Any) -> dt.date | None:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value).strip()
    for size in (10, 7):
        try:
            if size == 10:
                return dt.date.fromisoformat(text[:10])
            return dt.date(int(text[:4]), int(text[5:7]), 1)
        except (ValueError, IndexError):
            continue
    return None


def _table_exists(con, name: str) -> bool:
    try:
        return bool(con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [name]
        ).fetchone()[0])
    except Exception:  # noqa: BLE001
        return False


def _figure_key(doc_id: Any, quote: Any) -> tuple[str, str]:
    return (str(doc_id or ""), " ".join(str(quote or "").split()))


# ---------------------------------------------------------------------------
# The nearest-miss classifier (pure)
# ---------------------------------------------------------------------------


def classify_nearest_miss(
    rejected: Iterable[dict[str, Any]],
    admitted_by_lever: dict[str, set[tuple[str, str]]],
) -> list[dict[str, Any]]:
    """Label each rejected figure with what it would take to admit it.

    ``rejected`` is the replay's rejection list with every gate on.
    ``admitted_by_lever`` maps a lever name (the first element of a
    :data:`GATE_LEVERS` value) to the figure keys a replay with that gate
    alone turned off ADMITTED. Each returned record carries ``only_reason``:

    * the rejection reason, when turning its gate off admits the figure —
      it was rejected for that reason and nothing else;
    * ``several`` when its gate is switchable and turning it off does not
      admit the figure (another gate rejects it too);
    * ``untested:<class>`` for a parse-stage rejection, which never reached
      the later gates.
    """

    out: list[dict[str, Any]] = []
    for entry in rejected:
        reason = str(entry.get("reason") or "")
        if reason in NOT_A_MISS:
            continue
        key = _figure_key(entry.get("doc_id"), entry.get("quote"))
        lever = GATE_LEVERS.get(reason)
        if lever is not None:
            admitted = admitted_by_lever.get(lever[0]) or set()
            only = reason if key in admitted else "several"
        else:
            only = f"untested:{PARSE_REASONS.get(reason, 'other')}"
        out.append({
            "reason": reason,
            "only_reason": only,
            "doc_id": entry.get("doc_id"),
            "value": entry.get("value"),
            "unit": entry.get("unit"),
            "quote": entry.get("quote"),
        })
    return out


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def rulebook_with(rulebook, key: str, value: Any):
    """A copy of ``rulebook`` with one dotted key set (no validation)."""

    import copy

    from resolver.hazard_resolution.rulebook import Rulebook

    data = copy.deepcopy(rulebook.raw)
    node = data
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
    return Rulebook(data, rulebook.path)


def replay_levers(con, *, iso3: str, ym: str, hazard: str, rulebook) -> dict[str, Any]:
    """The cell's rung-2 replay with every gate on, and with each gate off."""

    from resolver.hazard_resolution import quote_period_audit as audit

    base = audit.replay_cell(
        con, iso3=iso3, ym=ym, hazard=hazard, rulebook=rulebook, with_ladder=False,
    )
    reasons = {str(r.get("reason") or "") for r in base["rejected"]}
    admitted: dict[str, set[tuple[str, str]]] = {}
    for reason in sorted(reasons):
        lever = GATE_LEVERS.get(reason)
        if lever is None or lever[0] in admitted:
            continue
        name, key, value = lever
        variant = audit.replay_cell(
            con, iso3=iso3, ym=ym, hazard=hazard,
            rulebook=rulebook_with(rulebook, key, value), with_ladder=False,
        )
        admitted[name] = {
            _figure_key((c.detail or {}).get("doc_id"), (c.detail or {}).get("quote"))
            for c in variant["rung2"]
        }
    return {
        "docs_cached": base["docs_cached"],
        "figures": base["figures"],
        "admitted_now": len(base["rung2"]),
        "rejected": base["rejected"],
        "misses": classify_nearest_miss(base["rejected"], admitted),
        "levers_that_resolve": resolving_levers(len(base["rung2"]), admitted),
    }


def resolving_levers(admitted_now: int, admitted: dict[str, set]) -> list[str]:
    """Levers that would turn a cell with no admissible figure into one with one.

    A cell the current rules already answer is not resolved BY a lever: its
    NO_DATA is the freeze keeping an old answer, which no gate change moves.
    """

    if admitted_now:
        return []
    return sorted(name for name, keys in admitted.items() if keys)


# ---------------------------------------------------------------------------
# Per-cell facts
# ---------------------------------------------------------------------------


def detection_class(triggered: Any, trigger_source: Any, detail: dict[str, Any]) -> tuple[str, str]:
    """(class, strongest GDACS alert colour) for a cell's trigger row."""

    events = list(detail.get("events_qualifying") or []) + list(
        detail.get("events_below_threshold") or []
    )
    levels = [str(e.get("alert_level") or "").lower() for e in events]
    levels = [lvl for lvl in levels if lvl in _ALERT_ORDER]
    top = max(levels, key=_ALERT_ORDER.get) if levels else ""
    source = str(trigger_source or "")
    if triggered is None and not source:
        return "no_trigger_row", top
    if source == "gdacs":
        return (f"gdacs_{top}" if top else "gdacs_unknown_level"), top
    if source in ("ibtracs", "reliefweb_ladder"):
        return source, top
    if source == "reliefweb_sweep":
        return "reliefweb_sweep_legacy", top
    if not triggered:
        return "not_triggered_legacy", top
    return source or "other", top


def _raw_counts(con, table: str, iso3: str, ym: str, hazard: str) -> tuple[int, int]:
    """(records for this hazard, records for any hazard) touching the month."""

    if not _table_exists(con, table):
        return -1, -1
    rows = con.execute(
        f"""
        SELECT hazard, payload_json FROM {table}
        WHERE iso3 = ? AND (ym = ? OR payload_json LIKE ?)
        """,
        [iso3, ym, f'%"{ym}"%'],
    ).fetchall()
    this = anyh = 0
    seen: set[tuple[str, str]] = set()
    for hz, payload in rows:
        pl = _loads(payload)
        months = pl.get("months_overlapped") or []
        if ym not in months and not str(pl.get("start_date") or "").startswith(ym):
            continue
        rid = (str(hz), str(pl.get("idu_id") or pl.get("report_id") or payload)[:80])
        if rid in seen:
            continue
        seen.add(rid)
        anyh += 1
        this += int(str(hz) == hazard)
    return this, anyh


def _late_records(con, table: str, iso3: str, ym: str, hazard: str,
                  frozen_at: dt.date | None) -> list[dict[str, Any]]:
    """Records of this hazard for the month dated after the cell froze."""

    if frozen_at is None or not _table_exists(con, table):
        return []
    out: list[dict[str, Any]] = []
    for payload, retrieved in con.execute(
        f"SELECT payload_json, retrieved_at FROM {table} "
        "WHERE iso3 = ? AND hazard = ? AND (ym = ? OR payload_json LIKE ?)",
        [iso3, hazard, ym, f'%"{ym}"%'],
    ).fetchall():
        pl = _loads(payload)
        if ym not in (pl.get("months_overlapped") or []):
            continue
        raw = pl.get("raw") or {}
        published = _parse_date(raw.get("created_at")) or _parse_date(pl.get("report_date"))
        figure = pl.get("displaced") if "displaced" in pl else pl.get("num_affected")
        if not figure:
            continue
        if published is not None and published > frozen_at:
            out.append({"published": published.isoformat(), "figure": figure})
    return out


@dataclass
class CellFacts:
    iso3: str
    ym: str
    hazard: str
    status: str
    frozen_at: str = ""
    detection: str = ""
    max_alert: str = ""
    no_row_reason: str = ""
    idmc_unread: bool = False
    idmc_this_hazard: int = 0
    idmc_any_hazard: int = 0
    ifrc_unread: bool = False
    ifrc_this_hazard: int = 0
    ifrc_any_hazard: int = 0
    reliefweb_fetch_ok: Any = ""
    docs_found: int = 0
    docs_read: int = 0
    docs_failed: int = 0
    extraction_skipped: str = ""
    budget_capped: bool = False
    figures_extracted: int = 0
    figures_rejected: int = 0
    reasons: Counter = field(default_factory=Counter)
    stored_reasons: Counter = field(default_factory=Counter)
    nearest_miss: str = ""
    nearest_miss_value: Any = None
    nearest_miss_quote: str = ""
    levers_that_resolve: list[str] = field(default_factory=list)
    late_idmc: int = 0
    late_ifrc: int = 0

    def as_row(self) -> dict[str, Any]:
        return {
            "iso3": self.iso3, "ym": self.ym, "hazard": self.hazard, "status": self.status,
            "frozen_at": self.frozen_at, "detection": self.detection,
            "max_gdacs_alert": self.max_alert, "no_row_reason": self.no_row_reason,
            "idmc_unread": self.idmc_unread, "idmc_records_this_hazard": self.idmc_this_hazard,
            "idmc_records_any_hazard": self.idmc_any_hazard,
            "ifrc_unread": self.ifrc_unread, "ifrc_records_this_hazard": self.ifrc_this_hazard,
            "ifrc_records_any_hazard": self.ifrc_any_hazard,
            "reliefweb_fetch_ok": self.reliefweb_fetch_ok,
            "rw_docs_found": self.docs_found, "rw_docs_read": self.docs_read,
            "rw_docs_failed": self.docs_failed,
            "rw_extraction_skipped": self.extraction_skipped,
            "rw_budget_capped": self.budget_capped,
            "rw_figures_extracted": self.figures_extracted,
            "rw_figures_rejected": self.figures_rejected,
            "rw_rejection_reasons": ";".join(f"{k}={v}" for k, v in sorted(self.reasons.items())),
            "rw_stored_rejection_reasons": ";".join(
                f"{k}={v}" for k, v in sorted(self.stored_reasons.items())),
            "nearest_miss": self.nearest_miss,
            "nearest_miss_value": self.nearest_miss_value,
            "nearest_miss_quote": (self.nearest_miss_quote or "")[:300],
            "levers_that_resolve": ";".join(self.levers_that_resolve),
            "idmc_records_after_freeze": self.late_idmc,
            "ifrc_records_after_freeze": self.late_ifrc,
        }


#: How a cell's nearest miss is chosen when it has several: a figure one
#: gate alone stood in front of beats one that several did, which beats a
#: parse rejection.
def _miss_rank(miss: dict[str, Any]) -> int:
    only = miss["only_reason"]
    if only == "several":
        return 1
    if only.startswith("untested:"):
        return 2
    return 0


def cell_facts(con, *, iso3: str, ym: str, hazard: str, status: str, frozen_at: Any,
               provenance: dict[str, Any], trigger: tuple | None, rulebook,
               replay: bool) -> CellFacts:
    facts = CellFacts(iso3=iso3, ym=ym, hazard=hazard, status=status)
    frozen = _parse_date(frozen_at)
    facts.frozen_at = frozen.isoformat() if frozen else ""

    if trigger is None:
        facts.detection, facts.max_alert = detection_class(None, None, {})
    else:
        detail = _loads(trigger[2])
        facts.detection, facts.max_alert = detection_class(trigger[0], trigger[1], detail)
        facts.no_row_reason = str(detail.get("no_row_reason") or "")

    decision = provenance.get("decision") or {}
    unread = set(decision.get("rungs_unavailable") or decision.get("sources_unavailable") or [])
    facts.idmc_unread = "idmc_idu" in unread
    facts.ifrc_unread = "ifrc_go" in unread
    facts.idmc_this_hazard, facts.idmc_any_hazard = _raw_counts(
        con, "haz_raw_idmc_idu", iso3, ym, hazard)
    facts.ifrc_this_hazard, facts.ifrc_any_hazard = _raw_counts(
        con, "haz_raw_ifrc_go", iso3, ym, hazard)
    facts.late_idmc = len(_late_records(con, "haz_raw_idmc_idu", iso3, ym, hazard, frozen))
    facts.late_ifrc = len(_late_records(con, "haz_raw_ifrc_go", iso3, ym, hazard, frozen))

    rw = provenance.get("reliefweb_extraction") or {}
    docs = rw.get("documents") or {}
    facts.reliefweb_fetch_ok = docs.get("ok", "")
    extraction = rw.get("extraction") or {}
    facts.extraction_skipped = str(rw.get("skipped_reason") or extraction.get("skipped_reason") or "")
    facts.budget_capped = bool(extraction.get("budget_capped"))
    for entry in extraction.get("rejected") or []:
        facts.stored_reasons[str(entry.get("reason") or "")] += 1

    if _table_exists(con, "haz_raw_reliefweb_docs"):
        facts.docs_found = int(con.execute(
            "SELECT COUNT(DISTINCT record_id) FROM haz_raw_reliefweb_docs "
            "WHERE iso3 = ? AND ym = ? AND hazard = ?", [iso3, ym, hazard],
        ).fetchone()[0])
    year, month = (int(p) for p in ym.split("-"))
    if _table_exists(con, "haz_doc_extractions"):
        row = con.execute(
            "SELECT COUNT(DISTINCT CASE WHEN status = 'ok' THEN doc_id END), "
            "COUNT(DISTINCT CASE WHEN status = 'error' THEN doc_id END) "
            "FROM haz_doc_extractions WHERE iso3 = ? AND hazard = ? AND year = ? AND month = ?",
            [iso3, hazard, year, month],
        ).fetchone()
        facts.docs_read, facts.docs_failed = int(row[0] or 0), int(row[1] or 0)

    if replay and facts.docs_read:
        try:
            result = replay_levers(con, iso3=iso3, ym=ym, hazard=hazard, rulebook=rulebook)
        except Exception as exc:  # noqa: BLE001 - one cell must not end the report
            LOG.warning("[no_data] replay failed for %s %s %s: %s", iso3, hazard, ym, exc)
            result = None
        if result is not None:
            facts.figures_extracted = int(result["figures"])
            facts.figures_rejected = len(result["rejected"])
            facts.reasons = Counter(str(r.get("reason") or "") for r in result["rejected"])
            facts.levers_that_resolve = result["levers_that_resolve"]
            if result["admitted_now"]:
                # The current rules admit a figure the stored row did not:
                # a frozen row written under older rules. Said, not hidden.
                facts.nearest_miss = "admitted_under_current_rules"
            elif result["misses"]:
                best = min(result["misses"], key=_miss_rank)
                facts.nearest_miss = best["only_reason"]
                facts.nearest_miss_value = best["value"]
                facts.nearest_miss_quote = str(best["quote"] or "")
            facts._misses = result["misses"]  # type: ignore[attr-defined]
    else:
        facts.figures_extracted = int(extraction.get("figures_extracted") or 0)
        facts.figures_rejected = sum(facts.stored_reasons.values())
    return facts


# ---------------------------------------------------------------------------
# The breakdown
# ---------------------------------------------------------------------------


def _window_sql() -> str:
    return "(year * 12 + month) BETWEEN (? * 12 + ?) AND (? * 12 + ?)"


def _bounds(first: str, last: str) -> list[int]:
    fy, fm = (int(p) for p in first.split("-"))
    ly, lm = (int(p) for p in last.split("-"))
    return [fy, fm, ly, lm]


def publication_lags(con, hazard: str) -> dict[str, Any]:
    """Days from month end to publication of every winning figure, per rung.

    Over the hazard's whole record (a lag is a property of the source, and
    the window alone is too few cells to say much). ReliefWeb figures are
    dated by their document; IDMC by the record's own ``created_at``;
    IFRC GO by the report date.
    """

    lags: dict[str, list[int]] = defaultdict(list)
    for year, month, prov_json in con.execute(
        "SELECT year, month, provenance_json FROM haz_resolutions "
        "WHERE hazard = ? AND status = 'RESOLVED_VALUE'", [hazard],
    ).fetchall():
        prov = _loads(prov_json)
        rung = str(prov.get("winning_rung") or "")
        refs = prov.get("source_record_ids") or []
        if not rung or not refs:
            continue
        ym = f"{int(year):04d}-{int(month):02d}"
        end = month_end(ym)
        published: dt.date | None = None
        if rung == "reliefweb_extracted":
            for cand in (prov.get("decision") or {}).get("candidates") or []:
                if cand.get("source_ref") == refs[0]:
                    detail = cand.get("detail") or {}
                    published = _parse_date(detail.get("doc_date")) or _parse_date(
                        detail.get("doc_date_original"))
                    break
        elif rung in ("idmc_idu", "ifrc_go"):
            table = f"haz_raw_{rung}"
            if _table_exists(con, table):
                row = con.execute(
                    f"SELECT payload_json FROM {table} WHERE record_id = ? LIMIT 1",
                    [str(refs[0])],
                ).fetchone()
                if row:
                    pl = _loads(row[0])
                    raw = pl.get("raw") or {}
                    published = _parse_date(raw.get("created_at")) or _parse_date(
                        pl.get("report_date"))
        if published is not None:
            lags[rung].append((published - end).days)
    out: dict[str, Any] = {}
    for rung, values in sorted(lags.items()):
        values.sort()
        n = len(values)
        out[rung] = {
            "n": n,
            "median_days": values[n // 2],
            "p90_days": values[min(n - 1, int(0.9 * n))],
            "share_after_14_days": round(sum(v > 14 for v in values) / n, 3),
            "share_after_60_days": round(sum(v > 60 for v in values) / n, 3),
            "share_after_90_days": round(sum(v > 90 for v in values) / n, 3),
        }
    return out


def revisions_on_no_data(con, hazard: str, first: str, last: str) -> dict[str, Any]:
    """Post-freeze attempts on window NO_DATA cells that carried a value."""

    if not _table_exists(con, "haz_revisions"):
        return {"available": False}
    rows = con.execute(
        f"""
        SELECT r.iso3, r.year, r.month, r.source, r.new_value, r.observed_at
        FROM haz_revisions r
        JOIN haz_resolutions h
          ON h.iso3 = r.iso3 AND h.year = r.year AND h.month = r.month AND h.hazard = r.hazard
        WHERE r.hazard = ? AND h.status = 'NO_DATA' AND r.new_value IS NOT NULL
          AND {_window_sql().replace('year', 'r.year').replace('month', 'r.month')}
        """,
        [hazard, *_bounds(first, last)],
    ).fetchall()
    cells: dict[tuple, int] = {}
    by_source: Counter = Counter()
    for iso3, year, month, source, _value, observed in rows:
        ym = f"{int(year):04d}-{int(month):02d}"
        lag = (_parse_date(observed) - month_end(ym)).days if _parse_date(observed) else None
        key = (iso3, ym)
        if lag is not None:
            cells[key] = min(cells.get(key, lag), lag)
        by_source[str(source)] += 1
    return {
        "available": True,
        "cells": len(cells),
        "by_source": dict(by_source),
        "first_seen_days_after_month_end": sorted(cells.values()),
    }


def build_breakdown(con, *, hazard: str, first: str, last: str, rulebook,
                    replay: bool = True, examples: int = 3,
                    replay_budget_sec: float | None = None) -> dict[str, Any]:
    """The five-part breakdown for one hazard over the window.

    ``replay_budget_sec`` bounds the time spent replaying cells (the debug
    bundle runs inside a step budget); cells reached after it are counted
    from their stored rejection reasons and named in ``cells_not_replayed``.
    """

    import time

    started = time.monotonic()
    not_replayed = 0
    bounds = _bounds(first, last)
    triggers = {
        (str(r[0]).upper(), f"{int(r[1]):04d}-{int(r[2]):02d}"): (r[3], r[4], r[5])
        for r in con.execute(
            f"SELECT iso3, year, month, triggered, trigger_source, trigger_detail_json "
            f"FROM haz_triggers WHERE hazard = ? AND {_window_sql()}",
            [hazard, *bounds],
        ).fetchall()
    }
    resolved = con.execute(
        f"SELECT iso3, year, month, status, frozen_at, provenance_json FROM haz_resolutions "
        f"WHERE hazard = ? AND {_window_sql()} ORDER BY year, month, iso3",
        [hazard, *bounds],
    ).fetchall()
    status_counts = Counter(str(r[3]) for r in resolved)
    written = {(str(r[0]).upper(), f"{int(r[1]):04d}-{int(r[2]):02d}") for r in resolved}

    cells: list[CellFacts] = []
    for iso3, year, month, status, frozen_at, prov_json in resolved:
        if status != "NO_DATA":
            continue
        ym = f"{int(year):04d}-{int(month):02d}"
        key = (str(iso3).upper(), ym)
        in_budget = replay_budget_sec is None or (
            time.monotonic() - started) < replay_budget_sec
        not_replayed += int(replay and not in_budget)
        cells.append(cell_facts(
            con, iso3=key[0], ym=ym, hazard=hazard, status=status, frozen_at=frozen_at,
            provenance=_loads(prov_json), trigger=triggers.get(key), rulebook=rulebook,
            replay=replay and in_budget,
        ))

    no_row = Counter(
        str(_loads(v[2]).get("no_row_reason") or "unstated")
        for k, v in triggers.items() if k not in written
    )

    detection = Counter(c.detection for c in cells)
    nearest = Counter(c.nearest_miss or "no_figure_extracted" for c in cells)
    reasons: Counter = Counter()
    stored_reasons: Counter = Counter()
    only_reason_figures: Counter = Counter()
    miss_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    lever_cells: Counter = Counter()
    for c in cells:
        reasons.update(c.reasons)
        stored_reasons.update(c.stored_reasons)
        for lever in c.levers_that_resolve:
            lever_cells[lever] += 1
        for miss in getattr(c, "_misses", []):
            only_reason_figures[miss["only_reason"]] += 1
            bucket = miss_examples[miss["only_reason"]]
            if len(bucket) < examples:
                bucket.append({"iso3": c.iso3, "ym": c.ym, "value": miss["value"],
                               "unit": miss["unit"], "quote": str(miss["quote"] or "")[:300]})

    def _rung(attr_unread: str, attr_this: str, attr_any: str) -> dict[str, int]:
        return {
            "unread": sum(bool(getattr(c, attr_unread)) for c in cells),
            "record_this_hazard": sum(getattr(c, attr_this) > 0 for c in cells),
            "record_other_hazard_only": sum(
                getattr(c, attr_this) == 0 and getattr(c, attr_any) > 0 for c in cells),
            "no_record": sum(getattr(c, attr_any) == 0 for c in cells),
        }

    return {
        "hazard": hazard,
        "window": [first, last],
        "status_counts": dict(status_counts),
        "cells_assessed": len(triggers),
        "no_data": len(cells),
        "cells_not_replayed": not_replayed,
        "no_row": sum(no_row.values()),
        "no_row_reasons": dict(no_row.most_common()),
        "detection": dict(detection.most_common()),
        "rungs": {
            "idmc_idu": _rung("idmc_unread", "idmc_this_hazard", "idmc_any_hazard"),
            "ifrc_go": _rung("ifrc_unread", "ifrc_this_hazard", "ifrc_any_hazard"),
            "reliefweb": {
                "fetch_failed": sum(c.reliefweb_fetch_ok is False for c in cells),
                "no_documents": sum(c.docs_found == 0 for c in cells),
                "documents_not_read": sum(c.docs_found > 0 and c.docs_read == 0 for c in cells),
                "budget_capped": sum(c.budget_capped for c in cells),
                "read_no_figure": sum(c.docs_read > 0 and c.figures_extracted == 0 for c in cells),
                "figures_all_rejected": sum(c.figures_extracted > 0 for c in cells),
                "docs_found_total": sum(c.docs_found for c in cells),
                "docs_read_total": sum(c.docs_read for c in cells),
                "figures_extracted_total": sum(c.figures_extracted for c in cells),
                "rejection_reasons_current_rules": dict(reasons.most_common()),
                "rejection_reasons_stored": dict(stored_reasons.most_common()),
                "extraction_skipped": dict(Counter(
                    c.extraction_skipped for c in cells if c.extraction_skipped).most_common()),
            },
        },
        "nearest_miss_cells": dict(nearest.most_common()),
        "nearest_miss_figures": dict(only_reason_figures.most_common()),
        "nearest_miss_examples": {k: v for k, v in miss_examples.items()},
        "cells_resolved_by_lever": dict(lever_cells.most_common()),
        "by_country": dict(Counter(c.iso3 for c in cells).most_common()),
        "by_month": dict(sorted(Counter(c.ym for c in cells).items())),
        "freeze": {
            "freeze_days": rulebook.get("freeze_days"),
            "reliefweb_publication_pad_days": rulebook.get(
                "reliefweb.documents.publication_pad_days"),
            "cells_with_idmc_record_after_freeze": sum(c.late_idmc > 0 for c in cells),
            "cells_with_ifrc_record_after_freeze": sum(c.late_ifrc > 0 for c in cells),
            "revisions_on_no_data": revisions_on_no_data(con, hazard, first, last),
            "winning_figure_publication_lag": publication_lags(con, hazard),
        },
        "_cells": cells,
    }


def headline(breakdown: dict[str, Any]) -> dict[str, Any]:
    """The few numbers the debug bundle and the review inputs carry."""

    misses = {k: v for k, v in (breakdown.get("nearest_miss_figures") or {}).items()
              if not k.startswith("untested:") and k != "several"}
    top = sorted(misses.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    return {
        "hazard": breakdown["hazard"],
        "window": breakdown["window"],
        "no_data": breakdown["no_data"],
        "no_row": breakdown["no_row"],
        "cells_not_replayed": breakdown.get("cells_not_replayed", 0),
        "no_data_by_detection": breakdown["detection"],
        "top_nearest_miss_reasons": dict(top),
        "cells_resolved_by_lever": breakdown.get("cells_resolved_by_lever") or {},
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(str(x).replace("|", "/") for x in row) + " |")
    return lines


def render_markdown(breakdowns: list[dict[str, Any]], *, replayed: bool) -> str:
    out: list[str] = ["# Why PA cells end NO_DATA", ""]
    if not replayed:
        out += ["_Replay disabled: rejection reasons are the stored ones, and no "
                "single-gate test was run._", ""]
    for b in breakdowns:
        n = b["no_data"] or 1
        out += [f"## {b['hazard']}, {b['window'][0]} to {b['window'][1]}", ""]
        out += _table(["status", "cells"], sorted(b["status_counts"].items()))
        out += [f"| no row | {b['no_row']} |", "",
                f"Cells assessed: {b['cells_assessed']}. NO_DATA: {b['no_data']}.", ""]
        if b["no_row_reasons"]:
            out += ["No-row cells by reason:", ""]
            out += _table(["reason", "cells"], b["no_row_reasons"].items()) + [""]

        out += ["### 1. Detection", ""]
        out += _table(["trigger", "NO_DATA cells", "share"],
                      [(k, v, f"{100 * v / n:.1f}%") for k, v in b["detection"].items()])
        out += [""]

        out += ["### 2. Rungs", ""]
        rows = []
        for rung in ("idmc_idu", "ifrc_go"):
            r = b["rungs"][rung]
            rows.append((rung, r["unread"], r["record_this_hazard"],
                         r["record_other_hazard_only"], r["no_record"]))
        out += _table(["rung", "unread", "record for this hazard and month",
                       "record for another hazard only", "no record"], rows)
        rw = b["rungs"]["reliefweb"]
        out += ["", "ReliefWeb:", ""]
        out += _table(["", "cells"], [
            ("fetch failed", rw["fetch_failed"]),
            ("no documents in the window", rw["no_documents"]),
            ("documents found, none read", rw["documents_not_read"]),
            ("extraction budget-capped", rw["budget_capped"]),
            ("read, no figure stated", rw["read_no_figure"]),
            ("figures extracted, all rejected", rw["figures_all_rejected"]),
        ])
        out += ["", f"Documents found {rw['docs_found_total']}, read {rw['docs_read_total']}, "
                f"figures extracted {rw['figures_extracted_total']}.", ""]
        out += ["Figures rejected, by reason (current rules):", ""]
        out += _table(["reason", "figures"], rw["rejection_reasons_current_rules"].items())
        out += ["", "Figures rejected, by reason (as stored when the cell was decided):", ""]
        out += _table(["reason", "figures"], rw["rejection_reasons_stored"].items())
        out += [""]

        out += ["### 3. The nearest miss", "",
                "A figure is counted under a reason when turning that gate alone off "
                "admits it. `several` means another gate rejects it too; `untested` "
                "rejections are made while parsing and never reach the later gates.", ""]
        out += _table(["nearest miss per cell", "cells"], b["nearest_miss_cells"].items())
        out += [""]
        out += _table(["rejected figures by what alone stood in the way", "figures"],
                      b["nearest_miss_figures"].items())
        out += ["", "Cells a single gate, turned off, would resolve:", ""]
        out += _table(["lever", "cells"], b["cells_resolved_by_lever"].items())
        for reason, items in sorted(b["nearest_miss_examples"].items()):
            if reason == "several" or reason.startswith("untested:parse"):
                continue
            out += ["", f"Examples, `{reason}`:", ""]
            for item in items:
                out.append(f"- {item['iso3']} {item['ym']}, {item['value']} {item['unit']}: "
                           f"\"{item['quote']}\"")
        out += [""]

        out += ["### 4. Concentration", ""]
        top = list(b["by_country"].items())[:20]
        top_sum = sum(v for _, v in top)
        out += [f"The top 20 countries hold {top_sum} of {b['no_data']} NO_DATA cells "
                f"({100 * top_sum / n:.0f}%).", ""]
        out += _table(["country", "cells"], top) + [""]
        out += _table(["month", "cells"], b["by_month"].items()) + [""]

        f = b["freeze"]
        out += ["### 5. Freeze timing", "",
                f"`freeze_days` = {f['freeze_days']}; ReliefWeb documents are read only "
                f"to month end + {f['reliefweb_publication_pad_days']} days, so a later "
                "freeze adds no ReliefWeb figure unless that window widens too.", "",
                f"NO_DATA cells now holding an IDMC IDU record for the month dated after "
                f"the cell froze: {f['cells_with_idmc_record_after_freeze']}; an IFRC GO "
                f"record: {f['cells_with_ifrc_record_after_freeze']}.", ""]
        rev = f["revisions_on_no_data"]
        if rev.get("available"):
            out += [f"NO_DATA cells a post-freeze re-walk found a value for: {rev['cells']} "
                    f"(by source {rev['by_source']}).", ""]
        lags = f["winning_figure_publication_lag"]
        if lags:
            out += ["Publication lag of every winning figure for this hazard (days after "
                    "month end, whole record):", ""]
            out += _table(["rung", "n", "median", "p90", "> 14 d", "> 60 d", "> 90 d"], [
                (r, v["n"], v["median_days"], v["p90_days"], v["share_after_14_days"],
                 v["share_after_60_days"], v["share_after_90_days"]) for r, v in lags.items()])
            out += [""]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(db_path: str, *, hazards: Sequence[str], first: str, last: str, out_dir: Path,
        replay: bool = True, examples: int = 3) -> dict[str, Any]:
    from resolver.hazard_resolution import quote_period_audit as audit
    from resolver.hazard_resolution.rulebook import load_rulebook

    rulebook = load_rulebook()
    con = audit.open_read_only(db_path)
    try:
        breakdowns = [build_breakdown(con, hazard=h, first=first, last=last,
                                      rulebook=rulebook, replay=replay, examples=examples)
                      for h in hazards]
    finally:
        con.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [c.as_row() for b in breakdowns for c in b["_cells"]]
    with open(out_dir / "pa_no_data_cells.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CellFacts("", "", "", "").as_row()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "window": [first, last],
        "replayed": replay,
        "hazards": {b["hazard"]: {k: v for k, v in b.items() if k != "_cells"}
                    for b in breakdowns},
        "headline": {b["hazard"]: headline(b) for b in breakdowns},
    }
    (out_dir / "pa_no_data_breakdown.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "pa_no_data_breakdown.md").write_text(
        render_markdown(breakdowns, replayed=replay), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", required=True)
    parser.add_argument("--hazard", action="append", default=None,
                        help="flood or cyclone (FL/TC); repeatable. Default: flood")
    parser.add_argument("--window", required=True, help="YYYY-MM:YYYY-MM")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--no-replay", action="store_true",
                        help="use the stored rejection reasons only")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    hazards = []
    for name in args.hazard or ["flood"]:
        code = HAZARD_CODES.get(str(name).lower())
        if code is None:
            parser.error(f"unknown hazard {name!r}; use flood or cyclone")
        hazards.append(code)
    first, last = parse_window(args.window)

    before = sha256_file(args.db)
    run(args.db, hazards=hazards, first=first, last=last, out_dir=Path(args.out_dir),
        replay=not args.no_replay, examples=args.examples)
    after = sha256_file(args.db)
    if before != after:
        print(f"::error::the DB file changed while it was read ({before} -> {after})")
        return 2
    print(f"DB unchanged (sha256 {before[:16]}); report in {args.out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
