# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""What the November 2026 figure rules would change, read from the cache alone.

The rules in :mod:`figures` (date a figure by what it says, a stated people
count beats a household conversion, one figure one hazard) change which
ReliefWeb figures may resolve a cell. This audit replays each flood and
cyclone cell's CACHED extractions through the current ``build_candidates``
and the ladder, with no network and no model call, and reports per cell:

* the stored status, value, winning quote, ``provisional`` and ``frozen_at``;
* the winner and value the current rules give;
* why the stored winner no longer wins, when it does not.

It also counts, across every accepted ReliefWeb figure the database holds,
the rejection reasons the current rules would give them. That count is
information only: frozen history is not changed by a re-walk.

It never writes. The connection is opened read-only and the schema module
is told the tables already exist, so nothing below tries to create one.

Run::

    python -m resolver.hazard_resolution.quote_period_audit --db PATH \\
        --months 2026-08 2026-09 2026-10 --json-out audit.json --md-out audit.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

LOG = logging.getLogger(__name__)

AUDIT_HAZARDS = ("FL", "TC")
RELIEFWEB_SOURCE = "reliefweb_extracted"

#: The switches the November 2026 rules added. Turned off together they
#: reproduce the rules as they stood before, which is the audit's baseline.
NOVEMBER_SWITCHES = (
    "reliefweb.documents.enforce_quote_period",
    "reliefweb.documents.reject_undated_in_dated_document",
    "reliefweb.documents.one_hazard_per_figure",
    "reliefweb.household_conversion.prefer_stated_people",
)


def rulebook_with_switches(rulebook, value: bool):
    """A copy of ``rulebook`` with every November switch set to ``value``."""

    import copy

    from resolver.hazard_resolution.rulebook import Rulebook

    data = copy.deepcopy(rulebook.raw)
    for key in NOVEMBER_SWITCHES:
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return Rulebook(data, rulebook.path)


class _NoCalls:
    """A model seam that refuses: the audit must never reach a provider."""

    calls = 0

    def __call__(self, *args: Any, **kwargs: Any):  # pragma: no cover - must not run
        _NoCalls.calls += 1
        raise RuntimeError("quote_period_audit must not call a model")


def open_read_only(db_path: str):
    """A read-only connection the haz_* helpers will not try to migrate."""

    import duckdb

    from resolver.hazard_resolution import schema as schema_mod

    con = duckdb.connect(str(db_path), read_only=True)
    schema_mod._ENSURED_CONNECTIONS.add(con)
    return con


def _zero_budget():
    from resolver.hazard_resolution.extract import ExtractionBudget

    # max 0: every uncached document is "budget capped" and nothing is read.
    return ExtractionBudget(max_calls_per_month=0, used_this_month=0)


def stored_resolution(con, iso3: str, ym: str, hazard: str) -> dict[str, Any] | None:
    year, month = (int(p) for p in ym.split("-"))
    row = con.execute(
        """
        SELECT status, value, rule_fired, COALESCE(provisional, FALSE), frozen_at,
               provenance_json
        FROM haz_resolutions
        WHERE iso3 = ? AND hazard = ? AND year = ? AND month = ?
        """,
        [iso3, hazard, year, month],
    ).fetchone()
    if row is None:
        return None
    try:
        provenance = json.loads(row[5] or "{}")
    except (TypeError, ValueError):
        provenance = {}
    winner: dict[str, Any] = {}
    refs = provenance.get("source_record_ids") or []
    for candidate in (provenance.get("decision") or {}).get("candidates") or []:
        if refs and candidate.get("source_ref") == refs[0]:
            winner = candidate
            break
    detail = winner.get("detail") or {}
    return {
        "status": row[0],
        "value": None if row[1] is None else float(row[1]),
        "rule_fired": row[2],
        "provisional": bool(row[3]),
        "frozen_at": row[4].isoformat() if hasattr(row[4], "isoformat") else row[4],
        "winning_rung": provenance.get("winning_rung"),
        "winning_source_ref": refs[0] if refs else None,
        "winning_doc_id": detail.get("doc_id"),
        "winning_quote": detail.get("quote"),
        "winning_unit": detail.get("unit_as_stated"),
    }


def cells_for(con, hazard: str, ym: str) -> list[str]:
    """Every country whose cell the LADDER decided, or that has a cached extraction.

    A ``RESOLVED_ZERO`` is written by the detection path (no trigger, a
    silent sweep), never by the ladder, so replaying the ladder for it
    would compare two different questions; those cells are left out.
    """

    year, month = (int(p) for p in ym.split("-"))
    rows = con.execute(
        """
        SELECT iso3 FROM haz_resolutions WHERE hazard = ? AND year = ? AND month = ?
          AND status <> 'RESOLVED_ZERO'
        UNION
        SELECT iso3 FROM haz_doc_extractions WHERE hazard = ? AND year = ? AND month = ?
        """,
        [hazard, year, month, hazard, year, month],
    ).fetchall()
    return sorted({str(r[0]).upper() for r in rows if r[0]})


def replay_cell(
    con, *, iso3: str, ym: str, hazard: str, rulebook, today: dt.date | None = None,
    with_ladder: bool = True,
) -> dict[str, Any]:
    """Replay one cell's cached extractions through the current rules.

    Returns the rung-2 candidates and rejections and, with ``with_ladder``,
    the ladder's verdict over every rung's cached candidates.
    """

    from resolver.hazard_resolution import candidates as cand_mod
    from resolver.hazard_resolution import extract as extract_mod
    from resolver.hazard_resolution import figures as figures_mod
    from resolver.hazard_resolution import reconcile as reconcile_mod
    from resolver.hazard_resolution.impact import national_population

    extraction = extract_mod.extract_for_cell(
        con, iso3=iso3, ym=ym, hazard=hazard, rulebook=rulebook,
        budget=_zero_budget(), call=_NoCalls(),
        model_ref=extract_mod.resolve_extraction_model(rulebook),
    )
    ceiling_basis = cand_mod.exposure_ceiling_basis(con, iso3, ym, hazard, rulebook)
    rung2, rejected = figures_mod.build_candidates(
        extraction, iso3=iso3, ym=ym, hazard=hazard, rulebook=rulebook,
        exposure_ceiling=ceiling_basis["value"], ceiling_basis=ceiling_basis,
    )
    out: dict[str, Any] = {
        "docs_cached": extraction.docs_cached,
        "figures": len(extraction.figures),
        "rung2": rung2,
        "rejected": rejected,
    }
    if with_ladder:
        year = int(ym.split("-")[0])
        found = cand_mod.build_candidates(con, iso3, ym, hazard, rulebook, extracted=rung2)
        verdict = reconcile_mod.reconcile(
            iso3=iso3, ym=ym, hazard=hazard, candidates=found, rulebook=rulebook,
            national_population=national_population(con, iso3, year), today=today,
        )
        winner = None
        refs = verdict.provenance.get("source_record_ids") or []
        for candidate in found:
            if refs and candidate.source_ref == refs[0]:
                winner = candidate
                break
        out["verdict"] = {
            "status": verdict.status,
            "value": verdict.value,
            "rule_fired": verdict.rule_fired,
            "winning_rung": verdict.provenance.get("winning_rung"),
            "winning_source_ref": refs[0] if refs else None,
            "winning_quote": (winner.detail or {}).get("quote") if winner else None,
        }
    return out


def why_stored_winner_failed(stored: dict[str, Any], replay: dict[str, Any]) -> str:
    """The reason the stored rung-2 winner no longer wins, or '' when it still does."""

    if not stored or stored.get("winning_rung") != RELIEFWEB_SOURCE:
        return ""
    ref = stored.get("winning_source_ref")
    unit = stored.get("winning_unit")
    # The source ref is the document plus the quote's hash, so the
    # households and the people readings of one sentence share it: the unit
    # tells them apart.
    def _same(candidate) -> bool:
        cand_unit = (candidate.detail or {}).get("unit_as_stated")
        return candidate.source_ref == ref and (not unit or cand_unit == unit)

    if any(_same(c) for c in replay.get("rung2") or []):
        verdict = replay.get("verdict") or {}
        if verdict and verdict.get("winning_source_ref") != ref:
            return "outranked"
        return ""
    from resolver.hazard_resolution.figures import _normalise

    quote = _normalise(stored.get("winning_quote") or "")
    doc = str(stored.get("winning_doc_id") or "")
    reasons = sorted({
        str(r.get("reason"))
        for r in replay.get("rejected") or []
        if str(r.get("doc_id")) == doc and _normalise(r.get("quote") or "") == quote
        and (not unit or not r.get("unit") or r.get("unit") == unit)
    })
    return ",".join(reasons) or "not_extracted"


def audit_months(
    con, months: Iterable[str], rulebook, *, hazards: Iterable[str] = AUDIT_HAZARDS,
    today: dt.date | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline = rulebook_with_switches(rulebook, False)
    for ym in months:
        for hazard in hazards:
            for iso3 in cells_for(con, hazard, ym):
                stored = stored_resolution(con, iso3, ym, hazard) or {}
                try:
                    replay = replay_cell(
                        con, iso3=iso3, ym=ym, hazard=hazard, rulebook=rulebook, today=today
                    )
                    before = replay_cell(
                        con, iso3=iso3, ym=ym, hazard=hazard, rulebook=baseline, today=today
                    )
                except Exception as exc:  # one bad cell must not end the audit
                    LOG.warning("[audit] %s %s %s failed: %s", iso3, hazard, ym, exc)
                    rows.append({"iso3": iso3, "hazard": hazard, "ym": ym, "error": str(exc)})
                    continue
                verdict = replay.get("verdict") or {}
                old_verdict = before.get("verdict") or {}
                reason = why_stored_winner_failed(stored, replay)
                # Changed BY THESE RULES: the same cache, the same ladder, the
                # November switches off and then on. A difference between the
                # stored row and today's ladder that both replays share (a
                # newer IDMC record, say) is not this change's doing.
                changed = (
                    old_verdict.get("status") != verdict.get("status")
                    or old_verdict.get("value") != verdict.get("value")
                )
                rows.append({
                    "iso3": iso3, "hazard": hazard, "ym": ym,
                    "stored_status": stored.get("status"),
                    "stored_value": stored.get("value"),
                    "stored_winning_rung": stored.get("winning_rung"),
                    "stored_winning_quote": stored.get("winning_quote"),
                    "provisional": stored.get("provisional"),
                    "frozen_at": stored.get("frozen_at"),
                    "new_status": verdict.get("status"),
                    "new_value": verdict.get("value"),
                    "new_winning_rung": verdict.get("winning_rung"),
                    "new_winning_quote": verdict.get("winning_quote"),
                    "old_rules_status": old_verdict.get("status"),
                    "old_rules_value": old_verdict.get("value"),
                    "stored_winner_fails_because": reason,
                    "changed": bool(changed),
                })
    return rows


def accepted_figure_reason_counts(con, rulebook) -> dict[str, Any]:
    """Across every accepted ReliefWeb figure, the reason the current rules give.

    "accepted" means the figure is a stored ``reliefweb_extracted``
    candidate. Each such cell is replayed once; a stored candidate the
    replay no longer keeps is counted under the reason it is rejected with.
    """

    rows = con.execute(
        """
        SELECT DISTINCT iso3, hazard,
               printf('%04d-%02d', year, month) AS ym
        FROM haz_impact_candidates
        WHERE source = ? AND hazard IN ('FL', 'TC')
        ORDER BY 2, 3, 1
        """,
        [RELIEFWEB_SOURCE],
    ).fetchall()
    from resolver.hazard_resolution.figures import _normalise

    counts: Counter = Counter()
    by_hazard: Counter = Counter()
    accepted_total = 0
    cells_failed = 0
    for iso3, hazard, ym in rows:
        stored = con.execute(
            """
            SELECT source_ref, detail_json FROM haz_impact_candidates
            WHERE source = ? AND iso3 = ? AND hazard = ? AND year = ? AND month = ?
            """,
            [RELIEFWEB_SOURCE, iso3, hazard, int(ym[:4]), int(ym[5:7])],
        ).fetchall()
        try:
            replay = replay_cell(
                con, iso3=iso3, ym=ym, hazard=hazard, rulebook=rulebook, with_ladder=False
            )
        except Exception as exc:
            LOG.warning("[audit] count replay %s %s %s failed: %s", iso3, hazard, ym, exc)
            cells_failed += 1
            continue
        kept = {c.source_ref for c in replay["rung2"]}
        for ref, detail_json in stored:
            accepted_total += 1
            if ref in kept:
                continue
            try:
                detail = json.loads(detail_json or "{}")
            except (TypeError, ValueError):
                detail = {}
            quote = _normalise(detail.get("quote") or "")
            doc = str(detail.get("doc_id") or str(ref).split("#")[0])
            reasons = sorted({
                str(r.get("reason")) for r in replay["rejected"]
                if str(r.get("doc_id")) == doc and _normalise(r.get("quote") or "") == quote
            })
            key = ",".join(reasons) or "not_extracted"
            counts[key] += 1
            by_hazard[(hazard, key)] += 1
    return {
        "cells": len(rows),
        "cells_failed": cells_failed,
        "accepted_figures": accepted_total,
        "now_rejected_by_reason": dict(counts.most_common()),
        "now_rejected_by_hazard_and_reason": {
            f"{h}:{r}": n for (h, r), n in sorted(by_hazard.items())
        },
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:,.0f}"
    return str(value)


def _short(text: Any, limit: int = 110) -> str:
    text = str(text or "").replace("|", "/").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_markdown(report: dict[str, Any], *, only_changed: bool = True) -> str:
    lines = [
        "# Quote-period audit",
        "",
        f"Database: `{report.get('db')}`  ",
        f"Months: {', '.join(report.get('months') or [])}  ",
        f"Rulebook switches: {json.dumps(report.get('switches') or {})}",
        "",
        "A cell is listed when the November switches change its ladder verdict",
        "(old rules against new rules, same cache), or when its stored winner",
        "no longer wins. `stored` is the row in the database; `old rules` is",
        "today's replay with the switches off.",
        "",
        "| cell | stored | old rules | new rules | provisional | frozen_at | stored winner fails because | stored winning quote | new winning quote |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report.get("cells") or []:
        if "error" in row:
            lines.append(f"| {row['iso3']} {row['hazard']} {row['ym']} | error: {_short(row['error'])} | | | | | | | |")
            continue
        if only_changed and not row.get("changed") and not row.get("stored_winner_fails_because"):
            continue
        stored = f"{row.get('stored_status') or '-'} {_fmt(row.get('stored_value'))}".strip()
        old = f"{row.get('old_rules_status') or '-'} {_fmt(row.get('old_rules_value'))}".strip()
        new = f"{row.get('new_status') or '-'} {_fmt(row.get('new_value'))}".strip()
        if row.get("new_winning_rung"):
            new += f" ({row['new_winning_rung']})"
        lines.append(
            f"| {row['iso3']} {row['hazard']} {row['ym']} | {stored} | {old} | {new} | "
            f"{row.get('provisional')} | {(row.get('frozen_at') or '')[:10]} | "
            f"{row.get('stored_winner_fails_because') or ''} | "
            f"{_short(row.get('stored_winning_quote'), 80)} | {_short(row.get('new_winning_quote'), 80)} |"
        )
    counts = report.get("accepted_figure_counts") or {}
    if counts:
        lines += [
            "",
            "## Accepted ReliefWeb figures the current rules reject (all months, information only)",
            "",
            f"Cells replayed: {counts.get('cells')}, accepted figures: {counts.get('accepted_figures')}, "
            f"cells that failed to replay: {counts.get('cells_failed')}",
            "",
            "| reason | figures |",
            "|---|---|",
        ]
        for reason, n in (counts.get("now_rejected_by_reason") or {}).items():
            lines.append(f"| {reason} | {n} |")
    return "\n".join(lines) + "\n"


def _switches(rulebook) -> dict[str, Any]:
    keys = (
        "reliefweb.documents.enforce_reporting_window",
        "reliefweb.documents.enforce_quote_period",
        "reliefweb.documents.reject_undated_in_dated_document",
        "reliefweb.documents.one_hazard_per_figure",
        "reliefweb.household_conversion.prefer_stated_people",
    )
    return {k: rulebook.get(k, None) for k in keys}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", required=True)
    parser.add_argument("--months", nargs="+", required=True)
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    parser.add_argument("--rulebook", default=None)
    parser.add_argument("--today", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument(
        "--no-counts", action="store_true",
        help="skip the all-months count of accepted figures (slow on a full DB)",
    )
    parser.add_argument("--all-cells", action="store_true",
                        help="list unchanged cells in the markdown too")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    from resolver.hazard_resolution.rulebook import load_rulebook

    rulebook = load_rulebook(args.rulebook) if args.rulebook else load_rulebook()
    today = dt.date.fromisoformat(args.today) if args.today else None
    con = open_read_only(args.db)
    try:
        report = {
            "db": str(args.db),
            "months": list(args.months),
            "switches": _switches(rulebook),
            "cells": audit_months(con, args.months, rulebook, today=today),
        }
        if not args.no_counts:
            report["accepted_figure_counts"] = accepted_figure_reason_counts(con, rulebook)
    finally:
        con.close()
    report["model_calls"] = _NoCalls.calls
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))
    markdown = render_markdown(report, only_changed=not args.all_cells)
    if args.md_out:
        Path(args.md_out).write_text(markdown)
    else:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
