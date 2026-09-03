# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Backcast — the identical rulebook and code, replayed over history.

    python -m resolver.hazard_resolution.backcast --hazard flood
    python -m resolver.hazard_resolution.backcast --hazard cyclone --to 2015-12
    python -m resolver.hazard_resolution.backcast --hazard drought --no-extract

Each hazard is replayed from ``rulebook.backcast.<hazard>`` through the
last month that has FROZEN, one month at a time, through exactly the
functions a live run uses (:func:`cli.run_cyclone_month` and its two
siblings). There is no backcast-specific detection, no backcast-specific
ladder, and no second copy of any rule. The only difference is the
``run_type`` stamped on every row it writes.

**Why "the same code" is the whole point.** Base rates drawn from a
history the machine did not actually produce would describe a different
machine. If the backcast had its own path, a rulebook change could move
the live answers without moving the base rates the forecaster compares
them against, and nothing would fail.

**What a backcast may and may not overwrite.** Every month it touches is
frozen by definition, so the freeze guard in :mod:`resolutions` applies in
full: a cell that already has an answer keeps it, and the attempt is
logged to ``haz_revisions``. A backcast therefore FILLS GAPS and never
rewrites history. That also makes the resume ledger
(``haz_backcast_progress``) more than a convenience: re-walking a
completed month would write one revision row per already-answered cell,
burying the genuine post-freeze revisions the audit table exists for.

**Two costs to know before starting one.** ReliefWeb silence sweeps run
once per non-triggered country-month — a 25-year cyclone backcast over
~200 countries is on the order of half a million paced requests, and the
driver prints an estimate before it begins. And ladder rung 2 spends real
money, bounded by ``extraction.max_calls_per_month`` (counted per calendar
month across all runs), so a long backcast collects extraction gradually
rather than in one bill. ``--no-extract`` turns that off entirely.

**A known gap, deliberately not papered over.** The drought path's
indicator feeds publish a *latest* snapshot with no observation date. A
backcast month therefore falls outside ``max_observation_age_months``, the
required indicator reads UNAVAILABLE, and the month resolves INCONCLUSIVE
rather than to a false zero. That is the correct behaviour, and it means a
drought backcast produces almost nothing until ``drought.indicators``
entries point at a per-month archive (their URLs accept ``{ym}``,
``{year}`` and ``{month}``). :func:`check_backcastable` says so up front
rather than letting the run look merely disappointing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from resolver.hazard_resolution.base_rates import backcast_window, last_frozen_month
from resolver.hazard_resolution.rulebook import (
    HAZARD_CODE_BY_RULEBOOK_NAME,
    Rulebook,
)
from resolver.hazard_resolution.schema import RUN_TYPE_BACKCAST, ensure_haz_schema

LOG = logging.getLogger(__name__)

_HAZARD_ALIASES = {
    "cyclone": "cyclone",
    "tc": "cyclone",
    "flood": "flood",
    "fl": "flood",
    "drought": "drought",
    "dr": "drought",
}


def months_between(first_ym: str, last_ym: str) -> list[str]:
    """Every ``YYYY-MM`` from ``first_ym`` to ``last_ym``, inclusive."""

    first = tuple(int(p) for p in first_ym.split("-"))
    last = tuple(int(p) for p in last_ym.split("-"))
    if last < first:
        return []
    out: list[str] = []
    year, month = first
    while (year, month) <= last:
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return out


def plan_months(
    hazard_name: str,
    rulebook: Rulebook,
    *,
    from_ym: str | None = None,
    to_ym: str | None = None,
    today: dt.date | None = None,
) -> list[str]:
    """The months a backcast of ``hazard_name`` covers, oldest first.

    ``--from`` / ``--to`` may narrow the rulebook's window (to resume by
    hand, or to split a long run) but never widen it: the start year is a
    statement about when the detector's record begins, and the end is the
    freeze boundary. Silently honouring a wider request would produce
    months whose zeros the coverage gate has to suppress anyway, plus
    months that can still change.
    """

    hazard = HAZARD_CODE_BY_RULEBOOK_NAME[hazard_name]
    window_first, window_last = backcast_window(hazard, rulebook, today=today)

    first = max(from_ym, window_first) if from_ym else window_first
    last = min(to_ym, window_last) if to_ym else window_last
    if from_ym and from_ym < window_first:
        LOG.warning(
            "[backcast] --from %s precedes backcast.%s (%s) — starting at %s",
            from_ym, hazard_name, window_first, window_first,
        )
    if to_ym and to_ym > window_last:
        LOG.warning(
            "[backcast] --to %s is past the last frozen month (%s) — stopping at %s",
            to_ym, window_last, window_last,
        )
    return months_between(first, last)


@dataclass
class BackcastableCheck:
    """Whether a hazard can produce meaningful backcast answers at all."""

    ok: bool
    warnings: list[str] = field(default_factory=list)


def check_backcastable(hazard_name: str, rulebook: Rulebook) -> BackcastableCheck:
    """Pre-flight warnings a human should see before a multi-hour run.

    This never blocks a run — the machine's own gates are already
    fail-closed, and an operator may legitimately want the months that DO
    work. It exists so a run that will mostly produce INCONCLUSIVE is
    announced as such at the start rather than discovered at the end.
    """

    warnings: list[str] = []
    if hazard_name == "drought":
        entries = [
            entry
            for entry in (rulebook.get("drought.indicators.entries", []) or [])
            if isinstance(entry, Mapping)
        ]

        def _addresses(entry: Mapping[str, Any]) -> list[str]:
            out = [str(u) for u in (entry.get("urls") or []) if str(u).strip()]
            single = str(entry.get("url") or "").strip()
            if single:
                out.append(single)
            return out

        # An HTTP feed serving only a "latest" snapshot cannot speak for a
        # past month: it is stamped with its RETRIEVAL month and correctly
        # falls outside the observation-age window.
        undated = [
            str(entry.get("name"))
            for entry in entries
            if str(entry.get("provider")) != "pythia_table"
            and _addresses(entry)
            and not any(
                token in address
                for address in _addresses(entry)
                for token in ("{ym}", "{year}", "{month}")
            )
        ]
        # A table this repo ingests carries its own dates per row, so it CAN
        # answer for a past month — as far back as the ingest goes.
        dated = [
            str(entry.get("name"))
            for entry in entries
            if str(entry.get("provider")) == "pythia_table"
            and str(entry.get("date_column") or "").strip()
        ]

        if undated:
            warnings.append(
                "drought indicator(s) "
                + ", ".join(undated)
                + " point at a LATEST snapshot with no per-month archive, so they "
                "cannot speak for a backcast month: their readings fall outside "
                f"drought.indicators.max_observation_age_months="
                f"{rulebook.get('drought.indicators.max_observation_age_months')} "
                "(fail-closed, not a false zero). Point their url at a per-month "
                "archive using {ym}/{year}/{month} to backcast on them."
            )
        if undated and not dated:
            warnings.append(
                "no drought indicator can speak for a past month, so EVERY "
                "backcast month will resolve INCONCLUSIVE. Configure a "
                "pythia_table entry with a date_column, or a per-month archive "
                "url, before spending hours on this run."
            )
        elif dated:
            warnings.append(
                "drought backcast will rest on "
                + ", ".join(dated)
                + ", which carry their own dates — so it reaches back only as far "
                "as those tables were ingested, and earlier months will resolve "
                "INCONCLUSIVE rather than zero."
            )
    return BackcastableCheck(ok=True, warnings=warnings)


# ---------------------------------------------------------------------------
# Progress ledger
# ---------------------------------------------------------------------------


def completed_months(con, hazard: str) -> set[str]:
    """Months this hazard's backcast has already finished successfully."""

    ensure_haz_schema(con)
    return {
        str(row[0])
        for row in con.execute(
            "SELECT ym FROM haz_backcast_progress WHERE hazard = ? AND status = 'ok'",
            [hazard],
        ).fetchall()
    }


def record_month(
    con,
    *,
    hazard: str,
    ym: str,
    status: str,
    counts: dict[str, Any] | None = None,
    duration_sec: float | None = None,
    error: str | None = None,
) -> None:
    """Upsert one month's outcome into the resume ledger."""

    ensure_haz_schema(con)
    counts = counts or {}
    try:
        con.execute("BEGIN TRANSACTION")
        con.execute(
            "DELETE FROM haz_backcast_progress WHERE hazard = ? AND ym = ?",
            [hazard, ym],
        )
        con.execute(
            """
            INSERT INTO haz_backcast_progress
                (hazard, ym, status, cells, resolved_value, resolved_zero,
                 no_data, frozen_skipped, extraction_calls, extraction_cost_usd,
                 duration_sec, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                hazard,
                ym,
                status,
                int(counts.get("cells") or 0),
                int(counts.get("resolved_value") or 0),
                int(counts.get("resolved_zero") or 0),
                int(counts.get("no_data") or 0),
                int(counts.get("frozen_skipped") or 0),
                int(counts.get("extraction_calls") or 0),
                float(counts.get("extraction_cost_usd") or 0.0),
                duration_sec,
                error,
            ],
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def month_counts(con, hazard: str, ym: str) -> dict[str, Any]:
    """What the resolution tables now hold for this cell-month.

    Read back from the tables rather than threaded out of the run objects:
    the cyclone, flood and drought runners return three different shapes,
    and the ledger should record what was actually written either way.

    The extraction and revision numbers are cumulative for the TARGET
    month, which for a backcast month is the same thing as "this run" —
    the resume ledger means each month is walked once, and frozen history
    has no live-run extractions to blur into. (Before this read-back the
    two extraction columns in ``haz_backcast_progress`` were always zero:
    ``record_month`` read keys ``month_counts`` never produced.)
    """

    year, month = (int(p) for p in ym.split("-"))
    rows = con.execute(
        """
        SELECT status, COUNT(*) FROM haz_resolutions
        WHERE hazard = ? AND year = ? AND month = ?
        GROUP BY status
        """,
        [hazard, year, month],
    ).fetchall()
    by_status = {str(status): int(n) for status, n in rows}
    cells = int(
        con.execute(
            "SELECT COUNT(*) FROM haz_triggers WHERE hazard = ? AND year = ? AND month = ?",
            [hazard, year, month],
        ).fetchone()[0]
    )
    extraction = con.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(cost_usd), 0.0)
        FROM haz_doc_extractions
        WHERE hazard = ? AND year = ? AND month = ?
          AND (status = 'ok'
               OR COALESCE(prompt_tokens, 0) + COALESCE(completion_tokens, 0) > 0)
        """,
        [hazard, year, month],
    ).fetchone()
    frozen_skipped = int(
        con.execute(
            "SELECT COUNT(*) FROM haz_revisions WHERE hazard = ? AND year = ? AND month = ?",
            [hazard, year, month],
        ).fetchone()[0]
    )
    return {
        "cells": cells,
        "resolved_value": by_status.get("RESOLVED_VALUE", 0),
        "resolved_zero": by_status.get("RESOLVED_ZERO", 0),
        "no_data": by_status.get("NO_DATA", 0),
        "frozen_skipped": frozen_skipped,
        "extraction_calls": int(extraction[0] or 0),
        "extraction_cost_usd": float(extraction[1] or 0.0),
    }


def month_is_complete(counts: Mapping[str, Any]) -> tuple[bool, str]:
    """Did this month actually finish, or did it merely not crash?

    The runner exits 0 when it walked every cell without raising — which it
    does even when a required source was unreadable and every cell came back
    INCONCLUSIVE. Recording that as complete is worse than a failure: the
    resume ledger then never retries the month, so a source outage becomes
    permanently missing history rather than a temporary gap. The August 2026
    ASAP outage did exactly this to twelve months of drought.

    A month with cells assessed and NOTHING written is therefore incomplete,
    whatever the exit code. A month with no cells to assess is complete —
    there was nothing to do.
    """

    cells = int(counts.get("cells") or 0)
    written = (
        int(counts.get("resolved_value") or 0)
        + int(counts.get("resolved_zero") or 0)
        + int(counts.get("no_data") or 0)
        + int(counts.get("frozen_skipped") or 0)
    )
    if cells and not written:
        return False, (
            f"{cells} cell(s) assessed and no row written — a source was "
            "unreadable, so the month is left incomplete and will be retried"
        )
    return True, ""


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass
class BackcastRun:
    """What one backcast pass did."""

    hazard: str
    hazard_name: str
    months_planned: int = 0
    months_run: int = 0
    months_skipped_done: int = 0
    months_failed: int = 0
    #: Months left unrun because the time budget was spent — not failures:
    #: the resume ledger picks them up on the next run.
    months_deferred: int = 0
    resolved_value: int = 0
    resolved_zero: int = 0
    no_data: int = 0
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> int:
        return self.resolved_value + self.resolved_zero


def _prefetch_ibtracs(con, rulebook: Rulebook) -> bool:
    """Load the full IBTrACS archive once for the whole backcast.

    One ~200 MB file covers every year, so downloading it per month would
    dominate the run. Returns whether the store is usable — a failed fetch
    on top of an empty store means the cyclone backcast has no detector and
    should not pretend otherwise.
    """

    from resolver.hazard_resolution import ibtracs as ibtracs_mod

    try:
        frame, url = ibtracs_mod.fetch_ibtracs(rulebook, scope="ALL")
        ibtracs_mod.store_ibtracs(con, frame, "ALL", url)
        return True
    except Exception as exc:  # noqa: BLE001 - an outage is data, not a crash
        summary = ibtracs_mod.store_summary(con)
        if summary["total_storms"] > 0:
            LOG.error(
                "[backcast] IBTrACS ALL fetch failed (%s) — continuing on the "
                "existing store (%d storms); months outside its coverage will "
                "have their zeros suppressed by the coverage gate",
                exc, summary["total_storms"],
            )
            return True
        LOG.error("[backcast] IBTrACS ALL fetch failed and the store is empty: %s", exc)
        return False


def _log_scale_estimate(hazard_name: str, months: list[str], no_extract: bool) -> None:
    if not months:
        return
    LOG.info(
        "[backcast] %s: %d months (%s .. %s). ReliefWeb silence sweeps run once "
        "per non-triggered country-month and are paced by the rulebook, so this "
        "is a long run — expect hours, not minutes.",
        hazard_name, len(months), months[0], months[-1],
    )
    if hazard_name != "drought":
        LOG.info(
            "[backcast] ladder rung 2 is %s. When on, spend is bounded by "
            "extraction.max_calls_per_month across ALL runs in a calendar month, "
            "so a long backcast accrues extraction gradually rather than at once.",
            "OFF (--no-extract)" if no_extract else "ON",
        )


def run_backcast(
    *,
    hazard_name: str,
    db_url: str | None = None,
    countries_filter: list[str] | None = None,
    from_ym: str | None = None,
    to_ym: str | None = None,
    no_extract: bool = False,
    no_sweep: bool = False,
    no_ladder: bool = False,
    dry_run: bool = False,
    resume: bool = True,
    time_budget_min: float | None = None,
    today: dt.date | None = None,
    rulebook: Rulebook | None = None,
    con=None,
    runner: Callable[..., int] | None = None,
) -> BackcastRun:
    """Replay one hazard over its full backcast window.

    ``time_budget_min`` bounds the run's wall clock: once spent, the run
    stops CLEANLY between months (never mid-month) and the remaining months
    are deferred, not failed — the resume ledger continues exactly where
    this run stopped. This is what lets a scheduled job chew through a
    multi-decade backcast one bounded chunk at a time and converge.
    """

    from resolver.db.duckdb_io import get_db
    from resolver.hazard_resolution import cli as cli_mod
    from resolver.hazard_resolution.rulebook import load_rulebook

    rulebook = rulebook or load_rulebook()
    con = con if con is not None else get_db(db_url)
    ensure_haz_schema(con)

    hazard = HAZARD_CODE_BY_RULEBOOK_NAME[hazard_name]
    run = BackcastRun(hazard=hazard, hazard_name=hazard_name)

    check = check_backcastable(hazard_name, rulebook)
    run.warnings = list(check.warnings)
    for warning in check.warnings:
        LOG.warning("[backcast] %s", warning)

    months = plan_months(
        hazard_name, rulebook, from_ym=from_ym, to_ym=to_ym, today=today
    )
    run.months_planned = len(months)
    if not months:
        LOG.warning(
            "[backcast] %s: no months in range (backcast.%s=%s, last frozen month=%s)",
            hazard_name, hazard_name, rulebook.get(f"backcast.{hazard_name}"),
            last_frozen_month(rulebook, today=today),
        )
        return run

    already = completed_months(con, hazard) if resume else set()
    todo = [ym for ym in months if ym not in already]
    run.months_skipped_done = len(months) - len(todo)
    if run.months_skipped_done:
        LOG.info(
            "[backcast] %s: %d of %d months already completed — resuming at %s "
            "(--no-resume to redo them)",
            hazard_name, run.months_skipped_done, len(months),
            todo[0] if todo else "(nothing left)",
        )
    _log_scale_estimate(hazard_name, todo, no_extract)

    # The cyclone detector's archive is one file for all of history.
    skip_detector_fetch = False
    if hazard_name == "cyclone" and todo and not dry_run:
        if not _prefetch_ibtracs(con, rulebook):
            return run
        skip_detector_fetch = True

    common: dict[str, Any] = dict(
        db_url=db_url,
        countries_filter=countries_filter,
        skip_fetch=False,
        no_sweep=no_sweep,
        no_ladder=no_ladder,
        no_extract=no_extract,
        dry_run=dry_run,
        run_type=RUN_TYPE_BACKCAST,
        rulebook=rulebook,
        con=con,
    )

    deadline = None
    if time_budget_min is not None and float(time_budget_min) > 0:
        deadline = time.monotonic() + float(time_budget_min) * 60.0

    for index, ym in enumerate(todo, start=1):
        if deadline is not None and time.monotonic() >= deadline:
            run.months_deferred = len(todo) - index + 1
            LOG.info(
                "[backcast] %s: time budget of %.0f min spent — %d month(s) "
                "deferred to the next run (the resume ledger continues from %s)",
                hazard_name, float(time_budget_min), run.months_deferred, ym,
            )
            break
        started = time.monotonic()
        LOG.info("[backcast] %s %s (%d/%d)", hazard_name, ym, index, len(todo))
        try:
            if runner is not None:
                rc = runner(ym=ym, **common)
            elif hazard_name == "cyclone":
                rc = cli_mod.run_cyclone_month(
                    ym=ym, scope="ALL", skip_detector_fetch=skip_detector_fetch, **common
                )
            elif hazard_name == "drought":
                rc = cli_mod.run_drought_month(ym=ym, **common)
            else:
                rc = cli_mod.run_flood_month(ym=ym, **common)
            error = None if rc == 0 else f"month runner exited {rc}"
        except Exception as exc:  # noqa: BLE001 - one bad month must not end the run
            rc, error = 1, f"{type(exc).__name__}: {exc}"
            LOG.exception("[backcast] %s %s raised", hazard_name, ym)

        duration = time.monotonic() - started
        counts = {} if dry_run else month_counts(con, hazard, ym)
        if rc == 0:
            run.months_run += 1
            run.resolved_value += int(counts.get("resolved_value") or 0)
            run.resolved_zero += int(counts.get("resolved_zero") or 0)
            run.no_data += int(counts.get("no_data") or 0)
        else:
            run.months_failed += 1
            run.failures.append(f"{ym}: {error}")
            LOG.error("[backcast] %s %s failed: %s", hazard_name, ym, error)

        # A dry run decides nothing and must not claim a month is done.
        if not dry_run:
            complete, incomplete_reason = month_is_complete(counts)
            if rc == 0 and not complete:
                # Exit code 0 only says the runner did not raise. A month that
                # assessed cells and wrote nothing is a source outage, and
                # marking it done would make that outage permanent.
                rc = 1
                error = incomplete_reason
                run.months_run -= 1
                run.months_failed += 1
                run.failures.append(f"{ym}: {error}")
                LOG.error("[backcast] %s %s: %s", hazard_name, ym, error)
            record_month(
                con,
                hazard=hazard,
                ym=ym,
                status="ok" if rc == 0 else "failed",
                counts=counts,
                duration_sec=duration,
                error=error,
            )

    LOG.info(
        "[backcast] %s finished: %d months run, %d already done, %d failed, "
        "%d deferred | %d values, %d zeros, %d no-data",
        hazard_name, run.months_run, run.months_skipped_done, run.months_failed,
        run.months_deferred, run.resolved_value, run.resolved_zero, run.no_data,
    )
    if run.failures:
        LOG.error(
            "[backcast] %d month(s) failed and were NOT marked complete — re-run "
            "to retry them: %s",
            len(run.failures), "; ".join(run.failures[:10]),
        )
    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="haz-backcast",
        description=(
            "Replay the resolution machine over history with run_type='backcast'"
        ),
    )
    parser.add_argument(
        "--hazard",
        required=True,
        choices=sorted(_HAZARD_ALIASES),
        type=str.lower,
        help="Hazard to backcast: cyclone/tc, flood/fl or drought/dr",
    )
    parser.add_argument("--db", default=os.getenv("RESOLVER_DB_URL", ""))
    parser.add_argument(
        "--from", dest="from_ym", default=None, metavar="YYYY-MM",
        help="Start later than the rulebook's backcast year (never earlier)",
    )
    parser.add_argument(
        "--to", dest="to_ym", default=None, metavar="YYYY-MM",
        help="Stop earlier than the last frozen month (never later)",
    )
    parser.add_argument("--countries", nargs="*", default=None, metavar="ISO3")
    parser.add_argument(
        "--no-extract", action="store_true",
        help="Skip ladder rung 2 — no model is called and nothing is spent",
    )
    parser.add_argument("--no-sweep", action="store_true", help="Skip silence sweeps (no zeros)")
    parser.add_argument("--no-ladder", action="store_true", help="Detection and zeros only")
    parser.add_argument(
        "--no-resume", action="store_true",
        help=(
            "Re-walk months already recorded complete. Every frozen cell they "
            "re-decide logs a haz_revisions row, so use this deliberately"
        ),
    )
    parser.add_argument(
        "--time-budget-min", type=float, default=None, metavar="MIN",
        help=(
            "Stop cleanly between months once this many minutes have elapsed; "
            "remaining months are deferred (exit 0) and the resume ledger "
            "continues on the next run. For scheduled chunked backcasts"
        ),
    )
    parser.add_argument(
        "--summary-out", default=None, metavar="PATH",
        help="Write a JSON summary of this run (months run/deferred/failed, counts) to PATH",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Plan and log the months without writing anything",
    )
    parser.add_argument("--log-level", default=os.getenv("RESOLVER_LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    run = run_backcast(
        hazard_name=_HAZARD_ALIASES[args.hazard],
        db_url=args.db or None,
        countries_filter=args.countries,
        from_ym=args.from_ym,
        to_ym=args.to_ym,
        no_extract=args.no_extract,
        no_sweep=args.no_sweep,
        no_ladder=args.no_ladder,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        time_budget_min=args.time_budget_min,
    )

    print(
        f"[backcast] {run.hazard_name}: {run.months_run} months run, "
        f"{run.months_skipped_done} already complete, {run.months_failed} failed, "
        f"{run.months_deferred} deferred"
    )
    print(
        f"[backcast] wrote {run.resolved_value} values, {run.resolved_zero} zeros, "
        f"{run.no_data} no-data"
    )
    for warning in run.warnings:
        print(f"[backcast] WARNING: {warning}")

    if args.summary_out:
        # The durable diagnostic — a summary that exists only in a job log
        # does not exist. Never fatal.
        try:
            import json
            from pathlib import Path

            payload = {
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "hazard": run.hazard,
                "hazard_name": run.hazard_name,
                "months_planned": run.months_planned,
                "months_run": run.months_run,
                "months_already_complete": run.months_skipped_done,
                "months_failed": run.months_failed,
                "months_deferred": run.months_deferred,
                "resolved_value": run.resolved_value,
                "resolved_zero": run.resolved_zero,
                "no_data": run.no_data,
                "failures": run.failures,
                "warnings": run.warnings,
            }
            Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.summary_out, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            LOG.info("[backcast] run summary written to %s", args.summary_out)
        except Exception as exc:  # noqa: BLE001 - diagnostics never fail a run
            LOG.warning(
                "[backcast] could not write the run summary to %s: %s",
                args.summary_out, exc,
            )

    return 1 if run.months_failed else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
