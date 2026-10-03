# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""On what day of the month does ICG publish each CrisisWatch edition?

The monthly forecast cycle is scheduled around this answer: a forecast that
runs before ICG publishes last month's edition reads an edition a month older
than it needs to. This script measures the day from the Wayback Machine, two
independent ways, for every edition in a range:

1. ``edition_page`` -- the first capture of the edition's own page,
   ``crisisgroup.org/crisiswatch/<month>-trends-and-<next month>-alerts-<year>``
   (both year spellings, since the December edition's alerts month is in the
   next year).
2. ``main_page`` -- the first capture of ``crisisgroup.org/crisiswatch`` whose
   parsed edition IS this edition, together with the last capture before it
   that still showed an older edition. Those two captures bracket the day
   ICG switched the page.

A first capture is an UPPER bound on the publication day: ICG may have
published earlier and nobody crawled it until later. The bracket's earlier
side is a LOWER bound. Read the table that way.

It writes nothing and always exits 0. Only a runner can reach archive.org,
so it runs in CI (``refresh-crisiswatch.yml``'s ``measure_publication_days``
input).
"""

from __future__ import annotations

import argparse
import calendar
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

log = logging.getLogger("crisiswatch_publication_days")

#: The schedule assumes ICG has published by this day of the month.
DEFAULT_LATEST_ACCEPTABLE_DAY = 9

_CDX_URL = "https://web.archive.org/cdx/search/cdx"

# (url_or_prefix, from, to, match_type) -> [(timestamp, statuscode), ...]
CdxFn = Callable[[str, str, str, str], list[tuple[str, str]]]
# timestamp -> (year, month) of the edition that capture shows, or None
EditionFn = Callable[[str], "tuple[int, int] | None"]


def add_months(year: int, month: int, delta: int) -> tuple[int, int]:
    index = year * 12 + (month - 1) + delta
    return index // 12, index % 12 + 1


def edition_slugs(year: int, month: int) -> list[str]:
    """The per-edition page paths ICG might use for edition ``year-month``."""
    ny, nm = add_months(year, month, 1)
    a = calendar.month_name[month].lower()
    b = calendar.month_name[nm].lower()
    years = [year] if ny == year else [year, ny]
    return [f"crisisgroup.org/crisiswatch/{a}-trends-and-{b}-alerts-{y}" for y in years]


def _window(year: int, month: int, months: int) -> tuple[str, str]:
    """CDX bounds from the first day of ``year-month`` for ``months`` months."""
    ey, em = add_months(year, month, months - 1)
    last = calendar.monthrange(ey, em)[1]
    return f"{year:04d}{month:02d}01", f"{ey:04d}{em:02d}{last:02d}"


@dataclass
class EditionDay:
    edition: str
    edition_page_first: str | None = None
    edition_page_url: str | None = None
    main_page_first: str | None = None
    main_page_last_older: str | None = None
    notes: list[str] = field(default_factory=list)

    @staticmethod
    def _day(ts: str | None, edition: str) -> int | None:
        """Day of month of a capture in the month AFTER the edition; else None."""
        if not ts:
            return None
        y, m = (int(x) for x in edition.split("-"))
        ny, nm = add_months(y, m, 1)
        if int(ts[:4]) == ny and int(ts[4:6]) == nm:
            return int(ts[6:8])
        # Captured later than the following month: an upper bound past the
        # month's end. Report it as a day beyond any month so it fails a gate.
        if (int(ts[:4]), int(ts[4:6])) > (ny, nm):
            return 99
        return None

    @property
    def upper_bound_day(self) -> int | None:
        """Earliest day either route proves the edition was live."""
        days = [
            d for d in (
                self._day(self.edition_page_first, self.edition),
                self._day(self.main_page_first, self.edition),
            ) if d is not None
        ]
        return min(days) if days else None

    @property
    def lower_bound_day(self) -> int | None:
        return self._day(self.main_page_last_older, self.edition)


def first_capture(cdx: CdxFn, urls: Iterable[str], start: str, end: str) -> tuple[str | None, str | None]:
    best: tuple[str, str] | None = None
    for url in urls:
        for match_type in ("exact", "prefix"):
            for ts, status in cdx(url, start, end, match_type):
                if not status.startswith(("2", "3")):
                    continue
                if best is None or ts < best[0]:
                    best = (ts, url)
            if best and best[1] == url:
                break
    return (best[0], best[1]) if best else (None, None)


def measure_edition(year: int, month: int, *, cdx: CdxFn, edition_of: EditionFn) -> EditionDay:
    row = EditionDay(edition=f"{year:04d}-{month:02d}")
    ny, nm = add_months(year, month, 1)
    start, end = _window(ny, nm, 2)

    row.edition_page_first, row.edition_page_url = first_capture(
        cdx, edition_slugs(year, month), f"{year:04d}{month:02d}01", end,
    )
    if row.edition_page_first is None:
        row.notes.append("no capture of the per-edition page")

    # Main page: walk the following two months oldest first. The first
    # capture showing this edition is the upper bound; the newest capture
    # before it that showed an older edition is the lower bound.
    timestamps = sorted(
        ts for ts, status in cdx("crisisgroup.org/crisiswatch", start, end, "exact")
        if status == "200"
    )
    target = (year, month)
    last_older: str | None = None
    for ts in timestamps:
        shown = edition_of(ts)
        if shown is None:
            continue
        if shown == target:
            row.main_page_first = ts
            break
        if shown < target:
            last_older = ts
        else:
            row.notes.append(f"{ts} already shows a later edition {shown[0]}-{shown[1]:02d}")
            break
    row.main_page_last_older = last_older if row.main_page_first else None
    if row.main_page_first is None:
        row.notes.append("no main-page capture showed this edition")
    return row


def editions_in(start: str, end: str) -> list[tuple[int, int]]:
    y, m = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append((y, m))
        y, m = add_months(y, m, 1)
    return out


def verdict(rows: list[EditionDay], latest_day: int) -> dict:
    late = [r.edition for r in rows if r.upper_bound_day is not None and r.upper_bound_day > latest_day]
    unknown = [r.edition for r in rows if r.upper_bound_day is None]
    return {
        "latest_acceptable_day": latest_day,
        "late_editions": late,
        "unmeasured_editions": unknown,
        "ok": not late,
    }


def render(rows: list[EditionDay], v: dict) -> str:
    def fmt(ts: str | None) -> str:
        return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if ts else "-"

    lines = [
        "| edition | edition page first capture | main page: last older | main page: first showing it | day (upper bound) | notes |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.edition} | {fmt(r.edition_page_first)} | {fmt(r.main_page_last_older)} "
            f"| {fmt(r.main_page_first)} | {r.upper_bound_day if r.upper_bound_day is not None else '-'} "
            f"| {'; '.join(r.notes)} |"
        )
    lines.append("")
    lines.append(
        "A first capture is an UPPER bound on the publication day; ICG may have "
        "published earlier. The 'last older' column is a LOWER bound."
    )
    lines.append("")
    if v["ok"]:
        lines.append(f"VERDICT: every measured edition was live by day {v['latest_acceptable_day']}.")
    else:
        lines.append(
            f"VERDICT: STOP. First seen after day {v['latest_acceptable_day']}: "
            + ", ".join(v["late_editions"])
        )
    if v["unmeasured_editions"]:
        lines.append("Not measured (no capture found): " + ", ".join(v["unmeasured_editions"]))
    return "\n".join(lines)


# --------------------------------------------------------------- live I/O


def _live_cdx(url: str, start: str, end: str, match_type: str) -> list[tuple[str, str]]:
    import requests  # noqa: PLC0415

    params = {
        "url": url, "output": "json", "from": start, "to": end,
        "fl": "timestamp,statuscode", "matchType": match_type, "limit": "500",
    }
    for attempt in range(3):
        try:
            resp = requests.get(
                _CDX_URL, params=params, timeout=60,
                headers={"User-Agent": "PythiaCrisisWatchRefresh/1.0 (+https://github.com/kwyjad/Pythia)"},
            )
            if resp.status_code == 200:
                rows = resp.json() if resp.text.strip() else []
                return [(r[0], r[1]) for r in rows[1:] if len(r) >= 2]
            log.warning("CDX %s answered %d", url, resp.status_code)
        except Exception as exc:  # noqa: BLE001
            log.warning("CDX %s failed: %s", url, exc)
        import time  # noqa: PLC0415

        time.sleep(10 * (attempt + 1))
    return []


def _live_edition_of(ts: str) -> tuple[int, int] | None:
    from scripts import refresh_crisiswatch as rc  # noqa: PLC0415

    html = rc._download_snapshot_html(ts, timeout_sec=90, max_attempts=2, backoff_sec=10)
    if not html or not rc._accept_snapshot_html(html, ts):
        return None
    try:
        data = rc.parse_edition(html, provenance=f"wayback:{ts}")
    except Exception:  # noqa: BLE001
        return None
    return rc._edition_key(data.get("month", ""), int(data.get("year") or 0))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--from-edition", default="2025-09")
    p.add_argument("--to-edition", default="2026-09")
    p.add_argument("--latest-day", type=int, default=DEFAULT_LATEST_ACCEPTABLE_DAY)
    p.add_argument("--out", default="diagnostics/crisiswatch_publication_days.json")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows: list[EditionDay] = []
    try:
        for y, m in editions_in(args.from_edition, args.to_edition):
            rows.append(measure_edition(y, m, cdx=_live_cdx, edition_of=_live_edition_of))
            log.info("measured %s: %s", rows[-1].edition, asdict(rows[-1]))
    except Exception as exc:  # noqa: BLE001 - a diagnostic never fails its job
        log.error("measurement stopped: %s", exc)
    v = verdict(rows, args.latest_day)
    text = render(rows, v)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("## CrisisWatch publication days\n\n" + text + "\n")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": [asdict(r) | {
        "upper_bound_day": r.upper_bound_day, "lower_bound_day": r.lower_bound_day,
    } for r in rows], "verdict": v}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
