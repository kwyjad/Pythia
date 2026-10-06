# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""IDMC conflict displacement: the series ACE/PA questions resolve against.

Until Oct 2026 every IDMC row reached ``facts_resolved`` as hazard ``IDU``,
whatever had displaced the people: ``resolver/ingestion/idmc/normalize.py``
summed every Internal Displacement Update in a country-month before anything
looked at its ``displacement_type``, and the adapter stamped the sum ``IDU``.
So China's 7.3 million typhoon evacuees in July 2026 and the Philippines'
1.7 million in September were the "conflict displacement" history every
ACE/PA prompt showed as THIS QUESTION'S SERIES, and no ACE/PA question ever
resolved, because the resolver looked for hazard ``ACE``.

This module reads the IDU ``all`` route (the one the PA machine's lower-bound
rung already reads, through the same transport and the same credential),
keeps each record whole, and splits it by ``displacement_type``:

* ``Conflict`` records are summed per country and per calendar month into
  the conflict displacement series, written as hazard ``ACE``, metric
  ``new_displacements``, ``series_semantics = 'new'``, publisher IDMC.
* ``Disaster``, ``Other`` and untyped records never enter it. They are
  counted, by cause, in people and in records, and the count is written to
  ``diagnostics/ingestion/idmc/conflict_exclusions.json``.

A record's figure is attributed WHOLLY to the month its displacement started
(the rule the PA machine applies to every event figure): a source states one
number for an event, and apportioning it across months invents a breakdown
nobody reported.

Three kinds of figure never enter the monthly sum (Oct 2026):

* a record IDMC does not recommend for totals (its ``role`` is a
  triangulation or a duplicate of another figure), counted and dropped;
* a record whose displacement spans more than ``MAX_SPAN_DAYS`` days. It
  is an aggregate over a period, and putting all of it in its start month
  invents a spike, while apportioning it invents a breakdown nobody
  reported. It is HELD OUT instead: every month its span touches is written
  under ``METRIC_HELD``, so no reader can take that month for a quiet one;
* a country-month total above the country's population, which cannot be a
  monthly count of people (PSE 2023-10 read 54,037,759). Held out the same
  way, and named in the run's diagnostics.

The month still in progress is never written: a flow for a month that has
not ended is a partial count, and a partial count read as a month is the
1 August 2026 ACLED fault.

Readers: one, :mod:`pythia.tools.base_rate_spd`'s conflict displacement
functions, which serve the prompt block, the ACE/PA anchor, the climatology
and persistence references and ``compute_resolutions``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

LOG = logging.getLogger(__name__)

#: The IDU route that serves the whole history. The PA machine reads it from
#: the rulebook (``idmc_idu.api_url``); this module reads the same address.
DEFAULT_IDU_ALL_URL = "https://helix-tools-api.idmcdb.org/external-api/idus/all/"
URL_ENV = "IDMC_IDU_ALL_URL"
MONTHS_ENV = "IDMC_CONFLICT_MONTHS"
#: How far back each run rewrites the series. 36 months is the window the
#: ACE/PA anchor reads (``base_rate_spd.CONFLICT_WINDOW_MONTHS``); a run writes
#: whatever the route serves inside it, and says how far back that was.
DEFAULT_MONTHS = 36
REQUEST_TIMEOUT_SEC = 300.0

#: The hazard, metric and series the conflict displacement rows are written
#: under. ``pythia.tools.base_rate_spd`` imports nothing from here (the
#: resolver package must not depend on the API's import chain), so the two
#: are pinned equal by forecaster/tests/test_base_rate_matches_resolution_source.py.
HAZARD_CODE = "ACE"
HAZARD_LABEL = "Armed conflict — internal displacement"
HAZARD_CLASS = "human-induced"
METRIC = "new_displacements"
#: Months that carry a figure IDMC reported but this series will not use as a
#: monthly count (a long span, or a total above the population). Read by
#: ``pythia.tools.base_rate_spd`` as UNKNOWN: neither a value nor a quiet month.
METRIC_HELD = "new_displacements_held"
SERIES_SEMANTICS = "new"
#: A displacement span longer than this is a period aggregate, not an event.
MAX_SPAN_DAYS = 31
#: ``role`` values that mark a figure IDMC recommends for totals. A record
#: carrying any other role (triangulation, duplicate) is dropped and counted.
RECOMMENDED_ROLE_PREFIX = "recommended"
POPULATION_CSV = Path("resolver/data/population.csv")
SOURCE = "IDMC"

#: Every value of ``displacement_type`` seen, mapped to the cause it names.
CAUSE_CONFLICT = "conflict"
CAUSE_DISASTER = "disaster"
CAUSE_OTHER = "other"
CAUSE_UNTYPED = "untyped"

STAGING_DIR = Path("resolver/staging/idmc")
DIAGNOSTICS_DIR = Path("diagnostics/ingestion/idmc")

#: Columns of the staging file the IDMC adapter reads. ``displacement_type``
#: is what the adapter keys on: a file without it (the HELIX path's all-cause
#: output) is refused there rather than written as conflict.
STAGING_COLUMNS = (
    "iso3", "as_of_date", "metric", "value", "series_semantics", "source",
    "displacement_type",
)

GetFn = Callable[[str, dict, float], list]

_FIGURE_FIELDS = ("figure", "total_figures", "displacement_figure", "new_displacements")
_START_FIELDS = (
    "displacement_start_date", "event_start_date", "displacement_date", "event_date",
)
_END_FIELDS = ("displacement_end_date", "event_end_date")


def cause_of(record: Mapping[str, Any]) -> str:
    """The cause an IDU record states: conflict, disaster, other or untyped."""

    value = str(record.get("displacement_type") or "").strip().lower()
    if not value:
        return CAUSE_UNTYPED
    if value.startswith("conflict"):
        return CAUSE_CONFLICT
    if value.startswith("disaster"):
        return CAUSE_DISASTER
    return CAUSE_OTHER


def _figure(record: Mapping[str, Any]) -> float | None:
    for field in _FIGURE_FIELDS:
        value = record.get(field)
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(str(value).replace(",", "").strip())
        except (TypeError, ValueError):
            continue
        if number < 0:
            return None
        return number
    return None


def _start_month(record: Mapping[str, Any]) -> str | None:
    for field in _START_FIELDS:
        value = record.get(field)
        if not value:
            continue
        stamp = pd.to_datetime(str(value)[:10], errors="coerce")
        if pd.isna(stamp):
            continue
        return stamp.strftime("%Y-%m")
    return None


def _date_field(record: Mapping[str, Any], fields: tuple[str, ...]) -> dt.date | None:
    for field in fields:
        value = record.get(field)
        if not value:
            continue
        stamp = pd.to_datetime(str(value)[:10], errors="coerce")
        if not pd.isna(stamp):
            return stamp.date()
    return None


def span_months(start: dt.date, end: dt.date) -> list[str]:
    """Every calendar month from ``start`` to ``end`` inclusive."""
    out: list[str] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


def load_population(path: Path = POPULATION_CSV) -> dict[str, float]:
    """ISO3 -> population from ``resolver/data/population.csv`` ({} on any
    failure: no cap is applied rather than a wrong one)."""
    try:
        frame = pd.read_csv(path, dtype=str)
    except Exception:  # noqa: BLE001
        return {}
    frame.columns = [c.strip() for c in frame.columns]
    out: dict[str, float] = {}
    for iso3, raw in zip(frame.get("iso3", []), frame.get("population", [])):
        try:
            out[str(iso3).strip().upper()] = float(str(raw).replace(",", "").strip())
        except (TypeError, ValueError):
            continue
    return out


def _iso3(record: Mapping[str, Any]) -> str | None:
    text = str(record.get("iso3") or record.get("iso") or record.get("ISO3") or "")
    text = text.strip().upper()
    return text if len(text) == 3 and text.isalpha() else None


def month_window(today: dt.date, months: int) -> tuple[str, str]:
    """(first, last) month of the window: ``months`` months ending with the
    last COMPLETE month before ``today`` (the month in progress is excluded)."""

    first_of_this = today.replace(day=1)
    last = (first_of_this - dt.timedelta(days=1)).replace(day=1)
    year, month = last.year, last.month
    for _ in range(max(1, int(months)) - 1):
        month -= 1
        if month == 0:
            year, month = year - 1, 12
    return f"{year:04d}-{month:02d}", last.strftime("%Y-%m")


def conflict_monthly_flows(
    records: Iterable[Mapping[str, Any]], first_ym: str, last_ym: str,
    *, population: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sum the recommended conflict records per (iso3, month) in the window.

    Returns ``(frame, report)``. ``frame`` has columns ``iso3``, ``ym``,
    ``value``, ``metric``: ``METRIC`` rows are the monthly series,
    ``METRIC_HELD`` rows mark months held out (a span longer than
    ``MAX_SPAN_DAYS``, or a total above the population; ``value`` is the
    figure held). ``report`` counts every record by what happened to it.
    """

    totals: dict[tuple[str, str], float] = defaultdict(float)
    held: dict[tuple[str, str], float] = defaultdict(float)
    held_reasons: dict[tuple[str, str], set] = defaultdict(set)
    records_by_cause: Counter = Counter()
    people_by_cause: Counter = Counter()
    dropped: Counter = Counter()
    roles: Counter = Counter()
    people_not_recommended = 0.0
    long_spans: list[dict[str, Any]] = []
    months_seen: set[str] = set()
    n_records = 0
    for record in records:
        if not isinstance(record, Mapping):
            dropped["not_a_record"] += 1
            continue
        n_records += 1
        cause = cause_of(record)
        figure = _figure(record)
        records_by_cause[cause] += 1
        if figure is not None:
            people_by_cause[cause] += figure
        if cause != CAUSE_CONFLICT:
            continue
        role = record.get("role")
        roles[str(role) if role else "<absent>"] += 1
        if role and not str(role).strip().lower().startswith(RECOMMENDED_ROLE_PREFIX):
            dropped["conflict_not_recommended_role"] += 1
            people_not_recommended += figure or 0.0
            continue
        iso3 = _iso3(record)
        if iso3 is None:
            dropped["conflict_no_country"] += 1
            continue
        if figure is None:
            dropped["conflict_no_figure"] += 1
            continue
        ym = _start_month(record)
        if ym is None:
            dropped["conflict_no_date"] += 1
            continue
        months_seen.add(ym)
        start = _date_field(record, _START_FIELDS)
        end = _date_field(record, _END_FIELDS)
        if start and end and (end - start).days > MAX_SPAN_DAYS:
            spanned = [m for m in span_months(start, end) if first_ym <= m <= last_ym]
            if spanned:
                for m in spanned:
                    held[(iso3, m)] += figure
                    held_reasons[(iso3, m)].add("span")
                dropped["conflict_long_span_held"] += 1
                long_spans.append({
                    "iso3": iso3, "start": start.isoformat(), "end": end.isoformat(),
                    "figure": figure, "event_name": str(record.get("event_name") or "")[:120],
                })
            else:
                dropped["conflict_out_of_window"] += 1
            continue
        if ym < first_ym or ym > last_ym:
            dropped["conflict_out_of_window"] += 1
            continue
        totals[(iso3, ym)] += figure

    over_population: list[dict[str, Any]] = []
    if population:
        for key in sorted(totals):
            iso3, ym = key
            pop = population.get(iso3)
            if pop and totals[key] > pop:
                over_population.append({
                    "iso3": iso3, "ym": ym, "total": totals[key], "population": pop,
                })
                held[key] += totals.pop(key)
                held_reasons[key].add("above_population")

    rows = [
        {"iso3": iso3, "ym": ym, "value": value, "metric": METRIC}
        for (iso3, ym), value in sorted(totals.items())
        if (iso3, ym) not in held
    ]
    rows += [
        {"iso3": iso3, "ym": ym, "value": value, "metric": METRIC_HELD}
        for (iso3, ym), value in sorted(held.items())
    ]
    frame = pd.DataFrame(rows, columns=["iso3", "ym", "value", "metric"])
    series = frame[frame["metric"] == METRIC] if not frame.empty else frame
    in_window = sorted(m for m in months_seen if first_ym <= m <= last_ym)
    report = {
        "window": {"first": first_ym, "last": last_ym},
        "records": n_records,
        "records_by_cause": dict(records_by_cause),
        "people_by_cause": {k: float(v) for k, v in people_by_cause.items()},
        "excluded_records": {
            cause: int(n) for cause, n in records_by_cause.items() if cause != CAUSE_CONFLICT
        },
        "excluded_people": {
            cause: float(v) for cause, v in people_by_cause.items() if cause != CAUSE_CONFLICT
        },
        "conflict_roles": dict(roles),
        "conflict_people_not_recommended": float(people_not_recommended),
        "conflict_records_dropped": dict(dropped),
        "held_months": [
            {"iso3": iso3, "ym": ym, "figure": float(v), "reasons": sorted(held_reasons[(iso3, ym)])}
            for (iso3, ym), v in sorted(held.items())
        ],
        "long_spans": long_spans[:200],
        "over_population": over_population,
        "rows": int(len(series)),
        "rows_held": int(len(held)),
        "countries": int(series["iso3"].nunique()) if not series.empty else 0,
        "first_conflict_month_served": min(months_seen) if months_seen else None,
        "months_in_window_with_conflict_rows": in_window,
    }
    if not series.empty:
        per_country = series.groupby("iso3")["ym"].nunique()
        report["months_per_country"] = {
            "min": int(per_country.min()),
            "median": float(per_country.median()),
            "max": int(per_country.max()),
        }
    return frame, report


def staging_frame(flows: pd.DataFrame) -> pd.DataFrame:
    """The flows in the shape the IDMC adapter reads (month-end dates)."""

    if flows.empty:
        return pd.DataFrame(columns=list(STAGING_COLUMNS))
    month_end = (
        pd.to_datetime(flows["ym"] + "-01").dt.to_period("M").dt.to_timestamp(how="end")
    ).dt.strftime("%Y-%m-%d")
    out = pd.DataFrame({
        "iso3": flows["iso3"],
        "as_of_date": month_end,
        "metric": flows["metric"] if "metric" in flows.columns else METRIC,
        "value": flows["value"].round().astype("int64"),
        "series_semantics": SERIES_SEMANTICS,
        "source": SOURCE,
        "displacement_type": CAUSE_CONFLICT,
    })
    return out.loc[:, list(STAGING_COLUMNS)]


def _client_id() -> tuple[str, str] | None:
    for env_name in ("IDMC_API_KEY", "IDMC_HELIX_CLIENT_ID"):
        key = os.getenv(env_name, "").strip()
        if key:
            return key, env_name
    return None


def _default_get(url: str, params: dict, timeout: float) -> list:
    # The PA machine's IDU transport, shared so a process that runs both
    # downloads the body once and the two cannot disagree about its shape.
    from resolver.hazard_resolution import idmc_idu

    return idmc_idu._default_get(url, params, timeout)


def run(
    *,
    months: int = DEFAULT_MONTHS,
    today: dt.date | None = None,
    get: GetFn | None = None,
    staging_dir: Path = STAGING_DIR,
    diagnostics_dir: Path = DIAGNOSTICS_DIR,
) -> int:
    """Fetch, split, write. Returns the process exit code.

    1 when the source could not be READ (no credential, a refused or failed
    request, a body that is not a list): that is not an empty series, and
    the workflow's terminal gate turns the run red for it. 0 otherwise,
    including a window with no conflict displacement at all.
    """

    today = today or dt.datetime.now(dt.timezone.utc).date()
    first_ym, last_ym = month_window(today, months)
    url = os.getenv(URL_ENV, "").strip() or DEFAULT_IDU_ALL_URL
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    summary_path = diagnostics_dir / "summary.json"
    exclusions_path = diagnostics_dir / "conflict_exclusions.json"

    def _write_summary(status: str, reason: str | None, report: dict[str, Any]) -> None:
        summary = {
            "connector": "idmc_conflict",
            "status": status,
            "reason": reason,
            "counts": {
                "fetched": int(report.get("records", 0)),
                "normalized": int(report.get("rows", 0)),
                "written": int(report.get("rows", 0)),
            },
            "window": {"first": first_ym, "last": last_ym},
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        exclusions_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    credential = _client_id()
    if credential is None:
        print(
            "::error title=IDMC could not be read::neither IDMC_API_KEY nor "
            "IDMC_HELIX_CLIENT_ID is set; no conflict displacement rows will be "
            "written this run. This is not an empty month."
        )
        _write_summary("error", "no_credentials", {"window": {"first": first_ym, "last": last_ym}})
        return 1
    key, env_name = credential
    LOG.info("[idmc_conflict] reading %s with %s; window %s..%s", url, env_name, first_ym, last_ym)
    fetch = get or _default_get
    try:
        rows = fetch(url, {"client_id": key}, REQUEST_TIMEOUT_SEC)
    except Exception as exc:  # noqa: BLE001 - a fetch failure is reported, never raised
        message = str(exc).replace(key, "REDACTED")
        print(f"::error title=IDMC could not be read::{type(exc).__name__}: {message}")
        _write_summary("error", f"fetch_failed: {type(exc).__name__}",
                       {"window": {"first": first_ym, "last": last_ym}})
        return 1
    if not isinstance(rows, list):
        print("::error title=IDMC could not be read::the IDU route did not return a list")
        _write_summary("error", "not_a_list", {"window": {"first": first_ym, "last": last_ym}})
        return 1

    population = load_population()
    if not population:
        print("::warning::population.csv unreadable; no population cap on conflict displacement")
    flows, report = conflict_monthly_flows(rows, first_ym, last_ym, population=population)
    for item in report.get("over_population", []):
        print(
            f"::warning title=IDMC month above population::{item['iso3']} {item['ym']}: "
            f"{item['total']:,.0f} against a population of {item['population']:,.0f}; held out"
        )
    staging = staging_frame(flows)
    staging.to_csv(staging_dir / "flow.csv", index=False)
    _write_summary("ok", None, report)
    excluded = report.get("excluded_people", {})
    print(
        "[idmc_conflict] records=%d conflict_rows=%d held_months=%d countries=%d "
        "window=%s..%s first_conflict_month_served=%s excluded_people=%s roles=%s dropped=%s"
        % (
            report["records"], report["rows"], report["rows_held"], report["countries"],
            first_ym, last_ym, report["first_conflict_month_served"],
            {k: int(v) for k, v in sorted(excluded.items())},
            report.get("conflict_roles"), report.get("conflict_records_dropped"),
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--months", type=int,
        default=int(os.getenv(MONTHS_ENV, "") or DEFAULT_MONTHS),
        help="Months of history to rewrite, ending with the last complete month",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        from resolver.diagnostics.http_recorder import maybe_install_from_env

        maybe_install_from_env()
    except Exception:  # noqa: BLE001 - recording is optional
        pass
    return run(months=args.months)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
