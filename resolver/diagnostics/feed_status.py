# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""What a committed feed currently covers, and whether it has gone quiet.

Some drought indicators are not fetched by a resolution run at all. SPEI-3
is a global raster and turning it into a country number is zonal statistics,
which needs a scientific stack the pipeline does not install; so a separate
scheduled producer writes ``resolver/data/spei3_country_means.csv`` and the
rulebook's ``tabular`` provider reads the committed file. That is the right
division of labour and it has one cost: **the producer can stop working
without any run noticing.** The rulebook entry is deliberately
``required: false`` with ``absence_means_no_drought: false``, so a missing
or stale file suppresses nothing and changes no verdict — which is correct,
and is also exactly why nothing would complain.

So the producer writes a status file beside the feed, and this module reads
it. Two consumers, one reader:

* the nightly backcast, for the restale request — the producer runs OUTSIDE
  the ``pythia-resolver-db`` concurrency group on purpose, so it can never
  cancel the nightly backcast or the monthly ingest, which means it cannot
  touch the canonical DB either. The committed status file is the channel,
  exactly as the CSV is;
* the run issue register, for staleness.

Pure and dependency-free: JSON and calendar arithmetic. A missing, empty or
unparseable status file is a STATE this module reports, never an exception
it raises — a diagnostic that raises is a diagnostic that takes the run
with it.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The SPEI-3 feed and the status file its producer writes beside it.
SPEI3_FEED = "spei3_country_means"
SPEI3_STATUS_PATH = REPO_ROOT / "resolver" / "data" / "spei3_status.json"
SPEI3_FEED_PATH = REPO_ROOT / "resolver" / "data" / "spei3_country_means.csv"

#: How far behind the previous complete month the newest covered month may
#: fall before the feed is reported stale. Three months, because the
#: producer runs monthly and ERA5T is about five days behind: one missed
#: cycle is not yet a fault, and three is not an accident.
MAX_LAG_MONTHS = 3

STATE_OK = "ok"
STATE_STALE = "stale"
STATE_ABSENT = "absent"
STATE_UNREADABLE = "unreadable"
STATE_INCOMPLETE = "incomplete"
STATE_FAILED = "failed"


def _previous_complete_month(today: dt.date | None = None) -> str:
    """The newest month that has ENDED, as ``YYYY-MM``."""

    day = today or dt.date.today()
    first = day.replace(day=1)
    last = first - dt.timedelta(days=1)
    return f"{last.year:04d}-{last.month:02d}"


def _month_index(ym: str) -> int | None:
    """``YYYY-MM`` as a count of months, so a lag is subtraction.

    Stepped in calendar months. Thirty-day arithmetic is how an ACAPS
    window asked for Mar, Jan, Dec, Dec and never February.
    """

    try:
        year, month = (int(part) for part in str(ym).split("-")[:2])
    except (TypeError, ValueError):
        return None
    if not 1 <= month <= 12:
        return None
    return year * 12 + (month - 1)


def months_behind(newest_ym: str, reference_ym: str) -> int | None:
    """How many months ``newest_ym`` falls behind ``reference_ym``."""

    newest, reference = _month_index(newest_ym), _month_index(reference_ym)
    if newest is None or reference is None:
        return None
    return reference - newest


@dataclass
class RestaleRequest:
    """A one-shot request to re-walk the resume ledger for named months.

    Committing a feed changes no rulebook key, so the hazard's fingerprint
    does not move, so ``completed_months`` returns every month already
    marked ``ok`` and the backcast skips them all. The file lands and
    nothing re-walks. This is what closes that loop.

    ``token`` is what makes it one-shot: applied once, recorded in
    ``haz_feed_restale``, and never applied again. Without it the nightly
    backcast would free the same months every night, re-walk them, and free
    them again forever.
    """

    feed: str
    hazard: str
    token: str
    months: list[str] = field(default_factory=list)
    requested_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "feed": self.feed,
            "hazard": self.hazard,
            "token": self.token,
            "months": list(self.months),
            "requested_at": self.requested_at,
        }


@dataclass
class FeedStatus:
    """A committed feed's own account of itself."""

    feed: str = ""
    state: str = STATE_ABSENT
    path: str = ""
    newest_month: str = ""
    oldest_month: str = ""
    months: int = 0
    rows: int = 0
    countries: int = 0
    months_behind: int | None = None
    reference_month: str = ""
    coverage: dict[str, int] = field(default_factory=dict)
    months_owed: list[str] = field(default_factory=list)
    last_success_run_id: str = ""
    last_failure: dict[str, Any] | None = None
    restale: RestaleRequest | None = None
    detail: str = ""

    @property
    def is_healthy(self) -> bool:
        return self.state == STATE_OK

    def as_dict(self) -> dict[str, Any]:
        return {
            "feed": self.feed,
            "state": self.state,
            "path": self.path,
            "newest_month": self.newest_month,
            "oldest_month": self.oldest_month,
            "months": self.months,
            "rows": self.rows,
            "countries": self.countries,
            "months_behind": self.months_behind,
            "reference_month": self.reference_month,
            "coverage": dict(self.coverage),
            "months_owed": list(self.months_owed),
            "last_success_run_id": self.last_success_run_id,
            "last_failure": self.last_failure,
            "restale": self.restale.as_dict() if self.restale else None,
            "detail": self.detail,
        }


def read_status_payload(path: Path | str | None = None) -> tuple[dict[str, Any] | None, str]:
    """``(payload, problem)``. A missing or broken file is a state, not a raise."""

    target = Path(path) if path is not None else SPEI3_STATUS_PATH
    if not target.exists():
        return None, f"{target} does not exist"
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"{target} could not be read: {type(exc).__name__}: {exc}"
    if not isinstance(payload, Mapping):
        return None, f"{target} does not carry a JSON object"
    return dict(payload), ""


def read_feed_status(
    path: Path | str | None = None,
    *,
    today: dt.date | None = None,
    max_lag_months: int = MAX_LAG_MONTHS,
    feed: str = SPEI3_FEED,
) -> FeedStatus:
    """The feed's state, ready for the register and for the backcast.

    ``absent`` is deliberately its own state rather than being folded into
    ``stale``: a feed whose producer has never succeeded and one whose
    producer stopped succeeding are different faults with different repairs,
    and reporting both as "stale" sends the reader to the wrong one.
    """

    target = Path(path) if path is not None else SPEI3_STATUS_PATH
    reference = _previous_complete_month(today)
    payload, problem = read_status_payload(target)
    if payload is None:
        state = STATE_ABSENT if "does not exist" in problem else STATE_UNREADABLE
        return FeedStatus(
            feed=feed, state=state, path=str(target),
            reference_month=reference, detail=problem,
        )

    newest = str(payload.get("newest_month") or "")
    lag = months_behind(newest, reference) if newest else None
    status = FeedStatus(
        feed=str(payload.get("feed") or feed),
        state=STATE_OK,
        path=str(target),
        newest_month=newest,
        oldest_month=str(payload.get("oldest_month") or ""),
        months=int(payload.get("months") or 0),
        rows=int(payload.get("rows") or 0),
        countries=int(payload.get("countries") or 0),
        months_behind=lag,
        reference_month=reference,
        coverage={str(k): int(v) for k, v in (payload.get("coverage") or {}).items()},
        months_owed=[str(m) for m in (payload.get("months_owed") or [])],
        last_success_run_id=str(payload.get("last_success_run_id") or ""),
        last_failure=dict(payload["last_failure"])
        if isinstance(payload.get("last_failure"), Mapping) else None,
        restale=_restale_from(payload, feed),
    )

    reported = str(payload.get("status") or "").strip().lower()
    if not newest:
        status.state = STATE_ABSENT
        status.detail = "the status file names no covered month"
    elif lag is not None and lag > max_lag_months:
        status.state = STATE_STALE
        status.detail = (
            f"the newest month covered is {newest}, {lag} month(s) behind "
            f"{reference} (limit {max_lag_months})"
        )
    elif reported == STATE_FAILED:
        status.state = STATE_FAILED
        status.detail = (
            "the last producer run failed its validation gates, so the feed "
            "was left as it was: "
            + str((status.last_failure or {}).get("reason") or "reason not recorded")
        )
    elif reported == STATE_INCOMPLETE or status.months_owed:
        status.state = STATE_INCOMPLETE
        status.detail = (
            f"{len(status.months_owed)} month(s) are still owed; the next "
            "producer run resumes on them: "
            + ",".join(status.months_owed[:12])
        )
    else:
        status.detail = (
            f"covers {status.oldest_month}..{newest} ({status.months} month(s), "
            f"{status.rows} row(s), {status.countries} country/countries)"
        )
    return status


def _restale_from(payload: Mapping[str, Any], feed: str) -> RestaleRequest | None:
    raw = payload.get("restale")
    if not isinstance(raw, Mapping):
        return None
    token = str(raw.get("token") or "").strip()
    hazard = str(raw.get("hazard") or "").strip().upper()
    months = [str(m) for m in (raw.get("months") or []) if str(m).strip()]
    if not token or not hazard or not months:
        return None
    return RestaleRequest(
        feed=str(payload.get("feed") or feed),
        hazard=hazard,
        token=token,
        months=months,
        requested_at=str(raw.get("requested_at") or ""),
    )


def spei3_restale_request(path: Path | str | None = None) -> RestaleRequest | None:
    """The pending SPEI-3 restale request, or None. Never raises."""

    payload, _problem = read_status_payload(path)
    if payload is None:
        return None
    return _restale_from(payload, SPEI3_FEED)


__all__ = [
    "MAX_LAG_MONTHS",
    "SPEI3_FEED", "SPEI3_FEED_PATH", "SPEI3_STATUS_PATH",
    "STATE_ABSENT", "STATE_FAILED", "STATE_INCOMPLETE", "STATE_OK",
    "STATE_STALE", "STATE_UNREADABLE",
    "FeedStatus", "RestaleRequest",
    "months_behind", "read_feed_status", "read_status_payload",
    "spei3_restale_request",
]
