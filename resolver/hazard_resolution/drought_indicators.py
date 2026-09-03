# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Drought indicators — the attribution gate, and the gate before a zero.

IPC states how many people are in Phase 3+; it does not state WHY in any
structured field. Driver attribution lives in narrative PDFs, so on its own
an IPC deterioration cannot be filed as drought impact without also filing
conflict- and price-driven food insecurity as drought. These indicators are
what closes that gap: a Phase 3+ increase is admitted as DROUGHT impact
only where an independent physical/agricultural indicator says the country
was in drought.

They do the same work at the other end. A drought zero rests on two
statements — the indicators saw no drought AND IPC recorded no
deterioration — so an indicator that could not be READ suppresses the zero
rather than permitting it. "We could not check" and "there was no drought"
are different facts, and only the second can justify a zero.

**What is stored is values, not verdicts.** A feed is fetched once per run
and cached in ``haz_raw_drought_indicators`` as the raw per-country classes
or anomalies it published. Thresholds from ``rulebook.yaml`` are applied at
evaluation time, so retuning a threshold changes behaviour on the next run
with no re-fetch and no code change.

**Two providers.**

``asap``
    JRC ASAP agricultural hotspot classification: free, no key,
    country-level. A country ABSENT from the feed has no warning, which is
    what makes ASAP usable as evidence of absence.
``tabular``
    A pre-computed ``(iso3, ym, value)`` anomaly feed — CSV or JSON —
    compared against a rulebook threshold. CHIRPS and SPEI are gridded
    rasters, and turning those into a country number needs zonal statistics
    that have no business running inside the resolution path; this provider
    lets a maintainer point the machine at whatever country-level anomaly
    product they already produce. Unlike ASAP, a country absent from an
    anomaly feed is UNKNOWN, not dry.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping

import requests

from resolver.hazard_resolution.rulebook import Rulebook
from resolver.hazard_resolution.sources import (
    FetchOutcome,
    RawRecord,
    load_raw_records,
    parse_date,
    shift_month,
    store_raw_records,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

SOURCE = "drought_indicators"

STATE_DROUGHT = "drought"
STATE_NO_DROUGHT = "no_drought"
STATE_UNAVAILABLE = "unavailable"

PROVIDER_ASAP = "asap"
PROVIDER_TABULAR = "tabular"
PROVIDER_PYTHIA_TABLE = "pythia_table"

#: Providers whose feeds enumerate only the countries they have something to
#: say about, so that a country's ABSENCE is itself a statement ("no warning
#: issued"). For the others, absence means the feed is silent about that
#: country and the indicator is unavailable there. An entry may override this
#: with its own ``absence_means_no_drought`` — the meaning of absence is a
#: property of the FEED, not of the transport that fetched it, and the
#: pythia_table provider serves both kinds.
_ABSENCE_MEANS_NO_DROUGHT = {
    PROVIDER_ASAP: True,
    PROVIDER_TABULAR: False,
    PROVIDER_PYTHIA_TABLE: False,
}


def _absence_means_no_drought(entry: Mapping[str, Any]) -> bool:
    explicit = entry.get("absence_means_no_drought")
    if isinstance(explicit, bool):
        return explicit
    return _ABSENCE_MEANS_NO_DROUGHT.get(str(entry.get("provider")), False)


def _entry_urls(entry: Mapping[str, Any]) -> list[str]:
    """Every candidate url for an entry, in order.

    ``urls`` (a list) exists because a third-party portal moves its download
    route without moving the data — the JRC ASAP warnings file did exactly
    that — and a moved route should be a YAML edit rather than an outage.
    Candidates are tried in order and the first that parses wins.
    """

    out: list[str] = []
    for candidate in entry.get("urls") or []:
        text = str(candidate).strip()
        if text:
            out.append(text)
    single = str(entry.get("url") or "").strip()
    if single and single not in out:
        out.append(single)
    return out

#: Injectable transport seam for tests: (url, timeout) -> (text, content_type).
GetFn = Callable[[str, float], "tuple[str, str]"]

#: Field names a feed might use for the country, the observation date and
#: the value. Deliberately generous: these are third-party feeds whose
#: naming has changed across versions, and a rename must degrade to an
#: explicit UNAVAILABLE rather than to a silent "no drought anywhere".
_ISO3_KEYS = ("iso3", "ISO3", "iso_3", "country_iso3", "adm0_iso3", "#country+code")
_NAME_KEYS = ("country", "country_name", "name", "adm0_name", "asap0_name")
_DATE_KEYS = ("ym", "month", "date", "period", "observed_ym", "reference_date")
_CLASS_KEYS = (
    "asap_warning",
    "warning",
    "warning_class",
    "hotspot_class",
    "class",
    "status",
    "level",
)
_VALUE_KEYS = ("value", "anomaly", "spi", "spei", "index", "score")

#: Class-shaped values (a concern level, a warning label) versus
#: number-shaped ones. A pythia_table entry says which it serves.
_MATCH_CLASSES = "classes"
_MATCH_THRESHOLD = "threshold"


@dataclass
class IndicatorReading:
    """What one indicator says about one country-month."""

    name: str
    provider: str
    state: str
    required: bool
    observed_ym: str | None = None
    value: Any = None
    source_url: str | None = None
    error: str | None = None
    #: The feed carried a reading FOR THIS COUNTRY. False for an answer
    #: inferred from absence (an alerting feed that did not list it) — that
    #: is still an answer, but it is not coverage.
    present: bool = False

    def as_evidence(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "state": self.state,
            "required": self.required,
            "observed_ym": self.observed_ym,
            "value": self.value,
            "source_url": self.source_url,
            "error": self.error,
            "present": self.present,
        }


@dataclass
class IndicatorVerdict:
    """The combined indicator answer for one country-month."""

    iso3: str
    ym: str
    state: str
    note: str
    combine: str
    readings: list[IndicatorReading] = field(default_factory=list)
    #: How many readings named the country itself (see IndicatorReading.present).
    present_count: int = 0
    #: The rulebook's floor on that count before a zero may rest on absence.
    min_present_readings: int = 0

    @property
    def shows_drought(self) -> bool:
        return self.state == STATE_DROUGHT

    @property
    def available(self) -> bool:
        return self.state != STATE_UNAVAILABLE

    @property
    def has_coverage(self) -> bool:
        """Did any feed actually look at this country?

        A verdict can be available (enough feeds answered) with every answer
        inferred from absence. That supports "no feed warned about it"; it
        does not support "it was quiet", and only the second earns a zero.
        """

        return self.present_count >= self.min_present_readings

    def as_evidence(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "combine": self.combine,
            "note": self.note,
            "present_count": self.present_count,
            "min_present_readings": self.min_present_readings,
            "readings": [r.as_evidence() for r in self.readings],
        }


def indicator_entries(rulebook: Rulebook) -> list[dict[str, Any]]:
    """The configured indicator entries, as plain dicts."""

    return [dict(entry) for entry in rulebook.get("drought.indicators.entries")]


def _default_get(url: str, timeout: float) -> tuple[str, str]:
    resp = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
    resp.raise_for_status()
    return resp.text, str(resp.headers.get("Content-Type") or "")


def _expand_url(url: str, ym: str) -> str:
    """Substitute ``{ym}``/``{year}``/``{month}`` in a feed URL.

    A feed that publishes only a "latest" snapshot cannot speak for a past
    month; pointing an entry at a per-month archive is how a backcast gets
    real indicator coverage instead of falling through to no zeros.
    """

    year, month = ym.split("-")
    return url.replace("{ym}", ym).replace("{year}", year).replace("{month}", month)


def _rows_from_text(text: str, content_type: str) -> list[dict[str, Any]]:
    """Parse a feed body into a list of records (JSON or CSV)."""

    body = text.strip()
    if not body:
        return []
    looks_json = "json" in content_type.lower() or body[:1] in "[{"
    if looks_json:
        payload = json.loads(body)
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, Mapping):
            for key in ("data", "records", "features", "warnings", "results", "items"):
                block = payload.get(key)
                if isinstance(block, list):
                    # GeoJSON-style feeds nest the record under "properties".
                    return [
                        dict(row.get("properties") or row)
                        for row in block
                        if isinstance(row, Mapping)
                    ]
            # A bare {iso3: value} mapping is a legitimate feed shape too.
            return [
                {"iso3": key, "value": value}
                for key, value in payload.items()
                if isinstance(key, str) and len(key) == 3
            ]
        return []
    return [dict(row) for row in csv.DictReader(io.StringIO(body))]


def _signed_number(value: Any) -> float | None:
    """Parse an anomaly index, sign included.

    Deliberately NOT ``sources.parse_number``: that one rejects negatives,
    because a negative people-affected count is malformed. A drought
    anomaly is the opposite — the negative values are the ones that matter,
    and reading SPI −1.8 as "unparseable" would silently make the driest
    countries the ones the indicator says nothing about.
    """

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return number


def _first(record: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _record_iso3(record: Mapping[str, Any]) -> str | None:
    """ISO3 for a feed record: explicit code first, then the country name.

    Same multi-key + name-fallback shape the ACLED connectors were forced
    into — a single ``.get("iso3")`` silently discards every record of a
    feed that happens to label the column differently.
    """

    raw = _first(record, _ISO3_KEYS)
    if raw is not None:
        code = str(raw).strip().upper()
        if len(code) == 3 and code.isalpha():
            return code
    try:
        from resolver.ingestion.utils.iso_normalize import resolve_iso3
    except Exception:  # pragma: no cover - ingestion utils always import
        return None
    iso3, _reason = resolve_iso3(dict(record), name_keys=_NAME_KEYS)
    return iso3


def _record_ym(record: Mapping[str, Any]) -> str | None:
    raw = _first(record, _DATE_KEYS)
    if raw is None:
        return None
    text = str(raw).strip()
    if len(text) == 7 and text[4] == "-":
        return text
    parsed = parse_date(text)
    return parsed.strftime("%Y-%m") if parsed else None


def _parse_feed(
    entry: Mapping[str, Any], text: str, content_type: str, fallback_ym: str
) -> dict[str, Any]:
    """Turn a feed body into the cached ``{iso3: value}`` snapshot.

    Values are stored verbatim. The thresholds that turn them into a verdict
    live in the rulebook and are applied at evaluation time, so a retuned
    threshold takes effect without a re-fetch.
    """

    provider = str(entry.get("provider"))
    keys = _CLASS_KEYS if provider == PROVIDER_ASAP else _VALUE_KEYS
    rows = _rows_from_text(text, content_type)

    values: dict[str, Any] = {}
    observed: set[str] = set()
    unresolved = 0
    for row in rows:
        iso3 = _record_iso3(row)
        if not iso3:
            unresolved += 1
            continue
        value = _first(row, keys)
        if value is None:
            continue
        values[iso3] = value
        ym = _record_ym(row)
        if ym:
            observed.add(ym)

    if observed:
        observed_ym, ym_source = max(observed), "feed"
    else:
        # No date anywhere in the feed: it is a "latest" snapshot and
        # describes the moment it was retrieved. Stamping it with the target
        # month instead would silently pass the staleness test for any
        # backcast month and manufacture indicator coverage we do not have.
        observed_ym, ym_source = fallback_ym, "retrieval"

    return {
        "values": values,
        "observed_ym": observed_ym,
        "observed_ym_source": ym_source,
        "n_records": len(rows),
        "n_countries": len(values),
        "n_unresolved": unresolved,
    }


def _snapshot_from_pythia_table(
    con: "duckdb.DuckDBPyConnection", entry: Mapping[str, Any], ym: str
) -> dict[str, Any]:
    """Build a snapshot from a table this repo already ingests. Never raises.

    Two indicator sources reach the drought gate this way, and neither needs
    a new fetch or a new credential:

    ``hdx_signals``
        OCHA's automated crisis monitoring, whose ``jrc_agricultural_hotspots``
        indicator is ASAP itself, reached by a path the repo already
        maintains. Like ASAP it is an ALERTING feed — it publishes signals,
        not a global status — so a country's absence is the statement that no
        signal was issued, and the entry sets
        ``absence_means_no_drought: true``.

    ``seasonal_forecasts``
        NMME country-mean precipitation anomalies in sigma units, already
        ingested monthly. This one is supporting evidence, not a primary
        gate: it is a FORECAST for the month, not an observation of it, which
        is why its entry is ``required: false``. A country absent from NMME
        is unknown, not dry.

    Deliberately NOT offered: FEWS NET / IPC Phase 3+. That is the very
    quantity the drought rule differences, so admitting it as the
    attribution indicator would reduce the gate to "IPC rose, therefore
    drought, because IPC rose" — which is exactly the conflict-driven food
    insecurity the gate exists to keep out.
    """

    table = str(entry.get("table") or "")
    iso_column = str(entry.get("iso3_column") or "iso3")
    value_column = str(entry.get("value_column") or "value")
    date_column = str(entry.get("date_column") or "")
    offset_column = str(entry.get("date_offset_column") or "")
    where = str(entry.get("where") or "").strip()
    lookback = int(entry.get("lookback_months") or 0)
    oldest = shift_month(ym, -max(lookback, 0))

    # The DB travels between runs and holds every month, so the snapshot has
    # to be cut to the window this cell may see. Reading the newest row
    # regardless of date would let a September signal answer for June.
    #
    # When the row's date is an ISSUE date and the row is ABOUT a later
    # month (an NMME forecast issued in M at lead L is about M+L), the
    # observation month is the issue month plus the offset column, and the
    # window is applied to that in Python — the SQL keeps only a generous
    # lower bound so a forecast issued well before the window is not read.
    clauses = [f"{iso_column} IS NOT NULL", f"{value_column} IS NOT NULL"]
    params: list[Any] = []
    if where:
        clauses.append(f"({where})")
    if date_column and not offset_column:
        clauses.append(f"substr(CAST({date_column} AS VARCHAR), 1, 7) <= ?")
        params.append(ym)
        clauses.append(f"substr(CAST({date_column} AS VARCHAR), 1, 7) >= ?")
        params.append(oldest)
    elif date_column:
        clauses.append(f"substr(CAST({date_column} AS VARCHAR), 1, 7) <= ?")
        params.append(ym)
        clauses.append(f"substr(CAST({date_column} AS VARCHAR), 1, 7) >= ?")
        params.append(shift_month(oldest, -_MAX_DATE_OFFSET_MONTHS))

    order = f"substr(CAST({date_column} AS VARCHAR), 1, 7)" if date_column else "'1'"
    offset_sel = f"{offset_column}" if offset_column else "0"
    # A fully specified ORDER BY: the pick between two rows for one country
    # and month below must not depend on the table's physical order.
    sql = (
        f"SELECT {iso_column} AS iso3, {value_column} AS value, "
        f"{order} AS observed_ym, {offset_sel} AS date_offset FROM {table} "
        f"WHERE {' AND '.join(clauses)} "
        f"ORDER BY observed_ym, iso3, CAST(value AS VARCHAR), date_offset"
    )
    rows = con.execute(sql, params).fetchall()

    # For a class-shaped feed, a drought-class reading outranks any other
    # reading from the same month: several rows for one country in one month
    # (several signal types, several issues) must resolve the same way on
    # every run, and the one that says "warning" is the one that matters.
    preferred = {
        str(c).strip().lower() for c in (entry.get("drought_classes") or [])
    }
    picked: dict[str, tuple[str, Any]] = {}
    observed: set[str] = set()
    unresolved = 0
    for iso3_raw, value, observed_raw, date_offset in rows:
        code = str(iso3_raw or "").strip().upper()
        if len(code) != 3 or not code.isalpha():
            unresolved += 1
            continue
        observed_ym = str(observed_raw or "") if date_column else ""
        if offset_column and observed_ym:
            try:
                observed_ym = shift_month(observed_ym, int(date_offset or 0))
            except (TypeError, ValueError):
                unresolved += 1
                continue
            if observed_ym > ym or observed_ym < oldest:
                continue
        current = picked.get(code)
        if current is None:
            picked[code] = (observed_ym, value)
        else:
            cur_ym, cur_value = current
            new_pref = str(value).strip().lower() in preferred
            cur_pref = str(cur_value).strip().lower() in preferred
            if observed_ym > cur_ym or (observed_ym == cur_ym and new_pref and not cur_pref):
                picked[code] = (observed_ym, value)
        if date_column and observed_ym:
            observed.add(observed_ym)
    values: dict[str, Any] = {code: pair[1] for code, pair in picked.items()}

    observed_ym = max(observed) if observed else ym
    return {
        "values": values,
        "observed_ym": observed_ym,
        "observed_ym_source": "table" if observed else "target_month",
        "n_records": len(rows),
        "n_countries": len(values),
        "n_unresolved": unresolved,
    }


def fetch_indicators(
    con: "duckdb.DuckDBPyConnection",
    ym: str,
    rulebook: Rulebook,
    *,
    get: GetFn | None = None,
) -> FetchOutcome:
    """Refresh every configured indicator feed. Never raises.

    One request per entry per run, not per country: these are global feeds.
    An entry with an empty url is skipped (it is configured off); an entry
    that fails is recorded as failed, which is what later suppresses zeros
    if it was required.
    """

    entries = indicator_entries(rulebook)
    getter = get or _default_get
    retrieval_ym = dt.date.today().strftime("%Y-%m")
    outcome = FetchOutcome(source=SOURCE, ok=False)
    detail: dict[str, Any] = {"entries": {}}

    records: list[RawRecord] = []
    consulted = 0
    succeeded = 0

    for entry in entries:
        name = str(entry.get("name"))
        provider = str(entry.get("provider"))

        # --- A table this repo already ingests. No request, no credential. ---
        if provider == PROVIDER_PYTHIA_TABLE:
            consulted += 1
            try:
                snapshot = _snapshot_from_pythia_table(con, entry, ym)
                used_url = f"pythia://{entry.get('table')}"
            except Exception as exc:  # noqa: BLE001 - an absent table is a fact
                LOG.error("[drought_indicators] %s read failed: %s", name, exc)
                detail["entries"][name] = {"ok": False, "error": str(exc)}
                continue
            if not snapshot["values"]:
                error = (
                    f"{entry.get('table')} holds no usable rows for {ym} — the "
                    "indicator is unavailable, which is not the same as no drought"
                )
                LOG.warning("[drought_indicators] %s: %s", name, error)
                detail["entries"][name] = {"ok": False, "error": error}
                continue
            attempts: list[dict[str, Any]] = []
        else:
            candidates = _entry_urls(entry)
            if not candidates:
                detail["entries"][name] = {"ok": False, "skipped": "no url configured"}
                continue
            consulted += 1

            # Try each candidate url in turn. A portal that moved its download
            # route has not lost the data, and the first candidate that parses
            # is the answer.
            snapshot = None
            used_url = ""
            attempts = []
            for candidate in candidates:
                expanded = _expand_url(candidate, ym)
                outcome.source_urls.append(expanded)
                try:
                    text, content_type = getter(
                        expanded, float(entry.get("request_timeout_sec") or 60)
                    )
                    parsed = _parse_feed(entry, text, content_type, retrieval_ym)
                except Exception as exc:
                    attempts.append({"url": expanded, "error": str(exc)})
                    LOG.warning(
                        "[drought_indicators] %s: %s failed (%s)", name, expanded, exc
                    )
                    continue

                # Every record failing to resolve to a country means the feed's
                # shape changed, not that the world is drought-free. Refuse the
                # snapshot rather than cache a feed that says nothing about
                # anywhere.
                if parsed["n_records"] and not parsed["values"]:
                    error = (
                        f"0 of {parsed['n_records']} records resolved to a country "
                        f"({parsed['n_unresolved']} unresolved) — the feed's shape has "
                        "probably changed"
                    )
                    attempts.append({"url": expanded, "error": error})
                    LOG.error("[drought_indicators] %s: %s", name, error)
                    continue

                snapshot = parsed
                used_url = expanded
                break

            if snapshot is None:
                # Name the LAST attempt's reason, not a generic summary: "the
                # feed's shape changed" and "the route is dead" call for
                # different repairs, and a caller reading only `error` must
                # still be able to tell them apart.
                last = attempts[-1]["error"] if attempts else "no attempt made"
                detail["entries"][name] = {
                    "ok": False,
                    "error": last,
                    "attempts": attempts,
                }
                continue

        succeeded += 1
        detail["entries"][name] = {
            "ok": True,
            "url": used_url,
            "observed_ym": snapshot["observed_ym"],
            "observed_ym_source": snapshot["observed_ym_source"],
            "countries": snapshot["n_countries"],
            "unresolved": snapshot["n_unresolved"],
            **({"attempts": attempts} if attempts else {}),
        }
        records.append(
            RawRecord(
                record_id=f"{name}-{snapshot['observed_ym']}",
                payload={
                    "name": name,
                    "provider": provider,
                    "url": used_url,
                    **snapshot,
                },
                ym=snapshot["observed_ym"],
                hazard="DR",
                source_url=used_url,
            )
        )

    if records:
        stored = store_raw_records(con, SOURCE, records)
        outcome.records = stored["records"]
        outcome.inserted = stored["inserted"]

    # No feed read means no indicator evidence, whatever the reason. Saying
    # ok here would put "drought_indicators" outside the run's
    # sources_unavailable list, and every row would then look as though the
    # attribution gate had been consulted.
    outcome.ok = succeeded > 0
    outcome.detail = detail
    if consulted == 0:
        outcome.error = "no drought indicator entry has a url configured"
        LOG.warning("[drought_indicators] %s", outcome.error)
    elif succeeded == 0:
        outcome.error = "every configured drought indicator failed"
        LOG.error("[drought_indicators] %s", outcome.error)
    LOG.info(
        "[drought_indicators] %s: %d/%d feeds read", ym, succeeded, consulted
    )
    return outcome


#: How far before the lookback window an ISSUE date may sit when the entry
#: carries a date_offset_column. NMME publishes seven leads; nothing offsets
#: further than that.
_MAX_DATE_OFFSET_MONTHS = 12


def _snapshot_for(
    con: "duckdb.DuckDBPyConnection",
    name: str,
    ym: str,
    max_age_months: int,
    cache: dict[Any, Any] | None = None,
) -> dict[str, Any] | None:
    """The newest cached snapshot of ``name`` that may speak for ``ym``.

    A snapshot speaks for a month when its observation month is that month
    or up to ``max_age_months`` earlier. A LATER snapshot is not used: it
    describes conditions after the month in question, and reading a
    September warning back onto June would invent hindsight.

    ``cache`` is a per-run dict the caller owns. Every call loads and parses
    the whole indicator cache (one JSON payload of ~250 countries per feed
    per month), and the drought pass asks the same question once per
    country — 3,750 loads per live month, ~140,000 per backcast — for an
    answer that cannot change between countries.
    """

    key = (name, ym, int(max_age_months))
    if cache is not None and key in cache:
        return cache[key]

    oldest = shift_month(ym, -int(max_age_months))
    best: dict[str, Any] | None = None
    for payload in load_raw_records(con, SOURCE, hazard="DR"):
        if str(payload.get("name")) != name:
            continue
        observed = str(payload.get("observed_ym") or "")
        if not observed or observed > ym or observed < oldest:
            continue
        if best is None or observed > str(best.get("observed_ym") or ""):
            best = payload
    if cache is not None:
        cache[key] = best
    return best


def _reading_for_entry(
    con: "duckdb.DuckDBPyConnection",
    entry: Mapping[str, Any],
    iso3: str,
    ym: str,
    max_age_months: int,
    cache: dict[Any, Any] | None = None,
) -> IndicatorReading:
    name = str(entry.get("name"))
    provider = str(entry.get("provider"))
    required = bool(entry.get("required"))
    url = (
        f"pythia://{entry.get('table')}"
        if provider == PROVIDER_PYTHIA_TABLE
        else (_entry_urls(entry)[0] if _entry_urls(entry) else "")
    )

    if not url:
        return IndicatorReading(
            name=name, provider=provider, state=STATE_UNAVAILABLE,
            required=required, error="no url configured",
        )

    snapshot = _snapshot_for(con, name, ym, max_age_months, cache=cache)
    if snapshot is None:
        return IndicatorReading(
            name=name, provider=provider, state=STATE_UNAVAILABLE, required=required,
            source_url=url,
            error=(
                f"no cached observation within {max_age_months} month(s) of {ym}"
            ),
        )

    values = snapshot.get("values") or {}
    observed_ym = str(snapshot.get("observed_ym") or "")
    raw_value = values.get(iso3.upper())

    if raw_value is None:
        if _absence_means_no_drought(entry):
            # An alerting feed lists the countries it has warned about; not
            # being on the list IS the statement that no warning was issued.
            # ASAP and HDX Signals are both of this kind; an anomaly feed is
            # not, and absence there is unknown rather than dry.
            return IndicatorReading(
                name=name, provider=provider, state=STATE_NO_DROUGHT, required=required,
                observed_ym=observed_ym, source_url=snapshot.get("url"),
            )
        return IndicatorReading(
            name=name, provider=provider, state=STATE_UNAVAILABLE, required=required,
            observed_ym=observed_ym, source_url=snapshot.get("url"),
            error=f"{iso3} is absent from the {name} feed",
        )

    # Class-shaped values (a warning class, a concern level) are matched
    # against the configured list; number-shaped ones go to the threshold.
    # Which of the two an entry serves is a property of the FEED, so a
    # pythia_table entry declares it with `match` rather than inheriting it
    # from its transport.
    match = str(entry.get("match") or "").strip().lower()
    if not match:
        match = _MATCH_CLASSES if provider == PROVIDER_ASAP else _MATCH_THRESHOLD

    if match == _MATCH_CLASSES:
        classes = {str(c).strip().lower() for c in entry.get("drought_classes") or []}
        state = (
            STATE_DROUGHT
            if str(raw_value).strip().lower() in classes
            else STATE_NO_DROUGHT
        )
        return IndicatorReading(
            name=name, provider=provider, state=state, required=required,
            observed_ym=observed_ym, value=raw_value, source_url=snapshot.get("url"),
            present=True,
        )

    number = _signed_number(raw_value)
    if number is None:
        return IndicatorReading(
            name=name, provider=provider, state=STATE_UNAVAILABLE, required=required,
            observed_ym=observed_ym, value=raw_value, source_url=snapshot.get("url"),
            error=f"value {raw_value!r} is not a number",
        )
    threshold = float(entry.get("threshold"))
    direction = str(entry.get("direction"))
    dry = number <= threshold if direction == "below" else number >= threshold
    return IndicatorReading(
        name=name, provider=provider,
        state=STATE_DROUGHT if dry else STATE_NO_DROUGHT,
        required=required, observed_ym=observed_ym, value=number,
        source_url=snapshot.get("url"), present=True,
    )


def evaluate_indicators(
    con: "duckdb.DuckDBPyConnection",
    iso3: str,
    ym: str,
    rulebook: Rulebook,
    *,
    cache: dict[Any, Any] | None = None,
) -> IndicatorVerdict:
    """Combine every configured indicator into one verdict for a cell.

    ``combine: any`` — one indicator showing drought is enough.
    ``combine: all`` — every indicator that answered must agree.

    Two ways the verdict comes back unavailable, and both mean the same
    thing: the machine did not check, so it may neither attribute a
    deterioration to drought nor write a zero.

    * an UNAVAILABLE **required** indicator — a feed nominated as
      indispensable did not answer;
    * fewer than ``min_available`` indicators answered at all. This is the
      one the shipped rulebook relies on. Nominating a single required feed
      makes that feed a single point of failure for every cell, which is how
      one dead ASAP URL left 756 drought cells inconclusive in a single run.
    """

    entries = indicator_entries(rulebook)
    combine = str(rulebook.get("drought.indicators.combine"))
    max_age = int(rulebook.get("drought.indicators.max_observation_age_months"))
    try:
        min_available = int(rulebook.get("drought.indicators.min_available"))
    except Exception:  # noqa: BLE001 - older rulebooks have no such key
        min_available = 1
    try:
        min_present = int(rulebook.get("drought.indicators.min_present_readings"))
    except Exception:  # noqa: BLE001 - older rulebooks have no such key
        min_present = 0

    readings = [
        _reading_for_entry(con, entry, iso3, ym, max_age, cache=cache)
        for entry in entries
    ]
    answered = [r for r in readings if r.state != STATE_UNAVAILABLE]
    present_count = sum(1 for r in answered if r.present)
    missing_required = [
        r for r in readings if r.state == STATE_UNAVAILABLE and r.required
    ]

    if missing_required:
        names = ", ".join(r.name for r in missing_required)
        return IndicatorVerdict(
            iso3=iso3, ym=ym, state=STATE_UNAVAILABLE, combine=combine,
            readings=readings, present_count=present_count,
            min_present_readings=min_present,
            note=(
                f"required indicator(s) unavailable: {names} — no drought verdict, "
                "and no zero"
            ),
        )

    if len(answered) < max(min_available, 1):
        unread = ", ".join(
            f"{r.name} ({r.error or 'no reading'})"
            for r in readings
            if r.state == STATE_UNAVAILABLE
        )
        return IndicatorVerdict(
            iso3=iso3, ym=ym, state=STATE_UNAVAILABLE, combine=combine,
            readings=readings, present_count=present_count,
            min_present_readings=min_present,
            note=(
                f"{len(answered)} of {len(readings)} indicator(s) answered, "
                f"below min_available={min_available}"
                + (f"; unread: {unread}" if unread else "")
            ),
        )

    dry = [r for r in answered if r.state == STATE_DROUGHT]
    if combine == "all":
        state = STATE_DROUGHT if len(dry) == len(answered) else STATE_NO_DROUGHT
    else:
        state = STATE_DROUGHT if dry else STATE_NO_DROUGHT

    note = (
        f"{len(dry)}/{len(answered)} indicator(s) show drought (combine={combine}); "
        f"{present_count} named the country"
    )
    return IndicatorVerdict(
        iso3=iso3, ym=ym, state=state, combine=combine, readings=readings, note=note,
        present_count=present_count, min_present_readings=min_present,
    )


def store_summary(con: "duckdb.DuckDBPyConnection") -> dict[str, Any]:
    """Provenance summary of the indicator cache (for evidence records)."""

    from resolver.hazard_resolution.sources import raw_store_summary

    return raw_store_summary(con, SOURCE, "DR")
