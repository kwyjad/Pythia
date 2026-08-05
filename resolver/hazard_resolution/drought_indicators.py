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
    utcnow_iso,
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

#: Providers whose feeds enumerate only the countries they have something to
#: say about, so that a country's ABSENCE is itself a statement ("no warning
#: issued"). For the others, absence means the feed is silent about that
#: country and the indicator is unavailable there.
_ABSENCE_MEANS_NO_DROUGHT = {PROVIDER_ASAP: True, PROVIDER_TABULAR: False}

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

    @property
    def shows_drought(self) -> bool:
        return self.state == STATE_DROUGHT

    @property
    def available(self) -> bool:
        return self.state != STATE_UNAVAILABLE

    def as_evidence(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "combine": self.combine,
            "note": self.note,
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
        url = str(entry.get("url") or "").strip()
        if not url:
            detail["entries"][name] = {"ok": False, "skipped": "no url configured"}
            continue
        consulted += 1
        expanded = _expand_url(url, ym)
        outcome.source_urls.append(expanded)
        try:
            text, content_type = getter(
                expanded, float(entry.get("request_timeout_sec") or 60)
            )
            snapshot = _parse_feed(entry, text, content_type, retrieval_ym)
        except Exception as exc:
            LOG.error("[drought_indicators] %s fetch failed: %s", name, exc)
            detail["entries"][name] = {"ok": False, "error": str(exc)}
            continue

        # Every record failing to resolve to a country means the feed's shape
        # changed, not that the world is drought-free. Refuse the snapshot
        # rather than cache a feed that says nothing about anywhere.
        if snapshot["n_records"] and not snapshot["values"]:
            error = (
                f"0 of {snapshot['n_records']} records resolved to a country "
                f"({snapshot['n_unresolved']} unresolved) — the feed's shape has "
                "probably changed"
            )
            LOG.error("[drought_indicators] %s: %s", name, error)
            detail["entries"][name] = {"ok": False, "error": error}
            continue

        succeeded += 1
        detail["entries"][name] = {
            "ok": True,
            "observed_ym": snapshot["observed_ym"],
            "observed_ym_source": snapshot["observed_ym_source"],
            "countries": snapshot["n_countries"],
            "unresolved": snapshot["n_unresolved"],
        }
        records.append(
            RawRecord(
                record_id=f"{name}-{snapshot['observed_ym']}",
                payload={
                    "name": name,
                    "provider": str(entry.get("provider")),
                    "url": expanded,
                    "fetched_at": utcnow_iso(),
                    **snapshot,
                },
                ym=snapshot["observed_ym"],
                hazard="DR",
                source_url=expanded,
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


def _snapshot_for(
    con: "duckdb.DuckDBPyConnection", name: str, ym: str, max_age_months: int
) -> dict[str, Any] | None:
    """The newest cached snapshot of ``name`` that may speak for ``ym``.

    A snapshot speaks for a month when its observation month is that month
    or up to ``max_age_months`` earlier. A LATER snapshot is not used: it
    describes conditions after the month in question, and reading a
    September warning back onto June would invent hindsight.
    """

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
    return best


def _reading_for_entry(
    con: "duckdb.DuckDBPyConnection",
    entry: Mapping[str, Any],
    iso3: str,
    ym: str,
    max_age_months: int,
) -> IndicatorReading:
    name = str(entry.get("name"))
    provider = str(entry.get("provider"))
    required = bool(entry.get("required"))
    url = str(entry.get("url") or "").strip()

    if not url:
        return IndicatorReading(
            name=name, provider=provider, state=STATE_UNAVAILABLE,
            required=required, error="no url configured",
        )

    snapshot = _snapshot_for(con, name, ym, max_age_months)
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
        if _ABSENCE_MEANS_NO_DROUGHT.get(provider, False):
            # ASAP lists the countries it has warned about; not being on the
            # list IS the statement that no warning was issued.
            return IndicatorReading(
                name=name, provider=provider, state=STATE_NO_DROUGHT, required=required,
                observed_ym=observed_ym, source_url=snapshot.get("url"),
            )
        return IndicatorReading(
            name=name, provider=provider, state=STATE_UNAVAILABLE, required=required,
            observed_ym=observed_ym, source_url=snapshot.get("url"),
            error=f"{iso3} is absent from the {name} feed",
        )

    if provider == PROVIDER_ASAP:
        classes = {str(c).strip().lower() for c in entry.get("drought_classes") or []}
        state = (
            STATE_DROUGHT
            if str(raw_value).strip().lower() in classes
            else STATE_NO_DROUGHT
        )
        return IndicatorReading(
            name=name, provider=provider, state=state, required=required,
            observed_ym=observed_ym, value=raw_value, source_url=snapshot.get("url"),
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
        source_url=snapshot.get("url"),
    )


def evaluate_indicators(
    con: "duckdb.DuckDBPyConnection", iso3: str, ym: str, rulebook: Rulebook
) -> IndicatorVerdict:
    """Combine every configured indicator into one verdict for a cell.

    ``combine: any`` — one indicator showing drought is enough.
    ``combine: all`` — every indicator that answered must agree.

    In both modes an UNAVAILABLE **required** indicator makes the verdict
    unavailable: the machine did not check, so it may neither attribute a
    deterioration to drought nor write a zero.
    """

    entries = indicator_entries(rulebook)
    combine = str(rulebook.get("drought.indicators.combine"))
    max_age = int(rulebook.get("drought.indicators.max_observation_age_months"))

    readings = [
        _reading_for_entry(con, entry, iso3, ym, max_age) for entry in entries
    ]
    answered = [r for r in readings if r.state != STATE_UNAVAILABLE]
    missing_required = [
        r for r in readings if r.state == STATE_UNAVAILABLE and r.required
    ]

    if missing_required:
        names = ", ".join(r.name for r in missing_required)
        return IndicatorVerdict(
            iso3=iso3, ym=ym, state=STATE_UNAVAILABLE, combine=combine,
            readings=readings,
            note=(
                f"required indicator(s) unavailable: {names} — no drought verdict, "
                "and no zero"
            ),
        )

    if not answered:
        return IndicatorVerdict(
            iso3=iso3, ym=ym, state=STATE_UNAVAILABLE, combine=combine,
            readings=readings, note="no indicator answered for this cell",
        )

    dry = [r for r in answered if r.state == STATE_DROUGHT]
    if combine == "all":
        state = STATE_DROUGHT if len(dry) == len(answered) else STATE_NO_DROUGHT
    else:
        state = STATE_DROUGHT if dry else STATE_NO_DROUGHT

    note = (
        f"{len(dry)}/{len(answered)} indicator(s) show drought (combine={combine})"
    )
    return IndicatorVerdict(
        iso3=iso3, ym=ym, state=state, combine=combine, readings=readings, note=note
    )


def store_summary(con: "duckdb.DuckDBPyConnection") -> dict[str, Any]:
    """Provenance summary of the indicator cache (for evidence records)."""

    from resolver.hazard_resolution.sources import raw_store_summary

    return raw_store_summary(con, SOURCE, "DR")
