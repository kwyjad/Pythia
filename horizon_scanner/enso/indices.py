# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ENSO indices — the numeric record the phase is computed from.

The August 2026 failure was not a missing guard. The connector asked a
rendered HTML page for a classification and believed the answer; guarding
that design yields an empty row instead of a wrong one. So the phase is no
longer read anywhere. It is COMPUTED here, locally, from a machine-readable
Niño 3.4 index, against NOAA's own operational definition.

**The source ladder.** Three numeric sources, tried in order. Ranks 1 to 3
are numeric tables in formats that have been stable for decades; the IRI
Quick Look and the CPC diagnostic discussion are decoration and never
load-bearing for the phase.

===== ============================================ =========================
Rank   Source                                       Gives
===== ============================================ =========================
1      NOAA ERDDAP ``ncepNinoSSTwk`` (tabledap)     Weekly Niño 3.4 anomaly
2      CPC ``data/indices/wksst9120.for``           Weekly Niño 3.4 anomaly
3      CPC ``data/indices/oni.ascii.txt``           Three-month ONI
===== ============================================ =========================

**Every rank is read on every run** (Sept 2026). Until then the ladder
stopped at the first usable answer, so the record could never say whether a
SECOND source agreed with it, and the corroboration machinery had nothing
to compare against. Three small requests a run is the price of knowing that
two independent sources are alive; each rank's newest observation is
recorded in the evidence whether or not it was the one used.

**Rank 2 moved from ``wksst8110.for`` to ``wksst9120.for``** (Sept 2026).
CPC changed its base period from 1981-2010 to 1991-2020 in early 2021 and
stopped updating the old file; its last line is dated 2021-01-27, which is
why the run of 2026-09-04 found it 2,046 days stale. The new file has the
same fixed-width layout, so the parser is unchanged.

**Rank 1 asks for the whole table, not named columns** (Sept 2026). The
request used to project ``time,NINO3_4,ANOM3_4`` and ERDDAP answered HTTP
400 — its response to an unrecognised variable name. The dataset publishes
its columns under names this code cannot verify from the build environment
(the info page reads ``Nino34_ssta``-style), so the request now asks for
every column and the parser finds the Niño 3.4 anomaly by pattern. A
renamed column is then a parse failure that names the columns it saw,
never an HTTP 400 that names nothing.

**The weekly reading and the ONI are two numbers.** A seasonal source (rank
3) IS the ONI and carries no weekly value; the record says the weekly
reading is absent rather than printing the ONI under a weekly label, so a
run that could only read the published table is visibly one source short.

**ONI, and what stands in for it.** The operational definition is stated on
the ONI: the three-month running mean of Niño 3.4 anomalies. Rank 3 serves
that number directly. Ranks 1 and 2 serve weekly anomalies, from which a
three-month mean is computed here and labelled as such
(``oni_basis='weekly_3month_mean'``) — close to the published ONI but not
identical to it, and the record says which it is rather than letting a
reader assume.

Every function in this module is pure apart from the one HTTP seam
(``GetFn``), which tests inject. Nothing here writes to a database and
nothing here parses HTML.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

LOG = logging.getLogger(__name__)

#: Injectable transport seam for tests: (url, timeout) -> body text.
GetFn = Callable[[str, float], str]

PHASE_EL_NINO = "El Niño"
PHASE_LA_NINA = "La Niña"
PHASE_NEUTRAL = "Neutral"

#: NOAA's operational threshold, applied to the ONI, in °C.
ONI_THRESHOLD = 0.5

#: Strength bands on |ONI|, in °C. Ordered weakest first; the last band is
#: open-ended upwards. "El Niño" alone under-describes what the Pacific is
#: doing, so the band travels with the phase everywhere it is printed.
STRENGTH_BANDS: tuple[tuple[float, float | None, str], ...] = (
    (0.5, 1.0, "weak"),
    (1.0, 1.5, "moderate"),
    (1.5, 2.0, "strong"),
    (2.0, None, "very strong"),
)

#: A Niño 3.4 anomaly outside this range is a parse failure, not a reading.
#: The observed record has never approached ±4 °C, so a value beyond it means
#: a column was misread — and a null index can then never accompany a stated
#: phase, which is the single rule that would have caught August 2026.
NINO34_MIN = -4.0
NINO34_MAX = 4.0

#: Days of weekly observations averaged into the ONI proxy. Three months of
#: weekly values, which is what the published ONI averages over.
ONI_PROXY_WINDOW_DAYS = 90

#: How far the ONI may move in a month, and how far two sources may differ,
#: before the reading needs a second opinion. The observed record moves well
#: under 1 °C month to month, so a larger step is likelier a changed column
#: than a changed ocean.
ONI_JUMP_LIMIT = 1.0

#: Two independent sources measuring the same ocean should agree closely. A
#: gap wider than this means one of them is not measuring what we think.
ONI_AGREEMENT_TOLERANCE = 0.5

#: The published ONI is a three-month mean, so its own observation is
#: centred a month or so back. This bounds how stale the newest observation
#: of ANY source may be before the record is refused as current.
MAX_OBSERVATION_AGE_DAYS = 120

#: ERDDAP tabledap. No variable projection: a leading ``&`` asks for every
#: column, and the parser picks the Niño 3.4 anomaly by name pattern. The
#: projected form (``?time,NINO3_4,ANOM3_4``) drew HTTP 400 from the live
#: server, which is what ERDDAP returns for a variable it does not have.
ERDDAP_URL = "https://coastwatch.pfeg.noaa.gov/erddap/tabledap/ncepNinoSSTwk.csv?"
#: The 1991-2020 base-period file, updated weekly. ``wksst8110.for`` (the
#: 1981-2010 base) froze at 2021-01-27 when CPC changed base period.
CPC_WEEKLY_URL = "https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for"
CPC_WEEKLY_URL_FROZEN = "https://www.cpc.ncep.noaa.gov/data/indices/wksst8110.for"
CPC_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

BASIS_ONI_TABLE = "oni_table"
BASIS_WEEKLY_MEAN = "weekly_3month_mean"

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

#: The ONI file labels rows by overlapping three-month season. The season's
#: CENTRE month is the month it describes, so "JJA 2026" is July 2026.
_SEASONS = (
    "DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
    "JJA", "JAS", "ASO", "SON", "OND", "NDJ",
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """One dated Niño 3.4 anomaly, in °C."""

    date: dt.date
    anomaly: float


@dataclass
class IndexReading:
    """What one source said, whether or not it answered.

    ``ok`` false is never converted into a phase anywhere downstream: the
    ladder moves to the next rank, and if every rank fails the record is
    carried forward rather than defaulted.
    """

    name: str
    rank: int
    url: str
    ok: bool = False
    error: str | None = None
    observations: list[Observation] = field(default_factory=list)
    #: True when the source already publishes a three-month mean (the ONI
    #: table), false when this module has to compute one from weekly values.
    seasonal: bool = False
    #: What this rank would have answered, recorded whether or not it was
    #: the rank used. The two-sources-alive check reads these.
    newest_observation: dt.date | None = None
    nino34: float | None = None
    oni: float | None = None
    oni_basis: str | None = None

    def as_evidence(self) -> dict:
        return {
            "name": self.name,
            "rank": self.rank,
            "url": self.url,
            "ok": self.ok,
            "error": self.error,
            "observations": len(self.observations),
            "seasonal": self.seasonal,
            "newest_observation": (
                self.newest_observation.isoformat() if self.newest_observation else None
            ),
            "nino34": self.nino34,
            "oni": self.oni,
            "oni_basis": self.oni_basis,
        }


@dataclass
class IndexResolution:
    """The numeric record, assembled from whichever rank answered first."""

    #: The Niño 3.4 anomaly the phase rests on. For a weekly source this is
    #: the newest weekly value; for the ONI table it is the ONI itself (a
    #: three-month mean of the same index).
    nino34: float | None = None
    #: The newest WEEKLY Niño 3.4 reading, or None when the rank that
    #: answered publishes only the seasonal ONI. Never filled from the ONI:
    #: an absent weekly reading must be visible as absent.
    nino34_weekly: float | None = None
    oni: float | None = None
    oni_basis: str | None = None
    observation_date: dt.date | None = None
    source_rank_used: int | None = None
    source_name: str | None = None
    source_url: str | None = None
    #: How many weekly observations went into an ONI proxy (0 for a table).
    n_observations: int = 0
    readings: list[IndexReading] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        """True when a numeric anomaly was read. The publishable test."""

        return self.nino34 is not None

    def as_evidence(self) -> dict:
        return {
            "nino34": self.nino34,
            "nino34_weekly": self.nino34_weekly,
            "oni": self.oni,
            "oni_basis": self.oni_basis,
            "observation_date": (
                self.observation_date.isoformat() if self.observation_date else None
            ),
            "source_rank_used": self.source_rank_used,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "n_observations": self.n_observations,
            "ranks_ok": sorted(r.rank for r in self.readings if r.ok),
            "readings": [r.as_evidence() for r in self.readings],
        }

    @property
    def usable_readings(self) -> list["IndexReading"]:
        """Every rank that answered with a current observation, best first."""

        return sorted((r for r in self.readings if r.ok), key=lambda r: r.rank)


# ---------------------------------------------------------------------------
# Classification — computed, never read
# ---------------------------------------------------------------------------

def classify_oni(oni: float | None) -> tuple[str, str]:
    """(phase, strength) for an ONI value, per NOAA's operational definition.

    El Niño at ONI >= +0.5, La Niña at <= -0.5, Neutral between. Strength is
    read off ``STRENGTH_BANDS`` on the absolute value; Neutral has no
    strength band and returns "".

    Returns ``("", "")`` for None. That is deliberate: there is no phase
    without a number, and a caller must never be handed "Neutral" as the
    answer to "we do not know".
    """

    if oni is None:
        return "", ""
    value = float(oni)
    if value >= ONI_THRESHOLD:
        phase = PHASE_EL_NINO
    elif value <= -ONI_THRESHOLD:
        phase = PHASE_LA_NINA
    else:
        return PHASE_NEUTRAL, ""

    magnitude = abs(value)
    strength = ""
    for low, high, label in STRENGTH_BANDS:
        if magnitude >= low and (high is None or magnitude < high):
            strength = label
    return phase, strength


def describe_phase(phase: str, strength: str) -> str:
    """"El Niño, strong" — the phase as it should be printed."""

    if not phase:
        return ""
    return f"{phase}, {strength}" if strength else phase


def valid_anomaly(value: object) -> float | None:
    """A Niño 3.4 anomaly, or None when the value is not a usable reading.

    Rejects non-numbers, NaN/inf, and anything outside ±4 °C. The range test
    is the one that turns a misread column into a parse failure instead of
    into a phase.
    """

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    if not (NINO34_MIN <= number <= NINO34_MAX):
        return None
    return number


# ---------------------------------------------------------------------------
# Parsers — one per numeric source, each pure
# ---------------------------------------------------------------------------

def parse_erddap_csv(body: str) -> list[Observation]:
    """Weekly observations from an ERDDAP tabledap CSV response.

    ERDDAP writes a column-name row, then a units row, then data. The units
    row is skipped by requiring the time cell to parse as a date, which also
    survives a column reordering.

    The anomaly column is found by NAME PATTERN, not by a fixed spelling:
    the request asks for every column (see ``ERDDAP_URL``) precisely because
    the live spelling could not be verified. A body with no recognisable
    Niño 3.4 anomaly column raises, naming the columns it saw, so the ladder
    records the real reason rather than "no usable observation".
    """

    out: list[Observation] = []
    reader = csv.DictReader(io.StringIO(body))
    if not reader.fieldnames:
        return out
    time_key = _match_key(reader.fieldnames, ("time", "date"))
    anom_key = erddap_anomaly_column(reader.fieldnames)
    if not time_key or not anom_key:
        raise ValueError(
            "no Niño 3.4 anomaly column recognised; columns were "
            + ", ".join(str(c) for c in reader.fieldnames)
        )
    for row in reader:
        moment = _parse_iso_date(str(row.get(time_key) or ""))
        anomaly = valid_anomaly(row.get(anom_key))
        if moment is None or anomaly is None:
            continue
        out.append(Observation(date=moment, anomaly=anomaly))
    return out


def parse_cpc_weekly(body: str) -> list[Observation]:
    """Weekly observations from CPC's ``wksst8110.for`` fixed-width table.

    Each data line is a ``DDMMMYYYY`` week label followed by four SST/SSTA
    pairs — Niño 1+2, 3, 3.4, 4 in that order — so the Niño 3.4 anomaly is
    the sixth number on the line. Header and blank lines have no week label
    and are skipped by that test alone, which is why a reworded header does
    not break the parse.
    """

    out: list[Observation] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"^(\d{2})([A-Z]{3})(\d{4})\b", stripped)
        if not match:
            continue
        moment = _week_label_date(match)
        if moment is None:
            continue
        numbers = re.findall(r"[-+]?\d+\.\d+", stripped[match.end():])
        if len(numbers) < 6:
            continue
        anomaly = valid_anomaly(numbers[5])
        if anomaly is None:
            continue
        out.append(Observation(date=moment, anomaly=anomaly))
    return out


def parse_cpc_oni(body: str) -> list[Observation]:
    """Seasonal ONI values from CPC's ``oni.ascii.txt``.

    Lines are ``SEAS YR TOTAL ANOM``. The season label names three
    overlapping months and the value describes the CENTRE one, so "JJA 2026"
    is dated July 2026 — dating it to the season's start would report the
    ONI as two months staler than it is and trip the staleness bound.
    """

    out: list[Observation] = []
    for line in body.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        season = parts[0].strip().upper()
        if season not in _SEASONS:
            continue
        try:
            year = int(parts[1])
        except (TypeError, ValueError):
            continue
        anomaly = valid_anomaly(parts[3])
        if anomaly is None:
            continue
        moment = _season_centre_date(season, year)
        if moment is None:
            continue
        out.append(Observation(date=moment, anomaly=anomaly))
    return out


def _match_key(fieldnames: Sequence[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {str(name).strip().lower(): str(name) for name in fieldnames if name}
    for candidate in candidates:
        hit = lowered.get(candidate.lower())
        if hit:
            return hit
    return None


#: Column-name patterns for the Niño 3.4 ANOMALY, tried in order. The first
#: two are the spellings this code has used; the rest cover the
#: ``Nino34_ssta`` / ``NINO3.4 anom`` families ERDDAP datasets publish. A
#: plain SST column (``Nino34_sst``) never matches — it must say anomaly.
_ERDDAP_ANOMALY_PATTERNS = (
    re.compile(r"^anom3[._]?4$", re.IGNORECASE),
    re.compile(r"^anom[._]?3[._]?4$", re.IGNORECASE),
    re.compile(r"^nino[._ ]?3[._]?4[._ ]?(ssta|anom\w*)$", re.IGNORECASE),
    re.compile(r"^(ssta|anom\w*)[._ ]?nino[._ ]?3[._]?4$", re.IGNORECASE),
    re.compile(r"nino[._ ]?3[._]?4.*(anom|ssta)", re.IGNORECASE),
    re.compile(r"(anom|ssta).*nino[._ ]?3[._]?4", re.IGNORECASE),
)


def erddap_anomaly_column(fieldnames: Sequence[str]) -> str | None:
    """The column holding the Niño 3.4 anomaly, whatever ERDDAP calls it."""

    names = [str(name).strip() for name in fieldnames if name]
    for pattern in _ERDDAP_ANOMALY_PATTERNS:
        for name in names:
            if pattern.search(name):
                return name
    return None


def _parse_iso_date(text: str) -> dt.date | None:
    cleaned = text.strip().replace("Z", "")
    if not cleaned:
        return None
    try:
        return dt.datetime.fromisoformat(cleaned).date()
    except ValueError:
        pass
    try:
        return dt.date.fromisoformat(cleaned[:10])
    except ValueError:
        return None


def _week_label_date(match: "re.Match[str]") -> dt.date | None:
    month = _MONTHS.get(match.group(2).upper())
    if month is None:
        return None
    try:
        return dt.date(int(match.group(3)), month, int(match.group(1)))
    except ValueError:
        return None


def _season_centre_date(season: str, year: int) -> dt.date | None:
    """The first day of the month a three-month season is centred on."""

    try:
        index = _SEASONS.index(season)
    except ValueError:
        return None
    # DJF is centred on January (index 0 -> month 1); the list is in order.
    month = index + 1
    try:
        return dt.date(year, month, 1)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# The ONI, from whichever series a rank gave us
# ---------------------------------------------------------------------------

def oni_from_observations(
    observations: Iterable[Observation], *, seasonal: bool
) -> tuple[float | None, float | None, dt.date | None, str | None, int]:
    """(nino34, oni, observation_date, oni_basis, n_used) for one series.

    A seasonal series already IS the ONI, so its newest value is taken as
    published. A weekly series is averaged over the trailing
    ``ONI_PROXY_WINDOW_DAYS``, and the result is labelled a proxy rather
    than passed off as the published index.
    """

    ordered = sorted(observations, key=lambda obs: obs.date)
    if not ordered:
        return None, None, None, None, 0
    latest = ordered[-1]
    if seasonal:
        return latest.anomaly, latest.anomaly, latest.date, BASIS_ONI_TABLE, 1

    cutoff = latest.date - dt.timedelta(days=ONI_PROXY_WINDOW_DAYS)
    window = [obs for obs in ordered if obs.date > cutoff]
    if not window:
        window = [latest]
    mean = sum(obs.anomaly for obs in window) / len(window)
    return latest.anomaly, round(mean, 3), latest.date, BASIS_WEEKLY_MEAN, len(window)


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

def _default_get(url: str, timeout: float) -> str:
    import requests

    resp = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "PythiaBot/1.0 (humanitarian forecasting research)"},
    )
    if resp.status_code >= 400:
        # ERDDAP puts the reason in the body ("Unrecognized variable=...");
        # a bare status code sent the September 2026 diagnosis nowhere.
        snippet = re.sub(r"\s+", " ", (resp.text or ""))[:200]
        raise RuntimeError(f"HTTP {resp.status_code} from {url}: {snippet}")
    return resp.text


def _erddap_url(today: dt.date, lookback_days: int) -> str:
    """ERDDAP with a trailing-window constraint, so we fetch weeks not decades."""

    start = today - dt.timedelta(days=lookback_days)
    return f"{ERDDAP_URL}&time%3E={start.isoformat()}T00:00:00Z"


def source_ladder(today: dt.date, *, lookback_days: int = 400) -> list[dict]:
    """The ranked numeric sources, newest-first by preference."""

    return [
        {
            "name": "noaa_erddap_ncepNinoSSTwk",
            "rank": 1,
            "url": _erddap_url(today, lookback_days),
            "parse": parse_erddap_csv,
            "seasonal": False,
        },
        {
            "name": "cpc_wksst9120",
            "rank": 2,
            "url": CPC_WEEKLY_URL,
            "parse": parse_cpc_weekly,
            "seasonal": False,
        },
        {
            "name": "cpc_oni_ascii",
            "rank": 3,
            "url": CPC_ONI_URL,
            "parse": parse_cpc_oni,
            "seasonal": True,
        },
    ]


def read_rank(
    spec: dict,
    *,
    getter: GetFn,
    today: dt.date,
    timeout: float,
    max_age_days: int,
) -> IndexReading:
    """Read ONE rank and say what it would answer. Never raises.

    A reading is ``ok`` only when the body parsed, at least one observation
    survived the range check, and the newest observation is within
    ``max_age_days``. An archived copy of last year's file is recorded as
    stale, never read as current.
    """

    reading = IndexReading(
        name=str(spec["name"]),
        rank=int(spec["rank"]),
        url=str(spec["url"]),
        seasonal=bool(spec["seasonal"]),
    )
    try:
        body = getter(reading.url, timeout)
        reading.observations = list(spec["parse"](body))
    except Exception as exc:  # noqa: BLE001 - a dead source is a fact, not a crash
        reading.error = f"{type(exc).__name__}: {exc}"
        LOG.warning("[enso] rank %d (%s) failed: %s", reading.rank, reading.name, exc)
        return reading

    if not reading.observations:
        reading.error = "no usable observation parsed from the response"
        LOG.warning("[enso] rank %d (%s): %s", reading.rank, reading.name, reading.error)
        return reading

    nino34, oni, observed, basis, _n_used = oni_from_observations(
        reading.observations, seasonal=reading.seasonal
    )
    reading.newest_observation = observed
    reading.nino34 = nino34
    reading.oni = oni
    reading.oni_basis = basis
    age = (today - observed).days if observed else None
    if age is not None and age > max_age_days:
        reading.error = (
            f"newest observation {observed.isoformat()} is {age} days old "
            f"(limit {max_age_days})"
        )
        LOG.warning("[enso] rank %d (%s): %s", reading.rank, reading.name, reading.error)
        return reading

    reading.ok = True
    LOG.info(
        "[enso] rank %d (%s): Niño 3.4 %+.2f °C on %s, ONI %+.2f (%s)",
        reading.rank, reading.name, nino34, observed.isoformat(), oni, basis,
    )
    return reading


def resolve_indices(
    *,
    get: GetFn | None = None,
    today: dt.date | None = None,
    timeout: float = 60.0,
    max_age_days: int = MAX_OBSERVATION_AGE_DAYS,
) -> IndexResolution:
    """Read EVERY rank, record what each said, and answer from the best. Never raises.

    The record is taken from the lowest-ranked usable reading. Every other
    rank's answer travels in the evidence, so a run can state how many
    independent sources were alive and whether they agreed — which is what
    the continuity and corroboration checks need, and what the
    two-ranks-alive contradiction check reads.

    The weekly Niño 3.4 (``nino34_weekly``) is filled ONLY from a weekly
    rank. When the ONI table is the only source that answered, the weekly
    reading is None and stays None: printing the ONI under a weekly label
    would hide that the run was one source short.
    """

    getter = get or _default_get
    day = today or dt.date.today()
    resolution = IndexResolution()

    for spec in source_ladder(day):
        resolution.readings.append(
            read_rank(spec, getter=getter, today=day, timeout=timeout,
                      max_age_days=max_age_days)
        )

    usable = resolution.usable_readings
    if not usable:
        LOG.error(
            "[enso] every numeric source failed — no phase may be computed; "
            "the caller must carry the last good record forward, never default"
        )
        return resolution

    chosen = usable[0]
    resolution.nino34 = chosen.nino34
    resolution.nino34_weekly = None if chosen.seasonal else chosen.nino34
    resolution.oni = chosen.oni
    resolution.oni_basis = chosen.oni_basis
    resolution.observation_date = chosen.newest_observation
    resolution.source_rank_used = chosen.rank
    resolution.source_name = chosen.name
    resolution.source_url = chosen.url
    resolution.n_observations = (
        1 if chosen.seasonal else _window_size(chosen.observations)
    )
    if resolution.nino34_weekly is None:
        LOG.info(
            "[enso] no weekly Niño 3.4 reading this run: rank %d (%s) publishes "
            "the seasonal ONI only; the weekly field is stored as absent",
            chosen.rank, chosen.name,
        )
    if len(usable) < 2:
        LOG.warning(
            "[enso] only one numeric source answered (rank %d); nothing can "
            "corroborate this reading", chosen.rank,
        )
    return resolution


def _window_size(observations: Iterable[Observation]) -> int:
    ordered = sorted(observations, key=lambda obs: obs.date)
    if not ordered:
        return 0
    cutoff = ordered[-1].date - dt.timedelta(days=ONI_PROXY_WINDOW_DAYS)
    return max(1, sum(1 for obs in ordered if obs.date > cutoff))


def fetch_oni_history(
    *, get: GetFn | None = None, timeout: float = 120.0
) -> list[Observation]:
    """The whole published ONI table, 1950 to present. Never raises.

    Used to seed history in one pass, so continuity checks have something to
    compare against and RC work sees a real ENSO record rather than three
    rows.
    """

    getter = get or _default_get
    try:
        return parse_cpc_oni(getter(CPC_ONI_URL, timeout))
    except Exception as exc:  # noqa: BLE001
        LOG.error("[enso] ONI history fetch failed: %s", exc)
        return []


def corroborate(
    resolution: IndexResolution,
    *,
    get: GetFn | None = None,
    today: dt.date | None = None,
    timeout: float = 60.0,
    max_age_days: int = MAX_OBSERVATION_AGE_DAYS,
) -> IndexResolution | None:
    """A SECOND numeric reading, from a rank the first answer did not use.

    Since every rank is read on every run, the second opinion is usually
    already in ``resolution.readings`` and costs nothing. A resolution
    assembled some other way (tests, the recompute path) falls back to
    fetching the other ranks.

    Returns None when no other rank answers, which the caller must treat as
    "uncorroborated", never as agreement.
    """

    used = resolution.source_rank_used
    for reading in resolution.usable_readings:
        if reading.rank == used or reading.oni is None:
            continue
        other = IndexResolution(
            nino34=reading.nino34,
            nino34_weekly=None if reading.seasonal else reading.nino34,
            oni=reading.oni,
            oni_basis=reading.oni_basis,
            observation_date=reading.newest_observation,
            source_rank_used=reading.rank,
            source_name=reading.name,
            source_url=reading.url,
        )
        LOG.info(
            "[enso] corroborating rank %d (%s): ONI %+.2f (already read this run)",
            other.source_rank_used, other.source_name, other.oni,
        )
        return other

    getter = get or _default_get
    day = today or dt.date.today()
    for spec in source_ladder(day):
        if int(spec["rank"]) == used:
            continue
        if any(r.rank == int(spec["rank"]) for r in resolution.readings):
            continue  # read this run and found unusable; do not ask again
        reading = read_rank(spec, getter=getter, today=day, timeout=timeout,
                            max_age_days=max_age_days)
        if not reading.ok or reading.oni is None:
            continue
        other = IndexResolution(
            nino34=reading.nino34,
            nino34_weekly=None if reading.seasonal else reading.nino34,
            oni=reading.oni,
            oni_basis=reading.oni_basis,
            observation_date=reading.newest_observation,
            source_rank_used=reading.rank,
            source_name=reading.name,
            source_url=reading.url,
        )
        LOG.info(
            "[enso] corroborating rank %d (%s): ONI %+.2f",
            other.source_rank_used, other.source_name, other.oni,
        )
        return other
    return None
