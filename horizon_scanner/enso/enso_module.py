"""
ENSO State and Forecast Module
===============================
Assembles the ENSO record Pythia's climate-sensitive hazard prompts read.

**The phase is computed, never read.** It comes from a machine-readable
Niño 3.4 index via :mod:`horizon_scanner.enso.indices`, classified against
NOAA's operational definition. Before September 2026 this module asked the
IRI Quick Look page for a classification and believed the answer; in August
2026 that stored "Neutral" through a strong El Niño, and every drought and
cyclone prompt in the run read it as current. A guard on that design would
have produced an empty row rather than a wrong one, so the design changed.

**Field-level assembly.** The record is built field by field, each with its
own source and its own fetch outcome, rather than as one all-or-nothing
scrape. A missing IRI probability table costs the probability table. It
does not cost the phase. Publishable requires exactly one thing: a numeric
Niño 3.4 anomaly from :func:`~horizon_scanner.enso.indices.resolve_indices`.

Sources:
  - Numeric (load-bearing), in rank order: NOAA ERDDAP ``ncepNinoSSTwk``,
    CPC ``wksst8110.for``, CPC ``oni.ascii.txt``. See ``indices.py``.
  - IRI ENSO Quick Look (decoration): the nine-season probability table,
    the multi-model plume, the IOD state and the narrative. Its own
    statement of the phase is kept as ``scraped_state`` and compared
    against the computed one — the computed phase always wins.

**Write policy.** The table gets a row every run. A run that resolved a
number writes ``status='fresh'``. A run that resolved none re-writes the
last good record under its ORIGINAL observation date as
``status='carried_forward'`` with an explicit age, so a prompt reads "ENSO
as of 19 August, 40 days old" instead of reading a stale row as current.
Neutral is a computed result and is never what the code says when it does
not know.

**Three kinds of row share the table** (``row_kind``, Sept 2026). ``live``
rows are what a run wrote about the state as of that run; ``historical``
rows are the published ONI table, one per season, seeded by
``--backfill-oni`` and keyed on their own observation date; ``repaired``
rows are pre-fix live rows that stated a phase with no measurement behind
it and were rebuilt from the ONI history. A consumer that wants "the state
as of today" reads live or repaired rows only — a 1950 row must never be
the newest thing a caller sees.

**The repair pass runs every run.** ``repair_unbacked_rows`` finds every
row stating a phase beside a null ONI or Niño 3.4, rebuilds it from the
newest ONI observation at or before the row's date, and DELETES it when no
such observation exists. It logs repaired, deleted and untouched counts, so
a run that repaired nothing is visible as such.

Usage:
    python -m horizon_scanner.enso.enso_module                 # show the record
    python -m horizon_scanner.enso.enso_module --prompt-context
    python -m horizon_scanner.enso.enso_module --backfill-oni  # seed 1950..now
    python -m horizon_scanner.enso.enso_module --recompute 2026-08-28
    python -m horizon_scanner.enso.enso_module --consumers 2026-08-28
"""

import re
import sys
import json
import argparse
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from horizon_scanner.enso import indices as idx
from horizon_scanner.enso.indices import (
    IndexResolution,
    classify_oni,
    describe_phase,
    resolve_indices,
    valid_anomaly,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

IRI_ENSO_URL = "https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/"

HEADERS = {
    "User-Agent": "PythiaBot/1.0 (humanitarian forecasting research)"
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SeasonProbability:
    """Probability forecast for a single 3-month season."""
    season: str = ""        # e.g. "NDJ", "DJF", "JFM"
    la_nina: float = 0.0    # probability 0-100
    neutral: float = 0.0
    el_nino: float = 0.0


@dataclass
class PlumeSeason:
    """Multi-model plume averages for a single season."""
    season: str = ""
    dyn_mean: Optional[float] = None    # dynamical model average Niño 3.4 anomaly
    stat_mean: Optional[float] = None   # statistical model average
    all_mean: Optional[float] = None    # all models average


@dataclass
class ENSOForecast:
    source: str = "IRI/CPC"
    fetch_date: str = ""
    publication_date: str = ""   # when the IRI forecast was published

    # --- Current state. COMPUTED from the numeric index, never scraped. ---
    current_state: str = ""      # "La Niña", "Neutral", "El Niño"
    strength: str = ""           # "weak" | "moderate" | "strong" | "very strong"
    oni: Optional[float] = None  # three-month mean Niño 3.4 anomaly, °C
    # The Niño 3.4 anomaly the phase rests on: the newest weekly value when a
    # weekly source answered, the ONI itself when only the seasonal table
    # did. This is what the DB column nino34_anomaly holds. It is distinct
    # from nino34_latest_weekly, which is ONLY ever a weekly reading.
    nino34_anomaly: Optional[float] = None
    oni_basis: str = ""          # "oni_table" | "weekly_3month_mean"
    observation_date: str = ""   # the date the index was OBSERVED, not fetched
    source_rank_used: Optional[int] = None
    nino34_source: str = ""      # which numeric source answered

    # Was this run's record read fresh, or carried forward from the last good
    # one? A carried-forward record keeps its ORIGINAL observation date and
    # states its own age, so nothing downstream reads it as current.
    status: str = "fresh"        # "fresh" | "carried_forward" | "historical"
    age_days: Optional[int] = None
    row_kind: str = "live"       # "live" | "historical" | "repaired"

    # What the IRI page said about the phase. Kept for the disagreement
    # check and never used as the answer.
    scraped_state: str = ""
    state_disagreement: str = ""

    alert_status: str = ""       # CPC alert: "La Niña Advisory", "El Niño Watch", etc.
    # The newest WEEKLY Niño 3.4 anomaly, or None when no weekly source
    # answered. Never filled from the ONI: a run that could only read the
    # seasonal table is one source short, and the prompt says so.
    nino34_latest_weekly: Optional[float] = None
    nino34_latest_season: Optional[float] = None  # latest 3-month Niño 3.4 anomaly
    nino34_latest_season_label: str = ""           # e.g. "Aug-Oct 2025"

    # Every source consulted this run and what it said, for provenance.
    index_evidence: dict = field(default_factory=dict)
    # Anything the validators flagged: a continuity jump, a scraped/computed
    # disagreement, a seasonal-TC cross-check mismatch.
    warnings: list = field(default_factory=list)

    # Probabilistic forecast (9 seasons)
    probability_forecast: list = field(default_factory=list)

    # Multi-model plume averages
    plume_averages: list = field(default_factory=list)

    # IOD
    iod_state: str = ""          # "Positive", "Neutral", "Negative"
    iod_dmi: Optional[float] = None
    iod_outlook: str = ""

    # Narrative
    summary: str = ""
    enso_context: str = ""  # concise 1-2 sentence context for prompt injection

    url: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items()
                if v is not None and v != "" and v != [] and v != {}}

    def to_prompt_context(self) -> str:
        """Generate a concise text block for injection into any hazard prompt."""
        # publication_date is regex-scraped and often empty — an unguarded
        # f-string rendered "published )" in prompts.
        bits = []
        if self.nino34_source:
            bits.append(f"index: {self.nino34_source}")
        if self.publication_date:
            bits.append(f"IRI outlook published {self.publication_date}")
        suffix = f" ({'; '.join(bits)})" if bits else ""
        lines = [f"## ENSO State and Forecast{suffix}"]

        # A carried-forward record says so in its first line. A stale phase
        # read as current is the whole failure this module exists to avoid.
        if self.status == "carried_forward":
            age = f"{self.age_days} days old" if self.age_days is not None else "age unknown"
            lines.append(
                f"STALE READING: no ENSO index could be read this run. The state "
                f"below is carried forward from {self.observation_date or 'an earlier run'} "
                f"and is {age}. Treat it as the last known state, not as current."
            )

        if self.current_state:
            state_line = f"Current state: {describe_phase(self.current_state, self.strength)}"
            if self.alert_status:
                state_line += f" ({self.alert_status})"
            if self.oni is not None:
                state_line += f". ONI (3-month mean Niño 3.4): {self.oni:+.2f}°C"
            if self.nino34_latest_weekly is not None:
                state_line += f". Latest weekly Niño 3.4: {self.nino34_latest_weekly:+.1f}°C"
            elif self.oni is not None:
                state_line += ". No independent weekly Niño 3.4 reading this run"
            if self.observation_date:
                state_line += f". Observed {self.observation_date}"
            state_line += "."
            lines.append(state_line)

        if self.summary:
            lines.append(self.summary)

        # Probability table — show key seasons
        if self.probability_forecast:
            prob_lines = []
            for sp in self.probability_forecast:
                if isinstance(sp, dict):
                    s, ln, n, en = sp["season"], sp["la_nina"], sp["neutral"], sp["el_nino"]
                else:
                    s, ln, n, en = sp.season, sp.la_nina, sp.neutral, sp.el_nino
                prob_lines.append(f"  {s}: La Niña {ln:.0f}%, Neutral {n:.0f}%, El Niño {en:.0f}%")
            lines.append("Probabilistic forecast (next 9 seasons):")
            lines.extend(prob_lines)

        if self.iod_state:
            iod_line = f"IOD: {self.iod_state}"
            if self.iod_dmi is not None:
                iod_line += f" (DMI: {self.iod_dmi:+.2f}°C)"
            if self.iod_outlook:
                iod_line += f". Outlook: {self.iod_outlook}"
            lines.append(iod_line)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def fetch_iri_page() -> str:
    """Fetch the IRI ENSO Quick Look page."""
    resp = requests.get(IRI_ENSO_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def extract_enso(html: str, url: str = IRI_ENSO_URL) -> ENSOForecast:
    """Extract ENSO data from the IRI Quick Look HTML."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    f = ENSOForecast(url=url)
    f.fetch_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Publication date: "Published: November 19, 2025"
    pub_m = re.search(r"Published:\s*([\w]+\s+\d{1,2},?\s+\d{4})", text)
    if pub_m:
        try:
            dt = datetime.strptime(pub_m.group(1).replace(",", ""), "%B %d %Y")
            f.publication_date = dt.strftime("%Y-%m-%d")
        except ValueError:
            f.publication_date = pub_m.group(1)

    # The narrative's own statement of the phase. It lands in scraped_state,
    # NOT in current_state: this is the field that said "Neutral" through a
    # strong El Niño in August 2026. It is kept only so the disagreement
    # between the page and the arithmetic can be logged.
    if re.search(r"La\s+Ni[ñn]a\s+(?:Advisory|conditions?\s+(?:are|is)\s+(?:firmly\s+)?established|state)", text, re.IGNORECASE):
        f.scraped_state = "La Niña"
    elif re.search(r"El\s+Ni[ñn]o\s+(?:Advisory|conditions?\s+(?:are|is)\s+established)", text, re.IGNORECASE):
        f.scraped_state = "El Niño"
    elif re.search(r"ENSO[- ]neutral", text):
        f.scraped_state = "Neutral"

    # Also check for state from "experiencing a ... La Niña" or "in a La Niña state"
    if not f.scraped_state:
        state_m = re.search(r"(?:experiencing|in)\s+(?:a\s+)?(?:declining\s+)?(La\s+Ni[ñn]a|El\s+Ni[ñn]o|ENSO[- ]neutral)", text, re.IGNORECASE)
        if state_m:
            raw = state_m.group(1)
            if "La" in raw:
                f.scraped_state = "La Niña"
            elif "El" in raw:
                f.scraped_state = "El Niño"
            else:
                f.scraped_state = "Neutral"

    # Alert status
    alert_m = re.search(r'(?:Alert\s+System\s+Status|maintained\s+a)\s*:?\s*"?([^".\n]+(?:Advisory|Watch|Warning))"?', text, re.IGNORECASE)
    if alert_m:
        f.alert_status = alert_m.group(1).strip().strip('"')

    # Latest weekly Niño 3.4
    weekly_m = re.search(r"(?:latest\s+weekly|week\s+centered)[^.]*?NINO3\.?4\s+index\s+was\s+([+-]?\d+\.?\d*)\s*°?C", text, re.IGNORECASE)
    if not weekly_m:
        weekly_m = re.search(r"NINO3\.?4\s+index[^.]*?was\s+([+-]?\d+\.?\d*)\s*°?C", text, re.IGNORECASE)
    if weekly_m:
        # Range-checked like every other numeric reading: a misread column
        # must be a parse failure, never a value that reaches a prompt.
        f.nino34_latest_weekly = valid_anomaly(weekly_m.group(1))

    # Latest seasonal Niño 3.4: "SST anomaly in the NINO3.4 region during the Aug–Oct 2025 season was -0.42 °C"
    seasonal_m = re.search(
        r"NINO3\.?4\s+region\s+during\s+(?:the\s+)?(\w+[–\-]\w+\s+\d{4})\s+season\s+was\s+([+-]?\d+\.?\d*)\s*°?C",
        text, re.IGNORECASE
    )
    if seasonal_m:
        f.nino34_latest_season_label = seasonal_m.group(1)
        f.nino34_latest_season = valid_anomaly(seasonal_m.group(2))

    # --- Probability forecast table ---
    # Look for the HTML table with Season | La Niña | Neutral | El Niño
    tables = soup.find_all("table")
    for table in tables:
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        header_text = " ".join(headers).lower()
        if "season" in header_text and ("niña" in header_text or "nina" in header_text or "neutral" in header_text):
            rows = table.find_all("tr")
            for row in rows:
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if len(cells) >= 4:
                    season_label = cells[0]
                    # Validate it looks like a season code (3 letters)
                    if re.match(r"^[A-Z]{3}$", season_label):
                        try:
                            sp = SeasonProbability(
                                season=season_label,
                                la_nina=float(cells[1]),
                                neutral=float(cells[2]),
                                el_nino=float(cells[3]),
                            )
                            f.probability_forecast.append(sp)
                        except (ValueError, IndexError):
                            pass
            if f.probability_forecast:
                break  # Use the first matching table

    # --- Plume averages ---
    # Look for rows like "Average, Dynamical models" | -0.654 | ...
    for table in tables:
        all_text = table.get_text()
        if "Average, Dynamical" in all_text or "Average, All models" in all_text:
            rows = table.find_all("tr")
            # Find season headers from the header row
            season_headers = []
            for row in rows:
                ths = [th.get_text(strip=True) for th in row.find_all("th")]
                tds = [td.get_text(strip=True) for td in row.find_all("td")]
                all_cells = ths + tds
                # Look for row with season codes
                season_codes = [c for c in all_cells if re.match(r"^[A-Z]{3}$", c)]
                if len(season_codes) >= 3:
                    season_headers = season_codes
                    break

            if not season_headers:
                # Try first header row
                header_row = rows[0] if rows else None
                if header_row:
                    cells = [c.get_text(strip=True) for c in header_row.find_all(["th", "td"])]
                    season_headers = [c for c in cells if re.match(r"^[A-Z]{3}$", c)]

            dyn_vals, stat_vals, all_vals = None, None, None
            for row in rows:
                cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                row_text = " ".join(cells)
                if "Average, Dynamical" in row_text or "Average,Dynamical" in row_text:
                    nums = re.findall(r"[+-]?\d+\.?\d*", row_text.split("models")[-1] if "models" in row_text else row_text)
                    dyn_vals = [float(x) for x in nums if abs(float(x)) < 5]
                elif "Average, Statistical" in row_text or "Average,Statistical" in row_text:
                    nums = re.findall(r"[+-]?\d+\.?\d*", row_text.split("models")[-1] if "models" in row_text else row_text)
                    stat_vals = [float(x) for x in nums if abs(float(x)) < 5]
                elif "Average, All" in row_text or "Average,All" in row_text:
                    nums = re.findall(r"[+-]?\d+\.?\d*", row_text.split("models")[-1] if "models" in row_text else row_text)
                    all_vals = [float(x) for x in nums if abs(float(x)) < 5]

            if season_headers and (dyn_vals or stat_vals or all_vals):
                for i, season in enumerate(season_headers):
                    ps = PlumeSeason(season=season)
                    if dyn_vals and i < len(dyn_vals):
                        ps.dyn_mean = dyn_vals[i]
                    if stat_vals and i < len(stat_vals):
                        ps.stat_mean = stat_vals[i]
                    if all_vals and i < len(all_vals):
                        ps.all_mean = all_vals[i]
                    f.plume_averages.append(ps)
            break  # Only process first matching table

    # --- IOD ---
    iod_m = re.search(r"Dipole\s+Mode\s+Index\s+measured\s+([–\-+]?\d+\.?\d*)\s*°?C", text)
    if iod_m:
        f.iod_dmi = float(iod_m.group(1).replace("–", "-"))
        if f.iod_dmi > 0.4:
            f.iod_state = "Positive"
        elif f.iod_dmi < -0.4:
            f.iod_state = "Negative"
        else:
            f.iod_state = "Neutral"

    iod_outlook_m = re.search(r"(transition\s+to\s+IOD[- ]neutral[^.]+\.)", text, re.IGNORECASE)
    if iod_outlook_m:
        f.iod_outlook = re.sub(r"\s+", " ", iod_outlook_m.group(1)).strip()

    # --- Summary / context ---
    # Grab the Quick Look summary paragraph
    summary_m = re.search(
        r"(As\s+of\s+mid-\w+\s+\d{4}[^.]+\.\s+The\s+(?:IRI\s+ENSO|CCSR)[^.]+\.)",
        text,
        re.DOTALL
    )
    if summary_m:
        f.summary = re.sub(r"\s+", " ", summary_m.group(1)).strip()

    # Build a concise context string
    _build_context(f)

    _log_forecast(f)
    return f


def _build_context(f: ENSOForecast):
    """Build a concise 1-2 sentence ENSO context for prompt injection."""
    parts = []
    if f.status == STATUS_CARRIED_FORWARD:
        age = f"{f.age_days} days old" if f.age_days is not None else "age unknown"
        parts.append(f"ENSO reading is STALE ({age}); no index could be read this run.")
    if f.current_state:
        parts.append(f"Current ENSO state: {describe_phase(f.current_state, f.strength)}")
        if f.oni is not None:
            parts.append(f"(ONI: {f.oni:+.2f}°C)")
        elif f.nino34_anomaly is not None:
            parts.append(f"(Niño 3.4: {f.nino34_anomaly:+.1f}°C)")

    if f.probability_forecast:
        # Find the transition point — when does the dominant state change?
        for sp in f.probability_forecast:
            probs = sp if isinstance(sp, dict) else asdict(sp)
            if probs["el_nino"] > probs["neutral"] and probs["el_nino"] > probs["la_nina"]:
                parts.append(f"El Niño becomes most likely by {probs['season']}.")
                break
            elif probs["la_nina"] < 20 and probs["la_nina"] > 0 and probs["neutral"] > 50:
                parts.append(f"Transition to ENSO-neutral expected by {probs['season']}.")
                break

    f.enso_context = " ".join(parts)


def _log_forecast(f: ENSOForecast):
    logger.info(
        "  ENSO state: %s (%s) [%s]",
        describe_phase(f.current_state, f.strength) or "unresolved",
        f.alert_status or "no alert",
        f.status,
    )
    if f.oni is not None:
        logger.info(
            "  ONI: %+.2f°C via %s (%s), observed %s",
            f.oni, f.nino34_source or "?", f.oni_basis or "?",
            f.observation_date or "?",
        )
    if f.nino34_latest_weekly is not None:
        logger.info(f"  Latest weekly Niño 3.4: {f.nino34_latest_weekly:+.1f}°C")
    logger.info(f"  Published: {f.publication_date}")
    if f.probability_forecast:
        logger.info(f"  Probability forecast: {len(f.probability_forecast)} seasons")
        for sp in f.probability_forecast[:3]:
            if isinstance(sp, SeasonProbability):
                logger.info(f"    {sp.season}: LN={sp.la_nina:.0f}% N={sp.neutral:.0f}% EN={sp.el_nino:.0f}%")
    if f.plume_averages:
        logger.info(f"  Plume averages: {len(f.plume_averages)} seasons")
    if f.iod_state:
        logger.info(f"  IOD: {f.iod_state} (DMI={f.iod_dmi:+.2f}°C)")


# ---------------------------------------------------------------------------
# Public API — for use by other Pythia modules
# ---------------------------------------------------------------------------

def get_enso_state(cache_path: Optional[Path] = None, max_age_days: int = 7) -> ENSOForecast:
    """
    Get current ENSO state and forecast.

    If a cached file exists and is recent enough, use it.
    Otherwise fetch fresh data from IRI.

    Args:
        cache_path: Path to cache file (default: ./output/enso_forecast.json)
        max_age_days: Maximum age of cache in days before re-fetching

    Returns:
        ENSOForecast object
    """
    if cache_path is None:
        cache_path = Path(__file__).parent / "output" / "enso_forecast.json"

    # Check cache
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            fetch_date = data.get("fetch_date", "")
            if fetch_date:
                age = datetime.now(timezone.utc) - datetime.fromisoformat(fetch_date.replace("Z", "+00:00"))
                if age.days < max_age_days:
                    logger.info(f"Using cached ENSO data from {fetch_date}")
                    # Reconstruct
                    f = ENSOForecast(**{k: v for k, v in data.items()
                                       if k in ENSOForecast.__dataclass_fields__})
                    return f
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")

    # Fetch fresh — numeric ladder first, IRI page for decoration.
    logger.info("Building a fresh ENSO record (numeric index first)...")
    forecast = build_enso_record()

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(forecast.to_dict(), indent=2, ensure_ascii=False))
    logger.info(f"Cached ENSO data to {cache_path}")

    return forecast


def get_enso_prompt_context(cache_path: Optional[Path] = None) -> str:
    """
    Convenience function: get ENSO state and return a prompt-ready text block.
    This is the main entry point for other Pythia hazard pipelines.

    Tries DB first, then falls back to live scrape (with JSON cache).
    """
    # 1. Try DB first
    db_forecast = load_enso_state_from_db()
    if db_forecast is not None:
        logger.info("Using ENSO data from DB")
        return db_forecast.to_prompt_context()

    # 2. Fall back to existing live scrape / JSON cache
    forecast = get_enso_state(cache_path)

    # 3. Store to DB for future use
    store_enso_state(forecast)

    return forecast.to_prompt_context()


# ---------------------------------------------------------------------------
# Assembly, classification and validation
# ---------------------------------------------------------------------------

STATUS_FRESH = "fresh"
STATUS_CARRIED_FORWARD = "carried_forward"
#: The published ONI table, seeded by --backfill-oni. Never "fresh": a row
#: for January 1950 is not a reading any run took.
STATUS_HISTORICAL = "historical"

ROW_KIND_LIVE = "live"
ROW_KIND_HISTORICAL = "historical"
ROW_KIND_REPAIRED = "repaired"

#: How old the newest stored observation may be before the DB record is
#: refused and a consumer falls back to a live fetch. 100, not 30: the ONI
#: is a three-month running mean published monthly, so its observation date
#: (the centre month) is structurally 30 to 70 days behind the calendar. At
#: 30 the seasonal TC module discarded a correctly detected strong El Niño
#: as stale on 2026-09-04 ("65 days old (max 30)") and built its context
#: with no ENSO signal at all. The ladder itself admits observations up to
#: indices.MAX_OBSERVATION_AGE_DAYS (120); this bound sits just under it.
DB_MAX_OBSERVATION_AGE_DAYS = 100


def _run_stamp(today=None) -> str:
    """The run's own timestamp, so a record is filed under the run's date.

    Tests inject ``today``; production passes None and gets the wall clock.
    Deriving it from the wall clock unconditionally made two runs on
    different simulated days collide on the table's primary key.
    """

    if today is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{today.isoformat()}T00:00:00Z"


def apply_indices(f: ENSOForecast, resolution: IndexResolution) -> ENSOForecast:
    """Stamp the numeric record onto a forecast and COMPUTE its phase.

    This is the only place ``current_state`` is ever set. With no number
    resolved it stays empty — there is no phase without an index, and the
    caller must carry a record forward rather than default to Neutral.
    """

    f.index_evidence = resolution.as_evidence()
    if not resolution.resolved:
        f.current_state = ""
        f.strength = ""
        return f

    f.nino34_anomaly = resolution.nino34
    f.nino34_latest_weekly = resolution.nino34_weekly
    f.oni = resolution.oni
    f.oni_basis = resolution.oni_basis or ""
    f.observation_date = (
        resolution.observation_date.isoformat() if resolution.observation_date else ""
    )
    f.source_rank_used = resolution.source_rank_used
    f.nino34_source = resolution.source_name or ""
    f.current_state, f.strength = classify_oni(resolution.oni)
    f.status = STATUS_FRESH
    f.age_days = 0
    return f


def check_state_disagreement(f: ENSOForecast) -> str:
    """Compare the scraped narrative's phase against the computed one.

    The computed phase wins, always. The disagreement is recorded because it
    is the cheapest possible signal that a page's wording has drifted, or
    that our arithmetic has.
    """

    if not f.scraped_state or not f.current_state:
        return ""
    if f.scraped_state == f.current_state:
        return ""
    note = (
        f"the IRI page states {f.scraped_state!r} but the index computes "
        f"{f.current_state!r} (ONI {f.oni:+.2f}) — the computed phase is used"
        if f.oni is not None
        else f"the IRI page states {f.scraped_state!r} but the index computes "
        f"{f.current_state!r} — the computed phase is used"
    )
    logger.warning("[enso] %s", note)
    return note


def continuity_problems(
    previous: Optional["ENSOForecast"], current: ENSOForecast
) -> list[str]:
    """Moves large enough to want a second opinion before they are believed.

    Two: an ONI step over :data:`indices.ONI_JUMP_LIMIT` since the last
    record, and a phase transition that skips Neutral (El Niño straight to
    La Niña, or back). Both are physically implausible month to month, so
    either is far likelier to be a changed column than a changed ocean.
    """

    problems: list[str] = []
    if previous is None or current.oni is None or previous.oni is None:
        return problems

    step = abs(float(current.oni) - float(previous.oni))
    if step > idx.ONI_JUMP_LIMIT:
        problems.append(
            f"ONI moved {step:.2f} °C since {previous.observation_date or 'the last record'} "
            f"({previous.oni:+.2f} -> {current.oni:+.2f}), over the "
            f"{idx.ONI_JUMP_LIMIT:.1f} °C limit"
        )

    poles = {idx.PHASE_EL_NINO, idx.PHASE_LA_NINA}
    if (
        previous.current_state in poles
        and current.current_state in poles
        and previous.current_state != current.current_state
    ):
        problems.append(
            f"phase went {previous.current_state} -> {current.current_state} "
            "without passing through Neutral"
        )
    return problems


def corroborated(
    current: ENSOForecast,
    problems: list[str],
    *,
    get=None,
    today=None,
) -> tuple[bool, str]:
    """Does a second numeric source agree with an extraordinary move?

    Returns (agreed, note). An uncorroborated jump is NOT written as a fresh
    reading — the caller carries the previous record forward instead, which
    is the conservative answer when two explanations are open and one of
    them is "a column moved".
    """

    if not problems or current.oni is None:
        return True, ""

    resolution = IndexResolution(
        source_rank_used=current.source_rank_used,
        source_name=current.nino34_source,
    )
    other = idx.corroborate(resolution, get=get, today=today)
    if other is None or other.oni is None:
        return False, (
            "; ".join(problems)
            + " — and no second numeric source could be read to confirm it"
        )

    gap = abs(float(other.oni) - float(current.oni))
    if gap > idx.ONI_AGREEMENT_TOLERANCE:
        return False, (
            "; ".join(problems)
            + f" — and {other.source_name} disagrees ({other.oni:+.2f} vs "
            f"{current.oni:+.2f}, gap {gap:.2f} °C)"
        )
    return True, (
        "; ".join(problems)
        + f" — confirmed by {other.source_name} ({other.oni:+.2f} °C)"
    )


def cross_check_seasonal_tc(con, phase: str) -> str:
    """A free second opinion from the seasonal TC context already in this run.

    The TSR and BoM outlook PDFs carry ENSO notes in prose, and in August
    2026 they disagreed with the stored record while nothing compared them.
    A string test over the cached TC context costs one query and no network.

    Returns a note when the cached text names a phase the computed one
    contradicts, "" otherwise. Never raises — a cross-check that cannot run
    is not a reason to lose a good reading.
    """

    if not phase:
        return ""
    try:
        rows = con.execute(
            "SELECT context_text FROM seasonal_tc_context_cache"
        ).fetchall()
    except Exception:  # noqa: BLE001 - table may not exist yet
        return ""

    hits: dict[str, int] = {idx.PHASE_EL_NINO: 0, idx.PHASE_LA_NINA: 0}
    for row in rows:
        text = str(row[0] or "")
        if not text:
            continue
        if re.search(r"El\s+Ni[ñn]o", text, re.IGNORECASE):
            hits[idx.PHASE_EL_NINO] += 1
        if re.search(r"La\s+Ni[ñn]a", text, re.IGNORECASE):
            hits[idx.PHASE_LA_NINA] += 1

    opposite = (
        idx.PHASE_LA_NINA if phase == idx.PHASE_EL_NINO else idx.PHASE_EL_NINO
    )
    if phase in hits and hits.get(opposite, 0) > 0 and hits.get(phase, 0) == 0:
        note = (
            f"the cached seasonal TC outlooks mention {opposite} in "
            f"{hits[opposite]} entr{'y' if hits[opposite] == 1 else 'ies'} and never "
            f"mention {phase}, which the index computes"
        )
        logger.warning("[enso] cross-check: %s", note)
        return note
    return ""


def validation_problems(f: ENSOForecast) -> list[str]:
    """What must be true before a record may be written as a state.

    One rule does the work: a null Niño 3.4 can never accompany a stated
    phase. That alone would have caught August 2026, where a page-scraped
    "Neutral" was stored beside no index at all.
    """

    problems: list[str] = []
    if f.status == STATUS_CARRIED_FORWARD:
        # A carried-forward record is a copy of one that already passed.
        return problems
    if f.current_state and valid_anomaly(f.nino34_anomaly) is None:
        problems.append(
            f"phase {f.current_state!r} stated with no valid Niño 3.4 anomaly "
            f"({f.nino34_anomaly!r})"
        )
    if f.current_state and f.oni is None:
        problems.append(f"phase {f.current_state!r} stated with no ONI")
    for label, value in (
        ("Niño 3.4", f.nino34_anomaly),
        ("weekly Niño 3.4", f.nino34_latest_weekly),
    ):
        if value is not None and valid_anomaly(value) is None:
            problems.append(
                f"{label} {value!r} is outside [{idx.NINO34_MIN}, {idx.NINO34_MAX}] °C"
            )
    return problems


def carry_forward(previous: ENSOForecast, *, today=None) -> ENSOForecast:
    """The last good record, re-stated under its ORIGINAL observation date.

    Its age is computed and printed, so a prompt says how old the reading is
    rather than presenting it as this month's.
    """

    from datetime import date as _date

    day = today or datetime.now(timezone.utc).date()
    f = ENSOForecast(**{
        k: v for k, v in asdict(previous).items()
        if k in ENSOForecast.__dataclass_fields__
    })
    f.status = STATUS_CARRIED_FORWARD
    f.fetch_date = _run_stamp(today)

    observed = None
    if f.observation_date:
        try:
            observed = _date.fromisoformat(f.observation_date[:10])
        except ValueError:
            observed = None
    f.age_days = (day - observed).days if observed else None
    _build_context(f)
    return f


def build_enso_record(
    *,
    html: Optional[str] = None,
    get=None,
    today=None,
    fetch_page: bool = True,
) -> ENSOForecast:
    """Assemble the record field by field: numeric first, decoration after.

    The numeric ladder decides whether there is a record at all. The IRI
    page is then read for the probability table, the plume, the IOD and the
    narrative — and a failure there costs only those fields.
    """

    resolution = resolve_indices(get=get, today=today)

    f = ENSOForecast(url=IRI_ENSO_URL)
    f.fetch_date = _run_stamp(today)

    if html is None and fetch_page:
        try:
            html = fetch_iri_page()
        except Exception as exc:  # noqa: BLE001 - decoration, not the record
            logger.warning(
                "[enso] IRI page unavailable (%s) — the phase is unaffected", exc
            )
            html = None

    if html is not None:
        try:
            scraped = extract_enso(html)
            # Everything the page contributes, and nothing it says about the
            # phase: current_state is set only by apply_indices.
            f.publication_date = scraped.publication_date
            f.alert_status = scraped.alert_status
            f.probability_forecast = scraped.probability_forecast
            f.plume_averages = scraped.plume_averages
            f.iod_state = scraped.iod_state
            f.iod_dmi = scraped.iod_dmi
            f.iod_outlook = scraped.iod_outlook
            f.summary = scraped.summary
            f.scraped_state = scraped.scraped_state
            f.nino34_latest_season = scraped.nino34_latest_season
            f.nino34_latest_season_label = scraped.nino34_latest_season_label
        except Exception as exc:  # noqa: BLE001
            logger.warning("[enso] IRI page parse failed (%s); numeric record kept", exc)

    apply_indices(f, resolution)

    note = check_state_disagreement(f)
    if note:
        f.state_disagreement = note
        f.warnings.append(note)

    _build_context(f)
    _log_forecast(f)
    return f


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------

def store_enso_state(forecast: ENSOForecast) -> bool:
    """
    Persist an ENSOForecast to the enso_state DuckDB table.

    Returns True on success, False on failure. Non-fatal on DB errors.
    """
    try:
        from pythia.db.schema import connect, ensure_schema
    except ImportError:
        logger.warning("pythia.db.schema not available; skipping ENSO DB store")
        return False

    try:
        # Parse fetch_date to a date string (YYYY-MM-DD)
        fd = forecast.fetch_date
        if fd:
            try:
                dt = datetime.fromisoformat(fd.replace("Z", "+00:00"))
                fetch_date_str = dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                fetch_date_str = fd[:10] if len(fd) >= 10 else fd
        else:
            fetch_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Serialize lists to JSON
        prob_data = []
        for sp in forecast.probability_forecast:
            if isinstance(sp, dict):
                prob_data.append(sp)
            else:
                prob_data.append(asdict(sp))
        forecast_json = json.dumps(prob_data) if prob_data else None

        plume_data = []
        for ps in forecast.plume_averages:
            if isinstance(ps, dict):
                plume_data.append(ps)
            else:
                plume_data.append(asdict(ps))
        plume_json = json.dumps(plume_data) if plume_data else None

        # Validate BEFORE the write. A null Niño 3.4 must never reach the
        # table beside a stated phase — that is the single rule that would
        # have caught August 2026, and refusing the write is the point of
        # having it.
        problems = validation_problems(forecast)
        if problems:
            for problem in problems:
                logger.error("[enso] refusing to store: %s", problem)
            return False

        raw_context = forecast.to_prompt_context()
        evidence_json = (
            json.dumps(forecast.index_evidence) if forecast.index_evidence else None
        )
        warnings_json = json.dumps(forecast.warnings) if forecast.warnings else None

        con = connect(read_only=False)
        try:
            ensure_schema(con)
            forecast.warnings.extend(
                w for w in [cross_check_seasonal_tc(con, forecast.current_state)] if w
            )
            warnings_json = (
                json.dumps(forecast.warnings) if forecast.warnings else None
            )
            con.execute(
                """
                INSERT OR REPLACE INTO enso_state
                    (fetch_date, enso_phase, nino34_anomaly, iod_phase,
                     forecast_json, plume_json, raw_context,
                     oni, enso_strength, oni_basis, observation_date,
                     source_rank_used, nino34_source, status, age_days,
                     scraped_phase, index_evidence_json, warnings_json,
                     nino34_weekly, row_kind)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    fetch_date_str,
                    forecast.current_state or None,
                    forecast.nino34_anomaly,
                    forecast.iod_state or None,
                    forecast_json,
                    plume_json,
                    raw_context,
                    forecast.oni,
                    forecast.strength or None,
                    forecast.oni_basis or None,
                    forecast.observation_date or None,
                    forecast.source_rank_used,
                    forecast.nino34_source or None,
                    forecast.status or STATUS_FRESH,
                    forecast.age_days,
                    forecast.scraped_state or None,
                    evidence_json,
                    warnings_json,
                    forecast.nino34_latest_weekly,
                    forecast.row_kind or ROW_KIND_LIVE,
                ],
            )
            logger.info(
                "Stored ENSO state to DB (fetch_date=%s, phase=%s, oni=%s, status=%s)",
                fetch_date_str,
                describe_phase(forecast.current_state, forecast.strength) or "none",
                f"{forecast.oni:+.2f}" if forecast.oni is not None else "none",
                forecast.status,
            )
            return True
        finally:
            con.close()
    except Exception as exc:
        logger.warning("Failed to store ENSO state to DB: %s", exc)
        return False


def load_enso_state_from_db(
    max_age_days: int = DB_MAX_OBSERVATION_AGE_DAYS,
) -> Optional[ENSOForecast]:
    """
    Load the most recent ENSO state from the DB: the newest LIVE (or
    repaired) row. Historical ONI rows are never a candidate — they are the
    record's past, not its present.

    Returns None if no data, data is too old (see
    ``DB_MAX_OBSERVATION_AGE_DAYS`` for why the bound is 100 days), or on
    any DB error.
    """
    try:
        from pythia.db.schema import connect, ensure_schema
    except ImportError:
        return None

    try:
        con = connect(read_only=True)
        try:
            ensure_schema(con)
            rows = con.execute(
                """
                SELECT fetch_date, enso_phase, nino34_anomaly, iod_phase,
                       forecast_json, plume_json, raw_context,
                       oni, enso_strength, oni_basis, observation_date,
                       source_rank_used, nino34_source, status, age_days,
                       scraped_phase, nino34_weekly, row_kind
                FROM enso_state
                WHERE COALESCE(row_kind, 'live') <> 'historical'
                ORDER BY fetch_date DESC
                LIMIT 1
                """
            ).fetchall()
            if not rows:
                return None

            row = rows[0]
            fetch_date_val = row[0]  # DATE type

            # Check staleness
            if fetch_date_val is not None:
                from datetime import date
                if isinstance(fetch_date_val, str):
                    fetch_date_obj = datetime.strptime(fetch_date_val, "%Y-%m-%d").date()
                elif isinstance(fetch_date_val, datetime):
                    fetch_date_obj = fetch_date_val.date()
                elif isinstance(fetch_date_val, date):
                    fetch_date_obj = fetch_date_val
                else:
                    fetch_date_obj = None

                if fetch_date_obj is not None:
                    # Age is measured against the OBSERVATION where there is
                    # one. A carried-forward row is re-written every run, so
                    # its fetch_date stays current while the reading it
                    # carries is months old — measuring the fetch would make
                    # a stale record look permanently fresh.
                    reference = fetch_date_obj
                    observed_raw = rows[0][10]
                    if observed_raw is not None:
                        try:
                            if isinstance(observed_raw, str):
                                reference = datetime.strptime(
                                    observed_raw[:10], "%Y-%m-%d"
                                ).date()
                            elif isinstance(observed_raw, datetime):
                                reference = observed_raw.date()
                            elif isinstance(observed_raw, date):
                                reference = observed_raw
                        except (ValueError, TypeError):
                            reference = fetch_date_obj
                    age_days = (datetime.now(timezone.utc).date() - reference).days
                    if age_days > max_age_days:
                        logger.info(
                            "ENSO DB observation is %d days old (max %d); treating as stale",
                            age_days, max_age_days,
                        )
                        return None

            # Reconstruct ENSOForecast
            f = ENSOForecast()
            f.fetch_date = str(fetch_date_val) if fetch_date_val else ""
            f.current_state = row[1] or ""
            f.nino34_anomaly = row[2]
            f.iod_state = row[3] or ""
            f.oni = row[7]
            f.strength = row[8] or ""
            f.oni_basis = row[9] or ""
            f.observation_date = str(row[10]) if row[10] else ""
            f.source_rank_used = row[11]
            f.nino34_source = row[12] or ""
            f.status = row[13] or STATUS_FRESH
            f.age_days = row[14]
            f.scraped_state = row[15] or ""
            f.nino34_latest_weekly = row[16]
            f.row_kind = row[17] or ROW_KIND_LIVE

            # Restore probability forecast
            if row[4]:
                try:
                    f.probability_forecast = json.loads(row[4])
                except (json.JSONDecodeError, TypeError):
                    pass

            # Restore plume averages
            if row[5]:
                try:
                    f.plume_averages = json.loads(row[5])
                except (json.JSONDecodeError, TypeError):
                    pass

            # raw_context is available but we regenerate via to_prompt_context()
            # for consistency; store it as a fallback attribute
            f._raw_context_from_db = row[6] or ""

            return f
        finally:
            con.close()
    except Exception as exc:
        logger.warning("Failed to load ENSO state from DB: %s", exc)
        return None


def load_last_good_record() -> Optional[ENSOForecast]:
    """The newest row that actually carries a computed phase, any age.

    Deliberately not age-gated: this is the record a failed run carries
    FORWARD, and refusing to return it because it is old would leave the
    caller with nothing and tempt a default. Age is stated on the copy
    instead.
    """

    try:
        from pythia.db.schema import connect, ensure_schema
    except ImportError:
        return None

    try:
        con = connect(read_only=True)
        try:
            ensure_schema(con)
            rows = con.execute(
                """
                SELECT enso_phase, nino34_anomaly, iod_phase, forecast_json,
                       plume_json, oni, enso_strength, oni_basis,
                       observation_date, source_rank_used, nino34_source,
                       nino34_weekly
                FROM enso_state
                WHERE oni IS NOT NULL AND enso_phase IS NOT NULL
                  AND nino34_anomaly IS NOT NULL
                ORDER BY observation_date DESC NULLS LAST,
                         CASE WHEN COALESCE(row_kind, 'live') = 'historical'
                              THEN 1 ELSE 0 END,
                         fetch_date DESC
                LIMIT 1
                """
            ).fetchall()
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load the last good ENSO record: %s", exc)
        return None

    if not rows:
        return None
    row = rows[0]
    f = ENSOForecast()
    f.current_state = row[0] or ""
    f.nino34_anomaly = row[1]
    f.iod_state = row[2] or ""
    if row[3]:
        try:
            f.probability_forecast = json.loads(row[3])
        except (json.JSONDecodeError, TypeError):
            pass
    if row[4]:
        try:
            f.plume_averages = json.loads(row[4])
        except (json.JSONDecodeError, TypeError):
            pass
    f.oni = row[5]
    f.strength = row[6] or ""
    f.oni_basis = row[7] or ""
    f.observation_date = str(row[8]) if row[8] else ""
    f.source_rank_used = row[9]
    f.nino34_source = row[10] or ""
    f.nino34_latest_weekly = row[11]
    return f


def fetch_and_store_enso(*, get=None, today=None, fetch_page: bool = True) -> bool:
    """Build this run's ENSO record and store it. Never raises.

    The table gets a row every run. Three outcomes:

    * a numeric source answered and the reading is ordinary — write it
      ``fresh``;
    * a numeric source answered but the move is extraordinary and no second
      source confirms it — carry the previous record forward rather than
      believe a jump whose cheapest explanation is a changed column;
    * no numeric source answered — carry the previous record forward under
      its original observation date, with its age stated.

    Neutral is never what this function writes because it did not know.

    Whatever the outcome, the row-kind migration and the repair pass run
    afterwards (see :func:`repair_unbacked_rows`): the repair sources its
    ONI from the history the workflow backfilled one step earlier, so it
    works with no network and runs on every Resolver Update.
    """

    try:
        ok = _fetch_and_store_this_run(get=get, today=today, fetch_page=fetch_page)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_and_store_enso failed: %s", exc)
        ok = False
    try:
        classify_row_kinds()
        repair_unbacked_rows(today=today)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[enso] repair pass failed: %s", exc)
    return ok


def _fetch_and_store_this_run(*, get=None, today=None, fetch_page: bool = True) -> bool:
    """The write for THIS run; see :func:`fetch_and_store_enso` for the policy."""

    try:
        previous = load_last_good_record()
        record = build_enso_record(get=get, today=today, fetch_page=fetch_page)

        if record.current_state:
            problems = continuity_problems(previous, record)
            if problems:
                agreed, note = corroborated(record, problems, get=get, today=today)
                record.warnings.append(note)
                if not agreed:
                    logger.error(
                        "[enso] refusing an uncorroborated move: %s", note
                    )
                    record = None if previous is None else carry_forward(
                        previous, today=today
                    )
                    if record is None:
                        logger.error(
                            "[enso] and no previous record exists to carry forward"
                        )
                        return False
                    record.warnings.append(note)

        if not record.current_state:
            if previous is None:
                logger.error(
                    "[enso] no numeric source answered and there is no previous "
                    "record to carry forward — writing nothing, which is correct: "
                    "an absent row is honest, a defaulted Neutral is not"
                )
                print(
                    "::warning::ENSO wrote no row this run (no numeric source "
                    "answered and no previous record exists) — the prompt block "
                    "will be absent until a source answers",
                    file=sys.stderr,
                )
                return False
            record = carry_forward(previous, today=today)
            logger.warning(
                "[enso] carrying forward the %s reading of %s (%s days old)",
                describe_phase(record.current_state, record.strength),
                record.observation_date or "unknown date",
                record.age_days if record.age_days is not None else "?",
            )

        ok = store_enso_state(record)
        if ok:
            logger.info(
                "fetch_and_store_enso: stored ENSO state (phase=%s, oni=%s, status=%s)",
                describe_phase(record.current_state, record.strength),
                f"{record.oni:+.2f}" if record.oni is not None else "none",
                record.status,
            )
        return ok
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_and_store_enso failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Repair: rows that state a phase with nothing behind it
# ---------------------------------------------------------------------------

_ROW_KIND_MIGRATION_SQL = """
    UPDATE enso_state
       SET row_kind = CASE
             WHEN oni_basis = 'oni_table'
              AND nino34_source = 'cpc_oni_ascii'
              AND observation_date IS NOT NULL
              AND fetch_date = observation_date
              AND COALESCE(age_days, 0) = 0
             THEN 'historical'
             ELSE 'live'
           END
     WHERE row_kind IS NULL
"""


def classify_row_kinds(con=None) -> dict:
    """Stamp ``row_kind`` on rows written before the column existed. Idempotent.

    A backfilled ONI row is recognisable by its own signature: the ONI
    table as basis and source, keyed on its observation date, age zero. A
    live run using rank 3 can never match, because the ONI for a centre
    month is published after the following month ends, so its fetch date
    is always later than its observation date. Everything else is live.
    Historical rows also get ``status='historical'``: "fresh" on a row for
    January 1950 said the run had read it, and no run had.
    """

    own = con is None
    if own:
        try:
            from pythia.db.schema import connect, ensure_schema
        except ImportError:
            return {"historical": 0, "live": 0}
        con = connect(read_only=False)
        ensure_schema(con)
    try:
        before = con.execute(
            "SELECT COUNT(*) FROM enso_state WHERE row_kind IS NULL"
        ).fetchone()[0]
        if not before:
            return {"historical": 0, "live": 0}
        con.execute(_ROW_KIND_MIGRATION_SQL)
        relabelled = con.execute(
            "SELECT COUNT(*) FROM enso_state "
            "WHERE row_kind = 'historical' AND COALESCE(status, '') <> 'historical'"
        ).fetchone()[0]
        con.execute(
            "UPDATE enso_state SET status = ? WHERE row_kind = 'historical'",
            [STATUS_HISTORICAL],
        )
        counts = dict(con.execute(
            "SELECT row_kind, COUNT(*) FROM enso_state GROUP BY row_kind"
        ).fetchall())
        logger.info(
            "[enso] row_kind migration: %d rows classified (%s); %d historical "
            "rows relabelled from 'fresh' to 'historical'",
            before, counts, relabelled,
        )
        return {"historical": int(counts.get("historical", 0)),
                "live": int(counts.get("live", 0))}
    finally:
        if own:
            con.close()


def repair_unbacked_rows(con=None, *, today=None) -> dict:
    """Rebuild every row that states a phase beside a null measurement.

    Runs at the end of every ``fetch_and_store_enso``, after the workflow's
    ONI backfill, so the newest ONI observation at or before each bad row's
    date is already in the table (``oni_basis='oni_table'``) and no network
    is needed. For each such row at date D:

    * found — write oni, nino34_anomaly (the ONI: a seasonal series is the
      index), observation_date, basis, source, rank 3, age_days and the
      phase and strength recomputed with ``classify_oni``; move the row's
      original phase word into ``scraped_phase``; append the repair to
      ``warnings_json``; ``row_kind='repaired'``, ``status='carried_forward'``
      (which is what it is: the last good observation as of D, its age on
      the row);
    * not found — DELETE the row. An absent row is honest; a phase with
      nothing behind it is not.

    Returns and logs ``{"repaired": N, "deleted": M, "untouched": K}`` where
    K counts the phase-bearing rows that needed nothing, so a run that
    repaired nothing is visible as such.
    """

    from datetime import date as _date

    own = con is None
    if own:
        try:
            from pythia.db.schema import connect, ensure_schema
        except ImportError:
            return {"repaired": 0, "deleted": 0, "untouched": 0}
        con = connect(read_only=False)
        ensure_schema(con)
    day = today or datetime.now(timezone.utc).date()
    stamp = day.isoformat()
    summary = {"repaired": 0, "deleted": 0, "untouched": 0}
    try:
        bad = con.execute(
            """
            SELECT fetch_date, enso_phase, warnings_json
            FROM enso_state
            WHERE enso_phase IS NOT NULL
              AND (oni IS NULL OR nino34_anomaly IS NULL)
            ORDER BY fetch_date
            """
        ).fetchall()
        summary["untouched"] = int(con.execute(
            "SELECT COUNT(*) FROM enso_state WHERE enso_phase IS NOT NULL "
            "AND oni IS NOT NULL AND nino34_anomaly IS NOT NULL"
        ).fetchone()[0])

        for fetch_date, phase_word, warnings_json in bad:
            row_day = (
                fetch_date if isinstance(fetch_date, _date)
                else _date.fromisoformat(str(fetch_date)[:10])
            )
            found = con.execute(
                """
                SELECT observation_date, oni
                FROM enso_state
                WHERE oni_basis = 'oni_table'
                  AND oni IS NOT NULL
                  AND observation_date IS NOT NULL
                  AND observation_date <= ?
                ORDER BY observation_date DESC, fetch_date DESC
                LIMIT 1
                """,
                [row_day],
            ).fetchone()
            if found is None:
                con.execute("DELETE FROM enso_state WHERE fetch_date = ?", [row_day])
                summary["deleted"] += 1
                logger.warning(
                    "[enso] repair: deleted the %s row — it stated %r with no "
                    "measurement and no ONI observation predates it",
                    row_day, phase_word,
                )
                continue

            observed_raw, oni = found
            observed = (
                observed_raw if isinstance(observed_raw, _date)
                else _date.fromisoformat(str(observed_raw)[:10])
            )
            oni = float(oni)
            phase, strength = classify_oni(oni)
            age_days = (row_day - observed).days
            try:
                warnings = json.loads(warnings_json) if warnings_json else []
                if not isinstance(warnings, list):
                    warnings = [warnings]
            except (json.JSONDecodeError, TypeError):
                warnings = [str(warnings_json)]
            warnings.append(
                f"repaired on {stamp}: this row stated {phase_word!r} with no "
                f"measurement; ONI {oni:+.2f} observed {observed.isoformat()} "
                f"({age_days} days before this row) substituted from the CPC "
                f"ONI table and the phase recomputed"
            )
            record = ENSOForecast()
            record.current_state, record.strength = phase, strength
            record.oni = oni
            record.nino34_anomaly = oni
            record.nino34_latest_weekly = None
            record.oni_basis = idx.BASIS_ONI_TABLE
            record.observation_date = observed.isoformat()
            record.status = STATUS_CARRIED_FORWARD
            record.age_days = age_days
            record.scraped_state = str(phase_word)
            _build_context(record)
            con.execute(
                """
                UPDATE enso_state
                   SET enso_phase = ?, enso_strength = ?, oni = ?,
                       nino34_anomaly = ?, nino34_weekly = NULL,
                       observation_date = ?, oni_basis = ?,
                       nino34_source = 'cpc_oni_ascii', source_rank_used = 3,
                       age_days = ?, status = ?, scraped_phase = ?,
                       warnings_json = ?, row_kind = ?, raw_context = ?
                 WHERE fetch_date = ?
                """,
                [
                    phase, strength or None, oni, oni, observed,
                    idx.BASIS_ONI_TABLE, age_days, STATUS_CARRIED_FORWARD,
                    str(phase_word), json.dumps(warnings), ROW_KIND_REPAIRED,
                    record.to_prompt_context(), row_day,
                ],
            )
            summary["repaired"] += 1
            logger.warning(
                "[enso] repair: %s row rebuilt — %r with no measurement became "
                "%s (ONI %+.2f observed %s, %d days old)",
                row_day, phase_word, describe_phase(phase, strength), oni,
                observed.isoformat(), age_days,
            )
    finally:
        if own:
            con.close()

    logger.info(
        "[enso] repair pass: repaired %d, deleted %d, untouched %d",
        summary["repaired"], summary["deleted"], summary["untouched"],
    )
    return summary


def backfill_oni_history(*, get=None) -> int:
    """Seed ``enso_state`` from the published ONI table, 1950 to present.

    ONI is a complete historical table, so this is one pass. It gives the
    continuity check something to compare against and gives base-rate and RC
    work a real ENSO history rather than three rows. Rows are written as
    ``row_kind='historical'`` / ``status='historical'`` under the season's
    own observation date; a historical row already present is replaced,
    since the ONI table itself is the record. A LIVE row whose fetch date
    happens to equal a season's centre date is left alone and counted — the
    table's key is the date, and a run's own reading must not be overwritten
    by the table it will be compared against.
    """

    observations = idx.fetch_oni_history(get=get)
    if not observations:
        logger.error("[enso] ONI history unavailable; nothing backfilled")
        return 0

    try:
        from pythia.db.schema import connect, ensure_schema
    except ImportError:
        logger.warning("pythia.db.schema not available; skipping ONI backfill")
        return 0

    written = 0
    skipped_live = 0
    try:
        con = connect(read_only=False)
        try:
            ensure_schema(con)
            classify_row_kinds(con)
            live_dates = {
                str(row[0])[:10] for row in con.execute(
                    "SELECT fetch_date FROM enso_state "
                    "WHERE COALESCE(row_kind, 'live') <> 'historical'"
                ).fetchall()
            }
            for observation in observations:
                phase, strength = classify_oni(observation.anomaly)
                stamp = observation.date.isoformat()
                if stamp in live_dates:
                    skipped_live += 1
                    continue
                con.execute(
                    """
                    INSERT OR REPLACE INTO enso_state
                        (fetch_date, enso_phase, nino34_anomaly, oni,
                         enso_strength, oni_basis, observation_date,
                         source_rank_used, nino34_source, status, age_days,
                         raw_context, row_kind, nino34_weekly)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    [
                        stamp,
                        phase or None,
                        observation.anomaly,
                        observation.anomaly,
                        strength or None,
                        idx.BASIS_ONI_TABLE,
                        stamp,
                        3,
                        "cpc_oni_ascii",
                        STATUS_HISTORICAL,
                        0,
                        f"## ENSO State (CPC ONI table)\n"
                        f"ENSO state for {stamp[:7]}: {describe_phase(phase, strength)}. "
                        f"ONI: {observation.anomaly:+.2f}°C. Observed {stamp}.",
                        ROW_KIND_HISTORICAL,
                    ],
                )
                written += 1
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        logger.error("[enso] ONI backfill failed after %d rows: %s", written, exc)
        return written

    logger.info(
        "[enso] backfilled %d ONI observations (%s .. %s) as historical rows; "
        "%d dates left alone because a live row holds them",
        written,
        observations[0].date.isoformat(),
        observations[-1].date.isoformat(),
        skipped_live,
    )
    return written


def recompute_record(fetch_date: str, *, get=None) -> Optional[ENSOForecast]:
    """Recompute one stored row's phase from the ONI table and overwrite it.

    The August 2026 row is wrong now and is being read now. This is how it
    is corrected: the ONI value for the month the row describes is looked up
    in the published table and reclassified, so the correction rests on the
    same arithmetic every future run will use.
    """

    from datetime import date as _date

    try:
        target = _date.fromisoformat(fetch_date[:10])
    except ValueError:
        logger.error("[enso] --recompute needs a YYYY-MM-DD date, got %r", fetch_date)
        return None

    observations = idx.fetch_oni_history(get=get)
    if not observations:
        logger.error("[enso] cannot recompute %s: the ONI table is unavailable", fetch_date)
        return None

    # The newest ONI observation at or before the stored row's date. A LATER
    # one would be hindsight the run could not have had.
    usable = [o for o in observations if o.date <= target]
    if not usable:
        logger.error("[enso] the ONI table has nothing at or before %s", fetch_date)
        return None
    observation = max(usable, key=lambda o: o.date)

    phase, strength = classify_oni(observation.anomaly)
    record = ENSOForecast()
    record.fetch_date = target.isoformat()
    record.current_state = phase
    record.strength = strength
    record.oni = observation.anomaly
    record.nino34_anomaly = observation.anomaly
    record.nino34_latest_weekly = None    # a seasonal table has no weekly reading
    record.oni_basis = idx.BASIS_ONI_TABLE
    record.observation_date = observation.date.isoformat()
    record.source_rank_used = 3
    record.nino34_source = "cpc_oni_ascii"
    record.status = STATUS_FRESH
    record.age_days = (target - observation.date).days
    record.warnings.append(
        f"recomputed from the CPC ONI table on {datetime.now(timezone.utc).date()}"
    )
    _build_context(record)

    if store_enso_state(record):
        logger.info(
            "[enso] recomputed %s: %s (ONI %+.2f, observed %s)",
            fetch_date,
            describe_phase(phase, strength),
            observation.anomaly,
            observation.date.isoformat(),
        )
        return record
    return None


def consumers_of_record(fetch_date: str) -> dict:
    """Which HS runs and triage rows read the ENSO record stored on a date.

    A wrong ENSO record is not corrected by fixing the row alone — the
    prompts that already consumed it are the reason it mattered. This
    reports the runs and the hazard rows so a per-hazard rerun decision can
    be made on evidence. DR and TC weight ENSO most heavily.
    """

    out: dict = {"fetch_date": fetch_date, "hs_runs": [], "by_hazard": {}, "rows": 0}
    try:
        from pythia.db.schema import connect, ensure_schema
    except ImportError:
        return out

    try:
        con = connect(read_only=True)
        try:
            ensure_schema(con)
            # The window a stored record was the newest one, i.e. from its
            # own date until the next record superseded it.
            bounds = con.execute(
                """
                SELECT MIN(fetch_date) FROM enso_state
                WHERE fetch_date > ? AND COALESCE(row_kind, 'live') <> 'historical'
                """,
                [fetch_date],
            ).fetchall()
            next_date = bounds[0][0] if bounds and bounds[0] else None

            params: list = [fetch_date]
            clause = "hr.generated_at >= ?"
            if next_date is not None:
                clause += " AND hr.generated_at < ?"
                params.append(str(next_date))

            rows = con.execute(
                f"""
                SELECT ht.hs_run_id, ht.hazard_code, COUNT(*) AS n
                FROM hs_triage ht
                JOIN hs_runs hr ON hr.run_id = ht.hs_run_id
                WHERE {clause}
                GROUP BY ht.hs_run_id, ht.hazard_code
                ORDER BY ht.hs_run_id, ht.hazard_code
                """,
                params,
            ).fetchall()
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[enso] consumer report failed: %s", exc)
        return out

    seen: list[str] = []
    for run_id, hazard, count in rows:
        if run_id not in seen:
            seen.append(str(run_id))
        # ENSO is injected into the climate-sensitive hazards only.
        out["by_hazard"].setdefault(str(hazard), 0)
        out["by_hazard"][str(hazard)] += int(count)
        out["rows"] += int(count)
    out["hs_runs"] = seen
    out["enso_injected_hazards"] = ["DR", "FL", "HW", "TC"]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build, inspect and repair the ENSO record")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--prompt-context", action="store_true", help="Print prompt-ready block")
    parser.add_argument("--text-file", help="Read the IRI page from a saved HTML file")
    parser.add_argument(
        "--store", action="store_true",
        help="Store the record to the DB (with the carry-forward policy)",
    )
    parser.add_argument(
        "--backfill-oni", action="store_true",
        help="Seed enso_state from the published ONI table, 1950 to present",
    )
    parser.add_argument(
        "--recompute", metavar="YYYY-MM-DD",
        help="Recompute one stored row's phase from the ONI table and overwrite it",
    )
    parser.add_argument(
        "--consumers", metavar="YYYY-MM-DD",
        help="Report which HS runs and triage rows read the record stored on this date",
    )
    args = parser.parse_args()

    if args.backfill_oni:
        written = backfill_oni_history()
        print(f"backfilled {written} ONI observations")
        # Zero rows is a failure, whatever the reason: this step seeds the
        # record fetch_and_store_enso carries forward, and a green step that
        # wrote nothing is exactly the silent loss the step exists to end.
        if written == 0:
            print("::warning::ENSO ONI backfill wrote no rows", file=sys.stderr)
            raise SystemExit(1)
        return

    if args.recompute:
        record = recompute_record(args.recompute)
        if record is None:
            raise SystemExit(f"could not recompute {args.recompute}")
        print(record.to_prompt_context())
        report = consumers_of_record(args.recompute)
        print()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    if args.consumers:
        print(json.dumps(consumers_of_record(args.consumers), indent=2, ensure_ascii=False))
        return

    html = Path(args.text_file).read_text() if args.text_file else None
    forecast = build_enso_record(html=html, fetch_page=args.text_file is None)

    if args.store:
        store_enso_state(forecast)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(forecast.to_dict(), indent=2, ensure_ascii=False))
        logger.info(f"Wrote ENSO forecast to {out_path}")
    else:
        print(json.dumps(forecast.to_dict(), indent=2, ensure_ascii=False))

    if args.prompt_context:
        print("\n" + "=" * 70)
        print("PROMPT CONTEXT")
        print("=" * 70)
        print()
        print(forecast.to_prompt_context())


if __name__ == "__main__":
    main()
