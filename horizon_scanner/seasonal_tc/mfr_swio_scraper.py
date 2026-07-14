# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""
Météo-France La Réunion Seasonal Cyclone Outlook Scraper (SWI basin)
====================================================================
Extracts structured forecast data from the RSMC La Réunion / Météo-France
seasonal cyclone activity outlook for the South-West Indian Ocean.

Source format:
  - A French-language HTML article on meteofrance.re, issued each year in
    late October / early November for the Nov-Apr season.
  - Article slugs DRIFT year over year: the prefix flips between
    "prevision-saisonniere" and "tendance-saisonniere", the section between
    /fr/climat/ and /fr/actualites/, and the "-saison-YYYY" suffix is
    sometimes absent. A same-content mirror exists on meteofrance.yt
    (Météo-France Mayotte). We try a candidate-slug list, then discover the
    article from listing pages, then fall back to the mirror.
  - Extractable fields: expected number of systems ("entre 9 et 14
    systèmes"), number reaching TC stage ("dont 5 à 8 ... stade de cyclone
    tropical"), tercile probabilities (present some years), categorical
    outlook ("activité proche ou supérieure à la normale"), ENSO/IOD
    commentary, and per-region risk notes.

Mid-year degradation (like BoM): from ~May to October the latest available
article is the PRIOR season's outlook; it is scraped and labeled with its
own season string so prompts never misrepresent its vintage.

Usage:
    python mfr_swio_scraper.py                    # scrape current outlook
    python mfr_swio_scraper.py --url <url>        # scrape a specific page
    python mfr_swio_scraper.py --text <file>      # extract from saved text
    python mfr_swio_scraper.py --prompt-context   # print prompt-ready block
"""

import re
import json
import argparse
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# meteofrance.re 403s non-browser User-Agents (same failure mode as
# bom.gov.au) — present a standard desktop-browser UA + French locale.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.5",
}

_HOSTS = ("https://meteofrance.re", "https://meteofrance.yt")

_LISTING_URLS = (
    "https://meteofrance.re/fr/cyclone",
    "https://meteofrance.re/fr/actualites",
    "https://meteofrance.yt/fr/actualites",
)

# Matches the outlook article title across observed years:
# "PRÉVISION SAISONNIÈRE D'ACTIVITÉ CYCLONIQUE ..." /
# "TENDANCE SAISONNIÈRE D'ACTIVITÉ CYCLONIQUE ..."
_TITLE_PATTERN = re.compile(
    r"(?:PR[ÉE]VISION|TENDANCE)\s+SAISONNI[ÈE]RE\s+D.ACTIVIT[ÉE]\s+CYCLONIQUE",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SeasonalForecast:
    source: str = "MeteoFrance_LaReunion"
    basin: str = "SWI"
    basin_full: str = "South-West Indian Ocean"
    season: str = ""            # e.g. "2025-26"
    season_year: int = 0        # year the season starts in (Nov)
    issue_date: str = ""
    forecast_type: str = "seasonal_outlook"

    # Activity forecast
    systems_range: Optional[str] = None      # e.g. "9-14" (named systems)
    tc_stage_range: Optional[str] = None     # e.g. "5-8" (reaching TC stage)
    prob_above: Optional[float] = None
    prob_near: Optional[float] = None
    prob_below: Optional[float] = None
    categorical_outlook: str = ""            # e.g. "near to above normal"

    # Context
    enso_context: str = ""                    # ENSO + IOD commentary
    regional_risk_note: str = ""              # Mozambique channel / Madagascar etc.
    summary: str = ""

    # Metadata
    url: str = ""
    extracted_at: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None and v != "" and v != [] and v != {}}

    def to_prompt_context(self) -> str:
        basin_label = self.basin_full or self.basin
        season_label = self.season or "latest available"
        lines = [
            f"## {basin_label} — {season_label} Seasonal Outlook "
            f"(Météo-France La Réunion / RSMC)"
        ]
        if self.summary:
            lines.append(self.summary)
        if self.systems_range:
            line = f"Expected named systems: {self.systems_range}"
            if self.tc_stage_range:
                line += f" (of which {self.tc_stage_range} reaching tropical cyclone stage)"
            lines.append(line + ".")
        # Guard ALL terciles: parsed by independent regexes, so a partial
        # set must be omitted rather than half-formatted (TSR fix pattern).
        if None not in (self.prob_above, self.prob_near, self.prob_below):
            lines.append(
                f"Season probabilities: {self.prob_above:.0%} above-normal, "
                f"{self.prob_near:.0%} near-normal, {self.prob_below:.0%} below-normal."
            )
        if self.categorical_outlook:
            lines.append(f"Categorical outlook: {self.categorical_outlook}.")
        if self.enso_context:
            lines.append(f"ENSO/IOD context: {self.enso_context}")
        if self.regional_risk_note:
            lines.append(f"Regional note: {self.regional_risk_note}")
        lines.append(
            "Note: the SWI outlook is issued once per season (Oct/Nov, for "
            "Nov-Apr); outside that window this is the most recent edition."
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_page(url: str) -> str:
    """Fetch a page and return its main-content text."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    main = soup.find("main") or soup.find("article") or soup
    return main.get_text(separator="\n")


def build_candidate_urls(season_start_year: int) -> list:
    """Candidate article URLs, newest-season first.

    The slug drifts across years, so emit every observed variant for the
    target season year and the prior one, on both hosts.
    """
    slugs = []
    base = "prevision-saisonniere-dactivite-cyclonique-dans-le-sud-ouest-de-locean-indien-saison"
    tendance = "tendance-saisonniere-dactivite-cyclonique-dans-le-sud-ouest-de-locean-indien-saison"
    for year in (season_start_year, season_start_year - 1):
        for stem in (tendance, base):
            slugs.append(f"/fr/climat/{stem}-{year}")
            slugs.append(f"/fr/actualites/{stem}-{year}")
    # The 2023-24 edition used an unversioned slug.
    slugs.append(f"/fr/actualites/{base}")
    slugs.append(f"/fr/climat/{base}")

    urls = []
    for host in _HOSTS:
        for slug in slugs:
            urls.append(f"{host}{slug}")
    return urls


def discover_outlook_url() -> Optional[str]:
    """Scan listing pages for a link whose text matches the outlook title."""
    for listing in _LISTING_URLS:
        try:
            resp = requests.get(listing, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as exc:  # noqa: BLE001
            logger.info("SWI discovery: listing %s failed: %s", listing, exc)
            continue
        host = listing.split("/fr/")[0]
        for a in soup.find_all("a", href=True):
            text = a.get_text(" ", strip=True)
            if text and _TITLE_PATTERN.search(text):
                href = a["href"]
                if href.startswith("/"):
                    href = f"{host}{href}"
                logger.info("SWI discovery: found outlook link %s", href)
                return href
    return None


# ---------------------------------------------------------------------------
# Extraction (French-language regexes)
# ---------------------------------------------------------------------------

_SEASON_RE = re.compile(r"SAISON\s+(\d{4})\s*[-–/]\s*(\d{2,4})", re.IGNORECASE)
_SYSTEMS_RE = re.compile(
    r"entre\s+(\d+)\s+(?:et|à)\s+(\d+)\s+(?:syst[èe]mes|temp[êe]tes)", re.IGNORECASE
)
_TC_STAGE_RE = re.compile(
    r"(?:dont\s+)?(\d+)\s+(?:à|et)\s+(\d+)\s+[^.\n]{0,120}?stade\s+de\s+cyclone\s+tropical",
    re.IGNORECASE,
)
_TERCILE_RE = re.compile(
    r"(\d+)\s*%\s+de\s+(?:probabilit[ée]|chance)s?\s+[^.\n]{0,120}?"
    r"(sup[ée]rieure|proche|inf[ée]rieure)",
    re.IGNORECASE,
)
_ENSO_RE = re.compile(
    r"((?:El\s+Ni[ñn]o|La\s+Ni[ñn]a|Dip[ôo]le\s+de\s+l.Oc[ée]an\s+Indien|DOI\b|ENSO)"
    r"[^.]{0,300}\.)",
    re.IGNORECASE,
)
_REGIONAL_RE = re.compile(
    r"((?:canal\s+du\s+Mozambique|Madagascar|Mascareignes)[^.]{0,300}\.)",
    re.IGNORECASE,
)
_CATEGORICAL_PATTERNS = [
    (re.compile(r"(?:proche\s+ou\s+sup[ée]rieure|normale?\s+à\s+sup[ée]rieure)", re.I),
     "near to above normal"),
    (re.compile(r"(?:proche\s+ou\s+inf[ée]rieure|normale?\s+à\s+inf[ée]rieure)", re.I),
     "near to below normal"),
    (re.compile(r"activit[ée][^.\n]{0,80}sup[ée]rieure\s+à\s+la\s+normale", re.I),
     "above normal"),
    (re.compile(r"activit[ée][^.\n]{0,80}inf[ée]rieure\s+à\s+la\s+normale", re.I),
     "below normal"),
    (re.compile(r"activit[ée][^.\n]{0,80}proche\s+de\s+la\s+normale", re.I),
     "near normal"),
]


def extract_swio_outlook(text: str, url: str = "") -> SeasonalForecast:
    """Extract a SeasonalForecast from the French article text.

    Every field is best-effort: a phrasing change leaves that field
    None/empty rather than raising.
    """
    f = SeasonalForecast(url=url, extracted_at=datetime.now(timezone.utc).isoformat())

    m = _SEASON_RE.search(text)
    if m:
        start_year = int(m.group(1))
        end = m.group(2)
        end_short = end[-2:]
        f.season = f"{start_year}-{end_short}"
        f.season_year = start_year

    m = _SYSTEMS_RE.search(text)
    if m:
        f.systems_range = f"{m.group(1)}-{m.group(2)}"

    m = _TC_STAGE_RE.search(text)
    if m:
        f.tc_stage_range = f"{m.group(1)}-{m.group(2)}"

    for pct, keyword in _TERCILE_RE.findall(text):
        try:
            value = float(pct) / 100.0
        except ValueError:
            continue
        kw = keyword.lower()
        if kw.startswith("sup") and f.prob_above is None:
            f.prob_above = value
        elif kw.startswith("proche") and f.prob_near is None:
            f.prob_near = value
        elif kw.startswith("inf") and f.prob_below is None:
            f.prob_below = value

    for pattern, label in _CATEGORICAL_PATTERNS:
        if pattern.search(text):
            f.categorical_outlook = label
            break

    m = _ENSO_RE.search(text)
    if m:
        f.enso_context = " ".join(m.group(1).split())

    m = _REGIONAL_RE.search(text)
    if m:
        f.regional_risk_note = " ".join(m.group(1).split())

    return f


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _season_start_year(year: Optional[int] = None) -> int:
    """SWI seasons start in November; before July the 'current' outlook is
    the one issued the previous autumn."""
    if year:
        return year
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def _looks_like_outlook(f: SeasonalForecast) -> bool:
    """A parse counts only if it produced at least one substantive field."""
    return bool(
        f.season
        or f.systems_range
        or f.categorical_outlook
        or None not in (f.prob_above, f.prob_near, f.prob_below)
    )


def process_all(fetch_live: bool = True, year: Optional[int] = None) -> list:
    """Fetch and extract the latest SWI seasonal outlook.

    Returns a list with at most one SeasonalForecast (one article per
    season). Per-URL failures are contained — one bad candidate never
    discards the batch (TSR-fix pattern).
    """
    if not fetch_live:
        return []

    candidates = build_candidate_urls(_season_start_year(year))
    discovered = discover_outlook_url()
    if discovered and discovered not in candidates:
        candidates.insert(0, discovered)

    for url in candidates:
        try:
            text = fetch_page(url)
            if not text or len(text) < 500:
                continue
            forecast = extract_swio_outlook(text, url=url)
            if _looks_like_outlook(forecast):
                logger.info(
                    "SWI outlook extracted from %s (season=%s, systems=%s)",
                    url, forecast.season or "?", forecast.systems_range or "?",
                )
                return [forecast]
            logger.info("SWI: page %s fetched but no outlook fields parsed", url)
        except Exception as exc:  # noqa: BLE001
            logger.debug("SWI candidate failed (%s): %s", url, exc)
            continue

    logger.warning(
        "SWI: no seasonal outlook found on meteofrance.re/.yt — "
        "SWI basin will have no seasonal TC context this cycle."
    )
    return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Météo-France La Réunion SWI outlook scraper")
    parser.add_argument("--url", help="Scrape a specific article URL")
    parser.add_argument("--text", help="Extract from a saved text file")
    parser.add_argument("--live", action="store_true", help="Fetch the live outlook")
    parser.add_argument("--output", help="Write JSON to this path")
    parser.add_argument("--prompt-context", action="store_true", help="Print prompt block")
    args = parser.parse_args()

    forecasts: list = []
    if args.text:
        raw = open(args.text, encoding="utf-8").read()
        forecasts = [extract_swio_outlook(raw, url=f"file://{args.text}")]
    elif args.url:
        forecasts = [extract_swio_outlook(fetch_page(args.url), url=args.url)]
    else:
        forecasts = process_all(fetch_live=True)

    if args.prompt_context:
        for f in forecasts:
            print(f.to_prompt_context())
            print()
    else:
        payload = [f.to_dict() for f in forecasts]
        text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"Wrote {args.output}")
        else:
            print(text)


if __name__ == "__main__":
    main()
