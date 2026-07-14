# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""
North Indian Ocean Seasonal TC Context (IMD / RSMC New Delhi)
=============================================================
IMD / RSMC New Delhi publishes NO seasonal tropical-cyclone count outlook
(its "seasonal forecast" page covers rainfall/temperature only), so the NIO
basin cannot mirror the TSR/BoM/Météo-France scraper design. This module
instead provides a climatology-first context block, optionally enriched by
two best-effort live sources:

1. RSMC New Delhi's weekly "Extended Range Outlook for Cyclogenesis" PDF —
   a two-week probabilistic cyclogenesis outlook (nil/low/moderate/high)
   for the Bay of Bengal and Arabian Sea, scraped from the archive page and
   parsed with pdfplumber. Only attached when issued within the last
   ~3 weeks (the pipeline runs monthly).
2. The ENSO state already persisted in the Pythia DB.

The climatology block ALWAYS renders (zero network), so NIO countries
(BGD/IND/MMR/LKA on the Bay of Bengal side; PAK/OMN/YEM/SOM/IRN on the
Arabian Sea side) never lose seasonal TC context entirely.

Usage:
    python imd_nio_scraper.py                    # climatology + live enrich
    python imd_nio_scraper.py --no-live          # climatology only
    python imd_nio_scraper.py --prompt-context   # print prompt-ready block
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

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
}

# RSMC New Delhi Extended Range Outlook archive (menu ids are stable base64
# route params observed on the live site).
ARCHIVE_URL = (
    "https://rsmcnewdelhi.imd.gov.in/archive-information.php"
    "?internal_menu=MjQ%3D&menu_id=Mg%3D%3D"
)
_RSMC_HOST = "https://rsmcnewdelhi.imd.gov.in"

# Attach the extended-range note only when reasonably fresh.
_ERO_MAX_AGE_DAYS = 21

NIO_CLIMATOLOGY = (
    "The North Indian Ocean averages about 5 cyclonic storms per year "
    "(~4 in the Bay of Bengal, ~1 in the Arabian Sea). Activity is strongly "
    "bimodal: a pre-monsoon peak in April-June and a stronger post-monsoon "
    "peak in October-December; the monsoon months (July-September) are "
    "usually quiet. The highest-impact months are May and October-November. "
    "Bay of Bengal systems chiefly threaten Bangladesh, eastern India, "
    "Myanmar and Sri Lanka; Arabian Sea systems chiefly threaten Oman, "
    "Yemen, Pakistan, western India and occasionally Somalia."
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SeasonalForecast:
    source: str = "IMD_RSMC_NewDelhi"
    basin: str = "NIO"
    basin_full: str = "North Indian Ocean"
    season: str = ""
    season_year: int = 0
    issue_date: str = ""
    forecast_type: str = "climatology_context"

    climatology_note: str = ""
    extended_range_note: Optional[str] = None   # from the weekly ERO PDF
    enso_context: Optional[str] = None          # from the Pythia DB

    # Metadata
    url: str = ""
    extracted_at: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None and v != "" and v != [] and v != {}}

    def to_prompt_context(self) -> str:
        lines = [
            f"## {self.basin_full} — Seasonal TC Context (climatology; "
            f"IMD/RSMC New Delhi publishes no seasonal count outlook)"
        ]
        if self.climatology_note:
            lines.append(self.climatology_note)
        if self.extended_range_note:
            lines.append(
                f"RSMC New Delhi extended-range cyclogenesis outlook: {self.extended_range_note}"
            )
        if self.enso_context:
            lines.append(f"ENSO context: {self.enso_context}")
        lines.append(
            "Interpretation: weight monthly probabilities by the bimodal "
            "seasonality above; this block is climatological context, not a "
            "seasonal count forecast."
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extended Range Outlook (best-effort enrichment)
# ---------------------------------------------------------------------------

_ERO_LINK_RE = re.compile(r"Extended.?Range.?Outlook", re.IGNORECASE)
# Filenames observed: ..._Extended_Range_Outlook_11June2026.pdf /
# ..._Extended Range Outlook_05Sep2024.pdf
_ERO_DATE_RE = re.compile(r"_(\d{1,2})\s?([A-Za-z]+)\s?(\d{4})\.pdf", re.IGNORECASE)


def _parse_ero_filename_date(href: str) -> Optional[datetime]:
    m = _ERO_DATE_RE.search(href.replace("%20", " "))
    if not m:
        return None
    day, month_name, year = m.groups()
    for fmt in ("%d%B%Y", "%d%b%Y"):
        try:
            return datetime.strptime(f"{day}{month_name}{year}", fmt).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
    return None


def fetch_latest_ero_pdf_url() -> Optional[tuple]:
    """Return (url, issue_datetime) of the newest Extended Range Outlook PDF."""
    resp = requests.get(ARCHIVE_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    best: Optional[tuple] = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        label = a.get_text(" ", strip=True)
        if not (_ERO_LINK_RE.search(href) or _ERO_LINK_RE.search(label)):
            continue
        issued = _parse_ero_filename_date(href)
        if issued is None:
            continue
        if href.startswith("/"):
            href = f"{_RSMC_HOST}{href}"
        elif not href.startswith("http"):
            href = f"{_RSMC_HOST}/{href}"
        if best is None or issued > best[1]:
            best = (href, issued)
    return best


_ERO_BASIN_RE = re.compile(
    r"(Bay\s+of\s+Bengal|Arabian\s+Sea)[^.]{0,200}?(nil|low|moderate|high)",
    re.IGNORECASE,
)


def extract_ero(pdf_path: str) -> dict:
    """Parse basin cyclogenesis categories from an ERO PDF. Best-effort."""
    import pdfplumber  # heavy import, keep local

    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    text = "\n".join(text_parts)

    result: dict = {}
    for basin, category in _ERO_BASIN_RE.findall(text):
        key = "bay_of_bengal" if "bengal" in basin.lower() else "arabian_sea"
        # Keep the first (week-1) mention per basin.
        result.setdefault(key, category.lower())
    return result


def _build_extended_range_note(now: datetime) -> Optional[str]:
    """Fetch + parse the latest ERO PDF; None on any failure or staleness."""
    import tempfile

    found = fetch_latest_ero_pdf_url()
    if not found:
        return None
    url, issued = found
    age_days = (now - issued).days
    if age_days > _ERO_MAX_AGE_DAYS:
        logger.info("NIO ERO: newest outlook is %d days old — skipping", age_days)
        return None

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(resp.content)
        tmp.flush()
        parsed = extract_ero(tmp.name)
    if not parsed:
        return None

    parts = []
    if parsed.get("bay_of_bengal"):
        parts.append(f"Bay of Bengal cyclogenesis probability: {parsed['bay_of_bengal']}")
    if parsed.get("arabian_sea"):
        parts.append(f"Arabian Sea cyclogenesis probability: {parsed['arabian_sea']}")
    if not parts:
        return None
    return (
        f"issued {issued.date().isoformat()} (next ~2 weeks): " + "; ".join(parts) + "."
    )


def _load_enso_context() -> Optional[str]:
    """Current ENSO state from the Pythia DB (best-effort, never raises)."""
    try:
        from horizon_scanner.enso.enso_module import load_enso_state_from_db

        snapshot = load_enso_state_from_db()
        if snapshot and getattr(snapshot, "current_state", ""):
            return str(snapshot.current_state)
    except Exception as exc:  # noqa: BLE001
        logger.debug("NIO ENSO context unavailable: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_nio_context(fetch_live: bool = True) -> list:
    """Build the NIO context forecast. NEVER returns an empty list —
    the climatology block requires no network."""
    now = datetime.now(timezone.utc)
    f = SeasonalForecast(
        season=f"{now.year} pre-monsoon (Apr-Jun) / post-monsoon (Oct-Dec)",
        season_year=now.year,
        climatology_note=NIO_CLIMATOLOGY,
        url=ARCHIVE_URL,
        extracted_at=now.isoformat(),
    )

    if fetch_live:
        try:
            f.extended_range_note = _build_extended_range_note(now)
        except Exception as exc:  # noqa: BLE001
            logger.info("NIO extended-range enrichment failed (non-fatal): %s", exc)
        try:
            f.enso_context = _load_enso_context()
        except Exception as exc:  # noqa: BLE001
            logger.debug("NIO ENSO enrichment failed (non-fatal): %s", exc)

    return [f]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="NIO seasonal TC context builder")
    parser.add_argument("--no-live", action="store_true", help="Climatology only")
    parser.add_argument("--output", help="Write JSON to this path")
    parser.add_argument("--prompt-context", action="store_true", help="Print prompt block")
    args = parser.parse_args()

    forecasts = build_nio_context(fetch_live=not args.no_live)

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
