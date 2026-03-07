"""
NOAA CPC Seasonal Hurricane Outlook Scraper
============================================
Extracts structured forecast data from NOAA's seasonal hurricane outlook
press releases on noaa.gov.

Covers:
  - North Atlantic (initial May + August update)
  - Eastern Pacific (May outlook)
  - Central Pacific (separate May press release)

NOAA's actual forecast numbers live in press releases (noaa.gov/news-release/...),
NOT on the CPC shtml pages (which are shell pages pointing to JPG infographics).

Usage:
    python noaa_cpc_scraper.py                          # scrape known 2025 URLs
    python noaa_cpc_scraper.py --url <press_release>    # scrape a specific press release
    python noaa_cpc_scraper.py --year 2025              # try to discover all outlooks for a year
    python noaa_cpc_scraper.py --prompt-context         # also print prompt-ready blocks
"""

import re
import json
import argparse
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model (same structure as TSR extractor for interop)
# ---------------------------------------------------------------------------

@dataclass
class SeasonalForecast:
    source: str = "NOAA_CPC"
    basin: str = ""
    basin_full: str = ""
    season_year: int = 0
    issue_date: str = ""
    forecast_type: str = ""  # "initial_outlook", "august_update"
    
    # Ranges (NOAA gives ranges, not point estimates)
    named_storms_range: Optional[list] = None  # [low, high]
    hurricanes_range: Optional[list] = None
    major_hurricanes_range: Optional[list] = None
    # For Central Pacific: "tropical cyclones" instead of named storms
    tropical_cyclones_range: Optional[list] = None
    
    # Probabilities
    prob_above_normal: Optional[float] = None
    prob_near_normal: Optional[float] = None
    prob_below_normal: Optional[float] = None
    
    # ENSO context
    enso_context: str = ""
    
    # Summary
    summary: str = ""
    
    # Metadata
    url: str = ""
    extracted_at: str = ""
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None and v != "" and v != {} and v != []}
    
    def to_prompt_context(self) -> str:
        basin_label = self.basin_full or self.basin
        lines = [
            f"## {basin_label} — {self.season_year} Seasonal Outlook (NOAA CPC, {self.forecast_type}, issued {self.issue_date})"
        ]
        if self.summary:
            lines.append(self.summary)
        
        parts = []
        if self.named_storms_range:
            parts.append(f"{self.named_storms_range[0]}-{self.named_storms_range[1]} named storms")
        if self.hurricanes_range:
            parts.append(f"{self.hurricanes_range[0]}-{self.hurricanes_range[1]} hurricanes")
        if self.major_hurricanes_range:
            parts.append(f"{self.major_hurricanes_range[0]}-{self.major_hurricanes_range[1]} major hurricanes")
        if self.tropical_cyclones_range:
            parts.append(f"{self.tropical_cyclones_range[0]}-{self.tropical_cyclones_range[1]} tropical cyclones")
        if parts:
            lines.append("Forecast range: " + ", ".join(parts) + ".")
        
        if self.prob_above_normal is not None:
            lines.append(
                f"Season probabilities: {self.prob_above_normal:.0%} above-normal, "
                f"{self.prob_near_normal:.0%} near-normal, {self.prob_below_normal:.0%} below-normal."
            )
        
        if self.enso_context:
            lines.append(f"ENSO context: {self.enso_context}")
        
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML fetching
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": "PythiaBot/1.0 (humanitarian forecasting research)"
}

def fetch_page(url: str) -> str:
    """Fetch and return the text content of a web page."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Get just the main content text, stripping nav/footer
    main = soup.find("main") or soup.find("article") or soup
    return main.get_text(separator="\n", strip=True)


# ---------------------------------------------------------------------------
# Atlantic extraction
# ---------------------------------------------------------------------------

def extract_atlantic(text: str, url: str = "") -> SeasonalForecast:
    """Extract Atlantic hurricane season forecast from NOAA press release text."""
    f = SeasonalForecast(basin="ATL", basin_full="North Atlantic", url=url)
    f.extracted_at = datetime.utcnow().isoformat() + "Z"
    
    # Issue date — look for "May 22, 2025" or "August 7, 2025" pattern
    date_m = re.search(r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})", text)
    if date_m:
        try:
            dt = datetime.strptime(date_m.group(1), "%B %d, %Y")
            f.issue_date = dt.strftime("%Y-%m-%d")
            f.season_year = dt.year
            month = dt.month
            if month <= 6:
                f.forecast_type = "initial_outlook"
            else:
                f.forecast_type = "august_update"
        except ValueError:
            pass
    
    # Named storms range: "13 to 19 total named storms" or "13-18"
    # Be careful to match the ATLANTIC number, not East Pacific cross-references
    # Look for "expected named storms" or "number of expected named storms" near the top
    ns_m = re.search(
        r"(?:forecasting\s+a\s+range\s+of\s+|expected\s+named\s+storms\s+to\s+|updated\s+the\s+number\s+of\s+expected\s+named\s+storms\s+to\s+)(\d+)\s*(?:to|-)\s*(\d+)",
        text,
        re.IGNORECASE
    )
    if ns_m:
        f.named_storms_range = [int(ns_m.group(1)), int(ns_m.group(2))]
    else:
        # Fallback: first mention of "N to N named storms" before any "Pacific" mention
        for m in re.finditer(r"(\d+)\s*(?:to|-)\s*(\d+)\s+(?:total\s+)?named\s+storms", text):
            # Check that this isn't in a Pacific context
            preceding = text[max(0, m.start()-80):m.start()]
            if "Pacific" not in preceding and "Eastern" not in preceding:
                f.named_storms_range = [int(m.group(1)), int(m.group(2))]
                break
    
    # Hurricanes range: "6-10 ... hurricanes" or "5-9 could become hurricanes"
    h_m = re.search(r"(\d+)\s*(?:to|-)\s*(\d+)\s+(?:are forecast to become |could become )?hurricanes", text)
    if h_m:
        f.hurricanes_range = [int(h_m.group(1)), int(h_m.group(2))]
    
    # Major hurricanes: "3-5 major hurricanes" or "2-5 major hurricanes"
    mh_m = re.search(r"(\d+)\s*(?:to|-)\s*(\d+)\s+major\s+hurricanes", text)
    if mh_m:
        f.major_hurricanes_range = [int(mh_m.group(1)), int(mh_m.group(2))]
    
    # Probabilities — multiple patterns
    # "60% chance of an above-normal season" or "likelihood of above-normal activity is 50%"
    above_m = re.search(r"(\d+)%\s+(?:chance|likelihood|probability)\s+(?:of\s+)?(?:an?\s+)?above[- ]normal", text)
    if not above_m:
        above_m = re.search(r"above[- ]normal\s+(?:activity\s+)?(?:is\s+)?(\d+)%", text)
    if above_m:
        f.prob_above_normal = int(above_m.group(1)) / 100
    
    near_m = re.search(r"(\d+)%\s+(?:chance|likelihood|probability)\s+(?:of\s+)?(?:an?\s+)?near[- ]normal", text)
    if near_m:
        f.prob_near_normal = int(near_m.group(1)) / 100
    
    below_m = re.search(r"(\d+)%\s+(?:chance|likelihood|probability)\s+(?:of\s+)?(?:an?\s+)?below[- ]normal", text)
    if below_m:
        f.prob_below_normal = int(below_m.group(1)) / 100
    
    # ENSO context
    enso_m = re.search(r"(ENSO[- ]neutral\s+conditions[^.]+\.)", text)
    if enso_m:
        f.enso_context = enso_m.group(1).strip()
    else:
        # Fallback: look for El Nino / La Nina mentions
        enso_m2 = re.search(r"((?:El\s+Ni[ñn]o|La\s+Ni[ñn]a|ENSO)[^.]+\.)", text)
        if enso_m2:
            f.enso_context = enso_m2.group(1).strip()
    
    # Summary — first substantive sentence about the outlook
    summary_m = re.search(
        r"(NOAA'?s\s+outlook\s+for\s+the\s+\d{4}\s+Atlantic[^.]+\.)",
        text
    )
    if summary_m:
        f.summary = summary_m.group(1).strip()
    else:
        # Fallback: look for "predicts above-normal" type headlines
        summary_m2 = re.search(r"(NOAA\s+predicts\s+[^.]+hurricane\s+season[^.]*\.)", text, re.IGNORECASE)
        if summary_m2:
            f.summary = summary_m2.group(1).strip()
    
    _log_forecast(f)
    return f


# ---------------------------------------------------------------------------
# Eastern Pacific extraction (from Atlantic press release or separate page)
# ---------------------------------------------------------------------------

def extract_eastern_pacific_from_atlantic(text: str, url: str = "") -> Optional[SeasonalForecast]:
    """
    The August Atlantic update often contains a brief mention of the Eastern Pacific outlook.
    e.g. "NOAA's outlook for a below-average Eastern Pacific season — with 12-18 named storms"
    """
    f = SeasonalForecast(basin="ENP", basin_full="Eastern North Pacific", url=url, source="NOAA_CPC")
    f.extracted_at = datetime.utcnow().isoformat() + "Z"
    
    # Look for East Pacific mention
    ep_m = re.search(
        r"(?:Eastern|East)\s+Pacific\s+(?:season|outlook)[^.]*?(\d+)\s*(?:to|-)\s*(\d+)\s+named\s+storms",
        text,
        re.IGNORECASE
    )
    if ep_m:
        f.named_storms_range = [int(ep_m.group(1)), int(ep_m.group(2))]
        # Try to inherit date from the Atlantic extraction context
        date_m = re.search(r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})", text)
        if date_m:
            try:
                dt = datetime.strptime(date_m.group(1), "%B %d, %Y")
                f.issue_date = dt.strftime("%Y-%m-%d")
                f.season_year = dt.year
            except ValueError:
                pass
        f.forecast_type = "cross_reference"
        f.summary = f"Eastern Pacific outlook: {f.named_storms_range[0]}-{f.named_storms_range[1]} named storms."
        logger.info(f"  Found East Pacific cross-reference: {f.named_storms_range}")
        return f
    
    return None


def extract_eastern_pacific(text: str, url: str = "") -> SeasonalForecast:
    """Extract Eastern Pacific outlook from its own dedicated press release."""
    f = SeasonalForecast(basin="ENP", basin_full="Eastern North Pacific", url=url, source="NOAA_CPC")
    f.extracted_at = datetime.utcnow().isoformat() + "Z"
    
    # Date
    date_m = re.search(r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})", text)
    if date_m:
        try:
            dt = datetime.strptime(date_m.group(1), "%B %d, %Y")
            f.issue_date = dt.strftime("%Y-%m-%d")
            f.season_year = dt.year
        except ValueError:
            pass
    f.forecast_type = "initial_outlook"
    
    # Named storms
    ns_m = re.search(r"(\d+)\s*(?:to|-)\s*(\d+)\s+(?:total\s+)?named\s+storms", text)
    if ns_m:
        f.named_storms_range = [int(ns_m.group(1)), int(ns_m.group(2))]
    
    # Hurricanes
    h_m = re.search(r"(\d+)\s*(?:to|-)\s*(\d+)\s+(?:are forecast to become |could become )?hurricanes", text)
    if h_m:
        f.hurricanes_range = [int(h_m.group(1)), int(h_m.group(2))]
    
    # Major hurricanes
    mh_m = re.search(r"(\d+)\s*(?:to|-)\s*(\d+)\s+major\s+hurricanes", text)
    if mh_m:
        f.major_hurricanes_range = [int(mh_m.group(1)), int(mh_m.group(2))]
    
    # Probabilities
    above_m = re.search(r"(\d+)%\s+(?:chance|likelihood|probability)\s+(?:of\s+)?(?:an?\s+)?above[- ]normal", text)
    if above_m:
        f.prob_above_normal = int(above_m.group(1)) / 100
    near_m = re.search(r"(\d+)%\s+(?:chance|likelihood|probability)\s+(?:of\s+)?(?:an?\s+)?near[- ]normal", text)
    if near_m:
        f.prob_near_normal = int(near_m.group(1)) / 100
    below_m = re.search(r"(\d+)%\s+(?:chance|likelihood|probability)\s+(?:of\s+)?(?:an?\s+)?below[- ]normal", text)
    if below_m:
        f.prob_below_normal = int(below_m.group(1)) / 100
    
    _log_forecast(f)
    return f


# ---------------------------------------------------------------------------
# Central Pacific extraction
# ---------------------------------------------------------------------------

def extract_central_pacific(text: str, url: str = "") -> SeasonalForecast:
    """Extract Central Pacific hurricane season forecast from its own press release."""
    f = SeasonalForecast(basin="CP", basin_full="Central Pacific", url=url, source="NOAA_CPC")
    f.extracted_at = datetime.utcnow().isoformat() + "Z"
    
    # Date
    date_m = re.search(r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})", text)
    if date_m:
        try:
            dt = datetime.strptime(date_m.group(1), "%B %d, %Y")
            f.issue_date = dt.strftime("%Y-%m-%d")
            f.season_year = dt.year
        except ValueError:
            pass
    f.forecast_type = "initial_outlook"
    
    # Central Pacific uses "tropical cyclones" not "named storms"
    # "1-4 tropical cyclones" or "1 to 4 tropical cyclones"
    tc_m = re.search(r"(\d+)\s*(?:to|-)\s*(\d+)\s+tropical\s+cyclones", text)
    if tc_m:
        f.tropical_cyclones_range = [int(tc_m.group(1)), int(tc_m.group(2))]
    
    # Also check for "named storms" as fallback from cross-references
    if not f.tropical_cyclones_range:
        ns_m = re.search(r"(\d+)\s*(?:to|-)\s*(\d+)\s+named\s+storms", text)
        if ns_m:
            f.tropical_cyclones_range = [int(ns_m.group(1)), int(ns_m.group(2))]
    
    # Probabilities
    above_m = re.search(r"(\d+)%\s+(?:chance|likelihood|probability)\s+(?:of\s+|that\s+)?(?:an?\s+)?(?:it\s+will\s+be\s+)?above[- ]normal", text)
    if above_m:
        f.prob_above_normal = int(above_m.group(1)) / 100
    
    near_m = re.search(r"(\d+)%\s+(?:chance|likelihood|probability)\s+(?:of\s+)?(?:an?\s+)?near[- ]normal", text)
    if near_m:
        f.prob_near_normal = int(near_m.group(1)) / 100
    
    below_m = re.search(r"(\d+)%\s+(?:chance|likelihood|probability)\s+(?:of\s+)?(?:an?\s+)?below[- ]normal", text)
    if below_m:
        f.prob_below_normal = int(below_m.group(1)) / 100
    
    # Summary
    summary_m = re.search(r"((?:forecast|outlook)\s+calls\s+for\s+[^.]+\.)", text, re.IGNORECASE)
    if summary_m:
        f.summary = summary_m.group(1).strip()
    
    _log_forecast(f)
    return f


def extract_central_pacific_from_atlantic(text: str, url: str = "") -> Optional[SeasonalForecast]:
    """Extract CP cross-reference from Atlantic press release."""
    cp_m = re.search(
        r"Central\s+Pacific\s+outlook[^,]*,\s+calling\s+for\s+(\d+)\s*(?:to|-)\s*(\d+)\s+named\s+storms",
        text,
        re.IGNORECASE
    )
    if cp_m:
        f = SeasonalForecast(basin="CP", basin_full="Central Pacific", url=url, source="NOAA_CPC")
        f.extracted_at = datetime.utcnow().isoformat() + "Z"
        f.tropical_cyclones_range = [int(cp_m.group(1)), int(cp_m.group(2))]
        f.forecast_type = "cross_reference"
        f.summary = f"Central Pacific outlook: {f.tropical_cyclones_range[0]}-{f.tropical_cyclones_range[1]} named storms."
        # Inherit date
        date_m = re.search(r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})", text)
        if date_m:
            try:
                dt = datetime.strptime(date_m.group(1), "%B %d, %Y")
                f.issue_date = dt.strftime("%Y-%m-%d")
                f.season_year = dt.year
            except ValueError:
                pass
        logger.info(f"  Found Central Pacific cross-reference: {f.tropical_cyclones_range}")
        return f
    return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _log_forecast(f: SeasonalForecast):
    logger.info(f"  {f.basin_full} ({f.basin}): year={f.season_year}, issued={f.issue_date}, type={f.forecast_type}")
    if f.named_storms_range:
        logger.info(f"    Named storms: {f.named_storms_range}")
    if f.hurricanes_range:
        logger.info(f"    Hurricanes: {f.hurricanes_range}")
    if f.major_hurricanes_range:
        logger.info(f"    Major hurricanes: {f.major_hurricanes_range}")
    if f.tropical_cyclones_range:
        logger.info(f"    Tropical cyclones: {f.tropical_cyclones_range}")
    if f.prob_above_normal is not None:
        logger.info(f"    Probs: above={f.prob_above_normal:.0%}, near={f.prob_near_normal:.0%}, below={f.prob_below_normal:.0%}")


# ---------------------------------------------------------------------------
# Known NOAA press release URLs by year
# These follow a semi-predictable pattern on noaa.gov but need to be
# maintained manually or discovered via search.
# ---------------------------------------------------------------------------

KNOWN_URLS = {
    2025: {
        "ATL_initial": "https://www.noaa.gov/news-release/noaa-predicts-above-normal-2025-atlantic-hurricane-season",
        "ATL_update": "https://www.noaa.gov/news-release/prediction-remains-on-track-for-above-normal-atlantic-hurricane-season",
        "CP": "https://www.noaa.gov/news-release/noaa-predicts-less-active-2025-central-pacific-hurricane-season",
        # ENP is usually part of the Atlantic initial + update, or a separate release
    },
}


def process_known_urls(year: int) -> list[SeasonalForecast]:
    """Process all known URLs for a given year."""
    urls = KNOWN_URLS.get(year, {})
    results = []
    
    for key, url in urls.items():
        try:
            logger.info(f"Fetching: {key} -> {url}")
            text = fetch_page(url)
            
            if key.startswith("ATL"):
                atl = extract_atlantic(text, url)
                results.append(atl)
                
                # Also try to extract cross-references for other basins
                enp = extract_eastern_pacific_from_atlantic(text, url)
                if enp:
                    results.append(enp)
                cp = extract_central_pacific_from_atlantic(text, url)
                if cp:
                    results.append(cp)
            
            elif key == "CP":
                results.append(extract_central_pacific(text, url))
            
            elif key == "ENP":
                results.append(extract_eastern_pacific(text, url))
        
        except Exception as e:
            logger.error(f"Failed to process {key} ({url}): {e}")
    
    return results


# ---------------------------------------------------------------------------
# Auto-detect basin from press release text
# ---------------------------------------------------------------------------

def auto_extract(text: str, url: str = "") -> list[SeasonalForecast]:
    """Auto-detect basin(s) and extract all forecasts from a press release."""
    results = []
    text_lower = text.lower()
    
    if "atlantic" in text_lower and "hurricane season" in text_lower:
        results.append(extract_atlantic(text, url))
        # Cross-references
        enp = extract_eastern_pacific_from_atlantic(text, url)
        if enp:
            results.append(enp)
        cp = extract_central_pacific_from_atlantic(text, url)
        if cp:
            results.append(cp)
    
    if "central pacific" in text_lower and "hurricane season" in text_lower and not any(r.basin == "CP" for r in results):
        results.append(extract_central_pacific(text, url))
    
    if "eastern pacific" in text_lower and "hurricane season" in text_lower and not any(r.basin == "ENP" for r in results):
        results.append(extract_eastern_pacific(text, url))
    
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract NOAA CPC seasonal hurricane outlooks")
    parser.add_argument("--url", help="URL of a specific NOAA press release")
    parser.add_argument("--year", type=int, help="Process all known URLs for a year")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--prompt-context", action="store_true", help="Also print prompt-ready context blocks")
    args = parser.parse_args()
    
    forecasts = []
    
    if args.url:
        text = fetch_page(args.url)
        forecasts = auto_extract(text, args.url)
    elif args.year:
        forecasts = process_known_urls(args.year)
    else:
        # Default: process 2025
        forecasts = process_known_urls(2025)
    
    output_data = [f.to_dict() for f in forecasts]
    
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(output_data, indent=2))
        logger.info(f"Wrote {len(forecasts)} forecasts to {out_path}")
    else:
        print(json.dumps(output_data, indent=2))
    
    if args.prompt_context:
        print("\n" + "=" * 70)
        print("PROMPT CONTEXT BLOCKS")
        print("=" * 70)
        for f in forecasts:
            print()
            print(f.to_prompt_context())
            print()


if __name__ == "__main__":
    main()
