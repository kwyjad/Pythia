"""
BoM Seasonal Tropical Cyclone Outlook Scraper
==============================================
Extracts structured forecast data from the Australian Bureau of Meteorology's
tropical cyclone season outlooks.

Covers:
  - Australian Region (whole region + Western, Northern, Eastern subregions)
  - South Pacific (Western + Eastern subregions)

BoM outlook format:
  - Published annually in October (for the Nov-Apr season)
  - Key metric: "X% chance of having more tropical cyclones than average"
  - Subregional breakdowns with per-region probabilities
  - ENSO context (El Niño / La Niña / neutral)

Source pages:
  - Current: embedded in https://www.bom.gov.au/climate/cyclones/australia/
  - Archive (pre-2015): https://www.bom.gov.au/climate/ahead/archive/tropical-cyclone/
  - South Pacific: https://www.bom.gov.au/climate/cyclones/south-pacific/

Usage:
    python bom_scraper.py                          # scrape current outlook
    python bom_scraper.py --url <outlook_url>      # scrape a specific page
    python bom_scraper.py --text <text_file>       # extract from saved text
    python bom_scraper.py --prompt-context         # print prompt-ready blocks
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

HEADERS = {
    "User-Agent": "PythiaBot/1.0 (humanitarian forecasting research)"
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SubregionForecast:
    """Forecast for a single subregion."""
    name: str = ""
    prob_above_average: Optional[float] = None  # e.g. 0.60 = 60% chance above avg
    expected_tc_count: Optional[str] = None  # e.g. "8-9" or "near average"
    average_tc_count: Optional[float] = None  # long-term average for this region
    skill_level: str = ""  # "high", "moderate", "low", "very low"


@dataclass
class SeasonalForecast:
    source: str = "BoM"
    basin: str = ""
    basin_full: str = ""
    season: str = ""  # e.g. "2025-26"
    season_year: int = 0  # the year the season starts in (for Southern Hemisphere)
    issue_date: str = ""
    forecast_type: str = "seasonal_outlook"

    # Whole-region forecast
    prob_above_average: Optional[float] = None
    expected_tc_count: Optional[str] = None
    average_tc_count: Optional[float] = None
    categorical_outlook: str = ""  # "above average", "near average", "below average"

    # Subregional forecasts
    subregions: list = field(default_factory=list)

    # Severe TC likelihood
    severe_tc_note: str = ""

    # ENSO context
    enso_context: str = ""

    # Summary
    summary: str = ""

    # Metadata
    url: str = ""
    extracted_at: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        # Clean up empty fields
        d = {k: v for k, v in d.items() if v is not None and v != "" and v != [] and v != {}}
        if "subregions" in d:
            d["subregions"] = [
                {k: v for k, v in sr.items() if v is not None and v != ""}
                for sr in d["subregions"]
            ]
        return d

    def to_prompt_context(self) -> str:
        basin_label = self.basin_full or self.basin
        lines = [
            f"## {basin_label} — {self.season} Seasonal Outlook (BoM, issued {self.issue_date})"
        ]
        if self.summary:
            lines.append(self.summary)

        if self.prob_above_average is not None:
            lines.append(
                f"Whole region: {self.prob_above_average:.0%} chance of above-average TC activity "
                f"(average is {self.average_tc_count} TCs/season)."
            )
        if self.categorical_outlook:
            lines.append(f"Categorical outlook: {self.categorical_outlook}.")

        if self.subregions:
            parts = []
            for sr in self.subregions:
                if isinstance(sr, dict):
                    name = sr.get("name", "")
                    prob = sr.get("prob_above_average")
                else:
                    name = sr.name
                    prob = sr.prob_above_average
                if prob is not None:
                    parts.append(f"{name}: {prob:.0%} chance above-avg")
            if parts:
                lines.append("Subregions: " + "; ".join(parts) + ".")

        if self.severe_tc_note:
            lines.append(f"Severe TC note: {self.severe_tc_note}")

        if self.enso_context:
            lines.append(f"ENSO context: {self.enso_context}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML fetching
# ---------------------------------------------------------------------------

def fetch_page(url: str) -> str:
    """Fetch and return text content."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    main = soup.find("main") or soup.find(id="content") or soup
    return main.get_text(separator="\n", strip=True)


# ---------------------------------------------------------------------------
# Australian Region extraction
# ---------------------------------------------------------------------------

def extract_australian_outlook(text: str, url: str = "") -> SeasonalForecast:
    """
    Extract Australian region TC outlook.

    Key patterns in BoM text:
    - "The 2024-25 Australian tropical cyclone season is expected to be like the long-term average"
    - "The Australian region has only a 9% chance of having more tropical cyclones than average"
    - "The Western region ... 25% chance of more tropical cyclones than average"
    - "The likelihood of severe (strong) tropical cyclones is higher than average"
    """
    f = SeasonalForecast(basin="AUS", basin_full="Australian Region", url=url)
    f.extracted_at = datetime.utcnow().isoformat() + "Z"

    # Season detection: "2024-25" or "2025-26" or "2025/26"
    season_m = re.search(r"(\d{4})[–\-/](\d{2,4})", text)
    if season_m:
        start_year = int(season_m.group(1))
        end_suffix = season_m.group(2)
        if len(end_suffix) == 2:
            end_year = int(str(start_year)[:2] + end_suffix)
        else:
            end_year = int(end_suffix)
        f.season = f"{start_year}-{str(end_year)[-2:]}"
        f.season_year = start_year

    # Issue date — BoM typically issues in October
    date_m = re.search(r"(?:Issued|Last\s+updated)[:\s]*(\w+\s+\d{4})", text)
    if date_m:
        f.issue_date = date_m.group(1)
    elif f.season_year:
        f.issue_date = f"October {f.season_year}"
    f.forecast_type = "seasonal_outlook"

    # Whole-region probability
    # "X% chance of having more tropical cyclones than average"
    aus_prob_m = re.search(
        r"Australian\s+region\s+(?:has\s+)?(?:only\s+)?(?:a\s+)?(\d+)%\s+chance\s+of\s+(?:having\s+)?more\s+tropical\s+cyclones\s+than\s+average",
        text,
        re.IGNORECASE
    )
    if aus_prob_m:
        f.prob_above_average = int(aus_prob_m.group(1)) / 100

    # Average count
    avg_m = re.search(
        r"(?:long-term\s+average|average)[^.]*?(\d+)\s+tropical\s+cyclones?\s+(?:form\s+)?in\s+the\s+Australian\s+region",
        text,
        re.IGNORECASE
    )
    if avg_m:
        f.average_tc_count = float(avg_m.group(1))
    else:
        f.average_tc_count = 11  # Known long-term average

    # Categorical outlook from summary sentence
    if re.search(r"expected\s+to\s+be\s+(?:like|close\s+to)\s+(?:the\s+)?(?:long-term\s+)?average", text, re.IGNORECASE):
        f.categorical_outlook = "near average"
    elif re.search(r"above\s*[-–]?\s*average|more\s+active", text[:500], re.IGNORECASE):
        f.categorical_outlook = "above average"
    elif re.search(r"below\s*[-–]?\s*average|less\s+active|fewer", text[:500], re.IGNORECASE):
        f.categorical_outlook = "below average"

    # Expected count if given
    count_m = re.search(r"(?:in\s+which\s+)?(\d+)\s+tropical\s+cyclones\s+(?:are\s+expected\s+to\s+)?form\s+in\s+the\s+Australian\s+region", text, re.IGNORECASE)
    if count_m:
        f.expected_tc_count = count_m.group(1)

    # Subregional probabilities
    for region_name in ["Western", "Northwestern", "Northern", "Eastern"]:
        sr_m = re.search(
            rf"{region_name}\s+(?:sub-?region|region)\s+[^.]*?(\d+)%\s+chance\s+of\s+(?:having\s+)?more\s+tropical\s+cyclones\s+than\s+average",
            text,
            re.IGNORECASE
        )
        if sr_m:
            sr = SubregionForecast(
                name=region_name,
                prob_above_average=int(sr_m.group(1)) / 100,
            )
            # Try to get average count for this region
            sr_avg_m = re.search(
                rf"{region_name}\s+(?:sub-?region|region)[^.]*?average\s+(?:value\s+)?(?:of\s+|is\s+)?(\d+)\s+tropical\s+cyclones",
                text,
                re.IGNORECASE
            )
            if sr_avg_m:
                sr.average_tc_count = float(sr_avg_m.group(1))
            f.subregions.append(sr)

    # Severe TC note
    severe_m = re.search(r"(likelihood\s+of\s+severe[^.]+\.)", text, re.IGNORECASE)
    if severe_m:
        f.severe_tc_note = severe_m.group(1).strip()

    # ENSO context
    enso_m = re.search(r"((?:El\s+Ni[ñn]o|La\s+Ni[ñn]a|neutral\s+(?:ENSO\s+|climatic\s+)?conditions)[^.]+\.)", text, re.IGNORECASE)
    if enso_m:
        f.enso_context = re.sub(r"\s+", " ", enso_m.group(1)).strip()

    # Summary — first sentence about the outlook
    summary_m = re.search(
        r"(The\s+\d{4}[–\-/]\d{2,4}\s+Australian\s+tropical\s+cyclone\s+season\s+is\s+expected[^.]+\.)",
        text,
        re.IGNORECASE
    )
    if summary_m:
        f.summary = re.sub(r"\s+", " ", summary_m.group(1)).strip()
    else:
        # Fallback: look for "above/below/near average" summary
        summary_m2 = re.search(
            r"((?:A\s+(?:less|more)\s+active|Above\s+average|Below\s+average|Near\s+average|Average\s+to)[^.]*Australian[^.]*\.)",
            text,
            re.IGNORECASE
        )
        if summary_m2:
            f.summary = re.sub(r"\s+", " ", summary_m2.group(1)).strip()

    _log_forecast(f)
    return f


# ---------------------------------------------------------------------------
# South Pacific extraction
# ---------------------------------------------------------------------------

def extract_south_pacific_outlook(text: str, url: str = "") -> SeasonalForecast:
    """
    Extract South Pacific TC outlook.

    Key patterns:
    - "a X% chance of seeing activity above its average of Y tropical cyclones"
    - "six to ten tropical cyclones would occur over the South Pacific"
    - Western region and Eastern region breakdowns
    """
    f = SeasonalForecast(basin="SP", basin_full="South Pacific", url=url)
    f.extracted_at = datetime.utcnow().isoformat() + "Z"

    # Season
    season_m = re.search(r"(\d{4})[–\-/](\d{2,4})", text)
    if season_m:
        start_year = int(season_m.group(1))
        end_suffix = season_m.group(2)
        if len(end_suffix) == 2:
            end_year = int(str(start_year)[:2] + end_suffix)
        else:
            end_year = int(end_suffix)
        f.season = f"{start_year}-{str(end_year)[-2:]}"
        f.season_year = start_year
    else:
        # Fallback: "July 2010 to June 2011"
        season_m2 = re.search(r"(?:July|from)\s+(\d{4})\s+to\s+(?:June\s+)?(\d{4})", text, re.IGNORECASE)
        if season_m2:
            start_year = int(season_m2.group(1))
            end_year = int(season_m2.group(2))
            f.season = f"{start_year}-{str(end_year)[-2:]}"
            f.season_year = start_year
    f.forecast_type = "seasonal_outlook"

    # Total count range — numeric or spelled-out numbers
    WORD_TO_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                   "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                   "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15}

    count_m = re.search(
        r"(\d+|\w+)\s+(?:to|-)\s+(\d+|\w+)\s+tropical\s+cyclones\s+(?:would\s+)?(?:occur|expected|are expected|predicted|form)",
        text,
        re.IGNORECASE
    )
    if count_m:
        low_str, high_str = count_m.group(1).lower(), count_m.group(2).lower()
        low = WORD_TO_NUM.get(low_str, int(low_str) if low_str.isdigit() else None)
        high = WORD_TO_NUM.get(high_str, int(high_str) if high_str.isdigit() else None)
        if low is not None and high is not None:
            f.expected_tc_count = f"{low}-{high}"

    # Average
    avg_m = re.search(
        r"average\s+(?:of\s+|is\s+)?(?:around\s+)?(\d+)\s+tropical\s+cyclones\s+(?:in|per)\s+(?:the\s+)?South\s+Pacific",
        text,
        re.IGNORECASE
    )
    if avg_m:
        f.average_tc_count = float(avg_m.group(1))
    else:
        f.average_tc_count = 8  # Known approximate average

    # Subregional probabilities — Western and Eastern
    for region_name in ["Western", "South-West Pacific", "South-East Pacific", "Eastern"]:
        # Pattern 1: "X% chance of ... above/higher"
        sr_m = re.search(
            rf"(?:{region_name})\s+(?:region)?[^.]*?(\d+)%\s+chance\s+(?:of\s+)?(?:seeing\s+)?(?:activity\s+)?(?:above|more|higher)",
            text,
            re.IGNORECASE
        )
        # Pattern 2: "chance that ... will be higher than average is X%"
        if not sr_m:
            sr_m = re.search(
                rf"{region_name}\s+region[^.]*?(?:higher|above)\s+(?:than\s+)?average\s+is\s+(\d+)%",
                text,
                re.IGNORECASE
            )
        if sr_m:
            sr = SubregionForecast(
                name=region_name,
                prob_above_average=int(sr_m.group(1)) / 100,
            )
            # Average for subregion
            sr_avg_m = re.search(
                rf"{region_name}[^.]*?average\s+of\s+(\d+)\s+tropical\s+cyclones",
                text,
                re.IGNORECASE
            )
            if sr_avg_m:
                sr.average_tc_count = float(sr_avg_m.group(1))
            f.subregions.append(sr)

    # ENSO context
    enso_m = re.search(r"((?:El\s+Ni[ñn]o|La\s+Ni[ñn]a|ENSO|neutral\s+conditions)[^.]+\.)", text, re.IGNORECASE)
    if enso_m:
        f.enso_context = re.sub(r"\s+", " ", enso_m.group(1)).strip()

    # Summary
    summary_m = re.search(
        r"((?:Tropical\s+cyclone\s+activity\s+for|The\s+\d{4}[–\-/]\d{2,4}\s+South\s+Pacific)[^.]+\.)",
        text,
        re.IGNORECASE
    )
    if summary_m:
        f.summary = re.sub(r"\s+", " ", summary_m.group(1)).strip()

    _log_forecast(f)
    return f


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _log_forecast(f: SeasonalForecast):
    logger.info(f"  {f.basin_full} ({f.basin}): season={f.season}, issued={f.issue_date}")
    if f.prob_above_average is not None:
        logger.info(f"    Prob above avg: {f.prob_above_average:.0%}")
    if f.categorical_outlook:
        logger.info(f"    Outlook: {f.categorical_outlook}")
    if f.expected_tc_count:
        logger.info(f"    Expected count: {f.expected_tc_count}")
    for sr in f.subregions:
        if isinstance(sr, SubregionForecast):
            logger.info(f"    {sr.name}: {sr.prob_above_average:.0%} above avg")
        elif isinstance(sr, dict):
            logger.info(f"    {sr['name']}: {sr.get('prob_above_average', 'N/A')}")


# ---------------------------------------------------------------------------
# Known URLs
# ---------------------------------------------------------------------------

KNOWN_URLS = {
    "AUS_current": "https://www.bom.gov.au/climate/cyclones/australia/",
    "SP_current": "https://www.bom.gov.au/climate/cyclones/south-pacific/",
    # Archive format (pre-2015):
    # "AUS_2014": "https://www.bom.gov.au/climate/ahead/archive/tropical-cyclone/2014-2015-tc.shtml",
    # "SP_2014": "https://www.bom.gov.au/climate/ahead/archive/tropical-cyclone/south-pacific/2014-2015-tc.shtml",
}


def process_all(fetch_live: bool = True) -> list[SeasonalForecast]:
    """Process all known BoM TC outlook sources."""
    results = []

    if fetch_live:
        for key, url in KNOWN_URLS.items():
            try:
                logger.info(f"Fetching: {key} -> {url}")
                text = fetch_page(url)
                if "AUS" in key:
                    results.append(extract_australian_outlook(text, url))
                elif "SP" in key:
                    results.append(extract_south_pacific_outlook(text, url))
            except Exception as e:
                logger.error(f"Failed {key}: {e}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract BoM seasonal TC outlooks")
    parser.add_argument("--url", help="URL of a specific BoM outlook page")
    parser.add_argument("--text", help="Path to a text file with saved page content")
    parser.add_argument("--basin", choices=["AUS", "SP"], default="AUS", help="Basin to extract")
    parser.add_argument("--live", action="store_true", help="Fetch current outlooks from BoM")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--prompt-context", action="store_true")
    args = parser.parse_args()

    forecasts = []

    if args.url:
        text = fetch_page(args.url)
        if args.basin == "SP":
            forecasts.append(extract_south_pacific_outlook(text, args.url))
        else:
            forecasts.append(extract_australian_outlook(text, args.url))
    elif args.text:
        text = Path(args.text).read_text()
        if args.basin == "SP":
            forecasts.append(extract_south_pacific_outlook(text))
        else:
            forecasts.append(extract_australian_outlook(text))
    elif args.live:
        forecasts = process_all(fetch_live=True)
    else:
        # Default: try live fetch
        forecasts = process_all(fetch_live=True)

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
