"""
TSR Seasonal Tropical Cyclone Forecast Extractor
=================================================
Extracts structured forecast data from Tropical Storm Risk (TSR) seasonal
forecast PDFs. Covers Atlantic, NW Pacific, and (when available) Australian
and SW Indian Ocean basins.

Primary extraction is regex-based (the PDFs are very consistently formatted).
An optional LLM fallback can be enabled for edge cases.

Output: JSON per forecast document, suitable for ingestion into Pythia's
TC prompt grounding pipeline.

Usage:
    python tsr_seasonal_extractor.py                    # fetch latest known URLs
    python tsr_seasonal_extractor.py --url <pdf_url>    # extract from a specific PDF
    python tsr_seasonal_extractor.py --file <pdf_path>  # extract from a local PDF
    python tsr_seasonal_extractor.py --discover 2026    # try to discover all PDFs for a year
"""

import re
import json
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict

import requests
import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SeasonalForecast:
    """Structured representation of a single seasonal TC forecast."""
    source: str = "TSR"
    basin: str = ""
    basin_full: str = ""
    season_year: int = 0
    issue_date: str = ""
    forecast_type: str = ""  # e.g. "extended_range", "pre_season", "july_update"
    
    # Core forecast numbers
    named_storms: Optional[int] = None
    hurricanes_or_typhoons: Optional[int] = None  # "Hurricanes" (ATL) or "Typhoons" (NWP)
    intense_hurricanes_or_typhoons: Optional[int] = None  # "Intense Hurricanes" or "Intense Typhoons"
    ace_index: Optional[int] = None
    
    # Climatology for context
    climate_norm_30yr: dict = field(default_factory=dict)
    climate_norm_10yr: dict = field(default_factory=dict)
    
    # Tercile probabilities
    tercile_above: Optional[float] = None
    tercile_near: Optional[float] = None
    tercile_below: Optional[float] = None
    
    # ENSO context (extracted from narrative)
    enso_context: str = ""
    
    # Confidence / narrative summary
    summary: str = ""
    
    # Metadata
    pdf_url: str = ""
    extracted_at: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None and v != "" and v != {}}

    def to_prompt_context(self) -> str:
        """Generate a concise text block suitable for injection into a TC question prompt."""
        basin_label = self.basin_full or self.basin
        if "Atlantic" in basin_label or "Eastern" in basin_label or "Central" in basin_label:
            storm_label = "hurricanes"
            intense_label = "major hurricanes"
        elif "Pacific" in basin_label and "Northwest" in basin_label:
            storm_label = "typhoons"
            intense_label = "intense typhoons"
        elif "Indian" in basin_label or "Australian" in basin_label or "South Pacific" in basin_label:
            storm_label = "cyclones"
            intense_label = "intense cyclones"
        else:
            storm_label = "tropical cyclones"
            intense_label = "intense tropical cyclones"
        
        lines = [
            f"## {basin_label} — {self.season_year} Seasonal Forecast (TSR, {self.forecast_type}, issued {self.issue_date})"
        ]
        if self.summary:
            lines.append(self.summary)
        
        parts = []
        if self.named_storms is not None:
            parts.append(f"{self.named_storms} named storms")
        if self.hurricanes_or_typhoons is not None:
            parts.append(f"{self.hurricanes_or_typhoons} {storm_label}")
        if self.intense_hurricanes_or_typhoons is not None:
            parts.append(f"{self.intense_hurricanes_or_typhoons} {intense_label}")
        if self.ace_index is not None:
            parts.append(f"ACE index {self.ace_index}")
        if parts:
            lines.append("Forecast: " + ", ".join(parts) + ".")
        
        if self.climate_norm_30yr:
            norm_parts = []
            for k, v in self.climate_norm_30yr.items():
                norm_parts.append(f"{v} {k}")
            lines.append("30-yr climate norm (1991-2020): " + ", ".join(norm_parts) + ".")
        
        if self.tercile_above is not None:
            lines.append(
                f"Tercile probabilities (ACE): {self.tercile_above:.0%} above-normal, "
                f"{self.tercile_near:.0%} near-normal, {self.tercile_below:.0%} below-normal."
            )
        
        if self.enso_context:
            lines.append(f"ENSO context: {self.enso_context}")
        
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Basin detection
# ---------------------------------------------------------------------------

BASIN_PATTERNS = {
    "ATL": {
        "regex": r"North\s+Atlantic",
        "basin": "ATL",
        "basin_full": "North Atlantic",
        "storm_col": "Hurricanes",
        "intense_col": "Intense Hurricanes",
    },
    "NWP": {
        "regex": r"Northwest\s+Pacific|NW\s+Pacific",
        "basin": "NWP",
        "basin_full": "Northwest Pacific",
        "storm_col": "Typhoons",
        "intense_col": "Intense Typhoons",
    },
    "AUS": {
        "regex": r"Austral",
        "basin": "AUS",
        "basin_full": "Australian Region",
        "storm_col": "Hurricanes",
        "intense_col": "Intense Hurricanes",
    },
    "SWI": {
        "regex": r"South\s*West\s+Indian|SW\s+Indian",
        "basin": "SWI",
        "basin_full": "South-West Indian Ocean",
        "storm_col": "Cyclones",
        "intense_col": "Intense Cyclones",
    },
    "SP": {
        "regex": r"South\s+Pacific",
        "basin": "SP",
        "basin_full": "South Pacific",
        "storm_col": "Hurricanes",
        "intense_col": "Intense Hurricanes",
    },
}


def detect_basin(text: str) -> dict:
    """Detect which basin this forecast covers."""
    for key, info in BASIN_PATTERNS.items():
        if re.search(info["regex"], text, re.IGNORECASE):
            return info
    return {"basin": "UNKNOWN", "basin_full": "Unknown", "storm_col": "Hurricanes", "intense_col": "Intense Hurricanes"}


# ---------------------------------------------------------------------------
# Core extraction (regex-based)
# ---------------------------------------------------------------------------

def extract_issue_date(text: str) -> str:
    """Extract the issue date from the PDF header."""
    # Pattern: "Issued: 11th December 2025" or "Issued: 8th July 2025"
    m = re.search(r"Issued:\s*(\d{1,2})\s*(?:st|nd|rd|th)?\s+(\w+)\s+(\d{4})", text)
    if m:
        day, month_str, year = m.group(1), m.group(2), m.group(3)
        try:
            dt = datetime.strptime(f"{day} {month_str} {year}", "%d %B %Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return f"{year}-{month_str}-{day}"
    return ""


def extract_forecast_type(text: str, issue_date: str) -> str:
    """Infer the forecast type from the title and issue date."""
    text_lower = text[:500].lower()
    if "extended range" in text_lower:
        return "extended_range"
    if "pre-season" in text_lower or "pre season" in text_lower:
        return "pre_season"
    
    # Infer from month
    if issue_date:
        try:
            month = datetime.strptime(issue_date, "%Y-%m-%d").month
            month_names = {
                12: "extended_range", 1: "extended_range",
                4: "early_april", 5: "pre_season",
                6: "june_update", 7: "july_update",
                8: "august_update"
            }
            return month_names.get(month, "update")
        except ValueError:
            pass
    return "seasonal"


def extract_season_year(text: str) -> int:
    """Extract the forecast target year."""
    # Look for patterns like "Hurricane Activity in 2026" or "Typhoon Activity in 2025"
    m = re.search(r"(?:Hurricane|Typhoon|Cyclone)\s+Activity\s+in\s+(\d{4})", text)
    if m:
        return int(m.group(1))
    # Fallback: look for "season" + year
    m = re.search(r"(\d{4})\s+(?:season|hurricane|typhoon)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0


def extract_forecast_table(text: str, basin_info: dict) -> dict:
    """
    Extract the main forecast table. TSR tables look like:
    
     ACE Intense Tropical
     Index Hurricanes Hurricanes Storms
    TSR Forecast 2026 125 3 7 14
    30-yr Climate Norm 1991-2020 122 3.2 7.2 14.4
    10-yr Climate Norm 2016-2025 149 3.9 8.2 18.1
    """
    result = {
        "ace_index": None,
        "intense": None,
        "hurricanes_typhoons": None,
        "named_storms": None,
        "climate_norm_30yr": {},
        "climate_norm_10yr": {},
    }
    
    # Match the TSR Forecast row: "TSR Forecast YYYY" followed by numbers
    # The numbers are: ACE, Intense, Hurricanes/Typhoons, Tropical Storms
    forecast_pattern = r"TSR\s+Forecast\s+\d{4}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    m = re.search(forecast_pattern, text)
    if m:
        result["ace_index"] = int(round(float(m.group(1))))
        result["intense"] = int(round(float(m.group(2))))
        result["hurricanes_typhoons"] = int(round(float(m.group(3))))
        result["named_storms"] = int(round(float(m.group(4))))
        logger.info(f"  Forecast row: ACE={result['ace_index']}, Intense={result['intense']}, "
                     f"H/T={result['hurricanes_typhoons']}, TS={result['named_storms']}")
    else:
        logger.warning("  Could not find TSR Forecast row in table")
    
    # 30-yr norm
    norm30_pattern = r"30-yr\s+Climate\s+Norm\s+[\d-]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    m = re.search(norm30_pattern, text)
    if m:
        storm_label = basin_info.get("storm_col", "hurricanes").lower()
        intense_label = basin_info.get("intense_col", "intense hurricanes").lower()
        result["climate_norm_30yr"] = {
            "ace_index": float(m.group(1)),
            intense_label: float(m.group(2)),
            storm_label: float(m.group(3)),
            "named_storms": float(m.group(4)),
        }
    
    # 10-yr norm
    norm10_pattern = r"10-yr\s+Climate\s+Norm\s+[\d-]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    m = re.search(norm10_pattern, text)
    if m:
        storm_label = basin_info.get("storm_col", "hurricanes").lower()
        intense_label = basin_info.get("intense_col", "intense hurricanes").lower()
        result["climate_norm_10yr"] = {
            "ace_index": float(m.group(1)),
            intense_label: float(m.group(2)),
            storm_label: float(m.group(3)),
            "named_storms": float(m.group(4)),
        }
    
    return result


def extract_tercile_probabilities(text: str) -> dict:
    """
    Extract tercile probabilities from text like:
    "a 32% probability of being upper tercile (>156)), a 49% likelihood of being middle 
    tercile (75 to 156)) and a 19% chance of being lower tercile (<75))"
    
    Also handles:
    "only a 14% probability of being upper tercile, a 32% likelihood of being middle 
    tercile and a 54% chance of being lower tercile"
    """
    result = {"above": None, "near": None, "below": None}
    
    # Upper tercile
    m = re.search(r"(\d+)%\s+(?:probability|chance|likelihood)\s+of\s+being\s+upper\s+tercile", text)
    if m:
        result["above"] = int(m.group(1)) / 100
    
    # Middle tercile
    m = re.search(r"(\d+)%\s+(?:probability|chance|likelihood)\s+of\s+being\s+middle\s+tercile", text)
    if m:
        result["near"] = int(m.group(1)) / 100
    
    # Lower tercile
    m = re.search(r"(\d+)%\s+(?:probability|chance|likelihood)\s+of\s+being\s+lower\s+tercile", text)
    if m:
        result["below"] = int(m.group(1)) / 100
    
    return result


def extract_enso_context(text: str) -> str:
    """Extract ENSO-related context from the narrative sections."""
    # Look for the ENSO paragraph in section 2 or 3
    # Try to grab the first 1-2 sentences of the ENSO section
    m = re.search(
        r"ENSO:\s*(.+?)(?:\n\n|\n[A-Z]|\nTrade\s+Wind|\nActivity|\nPacific\s+Decadal|\nSkill|\nSpring|\nIntra)",
        text,
        re.DOTALL
    )
    if m:
        enso_text = m.group(1).strip()
        # Clean up: collapse whitespace, take first 2 sentences
        enso_text = re.sub(r"\s+", " ", enso_text)
        sentences = re.split(r"(?<=[.!?])\s+", enso_text)
        return " ".join(sentences[:2]).strip()
    return ""


def extract_summary(text: str) -> str:
    """Extract the forecast summary line."""
    # Look for "TSR predicts that..." or "TSR slightly lowers..." 
    m = re.search(r"(TSR\s+(?:predicts|slightly\s+lowers|raises|maintains).+?(?:norm|average|normal|climatology)[\s.])", text, re.DOTALL)
    if m:
        summary = re.sub(r"\s+", " ", m.group(1)).strip()
        # Ensure it ends cleanly
        if not summary.endswith("."):
            summary += "."
        return summary
    # Fallback: "TSR predicts..." up to first period
    m = re.search(r"(TSR\s+predicts\s+that.+?\.)", text, re.DOTALL)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    # Fallback: look for "Forecast Summary" section
    m = re.search(r"Forecast\s+Summary\s*\n(.+?\.)", text, re.DOTALL)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


# ---------------------------------------------------------------------------
# PDF download + text extraction
# ---------------------------------------------------------------------------

def download_pdf(url: str, cache_dir: Path = Path(__file__).parent / "output" / "pdf_cache") -> Path:
    """Download a PDF and cache it locally."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    local_path = cache_dir / filename
    
    if local_path.exists():
        logger.info(f"  Using cached: {local_path}")
        return local_path
    
    logger.info(f"  Downloading: {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    local_path.write_bytes(resp.content)
    return local_path


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts)


# ---------------------------------------------------------------------------
# Main extraction orchestrator
# ---------------------------------------------------------------------------

def extract_forecast(text: str, pdf_url: str = "", pdf_path: str = "") -> SeasonalForecast:
    """Run the full extraction pipeline on extracted PDF text."""
    forecast = SeasonalForecast()
    forecast.pdf_url = pdf_url
    forecast.extracted_at = datetime.utcnow().isoformat() + "Z"
    
    # Basin detection
    basin_info = detect_basin(text)
    forecast.basin = basin_info["basin"]
    forecast.basin_full = basin_info["basin_full"]
    logger.info(f"  Basin: {forecast.basin_full}")
    
    # Issue date & forecast type
    forecast.issue_date = extract_issue_date(text)
    forecast.season_year = extract_season_year(text)
    forecast.forecast_type = extract_forecast_type(text, forecast.issue_date)
    logger.info(f"  Season: {forecast.season_year}, Type: {forecast.forecast_type}, Issued: {forecast.issue_date}")
    
    # Core forecast table
    table_data = extract_forecast_table(text, basin_info)
    forecast.ace_index = table_data["ace_index"]
    forecast.intense_hurricanes_or_typhoons = table_data["intense"]
    forecast.hurricanes_or_typhoons = table_data["hurricanes_typhoons"]
    forecast.named_storms = table_data["named_storms"]
    forecast.climate_norm_30yr = table_data["climate_norm_30yr"]
    forecast.climate_norm_10yr = table_data["climate_norm_10yr"]
    
    # Tercile probabilities
    terciles = extract_tercile_probabilities(text)
    forecast.tercile_above = terciles["above"]
    forecast.tercile_near = terciles["near"]
    forecast.tercile_below = terciles["below"]
    if forecast.tercile_above is not None:
        logger.info(f"  Terciles: above={forecast.tercile_above:.0%}, "
                     f"near={forecast.tercile_near:.0%}, below={forecast.tercile_below:.0%}")
    
    # ENSO context
    forecast.enso_context = extract_enso_context(text)
    if forecast.enso_context:
        logger.info(f"  ENSO: {forecast.enso_context[:80]}...")
    
    # Summary
    forecast.summary = extract_summary(text)
    
    return forecast


def process_url(url: str) -> SeasonalForecast:
    """Download a PDF from URL, extract text, run extraction."""
    logger.info(f"Processing: {url}")
    pdf_path = download_pdf(url)
    text = extract_text_from_pdf(pdf_path)
    return extract_forecast(text, pdf_url=url)


def process_file(filepath: str) -> SeasonalForecast:
    """Extract from a local PDF file."""
    logger.info(f"Processing local file: {filepath}")
    text = extract_text_from_pdf(Path(filepath))
    return extract_forecast(text, pdf_path=filepath)


# ---------------------------------------------------------------------------
# URL discovery — TSR uses predictable naming conventions
# ---------------------------------------------------------------------------

TSR_BASE = "https://www.tropicalstormrisk.com/docs/"

# Known URL patterns for TSR forecasts
# ATL: TSRATLForecast{Month}{Year}.pdf — months: December (ext range), April, PreSeason (May), July, August
# NWP: TSRNWPForecast{Month}{Year}.pdf — months: April, May, July, August
# AUS: TSRAUSForecast{Month}{Year}.pdf (if available)

def build_tsr_urls(year: int) -> list[dict]:
    """
    Build candidate URLs for a given forecast year.
    Note: 'year' is the SEASON year, not the issue year.
    Extended range forecasts are issued in December of year-1.
    """
    candidates = []
    
    # Atlantic
    atl_variants = [
        (f"TSRATLForecastDecember{year}.pdf", year - 1, "December"),  # extended range, issued Dec of prior year
        (f"TSRATLForecastApril{year}.pdf", year, "April"),
        (f"TSRATLForecastPreSeason{year}.pdf", year, "May/June"),
        (f"TSRATLForecastJuly{year}.pdf", year, "July"),
        (f"TSRATLForecastAugust{year}.pdf", year, "August"),
    ]
    for filename, issue_yr, month in atl_variants:
        candidates.append({
            "url": TSR_BASE + filename,
            "basin": "ATL",
            "season_year": year,
            "issue_month": month,
        })
    
    # NW Pacific
    nwp_variants = [
        (f"TSRNWPForecastApril{year}.pdf", year, "April"),
        (f"TSRNWPForecastMay{year}.pdf", year, "May"),
        (f"TSRNWPForecastJuly{year}.pdf", year, "July"),
        (f"TSRNWPForecastAugust{year}.pdf", year, "August"),
    ]
    for filename, issue_yr, month in nwp_variants:
        candidates.append({
            "url": TSR_BASE + filename,
            "basin": "NWP",
            "season_year": year,
            "issue_month": month,
        })
    
    # Australian (less certain on naming)
    # TSR has historically issued AUS forecasts but naming may vary
    aus_variants = [
        (f"TSRAUSForecastOctober{year}.pdf", year, "October"),
        (f"TSRAUSForecastNovember{year}.pdf", year, "November"),
    ]
    for filename, issue_yr, month in aus_variants:
        candidates.append({
            "url": TSR_BASE + filename,
            "basin": "AUS",
            "season_year": year,
            "issue_month": month,
        })
    
    return candidates


def discover_and_extract(year: int) -> list[SeasonalForecast]:
    """Try all candidate URLs for a year, extract from those that exist."""
    candidates = build_tsr_urls(year)
    results = []
    
    for candidate in candidates:
        url = candidate["url"]
        try:
            resp = requests.head(url, timeout=10, allow_redirects=True)
            if resp.status_code == 200:
                logger.info(f"  Found: {url}")
                forecast = process_url(url)
                results.append(forecast)
            else:
                logger.debug(f"  Not found ({resp.status_code}): {url}")
        except requests.RequestException as e:
            logger.debug(f"  Error checking {url}: {e}")
    
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract TSR seasonal TC forecasts from PDFs")
    parser.add_argument("--url", help="URL of a specific TSR forecast PDF")
    parser.add_argument("--file", help="Path to a local TSR forecast PDF")
    parser.add_argument("--discover", type=int, metavar="YEAR", help="Discover and extract all TSR forecasts for a season year")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--prompt-context", action="store_true", help="Also print prompt-ready context blocks")
    args = parser.parse_args()
    
    forecasts = []
    
    if args.url:
        forecasts.append(process_url(args.url))
    elif args.file:
        forecasts.append(process_file(args.file))
    elif args.discover:
        forecasts = discover_and_extract(args.discover)
    else:
        # Default: try the latest known Atlantic + NWP forecasts
        default_urls = [
            "https://www.tropicalstormrisk.com/docs/TSRATLForecastDecember2026.pdf",
            "https://www.tropicalstormrisk.com/docs/TSRNWPForecastJuly2025.pdf",
        ]
        for url in default_urls:
            try:
                forecasts.append(process_url(url))
            except Exception as e:
                logger.error(f"Failed to process {url}: {e}")
    
    # Output
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
