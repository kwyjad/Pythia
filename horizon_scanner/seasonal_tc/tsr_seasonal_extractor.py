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
    #: Why issue_date is approximate or absent (see seasonal_tc.dates); empty
    #: when the header's own Issued line supplied it.
    issue_date_reason: str = ""
    forecast_type: str = ""  # e.g. "extended_range", "pre_season", "july_update"
    #: Where forecast_type came from: "title", "url" or "issue_month".
    forecast_type_source: str = ""
    
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
            f"## {basin_label} — {self.season_year} Seasonal Forecast (TSR, {self.forecast_type}, "
            f"issued {self.issue_date or 'date not stated'})"
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
        
        # Require all three terciles — some PDFs report a partial set, and
        # formatting a None with :.0% raises TypeError.
        if None not in (self.tercile_above, self.tercile_near, self.tercile_below):
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

_ISSUED_RE = re.compile(
    r"Issued:\s*(\d{1,2})\s*(?:st|nd|rd|th)?\s+(\w+)\s+(\d{4})"
)

#: How much of a TSR PDF counts as its header. Its own issue line sits at the
#: top; the dates further down belong to the earlier forecasts it compares
#: itself against.
_HEADER_CHARS = 1500


def _issued_dates(text: str) -> list[datetime]:
    out = []
    for match in _ISSUED_RE.finditer(text):
        day, month_str, year = match.group(1), match.group(2), match.group(3)
        try:
            out.append(datetime.strptime(f"{day} {month_str} {year}", "%d %B %Y"))
        except ValueError:
            continue
    return out


def extract_issue_date(
    text: str, issue_month: str = "", document_date: str | None = None
) -> str:
    """The date THIS document was issued (ISO), or "" — see :func:`resolve_issue_date`."""

    return resolve_issue_date(text, issue_month=issue_month, document_date=document_date)[0]


def resolve_issue_date(
    text: str, issue_month: str = "", document_date: str | None = None
) -> tuple[str, str]:
    """``(iso_date, reason)`` for the date THIS document was issued.

    A TSR update quotes the forecasts it supersedes — "our pre-season
    forecast, Issued: 28th May 2026" — and on the August 2026 Atlantic update
    that quote sat inside the first 1,500 characters, ahead of the document's
    own line, so a header-first search still returned May. The stored August
    row carried the May date beside the August figures, and the two rows
    collided on the dedup key.

    Three sources, used in this order, and the URL month is the arbiter:

    1. an ``Issued:`` line whose month matches ``issue_month`` (the month
       TSR wrote into its own filename), the latest such line winning;
    2. failing that, the latest ``Issued:`` line anywhere — a document
       cannot have been issued before the forecasts it discusses — but when
       that month contradicts the filename and the PDF's own metadata
       creation date agrees with the filename, the metadata date wins
       (reason ``pdf_metadata_creation_date``);
    3. failing both, the metadata date alone, or nothing.

    A date that survives while contradicting the filename month is kept and
    flagged (``issued_line_disagrees_with_url_month``): an honest disagreement
    beats a silent one.
    """

    from horizon_scanner.seasonal_tc.dates import (
        REASON_NO_DATE_IN_SOURCE,
        REASON_PDF_METADATA,
        REASON_URL_MONTH_DISAGREES,
        month_number,
    )

    wanted = month_number(str(issue_month).split("/")[0]) if issue_month else None
    header = _issued_dates(text[:_HEADER_CHARS])
    everywhere = _issued_dates(text)
    metadata = None
    if document_date:
        try:
            metadata = datetime.strptime(document_date[:10], "%Y-%m-%d")
        except ValueError:
            metadata = None

    if wanted:
        matching = [d for d in (header or everywhere) if d.month == wanted]
        if matching:
            return max(matching).strftime("%Y-%m-%d"), ""
    candidates = header or everywhere
    if candidates:
        chosen = max(candidates)
        if wanted and chosen.month != wanted:
            if metadata is not None and metadata.month == wanted:
                logger.warning(
                    "  Issued line says %s but the filename says month %s; "
                    "using the PDF's own creation date %s",
                    chosen.date(), wanted, metadata.date(),
                )
                return metadata.strftime("%Y-%m-%d"), REASON_PDF_METADATA
            logger.warning(
                "  Issued line %s disagrees with the filename month %s and no "
                "PDF metadata date can settle it — kept and flagged",
                chosen.date(), wanted,
            )
            return chosen.strftime("%Y-%m-%d"), REASON_URL_MONTH_DISAGREES
        return chosen.strftime("%Y-%m-%d"), ""
    if metadata is not None:
        return metadata.strftime("%Y-%m-%d"), REASON_PDF_METADATA
    return "", REASON_NO_DATE_IN_SOURCE


#: TSR names its own products in its filenames, and a filename is not prose.
#: The August update's opening summary discusses the pre-season forecast, so
#: sniffing the first 500 characters typed it "pre_season" — the URL said
#: "August" all along.
_TYPE_BY_ISSUE_MONTH = {
    "december": "extended_range",
    "january": "extended_range",
    "april": "early_april",
    "may": "pre_season",
    "may/june": "pre_season",
    "preseason": "pre_season",
    "june": "june_update",
    "july": "july_update",
    "august": "august_update",
    "october": "october_outlook",
    "november": "november_outlook",
}


#: What TSR calls the product in its own title, first line of the PDF.
#: "Extended Range Forecast for ...", "Pre-Season Forecast for ...",
#: "July Forecast Update for ...", "August Forecast Update for ...".
_TITLE_TYPE_RES = (
    (re.compile(r"extended[\s-]*range", re.IGNORECASE), "extended_range"),
    (re.compile(r"pre[\s-]*season", re.IGNORECASE), "pre_season"),
    (re.compile(r"\bapril\b[^\n]{0,40}forecast", re.IGNORECASE), "early_april"),
    (re.compile(r"\bjune\b[^\n]{0,40}(?:forecast|update)", re.IGNORECASE), "june_update"),
    (re.compile(r"\bjuly\b[^\n]{0,40}(?:forecast|update)", re.IGNORECASE), "july_update"),
    (re.compile(r"\baugust\b[^\n]{0,40}(?:forecast|update)", re.IGNORECASE), "august_update"),
    (re.compile(r"\boctober\b[^\n]{0,40}(?:forecast|outlook)", re.IGNORECASE), "october_outlook"),
    (re.compile(r"\bnovember\b[^\n]{0,40}(?:forecast|outlook)", re.IGNORECASE), "november_outlook"),
)

#: The title is the first line or two of the PDF. The summary paragraph
#: that follows discusses earlier forecasts ("our pre-season forecast ...")
#: and is exactly what must not be read for the product name.
_TITLE_LINES = 2


def _title(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " | ".join(lines[:_TITLE_LINES])


def extract_forecast_type(text: str, issue_date: str, issue_month: str = "") -> str:
    """The forecast type, as the document names itself."""

    return resolve_forecast_type(text, issue_date, issue_month=issue_month)[0]


def resolve_forecast_type(
    text: str, issue_date: str, issue_month: str = ""
) -> tuple[str, str]:
    """``(forecast_type, source)`` with source in ``title``/``url``/``issue_month``/``none``.

    The same NWP outlook of 11 May 2026 was stored as ``extended_range`` by
    runs that sniffed the text and as ``pre_season`` by the run that trusted
    the URL month, because "May" means pre-season for the Atlantic and the
    extended-range product for the NW Pacific. Only the document knows which
    it is, and it says so in its title. The URL month is the fallback for a
    title that names no product, and the issue date's month the fallback for
    that.
    """

    title = _title(text)
    for pattern, kind in _TITLE_TYPE_RES:
        if pattern.search(title):
            return kind, "title"

    if issue_month:
        known = _TYPE_BY_ISSUE_MONTH.get(str(issue_month).strip().lower())
        if known:
            return known, "url"

    if issue_date:
        try:
            month = datetime.strptime(issue_date, "%Y-%m-%d").month
            month_names = {
                12: "extended_range", 1: "extended_range",
                4: "early_april", 5: "pre_season",
                6: "june_update", 7: "july_update",
                8: "august_update"
            }
            return month_names.get(month, "update"), "issue_month"
        except ValueError:
            pass
    return "seasonal", "none"


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


_PDF_DATE_RE = re.compile(r"D:?(\d{4})(\d{2})(\d{2})")


def extract_pdf_document_date(pdf_path: Path) -> str | None:
    """The PDF's own CreationDate (ISO), or None. Never raises.

    A second witness for the issue date: the file TSR wrote carries the day
    it was written, and unlike the prose it never quotes an earlier forecast.
    """

    try:
        with pdfplumber.open(pdf_path) as pdf:
            meta = pdf.metadata or {}
    except Exception:  # noqa: BLE001 - metadata is a bonus, never a blocker
        return None
    for key in ("CreationDate", "ModDate"):
        raw = str(meta.get(key) or "")
        m = _PDF_DATE_RE.search(raw)
        if m:
            try:
                return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return None


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

def extract_forecast(
    text: str,
    pdf_url: str = "",
    pdf_path: str = "",
    issue_month: str = "",
    document_date: str | None = None,
) -> SeasonalForecast:
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
    forecast.issue_date, forecast.issue_date_reason = resolve_issue_date(
        text, issue_month=issue_month, document_date=document_date
    )
    forecast.season_year = extract_season_year(text)
    forecast.forecast_type, forecast.forecast_type_source = resolve_forecast_type(
        text, forecast.issue_date, issue_month=issue_month
    )
    logger.info(
        f"  Season: {forecast.season_year}, Type: {forecast.forecast_type} "
        f"(from {forecast.forecast_type_source}), Issued: {forecast.issue_date}"
        + (f" [{forecast.issue_date_reason}]" if forecast.issue_date_reason else "")
    )
    
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
    if None not in (forecast.tercile_above, forecast.tercile_near, forecast.tercile_below):
        logger.info(f"  Terciles: above={forecast.tercile_above:.0%}, "
                     f"near={forecast.tercile_near:.0%}, below={forecast.tercile_below:.0%}")
    
    # ENSO context
    forecast.enso_context = extract_enso_context(text)
    if forecast.enso_context:
        logger.info(f"  ENSO: {forecast.enso_context[:80]}...")
    
    # Summary
    forecast.summary = extract_summary(text)
    
    return forecast


def process_url(url: str, issue_month: str = "") -> SeasonalForecast:
    """Download a PDF from URL, extract text, run extraction.

    ``issue_month`` comes from the discovery URL, which names TSR's own
    product ("...ForecastAugust2026.pdf"). It is not prose and cannot be
    confused by the document discussing an earlier forecast.
    """

    logger.info(f"Processing: {url}")
    pdf_path = download_pdf(url)
    text = extract_text_from_pdf(pdf_path)
    return extract_forecast(
        text, pdf_url=url, issue_month=issue_month,
        document_date=extract_pdf_document_date(pdf_path),
    )


def process_file(filepath: str) -> SeasonalForecast:
    """Extract from a local PDF file."""
    logger.info(f"Processing local file: {filepath}")
    text = extract_text_from_pdf(Path(filepath))
    return extract_forecast(
        text, pdf_path=filepath, document_date=extract_pdf_document_date(Path(filepath))
    )


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
                forecast = process_url(url, issue_month=candidate.get("issue_month", ""))
                results.append(forecast)
            else:
                logger.debug(f"  Not found ({resp.status_code}): {url}")
        except requests.RequestException as e:
            logger.debug(f"  Error checking {url}: {e}")
        except Exception as e:
            # A parse/format error on one PDF must not discard the forecasts
            # already extracted from the others — skip this URL and continue.
            logger.warning(f"  Skipping {url} — extraction failed: {e}")
    
    _warn_on_colliding_issue_dates(results)
    return results


def _warn_on_colliding_issue_dates(forecasts: list[SeasonalForecast]) -> list[str]:
    """Two outlooks for one basin and season cannot share an issue date.

    When they do, one of them has been misread — which is exactly what
    happened to the August 2026 Atlantic update, stored under the May issue
    date with the May figures. It is reported rather than raised: a
    diagnostic that stops the whole batch would lose the forecasts that
    parsed correctly.
    """

    seen: dict[tuple, list[SeasonalForecast]] = {}
    for forecast in forecasts:
        if not forecast.issue_date:
            continue
        seen.setdefault(
            (forecast.basin, forecast.season_year, forecast.issue_date), []
        ).append(forecast)

    problems = []
    for (basin, season, issued), group in seen.items():
        if len(group) < 2:
            continue
        types = ", ".join(sorted(f.forecast_type or "?" for f in group))
        problem = (
            f"{basin} {season}: {len(group)} outlooks share the issue date "
            f"{issued} ({types}) — at least one has been misread from its PDF"
        )
        problems.append(problem)
        logger.warning("  %s", problem)
    return problems


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
