"""
ENSO State and Forecast Module
===============================
Scrapes current ENSO conditions and probabilistic forecasts from the
IRI/CPC ENSO Quick Look page. Designed as a shared module that any
Pythia hazard pipeline (TC, flood, drought, heatwave) can import.

Sources:
  - IRI ENSO Quick Look: https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/
    Contains both the CPC official probabilistic forecast (early-month) and the
    IRI model-based probabilistic forecast (mid-month), plus the full prediction
    plume of 20+ dynamical and statistical models.

  - IOD (Indian Ocean Dipole) forecast is also extracted as a bonus — relevant
    for East Africa, South/Southeast Asia drought and flood.

Output:
  - Current ENSO state (La Niña / Neutral / El Niño)
  - Current Niño 3.4 SST anomaly
  - 9-season probabilistic forecast (La Niña / Neutral / El Niño %)
  - Multi-model plume averages (dynamical, statistical, all)
  - IOD state and forecast
  - Narrative summary
  - Prompt-ready context block for injection into any hazard prompt

Usage:
    python enso_module.py                    # fetch and display current ENSO state
    python enso_module.py --output enso.json # write JSON
    python enso_module.py --prompt-context   # print prompt-ready block
"""

import re
import json
import argparse
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

import requests
from bs4 import BeautifulSoup

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

    # Current state
    current_state: str = ""      # "La Niña", "Neutral", "El Niño"
    alert_status: str = ""       # CPC alert: "La Niña Advisory", "El Niño Watch", etc.
    nino34_latest_weekly: Optional[float] = None  # latest weekly Niño 3.4 anomaly
    nino34_latest_season: Optional[float] = None  # latest 3-month Niño 3.4 anomaly
    nino34_latest_season_label: str = ""           # e.g. "Aug-Oct 2025"

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
        lines = [
            f"## ENSO State and Forecast (IRI/CPC, published {self.publication_date})"
        ]

        if self.current_state:
            state_line = f"Current state: {self.current_state}"
            if self.alert_status:
                state_line += f" ({self.alert_status})"
            if self.nino34_latest_weekly is not None:
                state_line += f". Latest weekly Niño 3.4: {self.nino34_latest_weekly:+.1f}°C"
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

    # Current state
    if re.search(r"La\s+Ni[ñn]a\s+(?:Advisory|conditions?\s+(?:are|is)\s+(?:firmly\s+)?established|state)", text, re.IGNORECASE):
        f.current_state = "La Niña"
    elif re.search(r"El\s+Ni[ñn]o\s+(?:Advisory|conditions?\s+(?:are|is)\s+established)", text, re.IGNORECASE):
        f.current_state = "El Niño"
    elif re.search(r"ENSO[- ]neutral", text):
        f.current_state = "Neutral"

    # Also check for state from "experiencing a ... La Niña" or "in a La Niña state"
    if not f.current_state:
        state_m = re.search(r"(?:experiencing|in)\s+(?:a\s+)?(?:declining\s+)?(La\s+Ni[ñn]a|El\s+Ni[ñn]o|ENSO[- ]neutral)", text, re.IGNORECASE)
        if state_m:
            raw = state_m.group(1)
            if "La" in raw:
                f.current_state = "La Niña"
            elif "El" in raw:
                f.current_state = "El Niño"
            else:
                f.current_state = "Neutral"

    # Alert status
    alert_m = re.search(r'(?:Alert\s+System\s+Status|maintained\s+a)\s*:?\s*"?([^".\n]+(?:Advisory|Watch|Warning))"?', text, re.IGNORECASE)
    if alert_m:
        f.alert_status = alert_m.group(1).strip().strip('"')

    # Latest weekly Niño 3.4
    weekly_m = re.search(r"(?:latest\s+weekly|week\s+centered)[^.]*?NINO3\.?4\s+index\s+was\s+([+-]?\d+\.?\d*)\s*°?C", text, re.IGNORECASE)
    if not weekly_m:
        weekly_m = re.search(r"NINO3\.?4\s+index[^.]*?was\s+([+-]?\d+\.?\d*)\s*°?C", text, re.IGNORECASE)
    if weekly_m:
        f.nino34_latest_weekly = float(weekly_m.group(1))

    # Latest seasonal Niño 3.4: "SST anomaly in the NINO3.4 region during the Aug–Oct 2025 season was -0.42 °C"
    seasonal_m = re.search(
        r"NINO3\.?4\s+region\s+during\s+(?:the\s+)?(\w+[–\-]\w+\s+\d{4})\s+season\s+was\s+([+-]?\d+\.?\d*)\s*°?C",
        text, re.IGNORECASE
    )
    if seasonal_m:
        f.nino34_latest_season_label = seasonal_m.group(1)
        f.nino34_latest_season = float(seasonal_m.group(2))

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
    if f.current_state:
        parts.append(f"Current ENSO state: {f.current_state}")
        if f.nino34_latest_weekly is not None:
            parts.append(f"(Niño 3.4: {f.nino34_latest_weekly:+.1f}°C)")

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
    logger.info(f"  ENSO state: {f.current_state} ({f.alert_status})")
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

    # Fetch fresh
    logger.info("Fetching fresh ENSO data from IRI...")
    html = fetch_iri_page()
    forecast = extract_enso(html)

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

        raw_context = forecast.to_prompt_context()

        con = connect(read_only=False)
        try:
            ensure_schema(con)
            con.execute(
                """
                INSERT OR REPLACE INTO enso_state
                    (fetch_date, enso_phase, nino34_anomaly, iod_phase,
                     forecast_json, plume_json, raw_context)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    fetch_date_str,
                    forecast.current_state or None,
                    forecast.nino34_latest_weekly,
                    forecast.iod_state or None,
                    forecast_json,
                    plume_json,
                    raw_context,
                ],
            )
            logger.info("Stored ENSO state to DB (fetch_date=%s)", fetch_date_str)
            return True
        finally:
            con.close()
    except Exception as exc:
        logger.warning("Failed to store ENSO state to DB: %s", exc)
        return False


def load_enso_state_from_db(max_age_days: int = 30) -> Optional[ENSOForecast]:
    """
    Load the most recent ENSO state from the DB.

    Returns None if no data, data is too old, or on any DB error.
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
                       forecast_json, plume_json, raw_context
                FROM enso_state
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
                    age_days = (datetime.now(timezone.utc).date() - fetch_date_obj).days
                    if age_days > max_age_days:
                        logger.info(
                            "ENSO DB data is %d days old (max %d); treating as stale",
                            age_days, max_age_days,
                        )
                        return None

            # Reconstruct ENSOForecast
            f = ENSOForecast()
            f.fetch_date = str(fetch_date_val) if fetch_date_val else ""
            f.current_state = row[1] or ""
            f.nino34_latest_weekly = row[2]
            f.iod_state = row[3] or ""

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


def fetch_and_store_enso() -> bool:
    """
    Fetch fresh ENSO data from IRI and store it to the DB.

    Returns True on success, False on failure. Non-fatal.
    """
    try:
        logger.info("Fetching fresh ENSO data from IRI for DB storage...")
        html = fetch_iri_page()
        forecast = extract_enso(html)
        ok = store_enso_state(forecast)
        if ok:
            logger.info(
                "fetch_and_store_enso: stored ENSO state (phase=%s, nino34=%s)",
                forecast.current_state,
                forecast.nino34_latest_weekly,
            )
        return ok
    except Exception as exc:
        logger.warning("fetch_and_store_enso failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch and display ENSO state and forecast")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--prompt-context", action="store_true", help="Print prompt-ready block")
    parser.add_argument("--text-file", help="Read from saved HTML file instead of fetching")
    args = parser.parse_args()

    if args.text_file:
        html = Path(args.text_file).read_text()
    else:
        html = fetch_iri_page()

    forecast = extract_enso(html)

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
