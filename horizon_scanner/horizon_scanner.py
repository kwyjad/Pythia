# -*- coding: utf-8 -*-
"""
Main script for the Pythia Horizon Scanner.
This script reads a list of countries, generates a risk analysis for each using
the Gemini API, and then compiles a final report with a summary table.
"""

import os
import sys
import time
import logging
import json
import re
from datetime import datetime, date
from pathlib import Path

import duckdb
import backoff
import google.generativeai as genai

# Ensure package imports resolve when executed as a script
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Horizon Scanner package imports
from horizon_scanner.db_writer import upsert_hs_payload
from horizon_scanner.hs_prompt import COUNTRY_ANALYSIS_PROMPT

from resolver.db import duckdb_io
from pythia.db.init import init as init_db
from pythia.prompts.registry import load_prompt_spec
from pythia.config import load as load_cfg
from pythia.llm_profiles import get_current_models

# --- Configuration ---
# Set up basic logging to see progress in the GitHub Actions console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Prompt metadata for registry/auditing
PROMPT_KEY = "hs.scenario.v1"
PROMPT_VERSION = "1.0.0"
PROMPT_SPEC = load_prompt_spec(
    PROMPT_KEY,
    PROMPT_VERSION,
    COUNTRY_ANALYSIS_PROMPT,
    str(CURRENT_DIR / "hs_prompt.py"),
)

# Load the Gemini API key from GitHub Secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY secret not found. Please add it to your repository secrets.")

genai.configure(api_key=GEMINI_API_KEY)

# Configuration for the Gemini models
# Use a safety setting to be less restrictive, as the content is professional analysis
_profile_models = {}
try:
    _profile_models = get_current_models()
except Exception:
    _profile_models = {}

GEMINI_MODEL_NAME = _profile_models.get("google", "gemini-3-pro-preview")

generation_config = {
    "temperature": 1.0,
    "top_p": 0.9,
    "top_k": 32,
    "max_output_tokens": 8192,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Initialize the generative model for the analysis
# This uses your preferred model for the complex analysis task
country_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL_NAME,
    generation_config=generation_config,
    safety_settings=safety_settings
)

logging.info(
    "Configured Gemini model %s for Horizon Scanner (temperature=%.2f).",
    GEMINI_MODEL_NAME,
    generation_config.get("temperature", 0.0),
)

# --- Main Functions ---

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def generate_report_for_country(country: str) -> str | None:
    """
    Generate a markdown risk report for a single country using Gemini.
    Returns the cleaned markdown string or None on failure.
    """
    logging.info("Generating report for: %s...", country)

    # *** CRITICAL FIX ***
    # Do NOT use str.format(...) on this prompt; it contains JSON with `{}` braces.
    # Use a plain string replace instead so braces in JSON examples are untouched.
    prompt = COUNTRY_ANALYSIS_PROMPT.replace("{country}", country)
    logging.info("Prompt length for %s: %d characters", country, len(prompt))

    # Call Gemini
    response = country_model.generate_content(prompt)
    logging.info(
        "Gemini call succeeded for %s using model %s.",
        country,
        GEMINI_MODEL_NAME,
    )
    report_text = getattr(response, "text", "") or ""
    logging.info("Received report for %s: %d characters", country, len(report_text))

    cleaned_text = report_text.strip()
    if not cleaned_text:
        logging.warning("Empty report returned for %s after trimming whitespace.", country)
        return None

    # Strip unexpected ``` fences if present
    if cleaned_text.startswith("```"):
        logging.debug("Report for %s starts with a code fence; stripping.", country)
        cleaned_text = cleaned_text.lstrip("`")
        cleaned_text = cleaned_text.lstrip("markdown").lstrip()
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text.rstrip("`").rstrip()

    return cleaned_text

def parse_reports_and_build_table(all_reports_text: str) -> tuple[str, list[dict]]:
    logging.info(
        "Parsing generated reports to build summary table using Python... total_length=%d",
        len(all_reports_text),
    )

    json_blocks = re.findall(
        r'<!-- SCENARIO_DATA_BLOCK: (.*?) -->',
        all_reports_text,
        re.DOTALL,
    )
    logging.info("Found %d SCENARIO_DATA_BLOCK comment(s).", len(json_blocks))

    if not json_blocks:
        logging.warning("No SCENARIO_DATA_BLOCKs found in the generated reports.")
        fallback = (
            "| Country | Scenario Title | Hazard | Likely Month | Probability | PIN Best Guess | PA Best Guess |\n"
            "|---|---|---|---|---|---|---|\n"
            "| No data found | - | - | - | - | - | - |"
        )
        return fallback, []

    def _parse_int(value):
        if value is None:
            return 0
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            cleaned = re.sub(r"[^0-9\-\.]+", "", value)
            if cleaned in {"", "-", ".", "-."}:
                return 0
            try:
                return int(float(cleaned))
            except ValueError:
                return 0
        return 0

    scenarios_for_db = []
    table_entries = []
    dedupe_keys = set()

    for block in json_blocks:
        try:
            data = json.loads(block)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON block: {block}. Error: {e}")
            continue

        country_name = data.get("country") or data.get("country_name") or "Unknown"
        iso3 = (data.get("iso3") or "").strip().upper()
        for scenario in data.get("scenarios", []):
            title = (scenario.get("title") or scenario.get("name") or "").strip()
            hazard_code = (scenario.get("hazard_code") or "").strip().upper()
            hazard_label = (
                scenario.get("hazard_label")
                or scenario.get("hazard")
                or hazard_code
            )
            hazard_label = hazard_label.strip()
            likely_window_month = (scenario.get("likely_window_month") or "").strip()
            probability_text = (
                scenario.get("probability")
                or scenario.get("probability_of_occurrence")
                or ""
            ).strip()
            prob_pct = 0.0
            if probability_text:
                match = re.search(r"(\d+(?:\.\d+)?)", probability_text)
                if match:
                    try:
                        prob_pct = float(match.group(1))
                    except ValueError:
                        prob_pct = 0.0
            markdown = scenario.get("markdown") or scenario.get("narrative") or ""
            best_guess_raw = scenario.get("best_guess") or {}
            best_guess = {
                "PIN": _parse_int(best_guess_raw.get("PIN")),
                "PA": _parse_int(best_guess_raw.get("PA")),
            }

            scenario_json = dict(scenario)
            scenario_json.update(
                {
                    "title": title,
                    "hazard_code": hazard_code,
                    "hazard_label": hazard_label,
                    "likely_window_month": likely_window_month,
                    "best_guess": best_guess,
                    "scenario_title": title,
                    "probability_text": probability_text,
                    "probability_pct": prob_pct,
                    "pin_best_guess": best_guess["PIN"],
                    "pa_best_guess": best_guess["PA"],
                }
            )

            dedupe_key = (iso3 or country_name, hazard_code, title)
            if dedupe_key in dedupe_keys:
                continue
            dedupe_keys.add(dedupe_key)

            table_entries.append(
                {
                    "country": country_name,
                    "title": title or "N/A",
                    "hazard": hazard_code or hazard_label or "N/A",
                    "likely_month": likely_window_month or "N/A",
                    "probability": probability_text or "N/A",
                    "pin": best_guess["PIN"],
                    "pa": best_guess["PA"],
                }
            )

            if not iso3:
                logging.warning(
                    "Scenario '%s' for %s missing ISO3 code; skipping database persistence.",
                    title,
                    country_name,
                )
                continue

            scenarios_for_db.append(
                {
                    "iso3": iso3,
                    "country_name": country_name,
                    "hazard_code": hazard_code,
                    "hazard_label": hazard_label,
                    "likely_window_month": likely_window_month,
                    "best_guess": best_guess,
                    "title": title,
                    "markdown": markdown,
                    "probability": probability_text,
                    "scenario_title": title,
                    "probability_text": probability_text,
                    "probability_pct": prob_pct,
                    "pin_best_guess": best_guess["PIN"],
                    "pa_best_guess": best_guess["PA"],
                    "json": {**scenario_json, "country": country_name, "iso3": iso3},
                }
            )

    if not table_entries:
        logging.warning("JSON blocks were found, but no valid scenario data could be extracted.")
        fallback = (
            "| Country | Scenario Title | Hazard | Likely Month | Probability | PIN Best Guess | PA Best Guess |\n"
            "|---|---|---|---|---|---|---|\n"
            "| No data extracted | - | - | - | - | - | - |"
        )
        return fallback, []

    logging.info(
        "Scenario parsing complete: %d table entries; %d scenarios with iso3 for DB.",
        len(table_entries),
        len(scenarios_for_db),
    )

    table_entries.sort(key=lambda x: (x["country"], x["title"]))
    table_header = (
        "| Country | Scenario Title | Hazard | Likely Month | Probability | PIN Best Guess | PA Best Guess |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    table_rows = [
        f"| {entry['country']} | {entry['title']} | {entry['hazard']} | {entry['likely_month']} | {entry['probability']} | {entry['pin']} | {entry['pa']} |"
        for entry in table_entries
    ]

    return table_header + "\n".join(table_rows), scenarios_for_db

def main(countries: list[str] | None = None):
    """Main function to run the bot.

    If `countries` is None or empty, fall back to hs_country_list.txt.
    Otherwise, use the provided list verbatim (treated as country names).
    """
    logging.info("Starting Pythia Horizon Scanner...")
    start_time = datetime.utcnow()

    if not countries:
        try:
            country_list_path = CURRENT_DIR / "hs_country_list.txt"
            logging.info("Reading country list from %s...", country_list_path)
            with open(country_list_path, "r", encoding="utf-8") as f:
                # Ignore blank lines and lines starting with #
                countries = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
            logging.info("Found %d countries to process (from hs_country_list.txt).", len(countries))
        except FileNotFoundError:
            logging.error("hs_country_list.txt not found. Please create it in the repository root, or pass explicit countries.")
            return
    else:
        logging.info("Using %d countries passed in by caller.", len(countries))

    individual_reports: list[str] = []

    for idx, country in enumerate(countries, start=1):
        try:
            report = generate_report_for_country(country)
        except Exception as e:
            logging.error(
                "Failed to generate report for %s after retries: %s",
                country,
                e,
                exc_info=True,
            )
            report = None

        if report:
            individual_reports.append(report)
            logging.info(
                "Completed %s (%d/%d); report length=%d characters",
                country,
                idx,
                len(countries),
                len(report),
            )
        else:
            logging.warning(
                "No report captured for %s (%d/%d).",
                country,
                idx,
                len(countries),
            )

        # Add a delay between API calls to respect rate limits and avoid overload
        time.sleep(5)

    logging.info("All country reports have been generated.")

    # Combine all individual reports into a single string
    all_reports_text = "\n\n---\n\n".join(individual_reports)

    # --- New Step: Parse reports and build table using Python ---
    summary_table, scenarios = parse_reports_and_build_table(all_reports_text)

    # --- Persist structured payloads to DuckDB ---
    try:
        cfg = load_cfg()
        app_cfg = cfg.get("app", {}) if isinstance(cfg, dict) else {}
        db_url = str(app_cfg.get("db_url", "")).strip()
    except Exception:
        db_url = ""

    if not db_url:
        # Fallback to legacy default if config is missing or incomplete
        data_dir = REPO_ROOT / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        db_url = f"duckdb:///{data_dir / 'resolver.duckdb'}"
        logging.info("No app.db_url found in config; falling back to %s", db_url)
    else:
        # Ensure directory exists for configured DuckDB path
        db_path = db_url.replace("duckdb:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        logging.info("Using app.db_url from config for DuckDB: %s", db_url)

    init_db(db_url)
    run_id = f"hs_{start_time.strftime('%Y%m%dT%H%M%S')}"
    run_meta = {
        "run_id": run_id,
        "started_at": start_time.isoformat(),
        "finished_at": datetime.utcnow().isoformat(),
        "countries": json.dumps(countries),
        "prompt_key": PROMPT_SPEC.key,
        "prompt_version": PROMPT_SPEC.version,
        "prompt_sha256": PROMPT_SPEC.sha256,
    }
    upsert_hs_payload(
        db_url,
        run_meta,
        scenarios,
        today=date.today(),
        horizon_months=6,
    )

    # --- Diagnostics: list questions written to DuckDB for this run ---
    questions_md = ""
    conn = None
    try:
        conn = duckdb_io.get_db(db_url)
        query = """
            SELECT
                question_id,
                iso3,
                hazard_code,
                metric,
                target_month,
                wording
            FROM questions
            WHERE run_id = ?
            ORDER BY iso3, hazard_code, metric, target_month, question_id
        """
        question_rows = conn.execute(query, [run_meta["run_id"]]).fetchall()

        if question_rows:
            logging.info(
                "Questions written to DuckDB for run %s (total %d):",
                run_meta["run_id"],
                len(question_rows),
            )
            for q_id, iso3, hz, metric, target_month, wording in question_rows:
                logging.info(
                    "  question_id=%s | iso3=%s | hazard=%s | metric=%s | month=%s | wording=%s",
                    q_id,
                    iso3,
                    hz,
                    metric,
                    target_month,
                    wording,
                )

            lines = [
                "## Questions Written to DuckDB",
                "",
                f"Total questions written: {len(question_rows)}",
                "",
                "| Question ID | ISO3 | Hazard | Metric | Target Month | Wording |",
                "|---|---|---|---|---|---|",
            ]
            for q_id, iso3, hz, metric, target_month, wording in question_rows:
                safe_wording = (wording or "").replace("|", "\\|")
                lines.append(
                    f"| {q_id} | {iso3} | {hz} | {metric} | {target_month} | {safe_wording} |"
                )
            questions_md = "\n".join(lines)
        else:
            logging.warning(
                "No questions found in DuckDB for run_id=%s (check scenario parsing and DB write).",
                run_meta["run_id"],
            )
            questions_md = (
                "## Questions Written to DuckDB\n\n"
                f"No questions found for run_id `{run_meta['run_id']}`. "
                "Check scenario parsing and DB write steps.\n"
            )
    except duckdb.Error as e:
        logging.error(
            "Failed to fetch questions from DuckDB for diagnostics: %s",
            e,
            exc_info=True,
        )
        questions_md = (
            "## Questions Written to DuckDB\n\n"
            "An error occurred while fetching questions for diagnostics. "
            "See workflow logs for details.\n"
        )
    except Exception as e:  # pragma: no cover - unexpected exceptions
        logging.error(
            "Unexpected error during question diagnostics: %s",
            e,
            exc_info=True,
        )
        questions_md = (
            "## Questions Written to DuckDB\n\n"
            "An unexpected error occurred while fetching questions for diagnostics. "
            "See workflow logs for details.\n"
        )
    finally:
        duckdb_io.close_db(conn)

    # --- Final Report Assembly ---
    utc_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    final_report_content = (
        f"# Pythia Horizon Scan Report\n\n"
        f"**Generated on:** {utc_time}\n\n"
        f"## Summary Table\n\n{summary_table}\n\n"
        f"---\n\n{questions_md}\n\n"
        f"---\n\n# Individual Country Reports\n\n{all_reports_text}"
    )

    try:
        with open("Risk_Report.md", "w", encoding="utf-8") as f:
            f.write(final_report_content)
        logging.info("Final report successfully generated and saved to Risk_Report.md")
    except IOError as e:
        logging.error(f"Failed to write the final report file: {e}")

    logging.info("Bot run complete.")

if __name__ == "__main__":
    main()

