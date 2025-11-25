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
    model_name="gemini-3-pro-preview",
    generation_config=generation_config,
    safety_settings=safety_settings
)

# --- Main Functions ---

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def generate_report_for_country(country):
    """Generates a risk analysis report for a single country."""
    logging.info(f"Generating report for: {country}...")
    try:
        # Use the .format() method to insert the country name
        prompt = COUNTRY_ANALYSIS_PROMPT.format(country=country)
        response = country_model.generate_content(prompt)
        # Clean up the response text, removing markdown fences
        clean_text = response.text.strip().replace("```markdown", "").replace("```", "").strip()
        return clean_text
    except Exception as e:
        # Log the full error to help debug
        logging.error(f"An error occurred while generating report for {country}: {e}", exc_info=True)
        return None # Return None if an error occurs

def parse_reports_and_build_table(all_reports_text):
    """Parse SCENARIO_DATA_BLOCKs into structured rows and build a summary table."""
    logging.info("Parsing generated reports to build summary table using Python...")

    json_blocks = re.findall(r'<!-- SCENARIO_DATA_BLOCK: (.*?) -->', all_reports_text, re.DOTALL)
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
            title = scenario.get("title") or scenario.get("name") or ""
            hazard_code = (scenario.get("hazard_code") or "").strip().upper()
            hazard_label = scenario.get("hazard_label") or scenario.get("hazard") or hazard_code
            likely_window_month = scenario.get("likely_window_month") or ""
            probability = scenario.get("probability") or scenario.get("probability_of_occurrence") or ""
            markdown = scenario.get("markdown") or scenario.get("narrative") or ""
            best_guess_raw = scenario.get("best_guess") or {}
            best_guess = {
                "PIN": _parse_int(best_guess_raw.get("PIN")),
                "PA": _parse_int(best_guess_raw.get("PA")),
            }

            scenario_json = dict(scenario)
            scenario_json.setdefault("title", title)
            scenario_json.setdefault("hazard_code", hazard_code)
            scenario_json.setdefault("hazard_label", hazard_label)
            scenario_json.setdefault("likely_window_month", likely_window_month)
            scenario_json.setdefault("best_guess", best_guess)

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
                    "probability": probability or "N/A",
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
                    "probability": probability,
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

def main():
    """Main function to run the bot."""
    logging.info("Starting Pythia Horizon Scanner...")
    start_time = datetime.utcnow()

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
        logging.info(f"Found {len(countries)} countries to process.")
    except FileNotFoundError:
        logging.error("hs_country_list.txt not found. Please create it in the repository root.")
        return

    individual_reports = []
    for i, country in enumerate(countries):
        report = generate_report_for_country(country)
        if report:
            individual_reports.append(report)
        logging.info(f"Completed {country} ({i+1}/{len(countries)})")
        # Add a delay between API calls to respect rate limits and avoid overload
        time.sleep(5) 

    logging.info("All country reports have been generated.")

    # Combine all individual reports into a single string
    all_reports_text = "\n\n---\n\n".join(individual_reports)

    # --- New Step: Parse reports and build table using Python ---
    summary_table, scenarios = parse_reports_and_build_table(all_reports_text)

    # --- Persist structured payloads to DuckDB ---
    data_dir = REPO_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_url = f"duckdb:///{data_dir / 'resolver.duckdb'}"
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

