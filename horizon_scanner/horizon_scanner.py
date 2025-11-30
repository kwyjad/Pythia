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
import subprocess
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict

import asyncio
import backoff
import duckdb

# Ensure package imports resolve when executed as a script
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Horizon Scanner package imports
from horizon_scanner.db_writer import (
    log_hs_country_reports_to_db,
    log_hs_run_to_db,
    upsert_hs_payload,
)
from horizon_scanner.hs_prompt import COUNTRY_ANALYSIS_PROMPT
from horizon_scanner.llm_logging import log_hs_llm_call

from pythia.prompts.registry import load_prompt_spec
from pythia.llm_profiles import get_current_models, get_current_profile
from pythia.db.schema import ensure_schema, connect
from forecaster.providers import GEMINI_MODEL_ID, ModelSpec, call_chat_ms, estimate_cost_usd

# --- Configuration ---
# Set up basic logging to see progress in the GitHub Actions console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prompt metadata for registry/auditing
PROMPT_KEY = "hs.scenario.v1"
PROMPT_VERSION = "1.0.0"
PROMPT_SPEC = load_prompt_spec(
    PROMPT_KEY,
    PROMPT_VERSION,
    COUNTRY_ANALYSIS_PROMPT,
    str(CURRENT_DIR / "hs_prompt.py"),
)

# Configuration for the Gemini models
# Use a safety setting to be less restrictive, as the content is professional analysis
_profile_models = {}
try:
    _profile_models = get_current_models()
except Exception:
    _profile_models = {}


def _resolve_hs_model() -> str:
    if GEMINI_MODEL_ID:
        return GEMINI_MODEL_ID
    profile_model = _profile_models.get("google")
    if profile_model:
        return str(profile_model)
    return "gemini-2.5-flash-lite"


def _resolve_hs_temperature() -> float:
    try:
        return float(os.getenv("HS_GEMINI_TEMP", "1.0"))
    except Exception:
        return 1.0


GEMINI_MODEL_NAME = _resolve_hs_model()
HS_TEMPERATURE = _resolve_hs_temperature()
logging.info(
    "Configured Gemini model %s for Horizon Scanner (temperature=%.2f).",
    GEMINI_MODEL_NAME,
    HS_TEMPERATURE,
)


def _call_gemini_llm(prompt_text: str, **kwargs: Any) -> tuple[str, Dict[str, Any]]:
    del kwargs

    async def _call() -> tuple[str, Dict[str, Any], str]:
        spec = ModelSpec(
            name="Gemini",
            provider="google",
            model_id=GEMINI_MODEL_NAME,
            active=bool(GEMINI_MODEL_NAME),
        )
        return await call_chat_ms(
            spec,
            prompt_text,
            temperature=HS_TEMPERATURE,
            prompt_key="hs.scenario.v1",
            prompt_version=PROMPT_VERSION,
            component="HorizonScanner",
        )

    text, usage, error = asyncio.run(_call())
    if error:
        raise RuntimeError(error)

    if not usage or not any(usage.values()):
        from forecaster.research import _rough_token_count

        prompt_tokens = _rough_token_count(prompt_text)
        completion_tokens = _rough_token_count(text)
        usage.update(
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        )

    usage["cost_usd"] = estimate_cost_usd(GEMINI_MODEL_NAME, usage)
    return text, usage


def _call_hs_llm_for_country(*, hs_run_id: str, prompt: str) -> str:
    return log_hs_llm_call(
        hs_run_id=hs_run_id,
        call_type="hs_generation",
        model_name=GEMINI_MODEL_NAME,
        provider="google",
        model_id=GEMINI_MODEL_NAME,
        prompt_text=prompt,
        low_level_call=_call_gemini_llm,
        low_level_kwargs={},
        run_id=None,
        question_id=None,
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

    hs_run_id = os.getenv("PYTHIA_HS_RUN_ID", "")

    # Call Gemini
    t0 = time.time()
    report_text = _call_hs_llm_for_country(hs_run_id=hs_run_id, prompt=prompt)
    latency_ms = int((time.time() - t0) * 1000)

    logging.info(
        "Gemini call succeeded for %s using model %s (latency=%d ms).",
        country,
        GEMINI_MODEL_NAME,
        latency_ms,
    )
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


def _parse_scenario_block(block: str) -> dict | None:
    """
    Parse a SCENARIO_DATA_BLOCK JSON string into a Python dict.

    This is defensive against common LLM formatting mistakes:
      - Double braces: '{{ ... }}' instead of '{ ... }'
      - Code fences: ```json ... ``` around the JSON
    """
    if not block:
        return None

    raw = block.strip()

    # Strip common code fences like ```json ... ``` or ``` ... ```
    if raw.startswith("```"):
        # Remove leading ```... and trailing ``` if present
        # e.g. ```json\n{...}\n``` -> {...}
        parts = raw.split("```")
        # parts[0] is empty, parts[1] might be "json\n{...}", parts[-1] after last fence
        raw = "\n".join(p for p in parts[1:] if p).strip()
        # If there's still a leading 'json' or 'JSON' line, drop it
        if raw.lower().startswith("json"):
            raw = raw.split("\n", 1)[-1].strip()

    clean = raw

    # Fix classic double-brace pattern: leading/trailing {{ ... }}
    if clean.startswith("{{") and clean.endswith("}}"):
        clean = clean[1:-1].strip()

    # As a further guard, replace remaining '{{'/'}}' with '{'/' }'
    # This is blunt but works for the observed output.
    if "{{" in clean or "}}" in clean:
        clean = clean.replace("{{", "{").replace("}}", "}")

    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        # Log a truncated version of the cleaned payload for debugging
        snippet = clean[:200].replace("\n", " ")
        logger.error(
            "Failed to parse JSON block after cleanup: %r. Error: %s", snippet, e
        )
        return None

def parse_reports_and_build_table(all_reports_text: str) -> tuple[str, list[dict]]:
    logging.info(
        "Parsing generated reports to build summary table using Python... total_length=%d",
        len(all_reports_text),
    )

    pattern = r"SCENARIO_DATA_BLOCK\s*(\{.*?})\s*END_SCENARIO_BLOCK"
    json_blocks = re.findall(pattern, all_reports_text, re.DOTALL)
    logging.info("Found %d SCENARIO_DATA_BLOCK segments.", len(json_blocks))

    if not json_blocks:
        snippet = all_reports_text[:500].replace("\n", " ")
        logging.warning(
            "No SCENARIO_DATA_BLOCK segments found in the generated reports. First 500 chars: %r",
            snippet,
        )
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

    parsed_scenarios: list[dict] = []

    for block in json_blocks:
        data = _parse_scenario_block(block)
        if not data:
            continue

        country_name = data.get("country") or data.get("country_name")
        iso3 = (data.get("iso3") or "").strip().upper()
        scenarios = data.get("scenarios") or []

        if not country_name or not iso3 or not isinstance(scenarios, list) or not scenarios:
            logging.warning(
                "Scenario block for %r has missing fields or empty 'scenarios': %r",
                country_name,
                data,
            )
            continue

        for scenario in scenarios:
            if not isinstance(scenario, dict):
                continue
            scenario_copy = dict(scenario)
            scenario_copy.setdefault("country", country_name)
            scenario_copy.setdefault("iso3", iso3)
            parsed_scenarios.append(scenario_copy)

    if not parsed_scenarios:
        logging.warning("JSON blocks were found, but no valid scenario data could be extracted.")
        fallback = (
            "| Country | Scenario Title | Hazard | Likely Month | Probability | PIN Best Guess | PA Best Guess |\n"
            "|---|---|---|---|---|---|---|\n"
            "| No data extracted | - | - | - | - | - | - |"
        )
        return fallback, []

    for scenario in parsed_scenarios:
        country_name = scenario.get("country") or scenario.get("country_name") or "Unknown"
        iso3 = (scenario.get("iso3") or "").strip().upper()
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


def _get_git_sha(default: str = "") -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)
        )
        return output.decode().strip()
    except Exception:
        return default


def _normalize_iso3_list(countries: list[str]) -> list[str]:
    return [c.strip().upper() for c in countries if c and c.strip()]


def _extract_sources(report_text: str) -> list[str]:
    lines = report_text.splitlines()
    sources: list[str] = []
    collecting = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^#+\s*Sources", stripped, flags=re.IGNORECASE):
            collecting = True
            continue
        if collecting:
            if not stripped:
                if sources:
                    break
                continue
            if stripped.startswith("-"):
                sources.append(stripped.lstrip("- ").strip())
            elif sources:
                sources[-1] = f"{sources[-1]} {stripped}".strip()
    return sources


def _build_country_reports(
    countries: list[str],
    report_records: list[tuple[str, str | None]],
    scenarios: list[dict],
) -> dict[str, dict]:
    name_to_iso3: dict[str, str] = {}
    for scenario in scenarios:
        iso3 = (scenario.get("iso3") or "").upper()
        country_name = (scenario.get("country_name") or "").strip().lower()
        if iso3 and country_name and country_name not in name_to_iso3:
            name_to_iso3[country_name] = iso3

    reports_payload: dict[str, dict] = {}
    for country, report_text in report_records:
        if not report_text:
            continue

        iso3 = name_to_iso3.get(country.strip().lower())
        if not iso3 and len(country.strip()) == 3:
            iso3 = country.strip().upper()
        iso3 = iso3 or country.strip().upper()

        if not iso3:
            logging.warning("Unable to determine ISO3 for country report: %s", country)
            continue

        reports_payload[iso3] = {
            "markdown": report_text,
            "sources": _extract_sources(report_text),
        }

    # Fallback: if no reports were matched, try using scenario ISO3s
    if not reports_payload:
        for scenario in scenarios:
            iso3 = (scenario.get("iso3") or "").upper()
            if not iso3 or iso3 in reports_payload:
                continue
            reports_payload[iso3] = {
                "markdown": "",
                "sources": [],
            }

    return reports_payload

def main(countries: list[str] | None = None):
    """Main function to run the bot.

    If `countries` is None or empty, fall back to hs_country_list.txt.
    Otherwise, use the provided list verbatim (treated as country names).
    """
    logging.info("Starting Pythia Horizon Scanner...")
    start_time = datetime.utcnow()
    hs_run_id = f"hs_{start_time.strftime('%Y%m%dT%H%M%S')}"
    os.environ["PYTHIA_HS_RUN_ID"] = hs_run_id

    try:
        ensure_schema()
        logging.info("DuckDB schema ensured via ensure_schema().")
    except Exception:
        logging.exception("Failed to ensure DuckDB schema; aborting Horizon Scanner run.")
        return

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

    iso3_list = _normalize_iso3_list(countries)
    git_sha = _get_git_sha()
    config_profile = f"{GEMINI_MODEL_NAME}@temp{generation_config.get('temperature', 0.0):.2f}"
    llm_profile = get_current_profile()
    is_test_mode = llm_profile.strip().lower() == "test" or os.getenv("PYTHIA_TEST_MODE") == "1"
    if is_test_mode:
        config_profile = f"{config_profile}::test"

    try:
        log_hs_run_to_db(
            hs_run_id=hs_run_id,
            iso3_list=iso3_list,
            git_sha=git_sha,
            config_profile=config_profile,
        )
    except Exception:
        logging.exception("Failed to log Horizon Scanner run metadata to DuckDB.")

    individual_reports: list[str] = []
    report_records: list[tuple[str, str | None]] = []

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
            report_records.append((country, report))
        else:
            logging.warning(
                "No report captured for %s (%d/%d).",
                country,
                idx,
                len(countries),
            )
            report_records.append((country, None))

        # Add a delay between API calls to respect rate limits and avoid overload
        time.sleep(5)

    logging.info("All country reports have been generated.")

    # Combine all individual reports into a single string
    all_reports_text = "\n\n---\n\n".join(individual_reports)

    # --- New Step: Parse reports and build table using Python ---
    summary_table, scenarios = parse_reports_and_build_table(all_reports_text)

    # --- Persist structured payloads to DuckDB ---
    upsert_hs_payload(
        hs_run_id,
        scenarios,
        today=date.today(),
        horizon_months=6,
        is_test_mode=is_test_mode,
    )

    country_reports_payload = _build_country_reports(countries, report_records, scenarios)
    try:
        log_hs_country_reports_to_db(hs_run_id, country_reports_payload)
    except Exception:
        logging.exception("Failed to log HS country reports to DuckDB.")

    # --- Diagnostics: list questions written to DuckDB for this run ---
    questions_md = ""
    conn = None
    try:
        conn = connect()
        ensure_schema(conn)
        query = """
            SELECT
                question_id,
                iso3,
                hazard_code,
                metric,
                target_month,
                wording
            FROM questions
            WHERE hs_run_id = ?
            ORDER BY iso3, hazard_code, metric, target_month, question_id
        """
        question_rows = conn.execute(query, [hs_run_id]).fetchall()

        if question_rows:
            logging.info(
                "Questions written to DuckDB for run %s (total %d):",
                hs_run_id,
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
                hs_run_id,
            )
            questions_md = (
                "## Questions Written to DuckDB\n\n"
                f"No questions found for run_id `{hs_run_id}`. "
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
        try:
            if conn:
                conn.close()
        except Exception:
            pass

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

