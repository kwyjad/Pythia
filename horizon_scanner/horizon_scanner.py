# -*- coding: utf-8 -*-
"""
Main script for the Pythia Horizon Scanner.
This script reads a list of countries, generates a risk analysis for each using
the Gemini API, and then compiles a final report with a summary table.
"""

import os
import time
import logging
import google.generativeai as genai
import backoff
import json
import re
from datetime import datetime
from hs_prompt import COUNTRY_ANALYSIS_PROMPT

# --- Configuration ---
# Set up basic logging to see progress in the GitHub Actions console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the Gemini API key from GitHub Secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY secret not found. Please add it to your repository secrets.")

genai.configure(api_key=GEMINI_API_KEY)

# Configuration for the Gemini models
# Use a safety setting to be less restrictive, as the content is professional analysis
generation_config = {
    "temperature": 0.2,
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
    model_name="gemini-2.5-pro", 
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
    """
    Finds all JSON data blocks in the reports, parses them, and builds a Markdown table.
    This function is 100% reliable and replaces the second LLM call.
    """
    logging.info("Parsing generated reports to build summary table using Python...")
    all_scenarios = []
    
    # **BUG FIX 1**: This regex now correctly finds the JSON block.
    json_blocks = re.findall(r'<!-- SCENARIO_DATA_BLOCK: (.*?) -->', all_reports_text, re.DOTALL)

    if not json_blocks:
        logging.warning("No SCENARIO_DATA_BLOCKs found in the generated reports.")
        return "| Country | Scenario Name | Probability | People Affected |\n|---|---|---|---|\n| No data found | - | - | - |"

    for block in json_blocks:
        try:
            # Clean and parse the JSON data
            data = json.loads(block)
            country_name = data.get("country", "Unknown")
            for scenario in data.get("scenarios", []):
                all_scenarios.append({
                    "Country": country_name,
                    "Scenario Name": scenario.get("name", "N/A"),
                    "Probability": scenario.get("probability", "N/A"),
                    "People Affected": scenario.get("affected", "N/A")
                })
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON block: {block}. Error: {e}")
            continue

    if not all_scenarios:
        logging.warning("JSON blocks were found, but no valid scenario data could be extracted.")
        return "| Country | Scenario Name | Probability | People Affected |\n|---|---|---|---|\n| No data extracted | - | - | - |"

    # Sort the scenarios alphabetically by country name
    all_scenarios.sort(key=lambda x: x["Country"])

    # Build the Markdown table string
    table_header = "| Country | Scenario Name | Probability | People Affected |\n|---|---|---|---|\n"
    table_rows = [f"| {s['Country']} | {s['Scenario Name']} | {s['Probability']} | {s['People Affected']} |" for s in all_scenarios]
    
    return table_header + "\n".join(table_rows)

def main():
    """Main function to run the bot."""
    logging.info("Starting Pythia Horizon Scanner...")
    
    try:
        logging.info("Reading country list from hs_country_list.txt...")
        with open("hs_country_list.txt", "r", encoding="utf-8") as f:
            # **BUG FIX 2**: This now ignores blank lines and lines starting with #
            countries = [line.strip() for line in f if line.strip() and not line.startswith('#')]
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
    summary_table = parse_reports_and_build_table(all_reports_text)

    # --- Final Report Assembly ---
    utc_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    final_report_content = (
        f"# Pythia Horizon Scan Report\n\n"
        f"**Generated on:** {utc_time}\n\n"
        f"## Summary Table\n\n{summary_table}\n\n"
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

