# -*- coding: utf-8 -*-
"""
This file stores the prompt for the Gemini API calls.
Keeping it here makes the main script cleaner and easier to read.
"""

# Prompt for generating the individual country risk analysis.
# The placeholder {country} will be replaced by the script.
SCENARIO_DATA_INSTRUCTIONS = """
At the END of your answer, output a machine-readable JSON block for each country
using the following exact format. For each country, include ONE block:

SCENARIO_DATA_BLOCK
{
  "country": "<COUNTRY NAME>",
  "iso3": "<ISO3 CODE>",
  "scenarios": [
    {
      "title": "Short descriptive title of the main risk scenario",
      "hazard_label": "Short label for the hazard (e.g. 'Conflict', 'Flood', 'Drought')",
      "hazard_code": "STANDARD_HAZARD_CODE",  # e.g. 'FLOOD', 'DROUGHT', 'CONFLICT'
      "probability": "NN",                    # integer percent as a string, e.g. "60"
      "likely_window_month": "YYYY-MM",       # the target month of impact
      "best_guess": {
        "PIN": 12345,                         # expected people in need (integer)
        "PA": 6789                            # expected people affected (integer)
      },
      "markdown": "Short narrative describing the scenario, impact, and context."
    },
    ...
  ]
}
END_SCENARIO_BLOCK

IMPORTANT:
- Use **standard JSON** between SCENARIO_DATA_BLOCK and END_SCENARIO_BLOCK.
- Do NOT use any doubled braces like `{{` or `}}`.
- Do NOT wrap the JSON in code fences (no ```).
- The JSON must be parseable by a strict JSON parser.
- Use exactly the marker lines 'SCENARIO_DATA_BLOCK' and 'END_SCENARIO_BLOCK' as shown.
"""

COUNTRY_ANALYSIS_PROMPT = (
    """
You are an expert in contextual risk analysis working for UNICEF.
Your task is to develop a comprehensive political, armed conflict, and natural hazard risk analysis for the country in {country}, with a sole focus on shocks that could occur in the next six months.

---

## Instructions

### Country
{country}

### Scope of Analysis
- **You must search online for the latest information to inform your analysis.** Focus on credible, publicly available sources. Your goal is to find at least five high-quality sources.
- **Political Risk**: Focus on elections or legal/administrative changes that could cause civil unrest.
- **Natural Hazards**: Focus on cyclonic storms, floods, and drought.
  - Mention any relevant “seasons” in the next six months.
  - State whether current forecasts suggest an average, above average, or below average season.
- **Armed Conflict**: Focus on potential new inter- or intrastate armed conflicts, or possible escalation of ongoing conflicts.
- **Cross-Border Displcement Inflows: Focus on specfic situations in neighbouring countries that have potential to send an inflow of refugees, IDPs, or migrants into the target country.

### Judicious Use
- This is a standard template.
- If the country in question does not face any of these risks, do not fabricate information.

### Scenario Requirements
For any hazard or risk that you include, define a detailed scenario that specifies:
- The probability (0-100%) of the scenario occuring in the next 6 months, with rationale. Think carefully about the probability. Identify a reference class and base rate, and then apply bayesian reasoning. Red team your own analysis before producing the final output.
- The number of people and the number of children affected.
- The percentage of the population of the affected area that this represents.
- The sectors in which they could need assistance: Nutrition, WASH, Health, Protection, and Education, with description of the needs in each sector, with particular focus on children.
- Any operational challenges that could obstruct UNICEF in delivering assistance.

### Writing Style
- Use a professional tone, narrative form (not lists).
- Do not use words or sentence structures commonly associated with AI writing.
- Avoid “both-sidism.”
- Never use the em dash character.

### Detailed Output Structure
Present the final output as a single, fenced markdown code block that begins with ```markdown and ends with ```. The internal structure must follow these rules exactly:

1.  **Main Title (H1):** Start with a level 1 Markdown heading (`#`) formatted as: `**[Country Name]: Six-Month Risk Outlook ([Start Month] [Year] - [End Month] [Year])**`.
2.  **Introduction (H2):** Follow with a level 2 heading (`##`) titled `Introduction`. Write one narrative paragraph summarizing the key findings.
3.  **Risk Categories (H2):** For each risk area (Political, Armed Conflict, Natural Hazard, Cross-Border Displacement Influx),create a level 2 heading (`##`), e.g., `## Political Risk`. Beneath each heading, write a short narrative paragraph summarizingthe general situation for that risk.
4.  **Scenarios (H3):** Within each risk category, create one or more level 3 headings (`###`) for each distinct scenario. The heading must be formatted as: `### Scenario: [A short, descriptive title]`.
5.  **Scenario Details:** Beneath each scenario heading, provide:
    * A single narrative paragraph describing the scenario in extensive detail.
    * A bulleted list (`-`) with the following four items, using the exact bolded labels:
        * `- **Probability of Occurrence:**` [Provide percentage and detailed rationale that includes the reference class and based rate that you used, your Bayesian updating process, and your red team thinking]
        * `- **Affected Population:**` [Provide total people and children affected, with a detailed rationale]
        * `- **Percentage of Population in Affected Area:**` [Provide percentage and context.]
        * `- **Potential Sectors for Assistance:**` [List relevant sectors, with decription of the needs in each sector. Use a nested bulleted list if there are multiple.]
        * `- **Operational Challenges:**` [Describe any challenges that could obstruct UNICEF in delivering assistance.]
6.  **Sources (H2):** Conclude with a level 2 heading (`##`) titled `Sources Consulted`. **This section is mandatory.**
    * Your entire output for this section must be ONLY a bulleted list of the full, direct URLs of the primary sources used.
    * **The format for each source MUST be:** `- **[Website Name] - [Article Title (if available)]:** [Full, direct URL]`.
    * **CRITICAL RULE: You must not use placeholder text or add any commentary such as "Insert relevant URL here". If you cannot find a specific, valid source, omit that line item, but you must find and list at least five valid sources.**
7.  **Data Summary Block (Mandatory):** After the Sources section, you must include a machine-readable JSON summary of the scenarios using the exact marker lines shown below. Follow these instructions:
"""
    + SCENARIO_DATA_INSTRUCTIONS
    + """

    * `"hazard_code"` must be one of `[FL, DR, TC, HW, ACE, DI, CU, EC, PHE]`. **Do not** use `ACO`, `MULTI`, or `OT`.
      - Use `ACE` for any armed conflict hazard (onset or escalation). Do not use `ACO`.
    * `"likely_window_month"` must be the single most likely `YYYY-MM` within the next six months for the peak impact of the scenario.
    * `"best_guess"` must contain integer point estimates for affected people: `{"PIN": <persons>, "PA": <persons>}`.
    * Keep every scenario strictly single-hazard—never aggregate multiple hazards into one scenario.

"""
)
