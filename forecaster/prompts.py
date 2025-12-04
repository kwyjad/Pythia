# ANCHOR: prompts (paste whole file)
from __future__ import annotations
import importlib
import os
from datetime import date
from typing import Any, Dict, Optional
import json
from .config import CALIBRATION_PATH, ist_date


def _json_dumps_for_prompt(obj: Any, **kwargs: Any) -> str:
    """
    JSON-encode helper for prompts that tolerates Python objects like date
    by stringifying unknown types via default=str.
    """

    return json.dumps(obj, default=str, **kwargs)

_PYTHIA_CFG_LOAD = None
if importlib.util.find_spec("pythia.config") is not None:
    _PYTHIA_CFG_LOAD = getattr(importlib.import_module("pythia.config"), "load", None)


def _pythia_db_url_from_config() -> Optional[str]:
    try:
        if _PYTHIA_CFG_LOAD is None:
            return None
        cfg = _PYTHIA_CFG_LOAD()
        app_cfg = cfg.get("app", {}) if isinstance(cfg, dict) else {}
        db_url = str(app_cfg.get("db_url", "")).strip()
        return db_url or None
    except Exception:
        return None

def _load_calibration_note() -> str:
    """
    Pull the latest calibration guidance from DuckDB (calibration_advice).
    Returns "" if nothing readable is found, so prompts stay valid.
    """
    txt = ""
    try:
        from resolver.db import duckdb_io

        db_url = _pythia_db_url_from_config() or os.getenv("RESOLVER_DB_URL", "").strip()
        db_url = db_url or duckdb_io.DEFAULT_DB_URL
        con = duckdb_io.get_db(db_url)
        try:
            row = con.execute(
                """
                SELECT advice
                FROM calibration_advice
                ORDER BY as_of_month DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            duckdb_io.close_db(con)
        if row and row[0]:
            txt = str(row[0])
    except Exception:
        txt = ""

    if not txt and CALIBRATION_PATH:
        try:
            with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
                txt = f.read().strip()
        except Exception:
            txt = ""

    if not txt:
        return ""
    return txt if len(txt) <= 4000 else (txt[:3800] + "\n…[truncated]")

_CAL_NOTE = _load_calibration_note()
_CAL_PREFIX = (
    "CALIBRATION GUIDANCE (auto-generated weekly):\n"
    + (_CAL_NOTE if _CAL_NOTE else "(none available yet)")
    + "\n— end calibration —\n\n"
)


def build_scoring_resolution_block(
    *,
    hazard_code: str,
    metric: str,
    resolution_source: Optional[str] = None,
) -> str:
    """
    Build a short, LLM-friendly 'SCORING & RESOLUTION' block.

    Uses hazard_code + metric + optional resolution_source to explain:
      - What "affected" means for this question.
      - Which source is used (EM-DAT, IDMC/DTM, ACLED).
      - That we use Brier scores on the SPD buckets.

    This is inserted early in the SPD prompt.
    """

    hz = (hazard_code or "").upper()
    m = (metric or "").upper()
    src = (resolution_source or "").upper()

    if m == "PA" and (hz in {"ACO", "ACE", "CU", "DI"} or "IDMC" in src or "DTM" in src):
        meaning = (
            "“affected” means people who are internally displaced (IDPs), "
            "as recorded by the Internal Displacement Monitoring Centre (IDMC), "
            "with IOM Displacement Tracking Matrix (DTM) as a fallback source."
        )
        source_label = "IDMC/DTM displacement"
    elif m == "FATALITIES" or "ACLED" in src:
        meaning = (
            "“affected” means battle-related fatalities, as recorded by ACLED "
            "(armed conflict event data)."
        )
        source_label = "ACLED conflict fatalities"
    else:
        meaning = (
            "“affected” means people affected by the hazard as EM-DAT defines "
            "and records them for this hazard and country."
        )
        source_label = "EM-DAT people affected"

    lines = []
    lines.append("SCORING & RESOLUTION")
    lines.append("")
    lines.append(
        "- All forecasts will be resolved using Resolver's canonical metric for this "
        "hazard and country, and scored using **Brier scores** on your SPD buckets."
    )
    lines.append("- For this question:")
    lines.append(f"  - {meaning}")
    lines.append(f"  - Source for resolution: {source_label}.")
    lines.append("")
    return "\n".join(lines)


def build_time_horizon_block(
    *,
    window_start_date: Optional[date],
    window_end_date: Optional[date],
    month_labels: Optional[Dict[int, str]] = None,
    hazard_code: str,
    metric: str,
    resolution_source: Optional[str] = None,
) -> str:
    """
    Build a 'TIME HORIZON & RESOLUTION' block explaining:
      - The calendar window (start/end dates),
      - How month_1..month_6 map to calendar months,
      - That we resolve per-month, not just over the entire window.

    month_labels: optional mapping {1: "December 2025", ..., 6: "May 2026"}.
    """

    _ = hazard_code, metric, resolution_source  # quiet unused-parameter linting

    src = (resolution_source or "").upper()
    hz = (hazard_code or "").upper()
    m = (metric or "").upper()
    if "ACLED" in src or m == "FATALITIES":
        source_label = "ACLED"
    elif m == "PA" and (hz in {"ACO", "ACE", "CU", "DI"} or "IDMC" in src or "DTM" in src):
        source_label = "IDMC/DTM"
    else:
        source_label = "EM-DAT"

    ws = window_start_date.isoformat() if window_start_date else ""
    we = window_end_date.isoformat() if window_end_date else ""

    lines: list[str] = []
    lines.append("TIME HORIZON & RESOLUTION")
    lines.append("")
    if ws or we:
        lines.append(f"- This question covers the period from **{ws}** to **{we}**.")
    else:
        lines.append("- This question covers a six-month period (month_1 to month_6).")

    lines.append("- For scoring, we treat each month separately:")
    if month_labels:
        for idx in range(1, 7):
            label = month_labels.get(idx) or f"month_{idx}"
            lines.append(f"  - `month_{idx}` = {label}")
    else:
        lines.append("  - `month_1` = first calendar month in the forecast window.")
        lines.append("  - `month_2` = second calendar month in the forecast window.")
        lines.append("  - …")
        lines.append("  - `month_6` = sixth calendar month in the forecast window.")

    lines.append(
        "- For each month `m`, Resolver will compute a single monthly value for the "
        f"relevant metric from the underlying source ({source_label}), and "
        "your SPD for that month will be scored against that monthly value."
    )
    lines.append("")
    return "\n".join(lines)

# -------------------------------------------------------------------------------------
# FULL PROMPTS
# -------------------------------------------------------------------------------------

BINARY_PROMPT = _CAL_PREFIX + """
You are a careful probabilistic forecaster. Use the background context AND the research report AND your general knowlodge as an LLM.
Your task is to assign a probability (0–100%) to whether the binary event will occur, using Bayesian reasoning.

Follow these steps in your reasoning before giving the final probability:

1. **Base Rate (Prior) Selection**
   - Identify an appropriate base rate (prior probability P(H)) for the event.
   - Clearly explain why you chose this base rate (e.g., historical frequencies, reference class data, general statistics).
   - State the initial prior in probability or odds form.

2. **Comparison to Base Case**
   - Explain how the current situation is similar to the reference base case.
   - Explain how it is different, and why those differences matter for adjusting the probability.

3. **Evidence Evaluation (Likelihoods)**
   - For each key piece of evidence, consider how likely it would be if the event happens (P(E | H)) versus if it does not happen (P(E | ~H)).
   - Compute or qualitatively describe the likelihood ratio (P(E | H) / P(E | ~H)).
   - State clearly whether each piece of evidence increases or decreases the probability.

4. **Bayesian Updating (Posterior Probability)**
   - Use Bayes’ Rule conceptually:
       Posterior odds = Prior odds × Likelihood ratio
       Posterior probability = (Posterior odds) / (1 + Posterior odds)
   - Walk through at least one explicit update step, showing how the prior probability is adjusted by evidence.
   - Summarize the resulting posterior probability and explain how confident or uncertain it remains.

5. **Red Team Thinking**
    - Critically evaluate your own forecast for overconfidence or blind spots.
    - Consider tail risks and alternative scenarios that might affect the distribution.
    - Think of the best alternative forecast and why it might be plausible, as well as rebuttals
    - Adjust your percentiles if necessary to account for these considerations.
    
5. **Final Forecast**
   - Provide the final forecast as a single calibrated probability.
   - Ensure it reflects both the base rate and the impact of the evidence.

6. **Output Format**
   - End with EXACTLY this line (no other commentary):
Final: ZZ%

Question: {title}

Background:
{background}

Research Report (recent/contextual):
{research}

Resolution criteria:
{criteria}

Today (Istanbul time): {today}
"""

NUMERIC_PROMPT = _CAL_PREFIX + """You are a careful probabilistic forecaster. Use the background context AND the research report AND your general knowlodge as an LLM.
Your task is to produce a full probabilistic forecast for a numeric quantity using Bayesian reasoning.

Follow these steps in your reasoning before giving the final percentiles:

1. **Base Rate (Prior) Selection**
   - Identify an appropriate base rate or reference distribution for the target variable.
   - Clearly explain why you chose this base rate (e.g., historical averages, statistical reference classes, domain-specific priors).
   - State the mean/median and variance (or spread) of this base rate.

2. **Comparison to Base Case**
   - Explain how the current situation is similar to the reference distribution.
   - Explain how it is different, and why those differences matter for shifting or stretching the distribution.

3. **Evidence Evaluation (Likelihoods)**
   - For each major piece of evidence in the background or research report, consider how consistent it is with higher vs. lower values.
   - Translate this into a likelihood ratio or qualitative directional adjustment (e.g., “this factor makes higher outcomes 2× as likely as lower outcomes”).
   - Make clear which evidence pushes the forecast up or down, and by how much.

4. **Bayesian Updating (Posterior Distribution)**
   - Use Bayes’ Rule conceptually:
       Posterior ∝ Prior × Likelihood
   - Walk through at least one explicit update step to show how evidence modifies your prior distribution.
   - Describe how the posterior mean, variance, or skew has shifted.

5. **Red Team Thinking**
    - Critically evaluate your own forecast for overconfidence or blind spots.
    - Consider tail risks and alternative scenarios that might affect the distribution.
    - Think of the best alternative forecast and why it might be plausible, as well as rebuttals
    - Adjust your percentiles if necessary to account for these considerations.

6. **Final Percentiles**
   - Provide calibrated percentiles that summarize your posterior distribution.
   - Ensure they are internally consistent (P10 < P20 < P40 < P60 < P80 < P90).
   - Think carefully about tail risks and avoid overconfidence.

7. **Output Format**
   - End with EXACTLY these 6 lines (no other commentary):
P10: X
P20: X
P40: X
P60: X
P80: X
P90: X

Question: {title}
Units: {units}

Background:
{background}

Research Report (recent/contextual):
{research}

Resolution:
{criteria}

Today (Istanbul time): {today}
"""

MCQ_PROMPT = _CAL_PREFIX + """You are a careful probabilistic forecaster. Use the background context AND the research report AND your general knowlodge as an LLM.
Your task is to assign probabilities to each of the multiple-choice options using Bayesian reasoning.
Follow these steps clearly in your reasoning before giving your final answer:

1. **Base Rate (Prior) Selection** - Identify an appropriate base rate (prior probability P(H)) for each option.  
   - Clearly explain why you chose this base rate (e.g., historical frequencies, general statistics, or a reference class).  

2. **Comparison to Base Case** - Explain how the current case is similar to the base rate scenario.  
   - Explain how it is different, and why those differences matter.  

3. **Evidence Evaluation (Likelihoods)** - For each piece of evidence in the background or research report, consider how likely it would be if the option were true (P(E | H)) versus if it were not true (P(E | ~H)).  
   - State these likelihood assessments clearly, even if approximate or qualitative.  

4. **Bayesian Updating (Posterior)** - Use Bayes’ Rule conceptually:  
     Posterior odds = Prior odds × Likelihood ratio  
     Posterior probability = (Posterior odds) / (1 + Posterior odds)  
   - Walk through at least one explicit update step for key evidence, showing how the prior changes into a posterior.  
   - Explain qualitatively how other evidence shifts the probabilities up or down.  

5. **Red Team Thinking**
    - Critically evaluate your own forecast for overconfidence or blind spots.
    - Consider tail risks and alternative scenarios that might affect the distribution.
    - Think of the best alternative forecast and why it might be plausible, as well as rebuttals
    - Adjust your percentiles if necessary to account for these considerations.

6. **Final Normalization** - Ensure the probabilities across all options are consistent and sum to approximately 100%.  
   - Check calibration: if uncertain, distribute probability mass proportionally.  

7. **Output Format** - After reasoning, provide your final forecast as probabilities for each option.  
   - Use EXACTLY N lines, one per option, formatted as:  

Option_1: XX%  
Option_2: XX%  
Option_3: XX%  
...  
(sum ~100%)  

Question: {title}
Options: {options}

Background:
{background}

Research Report (recent/contextual):
{research}

Resolution criteria:
{criteria}

Today (Istanbul time): {today}
"""

SPD_PROMPT_TEMPLATE = _CAL_PREFIX + """
{scoring_block}
{time_horizon_block}
You are a careful probabilistic forecaster on a humanitarian early warning panel.

Your task is to forecast {quantity_description}.

You will express your beliefs as a SUBJECTIVE PROBABILITY DISTRIBUTION (SPD) over FIVE buckets
for each month.

SPD (Subjective Probability Distribution) means:
- You approximate your posterior belief about the monthly value using a small number of discrete buckets.
- Your probabilities over the buckets should reflect the relative plausibility of each range after
  considering the historical base rate and the evidence in the research bundle.

For each month, distribute 100% probability across these buckets:

{bucket_text}

One of these buckets MUST occur for each month. For each month m, your probabilities
[p1, p2, p3, p4, p5] must all be between 0 and 1 and sum to approximately 1.0.

Question:
{question}

Background:
{background}

Research bundle (recent/contextual information):
{research}

Resolution criteria (how this metric will be counted):
{resolution_text}

Today (Istanbul time): {today}

---

FORECASTING INSTRUCTIONS (Bayesian SPD)

1) Prior / base rates
   - Start from a prior SPD over the buckets based on historical data and relevant reference classes
     (for this hazard and country).
   - Make your prior explicit in your own thinking: which bucket would you expect *before* reading the evidence?

2) Evidence & likelihood
   - Use the research bundle and history to identify the most important pieces of evidence.
   - For each bucket, ask: “If the true value were in this bucket, how likely is this evidence?”
   - Note which buckets the evidence pushes up or down.

3) Posterior SPD sketch
   - Combine your prior and the evidence qualitatively to sketch a posterior SPD for each month.
   - Check that your SPD:
     - is not implausibly sharp (overconfident), and
     - is not completely flat (ignoring structure).

4) Red-team your forecast
   - Challenge your own forecast:
     - What scenarios might you be underweighting (e.g. rare breakdown of state control, extreme hazard)?
     - Are you systematically underweighting tail risks?
   - Adjust your SPD if needed to reflect realistic but low-probability extreme scenarios.

5) Final JSON output (IMPORTANT)
   - At the very end, output ONLY a single JSON object with this exact schema:

   {{
     "month_1": [p1, p2, p3, p4, p5],
     "month_2": [p1, p2, p3, p4, p5],
     "month_3": [p1, p2, p3, p4, p5],
     "month_4": [p1, p2, p3, p4, p5],
     "month_5": [p1, p2, p3, p4, p5],
     "month_6": [p1, p2, p3, p4, p5]
   }}

   - Do not include any text before or after the JSON.
   - Each list must contain exactly five numbers between 0 and 1 inclusive.
   - For each month, the probabilities must sum to roughly 1.0 (we allow small rounding error).
"""

SPD_BUCKET_TEXT_PA = """
People affected (PA) buckets (per month, country-level):
- Bucket 1: < 10,000 people affected (label: "<10k")
- Bucket 2: 10,000 to < 50,000 people affected (label: "10k-<50k")
- Bucket 3: 50,000 to < 250,000 people affected (label: "50k-<250k")
- Bucket 4: 250,000 to < 500,000 people affected (label: "250k-<500k")
- Bucket 5: >= 500,000 people affected (label: ">=500k")
"""

SPD_BUCKET_TEXT_FATALITIES = """
Conflict fatalities buckets (per month, country-level):
- Bucket 1: 0–4 deaths (label: "<5")
- Bucket 2: 5–24 deaths (label: "5-<25")
- Bucket 3: 25–99 deaths (label: "25-<100")
- Bucket 4: 100–499 deaths (label: "100-<500")
- Bucket 5: >= 500 deaths (label: ">=500")
"""

RESEARCHER_PROMPT = """You are a professional RESEARCHER for a Bayesian forecasting panel.
Your job is to produce a concise, decision-useful research brief that helps a statistician
update a prior. The forecasters will combine your brief with a statistical aggregator that
expects: base rates (reference class), recency-weighted evidence (relative to horizon),
key mechanisms, differences vs. the base rate, and indicators to watch. Provide a carefully reasoned, deeply through out research brief. Before answering, lay out for yoursefl your research plan step-by-step. First, identify the core questions to investigate. Second, for each question, propose the search queries you would use. Third, after gathering information, synthesize the key findings. Finally, draft the comprehensive answer."

QUESTION
Title: {title}
Type: {qtype}
Units/Options: {units_or_options}

BACKGROUND
{background}

RESOLUTION CRITERIA (what counts as “true”/resolution)
{criteria}

HORIZON & RECENCY
Today (Istanbul): {today}
Guideline: define “recent” relative to time-to-resolution:
- if >12 months to resolution: emphasize last 24 months
- if 3–12 months: emphasize last 12 months
- if <3 months: emphasize last 6 months

SOURCES (optional; may be empty)
Use these snippets primarily if present; if not present, rely on general knowledge.
Do NOT fabricate precise citations; if unsure, say “uncertain”.
{sources}

=== REQUIRED OUTPUT FORMAT (use headings exactly as written) ===
### Reference class & base rates
- Identify 1–3 plausible reference classes; give ballpark base rates or ranges and reasoning and on how these were derived; note limitations.

### Recent developments (timeline bullets)
- [YYYY-MM-DD] item — direction (↑/↓ for event effect on YES) — why it matters (≤25 words)
- Focus on events within the recency guideline above. Use grounding web search as needed. 

### Mechanisms & drivers (causal levers)
- List 3–6 drivers that move probability up/down; note typical size (small/moderate/large).

### Differences vs. the base rate (what’s unusual now)
- 3–6 bullets contrasting this case with the reference class (structure, actors, constraints, policy).

### Bayesian update sketch (for the statistician)
- Prior: brief sentence suggesting a plausible prior and “equivalent n” (strength).
- Evidence mapping: 3–6 bullets with sign (↑/↓) and rough magnitude (small/moderate/large).
- Net effect: one line describing whether the posterior should move up/down and by how much qualitatively.

### Indicators to watch (leading signals; next weeks/months)
- UP indicators: 3–5 short bullets.
- DOWN indicators: 3–5 short bullets.

### Caveats & pitfalls
- 3–5 bullets on uncertainty, data gaps, deception risks, regime changes, definitional gotchas.

Final Research Summary: One or two sentences for the forecaster. Keep the entire brief under ~3000 words.
"""

# -------------------------------------------------------------------------------------
# BUILDERS
# -------------------------------------------------------------------------------------

def build_binary_prompt(title: str, background: str, research_text: str, criteria: str) -> str:
    return BINARY_PROMPT.format(
        title=title,
        background=(background or "N/A"),
        research=(research_text or "N/A"),
        criteria=(criteria or "N/A"),
        today=ist_date(),
    )

def build_numeric_prompt(title: str, units: str, background: str, research_text: str, criteria: str) -> str:
    return NUMERIC_PROMPT.format(
        title=title,
        units=(units or "N/A"),
        background=(background or "N/A"),
        research=(research_text or "N/A"),
        criteria=(criteria or "N/A"),
        today=ist_date(),
    )

def build_mcq_prompt(title: str, options: list[str], background: str, research_text: str, criteria: str) -> str:
    return MCQ_PROMPT.format(
        title=title,
        options="\n".join([str(o) for o in (options or [])]) or "N/A",
        background=(background or "N/A"),
        research=(research_text or "N/A"),
        criteria=(criteria or "N/A"),
        today=ist_date(),
    )

def _format_resolution_text(base_text: str, criteria: str) -> str:
    extra = (criteria or "N/A").strip() or "N/A"
    return f"{base_text}\nAdditional resolution notes: {extra}"


def build_resolution_text_and_quantity_description(
    *,
    iso3: str,
    hazard_code: str,
    hazard_label: str,
    metric: str,
    resolution_source: Optional[str],
) -> tuple[str, str]:
    hz = (hazard_code or "").upper()
    m = (metric or "").upper()
    src = (resolution_source or "").upper()

    if m == "FATALITIES" or "ACLED" in src:
        resolution_text = (
            "Fatalities will be measured as battle-related deaths recorded by ACLED "
            "for this country and hazard code."
        )
        quantity_description = (
            f"Monthly battle-related fatalities in {iso3} associated with {hazard_label} "
            "events, as recorded by ACLED."
        )
        return resolution_text, quantity_description

    if m == "PA" and (hz in {"ACO", "ACE", "CU", "DI"} or "IDMC" in src or "DTM" in src):
        resolution_text = (
            "People affected (PA) will be measured as internally displaced people (IDPs), "
            "using IDMC displacement estimates for this hazard and country, with IOM DTM "
            "or comparable humanitarian estimates as fallback."
        )
        quantity_description = (
            f"Monthly internally displaced people (IDPs) in {iso3} due to {hazard_label.lower()}, "
            "as recorded by IDMC and DTM."
        )
        return resolution_text, quantity_description

    resolution_text = (
        "People affected (PA) will be measured using Resolver's canonical PA metric for "
        "this natural hazard and country, based primarily on EM-DAT data."
    )
    quantity_description = (
        f"Monthly people affected (PA) in {iso3} by {hazard_label} as recorded by EM-DAT "
        "(including both directly and indirectly affected people)."
    )
    return resolution_text, quantity_description


def build_spd_prompt_pa(
    *,
    question_title: str,
    iso3: str,
    hazard_code: str,
    hazard_label: str,
    metric: str,
    background: str,
    research_text: str,
    resolution_source: Optional[str],
    window_start_date: Optional[date],
    window_end_date: Optional[date],
    month_labels: Optional[Dict[int, str]],
    today: date,
    criteria: str,
) -> str:
    resolution_text_base, quantity_description = build_resolution_text_and_quantity_description(
        iso3=iso3,
        hazard_code=hazard_code,
        hazard_label=hazard_label,
        metric=metric,
        resolution_source=resolution_source,
    )
    resolution_text = _format_resolution_text(resolution_text_base, criteria)

    scoring_block = build_scoring_resolution_block(
        hazard_code=hazard_code,
        metric=metric,
        resolution_source=resolution_source,
    )
    time_horizon_block = build_time_horizon_block(
        window_start_date=window_start_date,
        window_end_date=window_end_date,
        month_labels=month_labels,
        hazard_code=hazard_code,
        metric=metric,
        resolution_source=resolution_source,
    )

    today_str = today.isoformat() if isinstance(today, date) else ist_date()

    return SPD_PROMPT_TEMPLATE.format(
        scoring_block=scoring_block,
        time_horizon_block=time_horizon_block,
        question=question_title,
        background=background or "",
        research=research_text or "",
        resolution_text=resolution_text,
        quantity_description=quantity_description,
        bucket_text=SPD_BUCKET_TEXT_PA,
        today=today_str,
    )


def build_spd_prompt_fatalities(
    *,
    question_title: str,
    iso3: str,
    hazard_code: str,
    hazard_label: str,
    metric: str,
    background: str,
    research_text: str,
    resolution_source: Optional[str],
    window_start_date: Optional[date],
    window_end_date: Optional[date],
    month_labels: Optional[Dict[int, str]],
    today: date,
    criteria: str,
) -> str:
    resolution_text_base, quantity_description = build_resolution_text_and_quantity_description(
        iso3=iso3,
        hazard_code=hazard_code,
        hazard_label=hazard_label,
        metric=metric,
        resolution_source=resolution_source,
    )
    resolution_text = _format_resolution_text(resolution_text_base, criteria)

    scoring_block = build_scoring_resolution_block(
        hazard_code=hazard_code,
        metric=metric,
        resolution_source=resolution_source,
    )
    time_horizon_block = build_time_horizon_block(
        window_start_date=window_start_date,
        window_end_date=window_end_date,
        month_labels=month_labels,
        hazard_code=hazard_code,
        metric=metric,
        resolution_source=resolution_source,
    )

    today_str = today.isoformat() if isinstance(today, date) else ist_date()

    return SPD_PROMPT_TEMPLATE.format(
        scoring_block=scoring_block,
        time_horizon_block=time_horizon_block,
        question=question_title,
        background=background or "",
        research=research_text or "",
        resolution_text=resolution_text,
        quantity_description=quantity_description,
        bucket_text=SPD_BUCKET_TEXT_FATALITIES,
        today=today_str,
    )


def build_spd_prompt(
    *,
    question_title: str,
    background: str,
    research_text: str,
    criteria: str,
    iso3: str = "",
    hazard_code: str = "",
    hazard_label: str = "",
    metric: str = "PA",
    resolution_source: Optional[str] = None,
    window_start_date: Optional[date] = None,
    window_end_date: Optional[date] = None,
    month_labels: Optional[Dict[int, str]] = None,
    today: Optional[date] = None,
) -> str:
    return build_spd_prompt_pa(
        question_title=question_title,
        iso3=iso3,
        hazard_code=hazard_code,
        hazard_label=hazard_label or hazard_code,
        metric=metric,
        background=background,
        research_text=research_text,
        resolution_source=resolution_source,
        window_start_date=window_start_date,
        window_end_date=window_end_date,
        month_labels=month_labels,
        today=today or date.today(),
        criteria=criteria,
    )

def build_research_prompt(
    title: str,
    qtype: str,
    units_or_options: str,
    background: str,
    criteria: str,
    today: str,
    sources_text: str,
) -> str:
    sources_text = sources_text.strip() if sources_text else "No external sources provided."
    return RESEARCHER_PROMPT.format(
        title=title,
        qtype=qtype,
        units_or_options=units_or_options or "N/A",
        background=(background or "N/A"),
        criteria=(criteria or "N/A"),
        today=today,
        sources=sources_text,
    )


def build_research_prompt_v2(
    question: Dict[str, Any],
    hs_triage_entry: Dict[str, Any],
    resolver_features: Dict[str, Any],
    model_info: Dict[str, Any] | None = None,
) -> str:
    """Structured research prompt for Researcher v2."""

    iso3 = question.get("iso3", "")
    hazard = question.get("hazard_code", "")
    metric = question.get("metric", "")
    resolution_source = question.get("resolution_source", "")
    model_info = model_info or {}

    return f"""You are a humanitarian risk analyst.\nYour task is to prepare machine-focused research for the forecaster.\n\nQuestion:\n- Country: {iso3}\n- Hazard: {hazard}\n- Metric: {metric}\n- Resolution dataset: {resolution_source}\n\nResolver history (noisy, incomplete base-rate data):\n```json\n{_json_dumps_for_prompt(resolver_features, indent=2)}\n```\n\nHS triage (tier, triage_score, drivers, regime_shifts, data_quality):\n\n```json\n{_json_dumps_for_prompt(hs_triage_entry, indent=2)}\n```\n\nModel/data notes:\n```json\n{_json_dumps_for_prompt(model_info, indent=2)}\n```\n\nUse Resolver as one imperfect signal. ACLED is generally strong for conflict fatalities; IDMC has short history for displacement; EM-DAT is patchy; DTM is contextual only.\n\nYour tasks:\n\n1. Summarise the base rate of {metric} for this hazard/country using:\n   * Resolver history (with its caveats),\n   * high-quality external analytical sources (UN, ACAPS, etc.),\n   * your own knowledge of the country context.\n2. Identify key update signals for the next 6 months that would push risk up or down.\n3. Identify specific regime-shift mechanisms that could make the next 6–12 months differ markedly from the past.\n4. Note important data gaps and uncertainties.\n\nReturn a single JSON object:\n\n```json\n{{\n  \"base_rate\": {{\n    \"qualitative_summary\": \"...\",\n    \"resolver_support\": {{\n      \"recent_level\": \"low|medium|high\",\n      \"trend\": \"up|down|flat|uncertain\",\n      \"data_quality\": \"low|medium|high\",\n      \"notes\": \"...\"\n    }},\n    \"external_support\": {{\n      \"consensus\": \"increasing|decreasing|mixed|uncertain\",\n      \"data_quality\": \"low|medium|high\",\n      \"recent_analyses\": [\"...\"]\n    }}\n  }},\n  \"update_signals\": [\n    {{\"description\": \"...\", \"direction\": \"up|down|unclear\", \"confidence\": 0.7, \"timeframe_months\": 6}}\n  ],\n  \"regime_shift_signals\": [\n    {{\"description\": \"...\", \"likelihood\": \"low|medium|high\", \"timeframe_months\": 3}}\n  ],\n  \"data_gaps\": [\"...\"]\n}}\n```\n\nDo not include any text outside the JSON.\n"""


def build_spd_prompt_v2(
    question: Dict[str, Any],
    history_summary: Dict[str, Any],
    hs_triage_entry: Dict[str, Any],
    research_json: Dict[str, Any],
) -> str:
    """Assemble the SPD v2 forecasting prompt with structured context."""

    base_rate_note = ""
    if (history_summary.get("source") or "").lower() == "none":
        base_rate_note = (
            "- Resolver does not currently provide a base-rate series for this hazard; treat the base rate as unknown and lean on HS triage + research.\n"
        )

    return (
        "You are a probabilistic forecaster. Produce a six-month SPD for the question below.\n\n"
        "Question metadata:\n"
        "```json\n"
        f"{_json_dumps_for_prompt(question, indent=2)}\n"
        "```\n\n"
        "Resolver history summary (Resolver is one imperfect source; ACLED strong, IDMC short, EM-DAT patchy):\n"
        "```json\n"
        f"{_json_dumps_for_prompt(history_summary, indent=2)}\n"
        "```\n\n"
        "HS triage output:\n"
        "```json\n"
        f"{_json_dumps_for_prompt(hs_triage_entry, indent=2)}\n"
        "```\n\n"
        "Research evidence:\n"
        "```json\n"
        f"{_json_dumps_for_prompt(research_json, indent=2)}\n"
        "```\n\n"
        "Instructions:\n"
        "- Provide a five-bucket SPD for each of the next 6 months using the bucket labels in the question metadata.\n"
        "- Use the history summary as one signal; do not over-trust it if short or missing.\n"
        f"{base_rate_note}"
        "- If Resolver history is missing (`source` = \"none\"), rely on HS + research to shape the base rate.\n"
        "- Return a single JSON object only.\n\n"
        "Return a JSON object with both probabilities and a short rationale:\n"
        "```json\n"
        "{\n"
        '  "spds": {\n'
        '    "YYYY-MM": {"buckets": ["<5","5-<25","25-<100","100-<500",">=500"], "probs": [0.7,0.2,0.07,0.02,0.01]},\n'
        '    "YYYY-MM+1": {"buckets": ["<5","5-<25","25-<100","100-<500",">=500"], "probs": [0.7,0.2,0.07,0.02,0.01]}\n'
        "  },\n"
        '  "human_explanation": "3–4 sentences explaining the drivers and uncertainties."\n'
        "}\n"
        "```\n"
        "Do not include any text outside the JSON.\n"
    )


def build_scenario_prompt(
    iso3: str,
    hazard_code: str,
    metric: str,
    spd_summary: Dict[str, Any],
    hs_triage_entry: Dict[str, Any],
    scenario_stub: str,
    forecaster_rationale: str,
) -> str:
    """Prompt the Scenario Writer to draft short scenarios from ensemble outputs."""

    scenario_text = scenario_stub or ""
    rationale_text = forecaster_rationale or ""

    return (
        f"You are a humanitarian scenario writer.\n\n"
        f"You are given a probabilistic forecast for {iso3}, hazard {hazard_code}, metric {metric}. "
        "The forecast is over 5 buckets and represents recorded impact in the resolution dataset "
        "(e.g. ACLED/EM-DAT/IDMC).\n\n"
        "Ensemble SPD summary:\n"
        "```json\n"
        f"{_json_dumps_for_prompt(spd_summary, indent=2)}\n"
        "```\n\n"
        "HS triage and drivers:\n"
        "```json\n"
        f"{_json_dumps_for_prompt(hs_triage_entry, indent=2)}\n"
        "```\n\n"
        "Optional HS scenario stub:\n"
        f"\"\"\"{scenario_text}\"\"\"\n\n"
        "Forecaster rationale:\n"
        f"\"\"\"{rationale_text}\"\"\"\n\n"
        "Your tasks:\n\n"
        "1. Write a short scenario (3–4 sentences) assuming the forecast's most likely bucket (bucket_max) occurs.\n"
        "2. If there is a different bucket with probability >= 0.05 (bucket_alt), write a second short "
        "scenario (3–4 sentences) assuming that alternative higher-impact bucket occurs.\n"
        "3. If there is no such alternative bucket, only write the primary scenario.\n\n"
        "Each scenario should:\n\n"
        "* Describe the situation and context vs. today.\n"
        "* Highlight key humanitarian needs by sector (high level; no numbers).\n"
        "* Highlight operational challenges for humanitarian actors (access, funding, politics).\n\n"
        "Return a JSON object:\n\n"
        "```json\n"
        "{\n"
        '  "primary": {\n'
        '    "bucket_label": "...",\n'
        '    "probability": 0.0,\n'
        '    "text": "..."\n'
        "  },\n"
        '  "alternative": null\n'
        "}\n"
        "```\n\n"
        "No extra keys. No text outside the JSON.\n"
    )
