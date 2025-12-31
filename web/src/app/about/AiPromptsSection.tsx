const linkClass =
  "text-sky-300 underline underline-offset-2 hover:text-sky-200";

const panelClass =
  "rounded-lg border border-slate-800 bg-slate-900/40 px-4 py-3 text-sm text-slate-300";

const preClass =
  "mt-3 whitespace-pre-wrap rounded-md bg-slate-950/50 p-4 text-xs text-slate-200 overflow-auto";

const sourceLink = (href: string, label = "Source") => (
  <a className={linkClass} href={href} rel="noreferrer" target="_blank">
    {label}
  </a>
);

export default function AiPromptsSection() {
  return (
    <section className="mt-10 space-y-4">
      <h2 className="text-2xl font-semibold text-white">AI Prompts</h2>
      <p className="text-sm text-slate-300">
        These are short, readable excerpts from the real prompts used in the
        pipeline. Each panel links to the source file in GitHub so you can inspect
        the exact code.
      </p>

      <details className={panelClass}>
        <summary className="cursor-pointer text-slate-100 font-medium">
          Web search / evidence pack
        </summary>
        <p className="mt-2">
          Used by the Gemini grounding retriever to produce a structured evidence
          pack (structural context + recent signals). It is a JSON-only response
          with no URLs in the text.
        </p>
        <pre className={preClass}>{`You are a research assistant using Google Search grounding.
Return strictly JSON with this shape:
{
  "structural_context": "max 8 lines",
  "recent_signals": ["<=8 bullets, last 120 days"],
  "notes": "optional"
}
- Focus on authoritative, recent sources (last 120 days).
- Do not include URLs in the JSON text.
Query: Afghanistan (AFG) humanitarian risk outlook - fetch grounded recent signals (last 120 days)
across conflict, displacement, disasters, food security, and political stability.`}</pre>
        <div className="mt-2 flex flex-wrap gap-3">
          {sourceLink(
            "https://github.com/kwyjad/Pythia/blob/main/pythia/web_research/backends/gemini_grounding.py"
          )}
          {sourceLink(
            "https://github.com/kwyjad/Pythia/blob/main/horizon_scanner/horizon_scanner.py",
            "Query builder"
          )}
        </div>
      </details>

      <details className={panelClass}>
        <summary className="cursor-pointer text-slate-100 font-medium">
          Horizon scan triage (HS v2)
        </summary>
        <p className="mt-2">
          The horizon scan prompt requires a single JSON object with a hazard entry
          for every code (ACE, DI, DR, FL, HW, TC). The model must output a numeric
          triage score and optional tier per hazard.
        </p>
        <pre className={preClass}>{`You are a strategic humanitarian risk analyst.
You are assessing {country_name} ({iso3}) for the next 1-6 months.
...
Output requirements (strict):
- Return a single JSON object only. No prose. No markdown fences. No extra keys.
- Provide a hazards entry for every hazard code in the catalog (ACE, DI, DR, FL, HW, TC).
- Each hazard must include a numeric triage_score (0.0 to 1.0).
- A tier is optional for interpretability (quiet/watchlist/priority).`}</pre>
        <div className="mt-2">{sourceLink(
          "https://github.com/kwyjad/Pythia/blob/main/horizon_scanner/prompts.py"
        )}</div>
      </details>

      <details className={panelClass}>
        <summary className="cursor-pointer text-slate-100 font-medium">
          Researcher
        </summary>
        <p className="mt-2">
          The researcher prompt produces a concise brief with explicit headings
          so forecasters can update their priors quickly.
        </p>
        <pre className={preClass}>{`You are a professional RESEARCHER for a Bayesian forecasting panel.
Your job is to produce a concise, decision-useful research brief that helps a statistician
update a prior.

QUESTION
Title: {title}
Type: {qtype}
Units/Options: {units_or_options}

BACKGROUND
{background}

RESOLUTION CRITERIA (what counts as “true”/resolution)
{criteria}

=== REQUIRED OUTPUT FORMAT (use headings exactly as written) ===
### Reference class & base rates
- Identify 1–3 plausible reference classes; give ballpark base rates or ranges and reasoning and on how these were derived; note limitations.

### Recent developments (timeline bullets)
- [YYYY-MM-DD] item — direction (↑/↓ for event effect on YES) — why it matters (≤25 words)

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

Final Research Summary: One or two sentences for the forecaster. Keep the entire brief under ~3000 words.`}</pre>
        <div className="mt-2">{sourceLink(
          "https://github.com/kwyjad/Pythia/blob/main/forecaster/prompts.py",
          "Source: forecaster/prompts.py"
        )}</div>
      </details>

      <details className={panelClass}>
        <summary className="cursor-pointer text-slate-100 font-medium">
          Forecast (SPD)
        </summary>
        <p className="mt-2">
          The forecaster produces a six‑month SPD (Subjective Probability Distribution)
          over five impact buckets. Output must be JSON only.
        </p>
        <pre className={preClass}>{`You are a careful probabilistic forecaster on a humanitarian early warning panel.

Your task is to forecast {quantity_description}.

You will express your beliefs as a SUBJECTIVE PROBABILITY DISTRIBUTION (SPD) over FIVE buckets
for each month.

For each month, distribute 100% probability across these buckets:

People affected (PA) buckets (per month, country-level):
- Bucket 1: < 10,000 people affected (label: "<10k")
- Bucket 2: 10,000 to < 50,000 people affected (label: "10k-<50k")
- Bucket 3: 50,000 to < 250,000 people affected (label: "50k-<250k")
- Bucket 4: 250,000 to < 500,000 people affected (label: "250k-<500k")
- Bucket 5: >= 500,000 people affected (label: ">=500k")

Conflict fatalities buckets (per month, country-level):
- Bucket 1: 0–4 deaths (label: "<5")
- Bucket 2: 5–24 deaths (label: "5-<25")
- Bucket 3: 25–99 deaths (label: "25-<100")
- Bucket 4: 100–499 deaths (label: "100-<500")
- Bucket 5: >= 500 deaths (label: ">=500")

FORECASTING INSTRUCTIONS (Bayesian SPD)

1) Prior / base rates
   - Start from a prior SPD over the buckets based on historical data and relevant reference classes
     (for this hazard and country).

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

   - Do not include any text before or after the JSON.`}</pre>
        <div className="mt-3 text-sm text-slate-300">
          <p className="font-medium text-slate-100">Hazard-specific notes</p>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            <li>
              Bucket set depends on metric: PA uses PA buckets; fatalities uses
              fatalities buckets.
            </li>
            <li>
              The system injects scoring + time-horizon blocks ahead of the
              template for each question.
            </li>
          </ul>
        </div>
        <div className="mt-2">{sourceLink(
          "https://github.com/kwyjad/Pythia/blob/main/forecaster/prompts.py",
          "Source: forecaster/prompts.py"
        )}</div>
      </details>

      <details className={panelClass}>
        <summary className="cursor-pointer text-slate-100 font-medium">
          Scenario generation (priority-only)
        </summary>
        <p className="mt-2">
          Scenario writing only runs when triage tier is <strong>priority</strong> and
          an ensemble SPD exists. The writer returns structured JSON with context,
          needs by sector, and operational impacts.
        </p>
        <pre className={preClass}>{`{
  "primary": {
    "bucket_label": "bucket_3",
    "probability": 0.6,
    "context": ["• brief bullet about context", "• another bullet"],
    "needs": {
      "WASH": ["• bullet"],
      "Health": ["• bullet"],
      "Nutrition": ["• bullet"],
      "Protection": ["• bullet"],
      "Education": ["• bullet"],
      "Shelter": ["• bullet"],
      "FoodSecurity": ["• bullet"]
    },
    "operational_impacts": ["• bullet about ops impact", "• another bullet"]
  },
  "alternative": null
}`}</pre>
        <div className="mt-2 flex flex-wrap gap-3">
          {sourceLink(
            "https://github.com/kwyjad/Pythia/blob/main/forecaster/prompts.py",
            "Scenario prompt"
          )}
          {sourceLink(
            "https://github.com/kwyjad/Pythia/blob/main/forecaster/scenario_writer.py",
            "Priority-only rule"
          )}
        </div>
      </details>

      <details className={panelClass}>
        <summary className="cursor-pointer text-slate-100 font-medium">
          How forecast questions are constructed
        </summary>
        <p className="mt-2">
          Horizon Scanner outputs are converted into question rows during HS
          upsert. The pipeline computes a target month and forecast window, then
          formats question text using hazard templates.
        </p>
        <pre className={preClass}>{`- upsert_hs_payload computes target_month + [opening_date, closing_date]
- _build_questions_for_scenario formats wording from templates:
  * Conflict fatalities (ACLED), displacement (IDMC/DTM)
  * Natural hazards (EM-DAT “people affected”)
- Persisted fields include:
  question_id, iso3, hazard_code, metric, target_month,
  window_start_date, window_end_date, wording, status`}</pre>
        <div className="mt-2">{sourceLink(
          "https://github.com/kwyjad/Pythia/blob/main/horizon_scanner/db_writer.py"
        )}</div>
      </details>

      <details className={panelClass}>
        <summary className="cursor-pointer text-slate-100 font-medium">
          Self-search escape hatch
        </summary>
        <p className="mt-2">
          If the model needs more evidence, it can respond with a single line
          containing <code>NEED_WEB_EVIDENCE:</code> and a query. The system then
          performs retrieval and retries the prompt with appended evidence.
        </p>
        <pre className={preClass}>{`NEED_WEB_EVIDENCE: {country} {hazard} {metric} outlook — include recent signals and structural drivers.`}</pre>
        <div className="mt-2 flex flex-wrap gap-3">
          {sourceLink(
            "https://github.com/kwyjad/Pythia/blob/main/forecaster/prompts.py",
            "Escape hatch in SPD prompt"
          )}
          {sourceLink(
            "https://github.com/kwyjad/Pythia/blob/main/forecaster/self_search.py",
            "Self-search retry"
          )}
        </div>
      </details>
    </section>
  );
}
