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
          Research v2
        </summary>
        <p className="mt-2">
          Research v2 combines question metadata, Resolver history, HS triage, and
          merged evidence packs, then returns a strict JSON brief for the forecaster.
        </p>
        <pre className={preClass}>{`Question metadata:
\`\`\`json
{ ...question... }
\`\`\`
Resolver history (noisy, incomplete base-rate data):
\`\`\`json
{ ...resolver_features... }
\`\`\`
HS triage (tier, triage_score, drivers, regime_shifts, data_quality):
\`\`\`json
{ ...hs_triage_entry... }
\`\`\`
Merged evidence (HS country pack + question-specific web research):
\`\`\`json
{ ...merged_evidence... }
\`\`\`
Return a single JSON object:
{ ...RESEARCH_V2_REQUIRED_OUTPUT_SCHEMA... }
Do not include any text outside the JSON.`}</pre>
        <div className="mt-2">{sourceLink(
          "https://github.com/kwyjad/Pythia/blob/main/forecaster/prompts.py"
        )}</div>
      </details>

      <details className={panelClass}>
        <summary className="cursor-pointer text-slate-100 font-medium">
          Forecasting (SPD v2)
        </summary>
        <p className="mt-2">
          The forecaster produces a six‑month SPD (Subjective Probability Distribution)
          over five impact buckets. Output must be JSON only.
        </p>
        <pre className={preClass}>{`Your task is to produce a six-month PROBABILITY DISTRIBUTION over five impact buckets.
...
If you need more evidence before forecasting, output EXACTLY one line:
NEED_WEB_EVIDENCE: <your query>
Otherwise, produce the forecast JSON.

Output schema:
{
  "spds": {
    "YYYY-MM": {"buckets": ["<10k","10k-<50k","50k-<250k","250k-<500k",">=500k"], "probs": [0.7,0.2,0.07,0.02,0.01]},
    "YYYY-MM+1": {"buckets": ["<10k","10k-<50k","50k-<250k","250k-<500k",">=500k"], "probs": [0.7,0.2,0.07,0.02,0.01]}
  },
  "human_explanation": "3–4 sentences summarising the base rate and update signals."
}`}</pre>
        <div className="mt-3 text-sm text-slate-300">
          <p className="font-medium text-slate-100">Hazard-specific notes</p>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            <li>
              Bucket labels differ for conflict fatalities vs. people‑affected
              (PA) questions.
            </li>
            <li>
              DI (displacement inflow) has no Resolver base rate; the model must
              construct a prior from research and focus on incoming cross‑border flows.
            </li>
            <li>
              Natural hazards with PA metrics use EM‑DAT “people affected”
              definitions (injured, displaced, or needing assistance).
            </li>
          </ul>
        </div>
        <div className="mt-2">{sourceLink(
          "https://github.com/kwyjad/Pythia/blob/main/forecaster/prompts.py",
          "SPD v2 prompt"
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
            "Escape hatch in SPD v2"
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
