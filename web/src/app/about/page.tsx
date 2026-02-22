import { renderSimpleMarkdown } from "../../lib/simple_markdown";
import AiPromptsSection from "./AiPromptsSection";

const ABOUT_MD = `## Welcome!

Fred is an experimental humanitarian impact forecasting system. Its objective is to test the effectiveness of LLM as horizon scanning and forecasting agents in the humanitarian space. Fred started operation in December 2025. Its first "stable" run (i.e. with a standard set of models and processes) was in January 2026. Fred runs monthly and will update with new forecasts at the start of each month.

To repeat: FRED IS AN EXPERIMENTAL SYSTEM. Do not use Fred's forecasts for anything, ever, for any reason, except pure entertainment. Consult an astrologer instead. Assume Fred's forecasts are rubbish. Whether or not they are any good is what we are here to figure out. Even if for some reason Fred's forecasts are not rubbish, they certainly don't foresee the future.

View at your own risk, and don't even think about using Fred's outputs. Beyond this, don't expect stability. Everything about Fred could change at any moment. In fact, it will change, that's the only sure thing around here. It is an experimental system. Question resolution is a major weak spot and this will negatively affect forecasting skill scoring.

## Fred: an end-to-end forecasting system for humanitarian risk

Fred is an AI forecasting pipeline built to answer a simple question in a rigorous way:

**How many people are likely to be affected by certain hazards in a set of countries in the next few months, and how (un)certain are we?**

Fred doesn’t just output a single prediction. It produces **probabilistic forecasts**—distributions over plausible outcomes—grounded in retrieved evidence and backed by a complete, inspectable audit trail.

Right now Fred covers the following hazards/events:

- Armed conflict (impact measured as conflict fatalities and people displaced by conflict; separate forecasts for each conflict question)
- Flood (people affected)
- Drought (people affected)
- Tropical cyclone (people affected)
- Displacement inflow (from neighbouring countries)
- Heat wave (people affected)

Future goals include adding public health emergencies and replacing people affected with people in need of humanitarian assistance, but resolution data limitations at present make this impossible (or at least a very big job).

## What Fred produces

For each country and hazard type, Fred generates:

- **A triage decision**: whether the situation is forecast-worthy right now, or “quiet.”
- **Forecast questions**: standardized questions with consistent metrics and horizons.
- **Evidence packs**: shared, reusable sets of retrieved sources.
- **Research briefs**: structured summaries designed to be consumed by forecasting prompts.
- **Probabilistic forecasts**: monthly probability distributions across severity bins.
- **Diagnostics**: model-by-model outputs, costs, latency, and run metadata.

Everything is written to a single **DuckDB system of record**, so each run is reproducible and auditable end-to-end.

## How Fred works (step by step)

### 1) Build the factual baseline (Resolver → database)

Fred starts with structured historical data: monthly facts, deltas, and snapshots. This baseline (ACLED for conflict fatalities, IDMC for displacement, and IFRC Montandon for people affected by flood, drought, cyclone or heat wave) serves two purposes:

1) it provides context and priors for triage, research, and forecasting, and  
2) it becomes the reference for later scoring and calibration.

### 2) Scan for emerging risks (Horizon Scanner / HS)

Fred runs a Horizon Scanner across a fixed hazard taxonomy for each country. HS produces structured triage rows that answer:

- What hazards might matter in this country right now?
- How urgent or material are they?
- Are they worth spending forecasting budget on?

A key design choice: **HS is allowed to produce zero forecast-worthy hazards.** “No questions” can be the correct outcome for a country-month. The system still records the triage outputs so the “quiet” decision is inspectable.

**Model used:**  
- HS triage runs on **Gemini 3 Flash** to balance intelligence, cost, and speed. The model is queried twice and the triage scores averaged to promote stability.

### 3) Turn triage into forecast questions

When HS flags something as forecast-worthy, Fred converts it into standardized forecast questions. Questions are designed to be stable over time so they can be scored consistently as outcomes arrive.

### 4) Gather a shared evidence pack (retriever web research)

Fred runs a shared retriever that performs web research once and injects the resulting evidence pack into multiple stages. This reduces cost, stabilizes sources, and avoids having each model “browse” independently.

**Model used:**  
- Web search and evidence pack generation uses **Gemini 2.5 Flash Lite** for speed and low cost where intelligence is less important.

### 5) Draft a structured research brief (Research v2)

Research v2 transforms the evidence pack into a structured brief: what changed recently, what signals matter, what is uncertain, and what to watch next. Research outputs are stored alongside run metadata and can be inspected independently.

**Model used:**  
- Research runs on **Gemini 3 Flash**.

### 6) Produce probabilistic forecasts (SPD v2 ensemble)

Fred’s core output is an SPD (Subjective Probability Distribution): a probability mass across five humanitarian impact severity bins for each month ahead in the forecast horizon. This allows the forecast to cover the different probabilities of various levels of impact that a hazard could have. The bins cover the range of zero to infinity, so one must be true each month.

Instead of a single number, you get answers like:

- a distribution over outcome ranges, and
- Fred calculates Expected Impact Values (EIVs) as a convenient summary statistic. This give you a single number that captures the average expected impact (by multiplying probabilities assigned to each bin by a bin centroid value and the summing these values), but the full distribution is still available for deeper analysis.

Fred first records each model’s raw distribution, then aggregates them into an ensemble forecast.

**Models used (forecast ensemble):**

- **OpenAI: GPT-5.1**
- **Anthropic: Claude Opus 4.5**
- **Google: Gemini 3 Flash**
- **Google: Gemini 3 Pro**

This multi-model approach reduces dependence on any single provider’s quirks and allows systematic evaluation of which models perform best for which hazards and contexts.

### 7) Optional: generate scenarios (priority-only)

For high-priority items, Fred can generate scenario narratives that explain plausible pathways consistent with the quantitative forecast. Scenarios are generated after the probability distribution exists, so narratives don’t drive the numbers—they explain them.

**Model used:**  
- Scenarios run on **Gemini 3 Flash**.

### 8) Log everything (accountability + scoring)

Every run writes structured artifacts into the system of record, including:

- run IDs and configuration fingerprints,
- per-model raw outputs,
- ensemble outputs,
- evidence packs and research briefs,
- and call-level diagnostics (tokens, costs, latency, success/failure).

This makes it possible to audit decisions, reproduce outputs, and systematically improve the system over time.

### 9) Resolve questions and provide calibration advice

- Fred uses the data in the DuckDB, updated monthly, to resolve the forecast questions. This produces Brier scores and variants.
- Fred also uses the resolution results to provide calibration advice back to the forecasting models, e.g., "on this kind of question you usually overestimate by X", to help the forecasts improve over time.

## Fred's schedule

- Horizon Scan and forecasts: First day of every month, with the forecast window starting one month forward and extending six months from that. E.g., forecasts made on 01 January start on 01 February and end 31 July
- Resolver updates: 15th day of every month, with a 3 month backfill to capture source data revisions
- Resolution and calibration advice: Automatically after Resolver updates, for active forecasts in the window for which resolution is available.

## Why distributions (not point predictions) matter

Humanitarian planning is rarely about “will X happen?” It’s about:

- **How big could it be?**
- **When might it peak?**
- **How uncertain is the situation?**
- **What’s the chance we cross a critical threshold?**

By producing monthly probability distributions, Fred supports:

- risk-based planning,
- transparent thresholds for action,
- and long-run scoring of calibration (whether predicted probabilities match observed frequencies).

---

## What makes Fred auditable

Fred is built so you can answer questions like:

- What evidence did the system use?
- Which model said what?
- How much did the run cost?
- Where did latency or failures occur?
- Did the system improve over time?

That’s possible because the system is designed around:

- a single canonical system of record (DuckDB), and
- consistent logging of intermediate artifacts and call-level diagnostics.

---

## Operational reality: failures, timeouts, and partial ensembles

Real-world runs face rate limits, provider outages, and key configuration issues. Fred is designed to degrade gracefully:

- If a provider is unavailable, the system can proceed with the remaining models.
- Timeouts and retry budgets are configurable so runs don’t hang indefinitely.
- “No hazards” outcomes are treated as valid and recorded as such.

---

## What Fred is (and isn’t)

**Fred is:**

- a forecasting pipeline that produces probability distributions,
- an evidence-backed system with an audit trail,
- a framework designed for scoring, calibration, and improvement over time.
- ***an experimental and completely unproven system.***

**Fred is not:**

- an oracle or source of ground truth,
- a substitute for expert judgment,
- a decision engine that dictates policy actions.

## Caveats

- The resolution component is currently Fred's weakest point. More work is needed to have comprehensive resolution data for all hazards. This will negatively impact forecast scoring, and probably produce poor scores until improved.

Fred provides probabilities and evidence, not instructions. If eventually these turn out to have value humans still decide what to do.

## Code and Contact

- Fred's code (almost entirely python) is open for non-profit or research use - not commercial. You can access the code at https://github.com/kwyjad/Pythia. Be aware that the code is a vibe-coded mess, and again, use at your own risk (Note: In GitHub Fred is called Pythia). 
- If you are interested in talking or collaborating, so am I. Contact me on LinkedIn at https://www.linkedin.com/in/kevinwyjad/
`;

export const metadata = {
  title: "About",
};

export default function AboutPage() {
  if (process.env.NODE_ENV !== "production" && ABOUT_MD.includes("Acceptance Criteria")) {
    console.warn("[About] ABOUT_MD contains Codex meta tail; remove it.");
  }

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold">About</h1>
        <p className="text-sm text-fred-text">
          About Fred and how the system works.
        </p>
      </header>
      <article className="max-w-none">
        {renderSimpleMarkdown(ABOUT_MD)}
        <AiPromptsSection />
      </article>
    </div>
  );
}
