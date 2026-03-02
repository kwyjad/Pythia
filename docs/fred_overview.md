## Changes in this Version

This version includes several significant improvements to the Horizon Scanner, forecasting pipeline, and evaluation system:

- **Regime Change is now assessed separately from triage.** RC assessment runs as a dedicated step before triage, with its own model context. This improves reliability by preventing RC reasoning from competing with triage scoring in a single prompt. RC and triage each run a two-pass pipeline for stability.
- **Per-hazard evidence and scoring.** Different hazard types now receive tailored evidence queries, prompts, and scoring criteria rather than a single generic prompt. For example, drought evidence retrieval uses different search terms, recency windows, and scoring considerations than conflict or flood. Seasonal awareness has been added: hazard scoring adjusts based on whether a hazard is in- or off-season for a given region, so that cyclone risk (for example) is scored lower during off-season months.
- **Larger, more diverse forecast ensemble.** The SPD ensemble expanded from 4 models across 3 providers to 7 models across 5 providers, adding Kimi K2.5 and DeepSeek Reasoner alongside upgrades to GPT-5.2, Gemini 3, and Claude Sonnet 4.6.
- **Track-based question routing.** A new track system differentiates question processing. Track 1 questions (those with elevated Regime Change, level > 0) go through the full multi-model ensemble. Track 2 questions (priority but no RC signal) use a lightweight single-model path, reducing cost while maintaining full analytical depth for the questions that need it most. Scenario generation is reserved for Track 1 questions only.
- **Simplified tier system.** The previous three-tier system (quiet / watchlist / priority) has been simplified to two tiers: quiet and priority, with a single threshold (default 0.50). The watchlist tier and its associated hysteresis logic were removed to reduce complexity.
- **Performance page and score exports.** The dashboard Performance page now includes a dropdown to compare different named ensemble methods (e.g., ensemble_mean vs. ensemble_bayesmc), displays median alongside average scores for all metrics, and offers expanded CSV downloads including per-question scored forecasts and model-level summaries.
- **Prompt and overview versioning.** Prompts and the system overview document are now version-tracked with dated snapshots, viewable on the About page for audit and comparison purposes.

---

## System at a Glance

Fred (also referred to as Pythia in the codebase) is an end-to-end humanitarian forecasting system. It takes raw event data (conflict, disasters, displacement) from three primary sources — ACLED, IDMC, and IFRC GO — and produces probabilistic forecasts for a set of standardized questions over a 1–6 month horizon. The system is designed for auditability: each run writes a canonical DuckDB database artifact that records inputs, intermediate judgments, model calls, and final outputs.

A key design tension in forecasting is balancing base rates (what usually happens) with out-of-pattern events (what can happen when the world changes). In Fred, this tension is handled explicitly through Horizon Scanner triage, Regime Change scoring, and — when needed — Hazard Tail Packs that surface trigger evidence.

| Stage | What it does | Primary inputs | Primary outputs |
| --- | --- | --- | --- |
| Resolver (data ingestion) | Ingests and normalizes raw data from external sources into monthly "facts" suitable for forecasting and evaluation. | Connector feeds: ACLED, IDMC, IFRC Montandon (GO API). | facts_deltas, facts_resolved, derived monthly tables; connector status. |
| Web Research (evidence packs) | Fetches grounded, source-cited qualitative evidence to complement quantitative facts, with an explicit recency window. | Country and hazard-tail queries; web research backend. | Evidence pack objects used by HS/Research/SPD; sources lists. |
| Horizon Scanner (HS) triage | Scores each country–hazard pair for elevated risk in the next 1–6 months; detects potential regime shifts. | Resolver facts + evidence pack + hazard catalog. | hs_runs, hs_country_reports, hs_triage (incl. RC fields). |
| Hazard Tail Packs (conditional) | Runs hazard-specific follow-up evidence retrieval for high RC cases to find triggers and counter-signals. | HS RC flags (Level ≥ 2), hazard code, forecast window. | hs_hazard_tail_packs (cached per run/country/hazard). |
| Research (question-level synthesis) | Produces a sourced research brief per question; audits HS Regime Change flags and assembles the reasoning substrate for forecasting. | HS triage + evidence pack + optional tail pack + question definition. | question_research; merged evidence embedded in prompts. |
| Forecasting (SPD v2, ensemble) | Produces a discrete probability distribution (SPD) over outcome bins for each question and month. Track 1 questions (RC-elevated) use a full multi-model ensemble; Track 2 questions (priority, no RC) use a lightweight single-model path. | Question + research brief + base-rate features + RC guidance + optional tail pack + track assignment. | forecasts_raw, forecasts_ensemble; llm_calls; scenario artifacts (Track 1 only). |
| Scoring + dashboard | Computes horizon-specific resolutions (ground truth per forecast month), evaluations (Brier, Log, CRPS scores), calibration, publishes artifacts and powers the dashboard, performance page, and exports. | Forecasts + resolved facts (per horizon month); weighting/calibration config. | resolutions, scores, calibration tables; performance dashboard; CSV downloads. |

## 1. Purpose, scope, and forecasting philosophy

Fred is built to support recurring humanitarian decision-making under uncertainty. It converts disparate, noisy real-world observations into probabilistic statements about near-term outcomes. Those statements are not point forecasts. Instead, the system produces distributions over outcome bins, allowing users to reason about risk, tails, and uncertainty rather than a single best guess.

The system assumes that base rates matter: for many humanitarian metrics, the best starting point is what typically happens in a country given its recent history and seasonality. However, base rates are not sufficient. Humanitarian outcomes often exhibit structural breaks driven by conflict escalation, sudden displacement, extreme weather, policy shifts, or compound shocks. A central design goal is therefore to detect and communicate when "business as usual" may not apply, and to help forecasters allocate probability mass to out-of-pattern outcomes when warranted.

Fred is designed as an auditable pipeline. Each stage writes its outputs into a single DuckDB database artifact, and the artifact is treated as the canonical record for that run. This enables reproducibility (rerunning with the same artifact) and forensic debugging (tracing why a forecast moved).

## 2. Core objects and terminology

To understand Fred, it helps to start with the system's core objects: facts, hazards, questions, evidence packs, and forecasts.

A **fact** is a normalized observation derived from an upstream source (for example, monthly conflict events or fatalities from ACLED, disaster impacts from EM-DAT, or displacement figures from IDMC). Facts are stored with metadata describing provenance and, where possible, a notion of "as-of" time. Facts are later "resolved" into ground truth used for scoring.

A **hazard** is a high-level class of humanitarian risk that Fred tracks consistently across countries. The primary hazard codes are: ACE (armed conflict events and impacts), DI (displacement inflows), DR (drought), FL (flood), HW (heat wave), and TC (tropical cyclone). With the February 2026 switch to IFRC GO as the natural-hazard data source, the system also recognizes additional hazard types including CW (cold wave), EQ (earthquake), VO (volcanic eruption), TS (tsunami), LS (landslide), FIRE (fire), and FI (food insecurity). These codes provide a stable vocabulary that connects HS triage, evidence gathering, question generation, and forecasting.

A **question** is the unit that gets forecast. A question typically represents a country–hazard–metric combination, evaluated across the next six months. The system forecasts each month's outcome using a discrete probability distribution over bins that correspond to meaningful ranges for the metric.

An **evidence pack** is a grounded research bundle used to incorporate qualitative signals. Evidence packs include a short structural context section and a set of recent-signal bullets, each ideally anchored to sources. Evidence packs are used by HS, the Researcher, and (optionally) the forecaster via self-search.

A **forecast** is represented as an SPD (a discrete probability distribution) per month. The system often runs multiple models and aggregates them into an ensemble forecast. The ensemble is treated as the system's primary forecast output for dashboarding and evaluation.

## 3. Resolver: data ingestion, normalization, and base-rate foundation

Resolver is the system's quantitative backbone. It ingests data from three active connectors and transforms them into monthly facts suitable for (a) base-rate estimation and (b) later scoring. Resolver is not a forecasting model. It is a data pipeline with strong opinions about consistency and auditability.

Resolver connectors pull from external providers. In a February 2026 refactoring, seven defunct or disused connectors (including EM-DAT, DTM, IPC, ReliefWeb, UNHCR, WFP, and WHO) were removed and the system was consolidated to three active connectors behind a formal protocol: ACLED (conflict event and fatality records), IDMC (internal displacement flows via the Helix API), and IFRC Montandon (natural-hazard people-affected data via the IFRC GO API, replacing the former EM-DAT connector). IFRC GO provides structured impact data across multiple metrics (affected, fatalities, injured, displaced, missing) and maps disaster types to hazard codes automatically using IFRC's own disaster-type classification.

The ingestion process writes fact rows into tables that track what changed (deltas) and what is considered resolved ground truth. In practice, this means there is a "facts_deltas" table that accumulates incremental connector writes, and a "facts_resolved" table used for evaluation, where each country–hazard–month has an authoritative value once it becomes stable. Two legacy tables (emdat_pa and acled_monthly_fatalities) are also consulted as fallbacks when more recent sources lack data for a given month.

Why this architecture matters for forecasting: the deltas table is the freshest view of what the system has most recently ingested; the resolved table provides a stable reference for scoring and calibration. This split helps prevent a common forecasting pitfall: scoring a forecast against data that later gets revised without keeping a record of what the system 'knew' at the time of forecasting. When the system resolves ground truth for scoring, it uses a four-table priority cascade: facts_resolved (highest priority), then facts_deltas, then emdat_pa (for people-affected metrics only), then acled_monthly_fatalities (for fatality metrics only). The cascade short-circuits: once any table returns a value for a country–hazard–month, lower-priority tables are not queried.

Operationally, Resolver is usually run through a GitHub Actions workflow (often called an initial backfill). This workflow can rebuild the database from scratch or incrementally backfill a set number of months. It also supports policies for how to treat '0 rows written' situations, which are important because external sources can occasionally return empty responses due to outages, rate limits, or schema changes. Two important safeguards were added in February 2026: a calendar cutoff prevents resolution of the current (incomplete) month — only fully completed calendar months are eligible for resolution — and a data-driven cutoff prevents resolution of months beyond the latest data actually present in any source table. A stale-resolution purge runs at the start of each resolution pass, deleting any resolution rows that were incorrectly written for future or partial months in earlier pipeline runs, along with their associated scores, and reverting affected questions to active status so they can be re-resolved correctly.

A key limitation — and one that users must internalize — is that these data sources are imperfect proxies for humanitarian impact. They reflect reporting incentives, access constraints, and definitional choices. Fred does not pretend this disappears. Instead, it tries to surface uncertainty explicitly through data-quality notes, model uncertainty, and regime-change flags.

## 4. Web research evidence packs: grounded qualitative signals

Evidence packs provide qualitative context that does not fit neatly into the structured datasets. The system uses evidence packs to detect emerging triggers, explain deviations, and support interpretability.

An evidence pack has two primary components. Structural context provides brief background drivers that change slowly (governance, exposure, long-running conflict dynamics). Recent signals provide a set of time-bounded observations intended to affect the 1–6 month horizon. Recent signals should be testable: they should point to observable developments rather than general commentary.

Fred's evidence packs are designed to be grounded: they carry explicit source URLs and are generated within a recency window (commonly 120 days). This is essential for credibility. When evidence is thin or contradictory, the system is instructed to say so rather than fabricate certainty.

In the upgraded Horizon Scanner configuration, the web research query is explicitly optimized to hunt for tail triggers. It requests that recent signals be framed in three types: TAIL-UP triggers (leading indicators of an out-of-pattern escalation or worsening), TAIL-DOWN dampeners (counter-signals suggesting mitigation or de-escalation), and BASELINE continuation signals. This design forces the system to consider why outcomes might depart from base rates, and also why they might not.

Evidence packs are used in three places: first, HS triage uses them to score risk and detect regime change; second, Research uses them to draft a sourced brief; and third, SPD forecasting may request one additional self-search query when evidence is insufficient, subject to strict rate limits.

## 5. Horizon Scanner (HS): triage-first risk screening

Horizon Scanner is the system's front-end risk screening stage. Its job is not to forecast numbers. Instead, it triages which country–hazard combinations merit deeper attention and provides structured context for downstream research and forecasting.

HS takes three major inputs: Resolver features (base-rate signals), the hazard catalog (what each hazard means and how it maps to metrics), and a grounded evidence pack. It outputs a structured JSON object per country, with an entry for each hazard code. Each hazard entry includes a triage_score on a 0–1 scale and optional interpretability fields such as drivers, data-quality notes, and a short scenario stub.

HS is typically run on a schedule (for example, monthly) and writes into the canonical DuckDB. Two tables are important for understanding HS outputs. The hs_country_reports table stores the evidence pack used for a country (including sources and markdown renderings). The hs_triage table stores one row per country–hazard, including the triage score and other structured fields.

A key architectural change is that HS now runs as a multi-stage pipeline rather than a single prompt. First, Regime Change assessment runs as a dedicated step with its own model context (see Section 6). The RC results are then passed into the triage step, which scores the country–hazard pair with the RC assessment already available. This separation improves reliability by preventing RC reasoning from competing with triage scoring in a single, overloaded prompt.

Both the RC and triage steps use per-hazard prompts and evidence queries. Rather than a single generic prompt for all hazard types, the system now constructs tailored prompts for each hazard category. For example, drought evidence retrieval emphasizes precipitation anomalies, food security signals, and agricultural indicators, while conflict evidence focuses on armed-group dynamics, ceasefire status, and political escalation. This hazard-specific prompting produces more relevant evidence and more calibrated scores.

Seasonal filtering has also been integrated into the HS pipeline. The system is aware of whether a hazard is in-season or off-season for a given region, and scoring adjusts accordingly. For instance, tropical cyclone risk is scored lower during months outside the historical cyclone season, reducing false positives from evergreen background risk language.

To reduce run-to-run variance, HS uses a stabilization pattern: both RC and triage run a two-pass pipeline and average numeric outputs such as triage_score and RC likelihood. The second triage pass uses a different model (currently GPT-5-mini) to introduce diversity. This is not a deep statistical ensemble; it is a practical technique that improves consistency for dashboard users.

HS computes a tier label — quiet or priority — based on a single configurable threshold (default 0.50). Scores at or above the threshold are "priority"; scores below are "quiet". Tiers are intended for interpretability and prioritization, while the numeric triage_score is the primary signal that downstream code can use.

## 6. Regime Change (RC): making out-of-pattern risk explicit

Regime Change is Fred's mechanism for flagging when base rates may be misleading in the forecast window. The intent is not to predict black swans on demand. Rather, it is to detect credible situations in which the generating process for a hazard may shift — for example, when a conflict enters a new phase, when a border policy changes, or when seasonal forecasts indicate unusually severe hazards.

RC is represented with three conceptual dimensions: likelihood, direction, and magnitude. Likelihood captures the chance that a regime shift occurs during the next 1–6 months. Direction indicates whether the shift would push outcomes up (worse), down (improving), mixed, or unclear relative to baseline. Magnitude captures how large the shift could be if it occurs.

Operationally, RC is assessed in a dedicated LLM pipeline step that runs before triage (see Section 5). The RC assessment has its own prompt context, which allows the model to focus entirely on detecting structural breaks without simultaneously having to produce triage scores. The RC step also uses per-hazard prompts: each hazard type receives tailored instructions and evidence that reflect its specific escalation dynamics. For example, the RC prompt for armed conflict emphasizes ceasefire breakdowns, new armed-group mobilization, and political crisis triggers, while the RC prompt for flood emphasizes upstream dam status, rainfall anomalies, and infrastructure degradation.

Fred stores RC results as numeric and categorical fields in hs_triage. A derived RC score is computed as likelihood multiplied by magnitude. This provides a single 'how much should I care' number that suppresses cases that are high-likelihood but low-impact, or high-impact but highly speculative.

RC scores are mapped into discrete levels used for UI and downstream gating. Level 0 is base-rate normal, Level 1 is a watch state, Level 2 indicates elevated regime-change plausibility, and Level 3 indicates a strong out-of-pattern hypothesis. In February 2026, the RC scoring was recalibrated to reduce false positives. The Level 1 likelihood threshold was raised from 0.35 to 0.45, and the Level 1 composite-score threshold was raised from 0.20 to 0.25, so that mildly elevated signals no longer trigger a watch flag. The triage prompt now includes explicit calibration guidance: it distinguishes between high triage scores (which capture overall risk, including chronic situations) and regime-change scores (which should only capture departures from established patterns). The prompt provides an expected distribution — roughly 80% of country–hazard pairs should have likelihood ≤ 0.10 — and examples of what is and is not a regime change. A run-level distribution check warns operators when the fraction of elevated flags exceeds configurable thresholds (e.g., more than 15% at Level 2).

Critically, RC does not merely decorate the dashboard. When RC is elevated, the system can override the 'need_full_spd' gating so that a full probabilistic forecast is produced even if the triage tier would otherwise be quiet. This is how Fred avoids an important failure mode: missing a structural break because it does not look like a high base-rate risk.

RC is surfaced to users across the dashboard: on country maps and lists as a 'highest RC' indicator, on triage tables as probability/direction/magnitude/score columns, and on question drilldown pages as summary boxes. This makes the system's out-of-pattern reasoning visible rather than hidden inside prompts.

## 7. Hazard Tail Packs (HTP): hazard-specific trigger evidence (Phase A/B)

Hazard Tail Packs are a second-stage evidence mechanism designed to answer a simple question: if RC is high, what are the concrete leading indicators and counter-indicators for this hazard in this country, in this window?

A key lesson in operational forecasting is that generic 'risk outlook' research tends to reinforce base-rate thinking. Tail forecasting requires trigger detection: signs that escalation or suppression mechanisms are activating. Tail packs exist to retrieve that trigger evidence in a structured way.

Tail packs are generated conditionally. In Phase A, the system generates tail packs only for hazards with RC Level 2 or 3, and it caps generation to at most two hazards per country per HS run (by RC score). This prevents the system from becoming noisy and keeps costs bounded. Tail packs are cached per (run_id, iso3, hazard_code) in the hs_hazard_tail_packs table so reruns do not repeatedly hit the web.

Tail pack signals are intentionally formatted with explicit prefixes. TRIGGER bullets are leading indicators supporting the regime-shift hypothesis; DAMPENER bullets are counter-signals suggesting the shift may not materialize; BASELINE bullets support continuation. Each bullet also carries an intended time window (for example, month_3-4) and a direction tag.

In Phase A, tail packs are injected into the Research stage so the Researcher can confirm, downgrade, or rebut the RC hypothesis with sources. In Phase B, tail packs are also injected into the SPD forecasting prompts for RC-elevated hazards, but with strict length limits: at most 12 bullets are included to prevent prompt bloat.

## 8. Question bank: what gets forecast and why

After HS triage, the system maintains a question set that defines what gets forecast. Conceptually, a question ties together: a country, a hazard, a metric, and a forecast horizon.

Questions are intended to be stable and comparable over time. They encode the metric to be forecast (for example, a displacement inflow proxy or conflict impact proxy) and the discrete bins that define the SPD outcomes. This standardization is what enables scoring and calibration. Without it, each run would effectively forecast a different target.

The question set is typically refreshed using HS triage outputs. Priority-tier hazards are included by default, but as described earlier, elevated RC (level > 0) can force inclusion even for otherwise quiet hazards. This ensures the system pays attention to plausible regime breaks. Each question is assigned a track (see Section 10) based on its RC status: Track 1 for RC-elevated questions, Track 2 for priority questions without RC. The track determines the depth of the forecasting pipeline that the question passes through. A critical change in February 2026 made the questions table append-only: when a new HS run generates a question with the same ID as an earlier run, the original metadata (window start date, target month, wording) is preserved rather than overwritten. This prevents downstream resolution and scoring from breaking when the HS runs again months later. As an additional safeguard, the system can derive the true forecast window from the LLM call log (which is always append-only) rather than relying solely on the questions table, falling back through a three-tier priority: (1) the earliest LLM call timestamp for that question, (2) the questions table's window_start_date, (3) a derivation from the target_month.

## 9. Research v2: sourced synthesis and RC audit

The Research stage converts raw evidence into a structured, sourced brief that the forecaster can use. Research is not merely summarization. It is designed to bridge from "what is happening" to "what might happen in the next 1–6 months" with explicit attention to evidence quality, uncertainty, and plausible mechanisms.

Research v2 ingests HS triage outputs, including RC fields, and it is explicitly instructed to audit them. If RC is elevated, the Researcher must either confirm it with at least one regime-shift signal and sources, or provide a sourced rebuttal explaining why the RC flag is not supported. This prevents RC from becoming a decorative number that is ignored downstream.

Research uses a merged evidence object that can include: the country evidence pack, any question-specific evidence pack, and (when present) the hazard tail pack. The merged object is embedded directly into the prompt so that the model can cite sources. Research outputs are stored for provenance in the database.

## 10. Forecasting: SPD v2, ensembles, and tail-aware behavior

Fred's primary forecast output is an SPD: a discrete probability distribution over outcome bins for each month in the horizon. This representation is more operationally useful than a point estimate because it preserves uncertainty and supports expected-value computations and tail-risk analysis. SPD data is stored in dual format: a numeric format (month_index and bucket_index with a probability value) for computational efficiency, and a labeled format (class_bin labels like "<10k" with a probability value) for human readability. Both representations are written simultaneously so that scoring and display code can each use whichever is more convenient.

SPD v2 is designed as a constrained forecasting task with explicit guardrails. It encourages disciplined base-rate anchoring while still requiring tail awareness. With the RC upgrades, SPD v2 includes a dedicated Regime Change guidance block. When RC is elevated, the forecaster is instructed that the base rate is less reliable and must either widen the distribution appropriately or rebut the regime-shift hypothesis.

To prevent hand-waving, SPD v2 requires accountability in the human-facing explanation: the forecaster must include a sentence beginning with "RC:" that states what was flagged, whether it was accepted, and how it affected the forecast. This makes the system's interaction between RC and the distribution explicit.

Not all questions require the same depth of forecasting. Fred uses a track system to route questions through different pipeline configurations based on their risk profile:

- **Track 1** questions are those where Regime Change is elevated (RC level > 0). These go through the full multi-model ensemble, which currently includes 7 models across 5 providers: GPT-5.2 and GPT-5-mini (OpenAI), Claude Sonnet 4.6 (Anthropic), Gemini 3.1 Pro and Gemini 3 Flash (Google), Kimi K2.5 (Moonshot), and DeepSeek Reasoner (DeepSeek). Track 1 questions also receive scenario narratives (see Section 11). The rationale is that RC-elevated questions represent the highest-uncertainty situations where model diversity matters most.
- **Track 2** questions are priority-tier questions with no elevated RC signal. These use a lightweight single-model path (currently Gemini Flash) that is faster and cheaper. Scenario generation is skipped. Track 2 reflects a practical judgment: when base rates are expected to hold, a single capable model produces adequate forecasts without the cost of running the full ensemble.

The track field is persisted in the database alongside triage and question records, so downstream analysis can distinguish how each question was processed.

For Track 1 questions, the system aggregates member SPDs into named ensemble forecasts. The default named ensembles include ensemble_mean (simple average of member SPDs) and ensemble_bayesmc (a Bayesian Monte Carlo aggregation). Named ensembles are tracked as explicit model names in the database, allowing the dashboard and scoring system to compare ensemble methods against individual models. The ensemble approach is a pragmatic response to model idiosyncrasies: it tends to reduce the chance that a single model's blind spot dominates a forecast.

In Phase B, if a hazard tail pack is present for an RC-elevated hazard, it is injected into the SPD prompt as a bounded set of bullets (maximum 12). The prompt explains how to interpret TRIGGER, DAMPENER, and BASELINE bullets. This turns tail packs into actionable evidence rather than background noise.

Finally, the system supports a narrow form of self-search: when evidence is insufficient, the model may request a single additional query. This is tightly bounded to prevent runaway cost or unpredictable behavior.

## 11. Scenarios, narratives, and human interpretability

In addition to numeric forecasts, Fred can produce scenario narratives for Track 1 questions (those with elevated RC). Scenarios are not treated as forecasts in their own right. They are interpretability artifacts meant to summarize plausible pathways that could lead to different bins of the SPD. Scenario generation is skipped for Track 2 questions, where the base rate is expected to hold and the additional interpretive cost is not warranted.

Scenario narratives are particularly useful when RC is elevated, because they help users understand what a regime shift would look like operationally. However, narratives are also easy to misuse. Fred therefore keeps scenarios optional and separates them from the numeric forecasting artifacts used for scoring.

## 12. Scoring, calibration, and evaluation loops

A forecasting system is only as good as its evaluation loop. Fred includes workflows to compute resolutions (ground truth), compute forecast scores, and assess calibration. A critical correctness fix in February 2026 changed scoring to be horizon-specific: each of the six forecast months is now scored against its own calendar month's ground truth, rather than all six being scored against a single target month. For example, if a question's forecast window starts in September, month 1 is scored against September data, month 2 against October data, and so on through month 6 against February data. The resolutions table stores one row per (question, horizon month) pair, with the resolved ground-truth value and the calendar month it corresponds to.

Three proper scoring rules are computed for each scored forecast: the Brier score (sum of squared differences between predicted probabilities and the one-hot truth vector; lower is better), the log score (negative log of the probability assigned to the true bin; heavily penalizes confident wrong predictions), and a CRPS-like score (compares the cumulative distribution against a step function at the true bin). All three metrics are stored per (question, horizon, model) and are aggregated into averages and medians on the dashboard. Calibration and weighting can be configured using stored weight files (for example, in the calibration directory). The calibration workflow chain was fixed in February 2026 so that each stage runs in the correct order: Resolver backfill, then Compute Resolutions, then Compute Scores, then Calibration. Previously, scores could run before resolutions were available, producing empty results.

## 13. The dashboard: how users consume outputs

Fred's dashboard is designed for two audiences: forecasters (who need to inspect evidence and produce forecasts) and decision-makers (who need a summary of risk and tail conditions).

Key dashboard pages include: the Forecast Index (country-level rollups and map), the Countries list and drilldowns, HS Triage, Forecast lists and drilldowns, Resolver (data inspection), Costs, Performance, Downloads, and About.

RC is a first-class UI concept. Countries with Level 1–3 RC flags are highlighted, and questions display RC probability/direction/magnitude/score so users can see when the system believes a base-rate deviation is plausible. Exports include RC fields so that analysis outside the dashboard preserves the same context.

The Resolver page exists because the system treats data provenance as a primary concern. Users can inspect the underlying facts feeding the forecasts and see connector status, including last updated timestamps.

The About page includes versioned snapshots of both the system's prompts and this system overview document. Users can select from a dropdown of dated versions to see how the system's documented behavior has evolved over time. This supports auditability: when investigating a historical forecast, operators can review the prompt and system configuration that were active at the time.

The Performance page surfaces forecast evaluation in the dashboard. It displays five KPI cards (scored questions, hazards, models, HS runs, and ensemble Brier score) and supports four views: Total (one row per model), By Hazard (scores broken down by hazard code), By Run (scores per HS run over time), and By Model. Users can filter by metric (People Affected or Fatalities). An ensemble selector dropdown lets users switch between named ensemble methods (e.g., ensemble_mean vs. ensemble_bayesmc) in the By Hazard and By Run views, enabling direct comparison of aggregation strategies. Tables show both average and median values for all three scoring rules (Brier, Log, CRPS); median scores are particularly useful because they are less sensitive to outliers from early-run or data-sparse questions.

The Downloads page was expanded to include performance score exports: per-question CSV files for each named ensemble (including the full 6-month x 5-bin forecast grid, expected impact values, resolution values, and scores), a model-level summary CSV with average, median, min, and max scores per model and hazard, and a rationale export that downloads the human-readable explanations that LLM models provide alongside their probabilistic forecasts, filtered by hazard code.

## 14. Operations: canonical DB artifacts, workflows, and debugging

Fred is typically operated through GitHub Actions. Workflows download a canonical database artifact, apply new computations, and re-upload an updated artifact. A concurrency lock prevents multiple workflows from mutating the canonical DB at the same time. Model configuration is centralized in a single ensemble list in pythia/config.yaml, where each entry is a provider:model_id string (for example, openai:gpt-5.2 or anthropic:claude-sonnet-4-6). To change the models used in production, operators edit this list — no Python code changes are needed. Purpose-specific overrides (for example, a cheaper model for scenario writing or an HS fallback model) are also configured in the same file. Model cost rates (input and output cost per 1,000 tokens) are stored in a separate JSON file (pythia/model_costs.json) so that costs can be updated by editing data rather than code.

A signature guardrail is used to ensure the database contains required tables before being treated as canonical. This protects against partial artifacts and prevents downstream stages from running on an incomplete database.

Debugging is treated as an operational requirement. Runs log LLM calls, costs, evidence pack diagnostics, and the reasons why certain questions were included or skipped. This makes it feasible to answer "why did the model think this?" after the fact.

## 15. Practical guidance for forecasters

The system is designed to support, not replace, human judgment. In practice, forecasters should treat HS triage as a prioritization and hypothesis-generation tool. The most productive workflow is: (1) inspect HS triage and RC flags; (2) read research briefs and tail packs for high-RC hazards; (3) decide what would move the distribution; (4) ensure the SPD expresses both the baseline and the plausible tail mechanisms; and (5) write explanations that are accountable and falsifiable.

When RC is elevated, the system is deliberately asking you to think in counterfactual terms: what would it take for outcomes to be much worse (or much better) than the recent base rate? If you cannot find credible triggers, the correct response is not to force a tail forecast. It is to document uncertainty and, when appropriate, rebut the RC hypothesis with sources.

Finally, remember that humanitarian metrics are mediated by reporting. A sharp change in recorded values can reflect a real change, a reporting shift, or both. Use data quality notes and cross-source triangulation to avoid overconfident narratives.

## Appendix: Glossary

- **Base rate**: The typical distribution of outcomes for a metric given historical data and seasonality. In Fred, base-rate information is primarily derived from Resolver facts.
- **SPD (discrete probability distribution)**: A set of probabilities over discrete outcome bins. Fred forecasts SPDs rather than point estimates.
- **Hazard**: A standardized category of risk (ACE, DI, DR, FL, HW, TC) that organizes evidence and forecasting targets.
- **Regime Change (RC)**: A structured estimate that the generating process for a hazard may shift during the forecast window, making base rates less reliable.
- **Hazard Tail Pack**: A hazard-specific evidence pack generated for elevated RC hazards, containing trigger and dampener signals scoped to a forecast window.
- **Canonical database artifact**: The DuckDB file produced by a run and treated as the authoritative record for that run's inputs, intermediate artifacts, and outputs.
- **IFRC Montandon**: The IFRC GO API connector that replaced EM-DAT as the primary natural-hazard people-affected data source. Provides structured impact data with automatic disaster-type classification across 18 hazard types.
- **Calendar cutoff**: A safeguard that prevents resolution of ground truth for the current (incomplete) month. Only fully completed calendar months are eligible for resolution.
- **Performance page**: A dashboard page that surfaces forecast evaluation metrics (Brier, Log, CRPS scores) with views by total, hazard, run, and model, including support for comparing named ensemble methods.
- **Named ensemble**: An ensemble method identified by an explicit name (e.g., ensemble_mean, ensemble_bayesmc) rather than the legacy unnamed ensemble. Named ensembles are tracked separately in the database and can be compared on the Performance page.
- **Brier score**: A proper scoring rule that measures the mean squared error between the predicted probability distribution and the actual outcome. Range 0 (perfect) to 1. Used alongside log score and CRPS in Fred's evaluation pipeline.
- **Rationale export**: A downloadable CSV of the human-readable explanations that LLM models provide alongside their probabilistic forecasts, filterable by hazard code and model.
- **Track 1 / Track 2**: The routing classification for questions in the forecasting pipeline. Track 1 questions (RC level > 0) receive the full multi-model ensemble and scenario generation. Track 2 questions (priority tier, no elevated RC) use a lightweight single-model path. The distinction reduces cost while concentrating analytical depth on the highest-uncertainty questions.
- **Per-hazard prompts**: Tailored evidence queries and scoring instructions constructed for each hazard category (conflict, drought, flood, etc.), replacing the earlier generic prompt approach. Per-hazard prompts improve evidence relevance and scoring calibration by incorporating hazard-specific indicators and escalation dynamics.
- **Seasonal filtering**: A mechanism that adjusts HS triage and RC scoring based on whether a hazard is in-season or off-season for a given region, reducing false positives from evergreen risk language during periods when a hazard is climatologically unlikely.
