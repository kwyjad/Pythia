## Changes in this Version

This version includes several significant improvements to the forecasting pipeline, evidence architecture, and evaluation system:

- **Structured data connectors replace the research LLM.** The old Research stage — which used an LLM to generate research briefs from web searches — has been retired. In its place, the system now pulls structured data directly from authoritative humanitarian, climate, and conflict-forecast APIs: ReliefWeb (situation reports), ACAPS (severity scores, risk radar, daily monitoring, humanitarian access), IPC (food security phase classifications), ACLED political events, NMME seasonal climate forecasts (via NOAA), VIEWS and conflictforecast.org (quantitative conflict predictions), and ICG CrisisWatch (expert conflict assessments). This data is stored in the database and injected directly into pipeline prompts, making the evidence pipeline deterministic and reproducible rather than dependent on variable web-search results.
- **Adversarial evidence checks for high-risk forecasts.** When Regime Change is elevated to Level 2 or above, the system now automatically runs targeted counter-evidence searches — looking for reasons the predicted regime shift might *not* materialize. The results (counter-evidence, historical analogs, stabilizing factors) are structured and injected into the forecasting prompt so that the model considers both sides of the hypothesis.
- **Hazard-specific forecasting guidance.** Each hazard type (conflict, flood, drought, cyclone, heatwave, displacement) now receives tailored reasoning instructions in the forecasting prompt. These replace the earlier generic guidance and provide hazard-specific advice on tail-risk coverage, outcome bucket calibration, and escalation dynamics.
- **Calibration advice generation.** The system now generates per-hazard, per-metric calibration advice based on historical forecast performance. This advice is injected into forecasting prompts to help models learn from past scoring patterns — for example, adjusting tail coverage or bucket calibration for hazards where the ensemble has historically been over- or under-confident.
- **Regime Change is now assessed separately from triage.** RC assessment runs as a dedicated step before triage, with its own model context. This improves reliability by preventing RC reasoning from competing with triage scoring in a single prompt. RC and triage each run a two-pass pipeline for stability.
- **Per-hazard evidence and scoring.** Different hazard types now receive tailored evidence queries, prompts, and scoring criteria rather than a single generic prompt. Seasonal awareness has been added: hazard scoring adjusts based on whether a hazard is in- or off-season for a given region.
- **Larger, more diverse forecast ensemble.** The SPD ensemble expanded from 4 models across 3 providers to 7 models across 5 providers, adding Kimi K2.5 and DeepSeek Reasoner alongside upgrades to GPT-5.2, Gemini 3, and Claude Sonnet 4.6., to test open-source performance.
- **Track-based question routing.** Track 1 questions (elevated Regime Change) go through the full multi-model ensemble. Track 2 questions (priority but no RC signal) use a lightweight single-model path, reducing cost while maintaining full analytical depth for the questions that need it most. Major change allowing for more time/money per in-dpeth forecast. 
- **Simplified tier system.** The previous three-tier system (quiet / watchlist / priority) has been simplified to two tiers: quiet and priority, with a single threshold.
- **Performance page and score exports.** The dashboard Performance page now includes a dropdown to compare different ensemble methods, displays median alongside average scores, and offers expanded CSV downloads.
- **Prompt and overview versioning.** Prompts and the system overview document are now version-tracked with dated snapshots, viewable on the About page for audit and comparison purposes.

---

## System at a Glance

Fred (also referred to as Pythia in the codebase) is an end-to-end humanitarian forecasting system. It takes raw event data (conflict, disasters, displacement) from three primary sources — ACLED, IDMC, and IFRC GO — supplements that data with structured feeds from humanitarian monitoring organizations (ACAPS, IPC, ReliefWeb), climate forecast services (NOAA/NMME), conflict forecast models (VIEWS, conflictforecast.org), and expert assessments (ICG CrisisWatch), and produces probabilistic forecasts for a set of standardized questions over a 1–6 month horizon. The system is designed for auditability: each run writes a canonical DuckDB database artifact that records inputs, intermediate judgments, model calls, and final outputs.

A key design tension in forecasting is balancing base rates (what usually happens) with out-of-pattern events (what can happen when the world changes). In Fred, this tension is handled explicitly through Horizon Scanner triage, Regime Change scoring, adversarial evidence checks, and — when needed — Hazard Tail Packs that surface trigger evidence.

| Stage | What it does | Primary inputs | Primary outputs |
| --- | --- | --- | --- |
| Resolver (data ingestion) | Ingests and normalizes raw data from external sources into monthly "facts" suitable for forecasting and evaluation. | Connector feeds: ACLED, IDMC, IFRC Montandon (GO API). | facts_deltas, facts_resolved, derived monthly tables; connector status. |
| Web Research (evidence packs) | Fetches grounded, source-cited qualitative evidence to complement quantitative facts, with an explicit recency window. Used primarily by the Horizon Scanner for triage and RC assessment. | Country and hazard-tail queries; web research backend. | Evidence pack objects used by HS triage and RC; sources lists. |
| Structured Data Connectors | Pulls authoritative humanitarian data and conflict/cliamte forecasts from specialist APIs and stores it for prompt injection. Replaces the former research LLM stage. | ReliefWeb, ACAPS (4 datasets), IPC, ACLED political events, ViEWS, ConflictForecast, NOAA | Database tables with structured evidence; formatted prompt blocks for forecasting. |
| Horizon Scanner (HS) triage | Scores each country–hazard pair for elevated risk in the next 1–6 months; detects potential regime shifts. | Resolver facts + evidence packs + structured data + hazard catalog. | hs_runs, hs_country_reports, hs_triage (incl. RC fields). |
| Hazard Tail Packs (conditional) | Runs hazard-specific follow-up evidence retrieval for high RC cases to find triggers and counter-signals. | HS RC flags (Level >= 2), hazard code, forecast window. | hs_hazard_tail_packs (cached per run/country/hazard). |
| Adversarial Evidence Checks (conditional) | Searches for counter-evidence to RC hypotheses — reasons the regime shift might not materialize. | HS RC flags (Level >= 2), country, hazard. | Structured counter-evidence, historical analogs, stabilizing factors, net assessment. |
| Prediction Markets (optional) | Retrieves crowd forecasts from Metaculus, Polymarket, and Manifold Markets that are thematically related to the question. | Question metadata (country, hazard, time horizon). | Prediction market signals embedded in forecasting prompts. |
| Forecasting (SPD v2, ensemble) | Produces a discrete probability distribution (SPD) over outcome bins for each question and month. Track 1 (RC-elevated) uses a full multi-model ensemble; Track 2 (priority, no RC) uses a single-model path. Receives structured data, RC guidance, calibration advice, adversarial checks, and hazard-specific reasoning instructions. | Question + structured data + base-rate features + RC guidance + adversarial check + calibration advice + optional tail pack + track assignment. | forecasts_raw, forecasts_ensemble; llm_calls; scenario artifacts (Track 1 only). |
| Scoring + Calibration | Computes horizon-specific resolutions (ground truth per forecast month), evaluations (Brier, Log, CRPS scores), calibration weights, and calibration advice. | Forecasts + resolved facts (per horizon month); historical scores. | resolutions, scores, calibration_weights, calibration_advice tables; performance dashboard; CSV downloads. |

## 1. Purpose, scope, and forecasting philosophy

Fred is built to support recurring humanitarian decision-making under uncertainty. It converts disparate, noisy real-world observations into probabilistic statements about near-term outcomes. Those statements are not point forecasts. Instead, the system produces distributions over outcome bins, allowing users to reason about risk, tails, and uncertainty rather than a single best guess.

The system assumes that base rates matter: for many humanitarian metrics, the best starting point is what typically happens in a country given its recent history and seasonality. However, base rates are not sufficient. Humanitarian outcomes often exhibit structural breaks driven by conflict escalation, sudden displacement, extreme weather, policy shifts, or compound shocks. A central design goal is therefore to detect and communicate when "business as usual" may not apply, and to help forecasters allocate probability mass to out-of-pattern outcomes when warranted.

Fred is designed as an auditable pipeline. Each stage writes its outputs into a single DuckDB database artifact, and the artifact is treated as the canonical record for that run. This enables reproducibility (rerunning with the same artifact) and forensic debugging (tracing why a forecast moved).

## 2. Core objects and terminology

To understand Fred, it helps to start with the system's core objects: facts, hazards, questions, evidence, and forecasts.

A **fact** is a normalized observation derived from an upstream source (for example, monthly conflict events or fatalities from ACLED, disaster impacts from IFRC GO, or displacement figures from IDMC). Facts are stored with metadata describing provenance and, where possible, a notion of "as-of" time. Facts are later "resolved" into ground truth used for scoring.

A **hazard** is a high-level class of humanitarian risk that Fred tracks consistently across countries. The primary hazard codes are: ACE (armed conflict events and impacts), DI (displacement inflows), DR (drought), FL (flood), HW (heat wave), and TC (tropical cyclone). The system also recognizes additional hazard types including CW (cold wave), EQ (earthquake), VO (volcanic eruption), TS (tsunami), LS (landslide), FIRE (fire), and FI (food insecurity). These codes provide a stable vocabulary that connects HS triage, evidence gathering, question generation, and forecasting.

A **question** is the unit that gets forecast. A question typically represents a country–hazard–metric combination, evaluated across the next six months. The system forecasts each month's outcome using a discrete probability distribution over bins that correspond to meaningful ranges for the metric.

**Structured data** refers to the authoritative humanitarian, climate, and conflict-forecast datasets pulled from specialist APIs (ReliefWeb, ACAPS, IPC, ACLED political events, NMME/NOAA, VIEWS, conflictforecast.org, ICG CrisisWatch). Unlike web-search-derived evidence packs, structured data is deterministic, reproducible, and sourced from organizations with formal data collection methodologies. Structured data is stored in the database and automatically formatted for injection into pipeline prompts.

An **evidence pack** is a grounded research bundle generated via web search, used primarily by the Horizon Scanner for triage and Regime Change assessment. Evidence packs include structural context (slow-changing background) and recent signals (time-bounded observations), each anchored to sources.

A **forecast** is represented as an SPD (a discrete probability distribution) per month. The system often runs multiple models and aggregates them into an ensemble forecast. The ensemble is treated as the system's primary forecast output for dashboarding and evaluation.

## 3. Resolver: data ingestion, normalization, and base-rate foundation

Resolver is the system's quantitative backbone. It ingests data from three active connectors and transforms them into monthly facts suitable for (a) base-rate estimation and (b) later scoring. Resolver is not a forecasting model. It is a data pipeline with strong opinions about consistency and auditability.

Resolver connectors pull from external providers. In a February 2026 refactoring, seven defunct or disused connectors (including EM-DAT, DTM, IPC, ReliefWeb, UNHCR, WFP, and WHO) were removed and the system was consolidated to three active connectors behind a formal protocol: ACLED (conflict event and fatality records), IDMC (internal displacement flows via the Helix API), and IFRC Montandon (natural-hazard people-affected data via the IFRC GO API, replacing the former EM-DAT connector). IFRC GO provides structured impact data across multiple metrics (affected, fatalities, injured, displaced, missing) and maps disaster types to hazard codes automatically using IFRC's own disaster-type classification.

The ingestion process writes fact rows into tables that track what changed (deltas) and what is considered resolved ground truth. In practice, this means there is a "facts_deltas" table that accumulates incremental connector writes, and a "facts_resolved" table used for evaluation, where each country–hazard–month has an authoritative value once it becomes stable. Two legacy tables (emdat_pa and acled_monthly_fatalities) are also consulted as fallbacks when more recent sources lack data for a given month.

Why this architecture matters for forecasting: the deltas table is the freshest view of what the system has most recently ingested; the resolved table provides a stable reference for scoring and calibration. This split helps prevent a common forecasting pitfall: scoring a forecast against data that later gets revised without keeping a record of what the system 'knew' at the time of forecasting. When the system resolves ground truth for scoring, it uses a four-table priority cascade: facts_resolved (highest priority), then facts_deltas, then emdat_pa (for people-affected metrics only), then acled_monthly_fatalities (for fatality metrics only). The cascade short-circuits: once any table returns a value for a country–hazard–month, lower-priority tables are not queried.

Operationally, Resolver is usually run through a GitHub Actions workflow (often called an initial backfill). This workflow can rebuild the database from scratch or incrementally backfill a set number of months. It also supports policies for how to treat '0 rows written' situations, which are important because external sources can occasionally return empty responses due to outages, rate limits, or schema changes. Two important safeguards were added in February 2026: a calendar cutoff prevents resolution of the current (incomplete) month — only fully completed calendar months are eligible for resolution — and a data-driven cutoff prevents resolution of months beyond the latest data actually present in any source table. A stale-resolution purge runs at the start of each resolution pass, deleting any resolution rows that were incorrectly written for future or partial months in earlier pipeline runs, along with their associated scores, and reverting affected questions to active status so they can be re-resolved correctly.

A key limitation — and one that users must internalize — is that these data sources are imperfect proxies for humanitarian impact. They reflect reporting incentives, access constraints, and definitional choices. Fred does not pretend this disappears. Instead, it tries to surface uncertainty explicitly through data-quality notes, model uncertainty, and regime-change flags.

## 4. Structured data connectors: authoritative humanitarian and forecast feeds

A major architectural shift in the March 2026 version replaced the former "Research LLM" stage with direct structured data connectors. Previously, the system used an LLM to generate research briefs by conducting web searches, synthesizing results, and writing a brief. This approach was effective but introduced variability — the same question could produce different evidence depending on which web results the LLM encountered. The new architecture pulls structured data directly from authoritative humanitarian, climate, and conflict-forecast APIs, stores it in the database, and injects it into prompts as pre-formatted text blocks. This makes the evidence pipeline deterministic, reproducible, and anchored to organizations with formal data collection methodologies.

Fred now maintains eleven structured data feeds, organized into four categories: humanitarian monitoring, food security, climate and seasonal forecasts, and conflict forecasts. These feeds are used across the entire pipeline — Horizon Scanner RC assessment, HS triage, and the forecaster — though not every feed appears in every prompt. The table at the end of this section summarizes which feeds are injected where.

### Humanitarian monitoring feeds

**ReliefWeb situation reports.** ReliefWeb, operated by the UN Office for the Coordination of Humanitarian Affairs (OCHA), publishes humanitarian situation reports, assessments, and updates from agencies operating in the field. Fred fetches recent reports for each country (looking back 45 days, up to 15 reports), extracts key metadata (title, source organizations, disaster types, themes, publication date), and stores them in the database. These reports provide on-the-ground operational context that complements quantitative data — for example, a ReliefWeb flash update about sudden flooding or a situation report about a cholera outbreak. ReliefWeb data is used across all pipeline stages and all hazard types.

**ACAPS INFORM Severity Index.** The INFORM Severity Index is a composite score maintained by ACAPS that measures the overall severity of humanitarian crises. It combines indicators of crisis impact, conditions of affected people, and crisis complexity into a single score with trend data (improving, stable, deteriorating). Fred fetches the latest severity scores and trends for each country. This provides a standardized, cross-comparable measure of crisis severity that helps calibrate expectations about where a country sits on the spectrum from stable to acute crisis.

**ACAPS Risk Radar.** Risk Radar is ACAPS's forward-looking risk assessment tool. It identifies countries and crises where humanitarian conditions are expected to deteriorate in the near term, along with specific triggers (the events or conditions most likely to cause deterioration). Because Risk Radar is explicitly forward-looking and trigger-oriented, it is particularly valuable for forecasting — it provides expert assessments of what could go wrong, rather than just what has already happened.

**ACAPS Daily Monitoring.** ACAPS analysts produce daily curated updates tracking significant developments across active crises. Fred fetches recent monitoring entries for each country, providing a stream of analyst-vetted situational awareness. These updates fill the gap between the relatively slow-moving Resolver facts (which are monthly aggregates) and the real-time developments that can affect near-term forecasts.

**ACAPS Humanitarian Access.** Humanitarian access scores measure the degree to which aid organizations can reach affected populations. Access constraints (insecurity, bureaucratic restrictions, physical barriers) are a critical contextual factor for humanitarian outcomes — a country with deteriorating access is likely to see worse outcomes even if the underlying hazard conditions are unchanged. Fred fetches access constraint scores and injects them into HS triage prompts (but not into the forecaster SPD prompt).

**ACLED Political Events.** In addition to the monthly aggregate conflict data already ingested through the Resolver, the system pulls event-level political data from ACLED. This includes specific incidents such as strategic developments, protests, riots, and political violence events. Unlike the Resolver's monthly summaries (which provide base-rate counts), political events provide narrative detail about what is actually happening — for example, a protest movement gaining momentum, a military coup, or a ceasefire announcement. Political events are used selectively: they are injected into prompts only for armed conflict (ACE) and displacement (DI) hazards, where event-level context is most relevant.

### Food security feed

**IPC Food Security Phase Classifications.** The Integrated Food Security Phase Classification (IPC) is the global standard for classifying food insecurity severity. IPC classifies affected populations into five phases: Phase 1 (Minimal), Phase 2 (Stressed), Phase 3 (Crisis), Phase 4 (Emergency), and Phase 5 (Famine). Phase 3 and above — "Crisis or worse" — is the primary metric for humanitarian need. Fred fetches both current and projected IPC classifications for each country, including population counts per phase. This is particularly valuable for drought and food-insecurity hazards, but also provides important context for conflict and displacement, since food insecurity often compounds other crises.

### Climate and seasonal forecast feeds

**NMME Seasonal Climate Forecasts.** The North American Multi-Model Ensemble (NMME) provides seasonal temperature and precipitation anomaly forecasts from the NOAA Climate Prediction Center. Fred downloads the ensemble mean forecasts, computes country-level area-weighted averages, and stores the results in the database. For each country, the system provides anomalies for the next seven months, expressed as standard deviations from climatology, along with derived tercile categories (above normal, below normal, near normal). These forecasts are used for climate-sensitive hazards — drought, flood, heatwave, and tropical cyclone — giving the pipeline concrete quantitative signals about expected climate conditions rather than relying solely on qualitative web-search evidence. The data is refreshed monthly via a scheduled pipeline, typically around the 10th of each month when new NMME data becomes available.

### Conflict forecast and assessment feeds

**VIEWS (Uppsala/PRIO).** VIEWS is a machine-learning-based conflict forecasting system developed at Uppsala University and the Peace Research Institute Oslo. It provides country-month predictions of state-based conflict fatalities and the probability of at least 25 battle-related deaths at 1–6 month lead times. VIEWS is generally stronger at capturing trends and baseline levels of conflict, though weaker at predicting sudden onset. Fred stores VIEWS predictions in the database and includes staleness warnings when the data is more than 45 days old.

**conflictforecast.org (Mueller/Rauh).** This project provides news-based armed conflict risk scores derived from media coverage patterns. It offers risk scores at 3-month and 12-month horizons, plus a violence intensity outlook at 3 months. Because it is driven by news signals rather than historical conflict patterns, conflictforecast.org is generally better than VIEWS at detecting shifts and escalation signals. Where the two quantitative sources disagree, the LLM is instructed to note the disagreement and reason about why.

**ICG CrisisWatch (International Crisis Group).** CrisisWatch is ICG's monthly conflict monitoring bulletin. The system fetches per-country directional assessments (Deteriorated, Improved, Unchanged) and the monthly "On the Horizon" feature, which highlights roughly three conflict risks and one resolution opportunity expected in the next three to six months. ICG is highly selective about what it flags, so countries appearing in "On the Horizon" receive a prominent note in their RC assessment. Unlike the other structured feeds (which are stored in the database), CrisisWatch assessments are retrieved via web research at prompt time and injected directly.

All three conflict forecast feeds are used exclusively for armed conflict (ACE) hazards. VIEWS and conflictforecast.org predictions are stored in the database and injected alongside the ACLED retrospective base rates as structured quantitative anchors. ICG CrisisWatch is injected into the RC assessment only.

### Where structured data is used: summary table

Not every feed appears in every prompt. The following table shows which data sources are available at each pipeline stage. "All" means all hazard types; specific codes mean the feed is only injected for those hazards.

| Data source | RC assessment | HS triage | Forecaster SPD |
| --- | --- | --- | --- |
| Resolver features (base rates) | All | All | All (via HS triage output) |
| Web research evidence packs | All | All | — |
| ReliefWeb situation reports | All | All | All |
| ACAPS INFORM Severity | ACE | All | All |
| ACAPS Risk Radar | ACE | All | All |
| ACAPS Daily Monitoring | ACE | All | All |
| ACAPS Humanitarian Access | — | All | — |
| ACLED summary (monthly aggregates) | ACE | ACE | — |
| ACLED political events | ACE | ACE | ACE, DI |
| IPC phases | DR | All | All |
| NMME seasonal forecasts | DR, FL, HW, TC | DR, FL, HW, TC | All (via research data) |
| VIEWS conflict forecasts | ACE | ACE | — |
| conflictforecast.org risk scores | ACE | ACE | — |
| ICG CrisisWatch | ACE | — | — |
| RC result (from prior step) | — | All | All |
| Adversarial evidence check | — | — | RC Level 2+ only |
| Calibration advice | — | — | All |
| Hazard-specific reasoning block | — | — | All |
| Prediction market signals | — | — | All (when available) |
| Hazard tail pack | — | — | RC-elevated only |

All structured data stored in the database can be inspected after the fact, supporting the system's auditability goals: when reviewing a historical forecast, operators can see exactly what data was available and injected into the prompt at the time of forecasting.

## 5. Web research evidence packs: grounded qualitative signals

Evidence packs provide qualitative context that does not fit neatly into the structured datasets. While structured data connectors (Section 4) handle the bulk of evidence for the forecaster, evidence packs remain the primary evidence mechanism for the Horizon Scanner's triage and Regime Change assessment.

An evidence pack has two primary components. Structural context provides brief background drivers that change slowly (governance, exposure, long-running conflict dynamics). Recent signals provide a set of time-bounded observations intended to affect the 1–6 month horizon. Recent signals should be testable: they should point to observable developments rather than general commentary.

Fred's evidence packs are designed to be grounded: they carry explicit source URLs and are generated within a recency window (commonly 120 days). This is essential for credibility. When evidence is thin or contradictory, the system is instructed to say so rather than fabricate certainty.

The Horizon Scanner generates two separate evidence packs per country — one for RC assessment (optimized to detect regime-change signals) and one for triage (optimized for operational picture). The RC evidence pack explicitly requests recent signals framed in three types: TAIL-UP triggers (leading indicators of an out-of-pattern escalation or worsening), TAIL-DOWN dampeners (counter-signals suggesting mitigation or de-escalation), and BASELINE continuation signals. This design forces the system to consider why outcomes might depart from base rates, and also why they might not.

Evidence packs are not used in the forecaster SPD prompt. The forecaster instead receives structured data from the connectors described in Section 4, which is more deterministic and reproducible.

## 6. Horizon Scanner (HS): triage-first risk screening

Horizon Scanner is the system's front-end risk screening stage. Its job is not to forecast numbers. Instead, it triages which country–hazard combinations merit deeper attention and provides structured context for downstream forecasting.

HS is typically run on a schedule (for example, monthly) and writes into the canonical DuckDB. Two tables are important for understanding HS outputs. The hs_country_reports table stores the evidence pack used for a country (including sources and markdown renderings). The hs_triage table stores one row per country–hazard, including the triage score and other structured fields.

### Multi-stage pipeline

HS runs as a multi-stage pipeline rather than a single prompt. First, Regime Change assessment runs as a dedicated step with its own model context (see Section 7). The RC results are then passed into the triage step, which scores the country–hazard pair with the RC assessment already available. This separation improves reliability by preventing RC reasoning from competing with triage scoring in a single, overloaded prompt.

### Data sources injected into HS triage

The triage step receives the following data for each country–hazard pair:

- **Resolver features** — historical base-rate data from the facts_resolved table (all hazards).
- **RC result** — the output of the prior RC assessment step, including likelihood, magnitude, direction, window, and rationale (all hazards).
- **Web research evidence pack** — a grounded evidence bundle from web search (all hazards).
- **Season context** — a human-friendly description of the current season for the region (all hazards).
- **ReliefWeb situation reports** — recent humanitarian reports from OCHA (all hazards).
- **ACAPS INFORM Severity** — crisis severity scores and trends (all hazards).
- **ACAPS Risk Radar** — forward-looking risk with triggers (all hazards).
- **ACAPS Daily Monitoring** — analyst-curated situational updates (all hazards).
- **ACAPS Humanitarian Access** — access constraint scores (all hazards).
- **IPC phases** — food security phase classifications (all hazards).
- **NMME seasonal forecasts** — temperature and precipitation anomalies (climate hazards: DR, FL, HW, TC).
- **ACLED summary** — monthly aggregate conflict data (ACE only).
- **ACLED political events** — event-level political incidents (ACE only).
- **VIEWS + conflictforecast.org** — quantitative conflict predictions (ACE only).

### Per-hazard prompts and seasonal filtering

Both the RC and triage steps use per-hazard prompts and evidence queries. Rather than a single generic prompt for all hazard types, the system now constructs tailored prompts for each hazard category. For example, drought evidence retrieval emphasizes precipitation anomalies, food security signals, and agricultural indicators, while conflict evidence focuses on armed-group dynamics, ceasefire status, and political escalation. This hazard-specific prompting produces more relevant evidence and more calibrated scores.

Seasonal filtering has also been integrated into the HS pipeline. The system is aware of whether a hazard is in-season or off-season for a given region, and scoring adjusts accordingly. For instance, tropical cyclone risk is scored lower during months outside the historical cyclone season, reducing false positives from evergreen background risk language.

### Stabilization and tier assignment

To reduce run-to-run variance, HS uses a stabilization pattern: both RC and triage run a two-pass pipeline and average numeric outputs such as triage_score and RC likelihood. The second triage pass uses a different model (currently GPT-5-mini) to introduce diversity. This is not a deep statistical ensemble; it is a practical technique that improves consistency for dashboard users.

HS computes a tier label — quiet or priority — based on a single configurable threshold (default 0.50). Scores at or above the threshold are "priority"; scores below are "quiet". Tiers are intended for interpretability and prioritization, while the numeric triage_score is the primary signal that downstream components can use.

## 7. Regime Change (RC): making out-of-pattern risk explicit

Regime Change is Fred's mechanism for flagging when base rates may be misleading in the forecast window. The intent is not to predict black swans on demand. Rather, it is to detect credible situations in which the generating process for a hazard may shift — for example, when a conflict enters a new phase, when a border policy changes, or when seasonal forecasts indicate unusually severe hazards.

RC is represented with three conceptual dimensions: likelihood, direction, and magnitude. Likelihood captures the chance that a regime shift occurs during the next 1–6 months. Direction indicates whether the shift would push outcomes up (worse), down (improving), mixed, or unclear relative to baseline. Magnitude captures how large the shift could be if it occurs.

### Data sources injected into RC assessment

RC assessment runs as a dedicated LLM pipeline step before triage. It has its own prompt context, which allows the model to focus entirely on detecting structural breaks. The RC step receives the following data:

- **Resolver features** — historical base-rate data (all hazards).
- **Web research evidence pack** — a grounded evidence bundle optimized for regime-change signal detection (all hazards).
- **ReliefWeb situation reports** — recent humanitarian reports (all hazards).
- **NMME seasonal forecasts** — temperature and precipitation anomalies (climate hazards: DR, FL, HW, TC).
- **ACLED summary** — monthly aggregate conflict data (ACE only).
- **ACLED political events** — event-level political incidents (ACE only).
- **ACAPS INFORM Severity** — crisis severity scores (ACE only).
- **ACAPS Risk Radar** — forward-looking risk assessments (ACE only).
- **ACAPS Daily Monitoring** — analyst-curated updates (ACE only).
- **VIEWS + conflictforecast.org** — quantitative conflict predictions (ACE only).
- **ICG CrisisWatch "On the Horizon"** — ICG's forward-looking conflict flags (ACE only). Countries flagged by ICG receive a prominent note, as ICG is highly selective about what it flags.
- **IPC phases** — food security data (DR only).

The RC step uses per-hazard prompts: each hazard type receives tailored instructions and evidence that reflect its specific escalation dynamics. For example, the RC prompt for armed conflict emphasizes ceasefire breakdowns, new armed-group mobilization, and political crisis triggers, while the RC prompt for flood emphasizes upstream dam status, rainfall anomalies, and infrastructure degradation.

### Scoring and levels

Fred stores RC results as numeric and categorical fields in hs_triage. A derived RC score is computed as likelihood multiplied by magnitude. This provides a single 'how much should I care' number that suppresses cases that are high-likelihood but low-impact, or high-impact but highly speculative.

RC scores are mapped into discrete levels used for UI and downstream gating. Level 0 is base-rate normal, Level 1 is a watch state, Level 2 indicates elevated regime-change plausibility, and Level 3 indicates a strong out-of-pattern hypothesis. The RC scoring was calibrated in February 2026 to reduce false positives. The Level 1 likelihood threshold is 0.45, and the Level 1 composite-score threshold is 0.25, so that mildly elevated signals do not trigger a watch flag. The triage prompt includes explicit calibration guidance: it distinguishes between high triage scores (which capture overall risk, including chronic situations) and regime-change scores (which should only capture departures from established patterns). The prompt provides an expected distribution — roughly 80% of country–hazard pairs should have likelihood of 0.10 or below — and examples of what is and is not a regime change. A run-level distribution check warns operators when the fraction of elevated flags exceeds configurable thresholds (e.g., more than 15% at Level 2).

### Behavioral implications

Critically, RC does not merely decorate the dashboard. When RC is elevated, the system can override the 'need_full_spd' gating so that a full probabilistic forecast is produced even if the triage tier would otherwise be quiet. This is how Fred avoids an important failure mode: missing a structural break because it does not look like a high base-rate risk.

RC is surfaced to users across the dashboard: on country maps and lists as a 'highest RC' indicator, on triage tables as probability/direction/magnitude/score columns, and on question drilldown pages as summary boxes. This makes the system's out-of-pattern reasoning visible rather than hidden inside prompts.

## 8. Adversarial evidence checks: testing the RC hypothesis

A new component added in March 2026, adversarial evidence checks provide a structured "devil's advocate" mechanism for high-risk forecasts. When Regime Change is elevated to Level 2 or above, the system automatically runs targeted web searches looking for counter-evidence — reasons the predicted regime shift might not materialize.

The adversarial check runs 2–3 targeted searches per elevated hazard, using hazard-specific search vocabularies. For example, if RC is elevated for armed conflict with an "up" direction (predicting escalation), the adversarial search looks for peace talks, ceasefire progress, and diplomatic resolution signals. If RC is elevated for drought, it searches for revised rainfall forecasts and improved water infrastructure.

The results are synthesized into a structured output with four components:

- **Counter-evidence**: specific claims with sources and relevance ratings that argue against the RC hypothesis.
- **Historical analogs**: past situations that appeared similar but did not result in the predicted regime shift, providing calibration.
- **Stabilizing factors**: structural or institutional factors that could prevent the shift from materializing (e.g., strong institutions, international engagement, resource buffers).
- **Net assessment**: an overall judgment of how strong the counter-evidence is (strong counter, moderate, weak counter, or inconclusive).

The adversarial check is stored in the database and injected into the SPD forecasting prompt. This ensures the forecasting models consider both the evidence supporting a regime shift and the evidence against it, reducing the risk of one-sided reasoning. The adversarial check is only generated for RC Level 2+ cases, keeping costs bounded while focusing counter-evidence where it matters most — on the forecasts most likely to deviate from base rates.

## 9. Hazard Tail Packs (HTP): hazard-specific trigger evidence (Phase A/B)

Hazard Tail Packs are a second-stage evidence mechanism designed to answer a simple question: if RC is high, what are the concrete leading indicators and counter-indicators for this hazard in this country, in this window?

A key lesson in operational forecasting is that generic 'risk outlook' research tends to reinforce base-rate thinking. Tail forecasting requires trigger detection: signs that escalation or suppression mechanisms are activating. Tail packs exist to retrieve that trigger evidence in a structured way.

Tail packs are generated conditionally. The system generates tail packs only for hazards with RC Level 2 or 3, and it caps generation to at most two hazards per country per HS run (by RC score). This prevents the system from becoming noisy and keeps costs bounded. Tail packs are cached per run, country, and hazard in the database so reruns do not repeatedly hit the web.

Tail pack signals are intentionally formatted with explicit prefixes. TRIGGER bullets are leading indicators supporting the regime-shift hypothesis; DAMPENER bullets are counter-signals suggesting the shift may not materialize; BASELINE bullets support continuation. Each bullet also carries an intended time window (for example, month 3–4) and a direction tag.

In Phase A, tail packs are injected into the forecasting evidence so the forecaster can confirm, downgrade, or rebut the RC hypothesis with sources. In Phase B, tail packs are also injected into the SPD forecasting prompts for RC-elevated hazards, but with strict length limits: at most 12 bullets are included to prevent prompt bloat.

## 10. Question bank: what gets forecast and why

After HS triage, the system maintains a question set that defines what gets forecast. Conceptually, a question ties together: a country, a hazard, a metric, and a forecast horizon.

Questions are intended to be stable and comparable over time. They encode the metric to be forecast (for example, a displacement inflow proxy or conflict impact proxy) and the discrete bins that define the SPD outcomes. This standardization is what enables scoring and calibration. Without it, each run would effectively forecast a different target.

The question set is typically refreshed using HS triage outputs. Priority-tier hazards are included by default, but as described earlier, elevated RC (level > 0) can force inclusion even for otherwise quiet hazards. This ensures the system pays attention to plausible regime breaks. Each question is assigned a track (see Section 12) based on its RC status: Track 1 for RC-elevated questions, Track 2 for priority questions without RC. The track determines the depth of the forecasting pipeline that the question passes through. A critical change in February 2026 made the questions table append-only: when a new HS run generates a question with the same ID as an earlier run, the original metadata (window start date, target month, wording) is preserved rather than overwritten. This prevents downstream resolution and scoring from breaking when the HS runs again months later. As an additional safeguard, the system can derive the true forecast window from the LLM call log (which is always append-only) rather than relying solely on the questions table, falling back through a three-tier priority: (1) the earliest LLM call timestamp for that question, (2) the questions table's window_start_date, (3) a derivation from the target_month.

## 11. Prediction markets: crowd forecast signals

Fred optionally retrieves crowd forecasts from three prediction market platforms: Metaculus, Polymarket, and Manifold Markets. The system uses search queries (generated by an LLM) and relevance filtering to find prediction markets that are thematically related to a given forecasting question — for example, a market about conflict escalation in a specific country might be relevant to an ACE question for that country.

Prediction market signals are presented as contextual evidence, not authoritative data. They are embedded in forecasting prompts alongside structured data and other evidence. When the dedicated prediction market retriever returns no relevant markets, the system falls back to a lightweight Manifold-only search as a last resort.

The rationale for including prediction markets is that they aggregate information from diverse participants with financial incentives to be accurate. However, market coverage is uneven — many humanitarian situations have no active prediction markets — and market liquidity varies widely. The system applies minimum thresholds (for example, minimum trading volume on Polymarket, minimum liquidity on Manifold) to filter out illiquid or unreliable markets.

## 12. Forecasting: SPD v2, ensembles, and structured evidence injection

Fred's primary forecast output is an SPD: a discrete probability distribution over outcome bins for each month in the horizon. This representation is more operationally useful than a point estimate because it preserves uncertainty and supports expected-value computations and tail-risk analysis. SPD data is stored in dual format: a numeric format (month index and bucket index with a probability value) for computational efficiency, and a labeled format (bin labels like "<10k" with a probability value) for human readability. Both representations are written simultaneously so that scoring and display can each use whichever is more convenient.

### Evidence injection

In the current architecture, the forecasting prompt receives a rich set of structured inputs rather than relying on LLM-generated research briefs. The forecaster does not use web-search evidence packs (those are reserved for the Horizon Scanner). Instead, it receives data from the structured connectors described in Section 4, along with several forecaster-specific inputs. The complete set of data injected into each SPD prompt is:

**Structured data from humanitarian APIs (loaded from the database for each country):**
- **ReliefWeb reports** — recent humanitarian situation reports (all hazards).
- **ACAPS INFORM Severity** — crisis severity scores and trend data (all hazards).
- **ACAPS Risk Radar** — forward-looking risk assessments with triggers (all hazards, when available).
- **ACAPS Daily Monitoring** — analyst-curated daily updates (all hazards, when available).
- **IPC phase data** — food security classifications with population counts per phase (all hazards, when available).
- **ACLED political events** — event-level political incidents (ACE and DI hazards only).

**Climate and forecast data (loaded from the database):**
- **NMME seasonal outlook** — temperature and precipitation anomalies for the next seven months (all hazards, via research data).

**HS-derived context (from the prior Horizon Scanner run):**
- **HS triage output** — triage score, tier, RC fields (likelihood, magnitude, direction, score, level), and other structured assessments. This is the forecaster's primary link to the Horizon Scanner's judgment.
- **Adversarial evidence check** — counter-evidence to the RC hypothesis, including historical analogs and stabilizing factors (RC Level 2+ only).
- **Hazard tail pack** — trigger, dampener, and baseline signals from the second-stage evidence retrieval (RC-elevated hazards only, capped at 12 bullets).

**Forecaster-specific inputs:**
- **Resolver history summary** — historical distribution data (source, recent level, trend, data quality) from the Resolver.
- **Calibration advice** — per-hazard/metric guidance text generated from historical scoring performance (all hazards).
- **Hazard-specific reasoning block** — tailored forecasting instructions for the hazard type (all hazards).
- **Prediction market signals** — crowd forecasts from Metaculus, Polymarket, and Manifold (all hazards, when available).

This evidence is presented to the forecasting models as structured text blocks, each clearly labeled with its source. The forecaster sees, for example, the latest ReliefWeb reports about the country, the ACAPS severity score and trend, the IPC phase populations, the NMME climate outlook, and any adversarial counter-evidence — all formatted for readability rather than raw data.

Note that some data sources available to the Horizon Scanner are not passed through to the forecaster. ACAPS Humanitarian Access, ACLED monthly summaries, VIEWS/conflictforecast.org quantitative predictions, and ICG CrisisWatch are used during HS triage and RC assessment but are not directly injected into the SPD prompt. Their influence reaches the forecaster indirectly through the HS triage output and RC fields.

### Hazard-specific reasoning guidance

Each hazard type now receives tailored reasoning instructions in the forecasting prompt, replacing the earlier generic guidance. These instructions reflect the fundamentally different generating processes behind different hazards. For example:

- **Armed conflict (fatalities)**: Guidance on tail-risk coverage for conflict escalation, bucket calibration for fatality counts, how to weigh ceasefire dynamics, and how different conflict types (state-based, one-sided, non-state) produce different fatality patterns.
- **Flood**: Guidance on seasonal patterns, upstream conditions, infrastructure resilience, and how flood impacts can spike dramatically in a single event.
- **Drought**: Guidance on slow-onset dynamics, cumulative effects, food security linkages, and how drought impacts often peak months after conditions begin deteriorating.
- **Displacement**: Guidance on push/pull factors, compound crisis effects, and how displacement flows can be sudden (conflict-driven) or gradual (climate-driven).

This hazard-specific reasoning helps models apply appropriate mental models rather than treating all forecasts identically.

### RC guidance and accountability

SPD v2 includes a dedicated Regime Change guidance block. When RC is elevated, the forecaster is instructed that the base rate is less reliable and must either widen the distribution appropriately or rebut the regime-shift hypothesis.

To prevent hand-waving, SPD v2 requires accountability in the human-facing explanation: the forecaster must include a sentence beginning with "RC:" that states what was flagged, whether it was accepted, and how it affected the forecast. This makes the system's interaction between RC and the distribution explicit.

### Track-based routing

Not all questions require the same depth of forecasting. Fred uses a track system to route questions through different pipeline configurations based on their risk profile:

- **Track 1** questions are those where Regime Change is elevated (RC level > 0). These go through the full multi-model ensemble, which currently includes 7 models across 5 providers: GPT-5.2 and GPT-5-mini (OpenAI), Claude Sonnet 4.6 (Anthropic), Gemini 3.1 Pro and Gemini 3 Flash (Google), Kimi K2.5 (Moonshot), and DeepSeek Reasoner (DeepSeek). Track 1 questions also receive scenario narratives (see Section 13). The rationale is that RC-elevated questions represent the highest-uncertainty situations where model diversity matters most.
- **Track 2** questions are priority-tier questions with no elevated RC signal. These use a lightweight single-model path (currently Gemini Flash) that is faster and cheaper. Scenario generation is skipped. Track 2 reflects a practical judgment: when base rates are expected to hold, a single capable model produces adequate forecasts without the cost of running the full ensemble.

The track field is persisted in the database alongside triage and question records, so downstream analysis can distinguish how each question was processed.

### Ensemble aggregation

For Track 1 questions, the system aggregates member SPDs into named ensemble forecasts. The default named ensembles include ensemble_mean (simple average of member SPDs) and ensemble_bayesmc (a Bayesian Monte Carlo aggregation). Named ensembles are tracked as explicit model names in the database, allowing the dashboard and scoring system to compare ensemble methods against individual models. The ensemble approach is a pragmatic response to model idiosyncrasies: it tends to reduce the chance that a single model's blind spot dominates a forecast.

### Self-search

The system supports a narrow form of self-search: when evidence is insufficient, the model may request a single additional query. This is tightly bounded to prevent runaway cost or unpredictable behavior.

## 13. Scenarios, narratives, and human interpretability

In addition to numeric forecasts, Fred can produce scenario narratives for Track 1 questions (those with elevated RC). Scenarios are not treated as forecasts in their own right. They are interpretability artifacts meant to summarize plausible pathways that could lead to different bins of the SPD. Scenario generation is skipped for Track 2 questions, where the base rate is expected to hold and the additional interpretive cost is not warranted.

Scenario narratives are particularly useful when RC is elevated, because they help users understand what a regime shift would look like operationally. However, narratives are also easy to misuse. Fred therefore keeps scenarios optional and separates them from the numeric forecasting artifacts used for scoring.

## 14. Scoring, calibration, and evaluation loops

A forecasting system is only as good as its evaluation loop. Fred includes workflows to compute resolutions (ground truth), compute forecast scores, assess calibration, and — in the latest version — generate calibration advice that feeds back into the forecasting prompts.

### Horizon-specific scoring

A critical correctness fix in February 2026 changed scoring to be horizon-specific: each of the six forecast months is now scored against its own calendar month's ground truth, rather than all six being scored against a single target month. For example, if a question's forecast window starts in September, month 1 is scored against September data, month 2 against October data, and so on through month 6 against February data. The resolutions table stores one row per (question, horizon month) pair, with the resolved ground-truth value and the calendar month it corresponds to.

### Scoring rules

Three proper scoring rules are computed for each scored forecast: the Brier score (sum of squared differences between predicted probabilities and the one-hot truth vector; lower is better), the log score (negative log of the probability assigned to the true bin; heavily penalizes confident wrong predictions), and a CRPS-like score (compares the cumulative distribution against a step function at the true bin). All three metrics are stored per (question, horizon, model) and are aggregated into averages and medians on the dashboard.

### Calibration weights

Calibration weights are computed per model, per hazard, and per metric. The system uses an exponential half-life weighting scheme (with a 12-month half-life) so that more recent forecast performance is weighted more heavily than older results. A minimum number of scored questions (currently 20) is required before calibration weights are considered valid. These weights are used by the Bayesian Monte Carlo ensemble aggregation to give more influence to models with better historical performance on similar questions.

### Calibration advice

New in March 2026, the system generates human-readable calibration advice per hazard and metric. This advice is based on analysis of historical forecast performance — for example, it might note that the ensemble has historically assigned too little probability mass to the highest outcome bins for armed conflict fatalities, or that drought forecasts tend to be overconfident in months 4–6. Calibration advice is stored in the database and injected into forecasting prompts, creating a feedback loop where past scoring performance directly informs future forecasts. For hazards or metrics where insufficient scored data exists, seed advice (based on domain knowledge) is used until enough performance data accumulates.

### Workflow ordering

The calibration workflow chain runs in a specific order: Resolver backfill, then Compute Resolutions, then Compute Scores, then Calibration (including advice generation). This ordering was fixed in February 2026 to prevent scores from running before resolutions were available.

## 15. The dashboard: how users consume outputs

Fred's dashboard is designed for two audiences: forecasters (who need to inspect evidence and produce forecasts) and decision-makers (who need a summary of risk and tail conditions).

Key dashboard pages include: the Forecast Index (country-level rollups and map), the Countries list and drilldowns, HS Triage, Forecast lists and drilldowns, Resolver (data inspection), Costs, Performance, Downloads, and About.

RC is a first-class UI concept. Countries with Level 1–3 RC flags are highlighted, and questions display RC probability/direction/magnitude/score so users can see when the system believes a base-rate deviation is plausible. Exports include RC fields so that analysis outside the dashboard preserves the same context.

The Resolver page exists because the system treats data provenance as a primary concern. Users can inspect the underlying facts feeding the forecasts and see connector status, including last updated timestamps.

The About page includes versioned snapshots of both the system's prompts and this system overview document. Users can select from a dropdown of dated versions to see how the system's documented behavior has evolved over time. This supports auditability: when investigating a historical forecast, operators can review the prompt and system configuration that were active at the time.

The Performance page surfaces forecast evaluation in the dashboard. It displays five KPI cards (scored questions, hazards, models, HS runs, and ensemble Brier score) and supports four views: Total (one row per model), By Hazard (scores broken down by hazard code), By Run (scores per HS run over time), and By Model. Users can filter by metric (People Affected or Fatalities). An ensemble selector dropdown lets users switch between named ensemble methods (e.g., ensemble_mean vs. ensemble_bayesmc) in the By Hazard and By Run views, enabling direct comparison of aggregation strategies. Tables show both average and median values for all three scoring rules (Brier, Log, CRPS); median scores are particularly useful because they are less sensitive to outliers from early-run or data-sparse questions.

The Downloads page was expanded to include performance score exports: per-question CSV files for each named ensemble (including the full 6-month x 5-bin forecast grid, expected impact values, resolution values, and scores), a model-level summary CSV with average, median, min, and max scores per model and hazard, and a rationale export that downloads the human-readable explanations that LLM models provide alongside their probabilistic forecasts, filtered by hazard code.

## 16. Operations: canonical DB artifacts, workflows, and debugging

Fred is typically operated through GitHub Actions. Workflows download a canonical database artifact, apply new computations, and re-upload an updated artifact. A concurrency lock prevents multiple workflows from mutating the canonical DB at the same time. Model configuration is centralized in a single ensemble list, where each entry specifies a provider and model identifier (for example, OpenAI's GPT-5.2 or Anthropic's Claude Sonnet 4.6). To change the models used in production, operators edit this list — no code changes are needed. Purpose-specific overrides (for example, a cheaper model for scenario writing or an HS fallback model) are also configured in the same file. Model cost rates (input and output cost per 1,000 tokens) are stored separately so that costs can be updated by editing data rather than code.

A signature guardrail is used to ensure the database contains required tables before being treated as canonical. This protects against partial artifacts and prevents downstream stages from running on an incomplete database.

Debugging is treated as an operational requirement. Runs log LLM calls, costs, evidence pack diagnostics, and the reasons why certain questions were included or skipped. A comprehensive debug bundle can be generated that includes triage summaries, research briefs, forecast details, and scoring results for a given run. This makes it feasible to answer "why did the model think this?" after the fact.

## 17. Practical guidance for forecasters

The system is designed to support, not replace, human judgment. In practice, forecasters should treat HS triage as a prioritization and hypothesis-generation tool. The most productive workflow is: (1) inspect HS triage and RC flags; (2) review the structured data feeds (ReliefWeb reports, ACAPS assessments, IPC phases) and tail packs for high-RC hazards; (3) check the adversarial evidence to understand counter-arguments; (4) decide what would move the distribution; (5) ensure the SPD expresses both the baseline and the plausible tail mechanisms; and (6) write explanations that are accountable and falsifiable.

When RC is elevated, the system is deliberately asking you to think in counterfactual terms: what would it take for outcomes to be much worse (or much better) than the recent base rate? If you cannot find credible triggers, the correct response is not to force a tail forecast. It is to document uncertainty and, when appropriate, rebut the RC hypothesis with sources.

Finally, remember that humanitarian metrics are mediated by reporting. A sharp change in recorded values can reflect a real change, a reporting shift, or both. Use data quality notes and cross-source triangulation to avoid overconfident narratives.

## Appendix: Glossary

- **Base rate**: The typical distribution of outcomes for a metric given historical data and seasonality. In Fred, base-rate information is primarily derived from Resolver facts.
- **SPD (discrete probability distribution)**: A set of probabilities over discrete outcome bins. Fred forecasts SPDs rather than point estimates.
- **Hazard**: A standardized category of risk (ACE, DI, DR, FL, HW, TC, etc.) that organizes evidence and forecasting targets.
- **Regime Change (RC)**: A structured estimate that the generating process for a hazard may shift during the forecast window, making base rates less reliable.
- **Hazard Tail Pack**: A hazard-specific evidence pack generated for elevated RC hazards, containing trigger and dampener signals scoped to a forecast window.
- **Adversarial evidence check**: A structured counter-evidence search run for RC Level 2+ forecasts. Produces counter-evidence, historical analogs, stabilizing factors, and a net assessment to ensure the forecaster considers reasons the regime shift might not materialize.
- **Structured data connectors**: Modules that pull authoritative humanitarian, climate, and conflict-forecast data from specialist APIs (ReliefWeb, ACAPS, IPC, ACLED, NMME, VIEWS, conflictforecast.org, ICG CrisisWatch) and store it in the database for prompt injection. Replaced the former Research LLM stage.
- **Canonical database artifact**: The DuckDB file produced by a run and treated as the authoritative record for that run's inputs, intermediate artifacts, and outputs.
- **IFRC Montandon**: The IFRC GO API connector that replaced EM-DAT as the primary natural-hazard people-affected data source. Provides structured impact data with automatic disaster-type classification across 18 hazard types.
- **Calendar cutoff**: A safeguard that prevents resolution of ground truth for the current (incomplete) month. Only fully completed calendar months are eligible for resolution.
- **Performance page**: A dashboard page that surfaces forecast evaluation metrics (Brier, Log, CRPS scores) with views by total, hazard, run, and model, including support for comparing named ensemble methods.
- **Named ensemble**: An ensemble method identified by an explicit name (e.g., ensemble_mean, ensemble_bayesmc) rather than the legacy unnamed ensemble. Named ensembles are tracked separately in the database and can be compared on the Performance page.
- **Brier score**: A proper scoring rule that measures the mean squared error between the predicted probability distribution and the actual outcome. Range 0 (perfect) to 1. Used alongside log score and CRPS in Fred's evaluation pipeline.
- **Rationale export**: A downloadable CSV of the human-readable explanations that LLM models provide alongside their probabilistic forecasts, filterable by hazard code and model.
- **Track 1 / Track 2**: The routing classification for questions in the forecasting pipeline. Track 1 questions (RC level > 0) receive the full multi-model ensemble and scenario generation. Track 2 questions (priority tier, no elevated RC) use a lightweight single-model path.
- **Per-hazard prompts**: Tailored evidence queries and scoring instructions constructed for each hazard category (conflict, drought, flood, etc.), replacing the earlier generic prompt approach.
- **Hazard-specific reasoning guidance**: Tailored forecasting instructions injected into SPD prompts for each hazard type. Provides hazard-specific advice on tail coverage, bucket calibration, escalation dynamics, and generating processes.
- **Seasonal filtering**: A mechanism that adjusts HS triage and RC scoring based on whether a hazard is in-season or off-season for a given region, reducing false positives during periods when a hazard is climatologically unlikely.
- **Conflict forecasts**: External quantitative predictions for armed conflict, sourced from VIEWS (ML-based fatality predictions) and conflictforecast.org (news-based risk scores). Injected into ACE prompts alongside ACLED base rates.
- **ICG CrisisWatch**: International Crisis Group's monthly conflict monitoring bulletin. Per-country directional assessments and forward-looking "On the Horizon" flags are fetched via web research at prompt time for armed conflict assessments.
- **ACAPS INFORM Severity**: A composite index measuring overall humanitarian crisis severity, maintained by ACAPS. Includes trend data (improving, stable, deteriorating).
- **ACAPS Risk Radar**: A forward-looking risk assessment tool maintained by ACAPS that identifies countries where humanitarian conditions are expected to deteriorate, along with specific triggers.
- **IPC phases**: The Integrated Food Security Phase Classification, a global standard that classifies food insecurity from Phase 1 (Minimal) to Phase 5 (Famine). Phase 3+ ("Crisis or worse") is the primary metric for humanitarian need.
- **Calibration advice**: Per-hazard, per-metric guidance text generated from historical forecast performance. Injected into forecasting prompts to help models learn from past scoring patterns. Includes seed advice for bootstrapping when insufficient scored data exists.
- **NMME seasonal forecasts**: Seasonal temperature and precipitation anomaly forecasts from the North American Multi-Model Ensemble (via NOAA Climate Prediction Center). Provides country-level anomalies in standard deviations from climatology for the next seven months. Used for climate-sensitive hazards (DR, FL, HW, TC).
- **VIEWS (Uppsala/PRIO)**: Machine-learning-based conflict forecasting system providing country-month predictions of fatalities and conflict onset probability at 1–6 month lead times. Stronger at trends and baseline levels.
- **conflictforecast.org (Mueller/Rauh)**: News-based armed conflict risk scores at 3-month and 12-month horizons. Better at detecting shifts and escalation signals from media coverage patterns.
- **Prediction markets**: Crowd forecast signals from Metaculus, Polymarket, and Manifold Markets. Presented as contextual evidence with relevance filtering and minimum liquidity/volume thresholds.
