## Changes in this Version

This version reflects significant architectural changes introduced in PR #641, expanding the system's data foundation and restructuring the Horizon Scanner pipeline with per-hazard separation, dedicated grounding, and new climate and conflict data sources.

- **Per-hazard Regime Change and triage with separate grounding.** The Horizon Scanner pipeline has been restructured so that each hazard (ACE, DR, FL, HW, TC) receives its own dedicated RC assessment and its own dedicated triage assessment, each with a separate Google Search grounding call. Previously, RC and triage shared a single grounding call per country. Now, the RC grounding call uses signal categories tuned for novelty detection (TRIGGER, DAMPENER, BASELINE), while the triage grounding call uses categories tuned for the operational picture (SITUATION, RESPONSE, FORECAST, VULNERABILITY). The recency windows also differ: RC grounding uses shorter windows (e.g. 90 days for ACE) to focus on recent disruptions, while triage uses longer windows (e.g. 120 days for ACE) to capture the broader situation.
- **ENSO state and forecast module.** A new module scrapes the IRI/CPC ENSO Quick Look page for current ENSO conditions, the Niño 3.4 sea-surface temperature anomaly, a 9-season probabilistic outlook (La Niña / Neutral / El Niño probabilities), multi-model plume averages (dynamical and statistical), and Indian Ocean Dipole (IOD) state. This data is cached for 7 days and injected into RC and triage prompts for all climate-sensitive hazards (DR, FL, HW, TC), giving the pipeline concrete, quantitative ENSO context instead of relying on web-search mentions.
- **Seasonal tropical cyclone forecasts.** A new module aggregates basin-level seasonal TC forecasts from three authoritative sources: Tropical Storm Risk (TSR, via PDF extraction), NOAA CPC (seasonal outlooks for Atlantic, Eastern/Central Pacific), and the Australian Bureau of Meteorology (BoM, covering the Australian and South Pacific regions). Forecasts cover eight named basins and are mapped to individual countries via a basin-to-country lookup. Pre-scraped, formatted forecast text is injected into TC-specific RC and triage prompts, providing quantitative seasonal context (named storm counts, above/below-normal probabilities, ACE index forecasts) that would otherwise require the LLM to find and interpret on its own.
- **HDX Signals (OCHA automated crisis monitoring).** A new connector pulls OCHA's HDX Signals dataset, which provides automated crisis indicator alerts across multiple domains (conflict, displacement, food insecurity, agricultural stress, market disruption). Each indicator type is mapped to one or more Fred hazard codes (for example, ACLED conflict signals map to ACE, IPC food insecurity signals map to DR, JRC agricultural hotspots map to DR/HW). Alerts are filtered by country, hazard, and recency (180 days), and injected as evidence blocks into both RC and triage prompts for all hazard types.
- **ACLED CAST (Conflict Alert System Tool).** A new conflict forecast connector fetches predictions from ACLED's CAST API, which provides admin1-level event count forecasts disaggregated by event type: total events, battles, explosions/remote violence (ERV), and violence against civilians (VAC). Forecasts are aggregated to country level and stored alongside VIEWS and conflictforecast.org predictions. Unlike the other two conflict forecast sources, CAST forecasts event counts (not fatalities) and provides event-type breakdown — battles vs. ERV vs. VAC — which can reveal shifts in the character of violence even when overall levels remain stable.
- **Per-hazard calibration anchors in RC and triage prompts.** Each hazard type now receives its own calibrated scoring guidance. RC prompts include hazard-specific early warning signal categories (e.g., 6 conflict early warning categories for ACE; ENSO/precipitation categories for DR), expected score distributions (e.g., roughly 80% of country-hazard pairs should have RC likelihood at or below 0.10), and concrete anchoring rules (e.g., TC out-of-season likelihood capped at 0.03). Triage prompts include numeric scoring anchors tied to impact levels (e.g., ACE: 0 fatalities maps to 0.02–0.10, 10,000+ maps to 0.85–1.00) and explicit priority ordering for scoring inputs: current situation first, then RC result, then seasonal context, then structural exposure, then vulnerability/capacity.
- **Two-pass model diversity for RC and triage.** Both the RC and triage LLM steps now run two passes with different models (Pass 1: Gemini Flash, Pass 2: GPT-5-mini). Results are merged by averaging numeric fields and reconciling directional labels. This is a practical stability measure rather than a deep ensemble, but it reduces the influence of any single model's idiosyncrasies on the HS output.
- **ACLED low-activity filter for ACE triage.** The triage step now skips ACE assessment for countries with minimal recent conflict: specifically, countries with 0 fatalities in the most recent 2 months and fewer than 25 fatalities in the trailing 12 months. This prevents the pipeline from spending LLM calls on countries where armed conflict is structurally implausible, and avoids false-positive triage scores driven by generic risk language.
- **New data sources injected into prompts.** Both RC and triage now receive a broader suite of structured data: ReliefWeb situation reports, ACLED political events (ACE/DI only), IPC food security phases, ACAPS INFORM severity, ACAPS Humanitarian Access, HDX Signals, ENSO context (climate hazards), seasonal TC forecasts (TC only), and conflict forecasts including ACLED CAST (ACE only).
- **New GitHub Actions workflows.** Two new scheduled workflows refresh the ENSO and seasonal TC caches, ensuring the pipeline always has recent climate context available without requiring a full HS run to trigger the scrapers.

---

## System at a Glance

Fred (also referred to as Pythia in the codebase) is an end-to-end humanitarian forecasting system. It takes raw event data (conflict, disasters, displacement) from three primary sources — ACLED, IDMC, and IFRC GO — supplements that data with structured feeds from humanitarian monitoring organizations (ACAPS, IPC, ReliefWeb, HDX Signals), climate forecast services (NOAA/NMME, IRI/CPC ENSO forecasts, seasonal TC outlooks from TSR/NOAA CPC/BoM), conflict forecast models (VIEWS, conflictforecast.org, ACLED CAST), and expert assessments (ICG CrisisWatch), and produces probabilistic forecasts for a set of standardized questions over a 1–6 month horizon. The system is designed for auditability: each run writes a canonical DuckDB database artifact that records inputs, intermediate judgments, model calls, and final outputs.

A key design tension in forecasting is balancing base rates (what usually happens) with out-of-pattern events (what can happen when the world changes). In Fred, this tension is handled explicitly through Horizon Scanner triage, Regime Change scoring, adversarial evidence checks, and — when needed — Hazard Tail Packs that surface trigger evidence.

| Stage | What it does | Primary inputs | Primary outputs |
| --- | --- | --- | --- |
| Resolver (data ingestion) | Ingests and normalizes raw data from external sources into monthly "facts" suitable for forecasting and evaluation. | Connector feeds: ACLED, IDMC, IFRC Montandon (GO API). | facts_deltas, facts_resolved, derived monthly tables; connector status. |
| Web Research (evidence packs) | Fetches grounded, source-cited qualitative evidence to complement quantitative facts, with an explicit recency window. Used primarily by the Horizon Scanner for triage and RC assessment. | Country and hazard-tail queries; web research backend. | Evidence pack objects used by HS triage and RC; sources lists. |
| Structured Data Connectors | Pulls authoritative humanitarian data and conflict/climate forecasts from specialist APIs and stores it for prompt injection. Replaces the former research LLM stage. | ReliefWeb, ACAPS (4 datasets), IPC, ACLED political events, VIEWS, ConflictForecast, NOAA, ACLED CAST, HDX Signals, IRI/CPC ENSO, TSR/NOAA CPC/BoM seasonal TC. | Database tables with structured evidence; formatted prompt blocks for forecasting. |
| Horizon Scanner (HS) per-hazard pipeline | Scores each country–hazard pair for elevated risk in the next 1–6 months; detects potential regime shifts. Runs per-hazard RC assessment, then per-hazard triage, each with dedicated grounding calls. | Resolver facts + evidence packs + structured data + ENSO context + seasonal TC + HDX Signals + conflict forecasts + hazard catalog. | hs_runs, hs_country_reports, hs_triage (incl. RC fields). |
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

**Structured data** refers to the authoritative humanitarian, climate, and conflict-forecast datasets pulled from specialist APIs (ReliefWeb, ACAPS, IPC, ACLED political events, NMME/NOAA, IRI/CPC ENSO, seasonal TC forecasts, VIEWS, conflictforecast.org, ACLED CAST, HDX Signals, ICG CrisisWatch). Unlike web-search-derived evidence packs, structured data is deterministic, reproducible, and sourced from organizations with formal data collection methodologies. Structured data is stored in the database and automatically formatted for injection into pipeline prompts.

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

Fred now maintains fourteen structured data feeds, organized into five categories: humanitarian monitoring, food security, climate and seasonal forecasts, conflict forecasts, and automated crisis monitoring. These feeds are used across the entire pipeline — Horizon Scanner RC assessment, HS triage, and the forecaster — though not every feed appears in every prompt. The table at the end of this section summarizes which feeds are injected where.

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

**ENSO State and Forecast (IRI/CPC).** New in this version, the ENSO module scrapes the International Research Institute (IRI) and Climate Prediction Center (CPC) ENSO Quick Look page to retrieve comprehensive ENSO conditions. The scraped data includes: the current ENSO state classification (La Niña, Neutral, or El Niño), the latest Niño 3.4 SST anomaly value, a 9-season probabilistic forecast table showing the probability of La Niña, Neutral, and El Niño for each upcoming 3-month season (e.g., MAM, JJA, SON), multi-model plume averages broken down by model type (dynamical models, statistical models, and all models combined), Indian Ocean Dipole (IOD) state and the Dipole Mode Index (DMI) value, and the CPC's narrative summary. The data is cached locally as JSON with a 7-day expiry. The public API provides two interfaces: `get_enso_state()` returns a structured dataclass for programmatic use, and `get_enso_prompt_context()` returns a formatted text block ready for injection into prompts. ENSO context is injected into RC and triage prompts for all climate-sensitive hazards (DR, FL, HW, TC), because ENSO phase is a major driver of seasonal precipitation patterns, tropical cyclone activity, and drought risk globally. For example, El Niño conditions typically suppress Atlantic hurricane activity while enhancing Pacific activity, and increase drought risk in parts of Southeast Asia and Australia — information that the pipeline previously had to discover anew through web searches on every run.

**Seasonal Tropical Cyclone Forecasts (TSR, NOAA CPC, BoM).** New in this version, this module aggregates basin-level seasonal tropical cyclone forecasts from three independent authoritative sources, covering eight named ocean basins:

- **Tropical Storm Risk (TSR)**: TSR publishes detailed PDF forecast documents for the Atlantic (ATL) and Northwest Pacific (NWP) basins. The extractor uses `pdfplumber` with regex-based parsing to discover and download TSR PDFs by their predictable URL naming convention (e.g., `TSRATLForecastApr2026.pdf`), then extracts named storm counts, hurricane/typhoon counts, intense hurricane/typhoon counts, Accumulated Cyclone Energy (ACE) index predictions, comparisons against 30-year and 10-year norms, tercile probability distributions (above/near/below normal), and ENSO context notes. TSR provides the most granular quantitative seasonal forecasts available.

- **NOAA Climate Prediction Center (CPC)**: NOAA publishes seasonal hurricane outlook press releases for the Atlantic (with an initial May forecast and an August update), Eastern North Pacific (ENP), and Central Pacific (CP) basins. The scraper extracts named storm, hurricane, and major hurricane ranges (e.g., "14–21 named storms"), above/near/below-normal probability assessments, and key explanatory factors. A single NOAA press release sometimes covers multiple basins, and the scraper handles cross-basin references by parsing the document for each relevant basin.

- **Australian Bureau of Meteorology (BoM)**: BoM publishes seasonal TC outlooks for the Australian region (AUS) and South Pacific (SP). The scraper extracts the probability of above-average activity, categorical outlooks (above/below/near average), subregional breakdowns (e.g., Western, Northern, and Eastern Australia), and notes on severe TC likelihood. BoM outlooks use a different structure from TSR and NOAA — they emphasize probability of above-average rather than specific count ranges — and the scraper normalizes these into the same output format.

All three sources are deduplicated by (source, basin, forecast_type), keeping the most recent forecast when multiple versions exist. The results are unified into a JSON cache and an optional `prompt_context.txt` file. At prompt time, the system maps each country to its relevant ocean basin(s) using a comprehensive ISO3-to-basin lookup table (for example, the Philippines maps to the Northwest Pacific, Madagascar maps to the Southwest Indian Ocean). The function `get_seasonal_tc_context_for_country(iso3)` returns a formatted text block containing all available seasonal forecasts for that country's basin(s), or `None` for landlocked or non-TC-exposed countries. A staleness warning is included if the cached forecasts are more than 120 days old. Seasonal TC context is injected into TC-specific RC and triage prompts, providing the pipeline with expert consensus on basin-level activity expectations — information that directly calibrates whether the upcoming season is expected to be above, near, or below average for a given region.

### Conflict forecast and assessment feeds

**VIEWS (Uppsala/PRIO).** VIEWS is a machine-learning-based conflict forecasting system developed at Uppsala University and the Peace Research Institute Oslo. It provides country-month predictions of state-based conflict fatalities and the probability of at least 25 battle-related deaths at 1–6 month lead times. VIEWS is generally stronger at capturing trends and baseline levels of conflict, though weaker at predicting sudden onset. Fred stores VIEWS predictions in the database and includes staleness warnings when the data is more than 45 days old.

**conflictforecast.org (Mueller/Rauh).** This project provides news-based armed conflict risk scores derived from media coverage patterns. It offers risk scores at 3-month and 12-month horizons, plus a violence intensity outlook at 3 months. Because it is driven by news signals rather than historical conflict patterns, conflictforecast.org is generally better than VIEWS at detecting shifts and escalation signals. Where the two quantitative sources disagree, the LLM is instructed to note the disagreement and reason about why.

**ACLED CAST (Conflict Alert System Tool).** New in this version, ACLED CAST provides event-count forecasts disaggregated by event type: total events, battles, explosions/remote violence (ERV), and violence against civilians (VAC). CAST uses an OAuth2-authenticated API and returns admin1-level forecasts that the connector aggregates to country level. Four distinct metrics are stored: `cast_total_events`, `cast_battles_events`, `cast_erv_events`, and `cast_vac_events`, each covering a 6-month forecast horizon. CAST differs from VIEWS and conflictforecast.org in two important ways. First, it forecasts event counts rather than fatalities, providing a complementary signal — a country could see rising event counts (more frequent clashes) without a proportional rise in fatalities, or vice versa. Second, the event-type breakdown (battles vs. ERV vs. VAC) reveals shifts in the character of violence that aggregate counts would mask. For example, a shift from battles to violence against civilians may indicate a change in conflict dynamics even if total event counts remain stable. CAST forecasts are rendered alongside VIEWS and conflictforecast.org in a unified conflict forecast block, with a note explaining that CAST measures event counts rather than fatalities. Country name aliases are used for ISO3 resolution (for example, mapping "Ivory Coast" to CIV, or "Congo DRC" to COD).

**ICG CrisisWatch (International Crisis Group).** CrisisWatch is ICG's monthly conflict monitoring bulletin. The system fetches per-country directional assessments (Deteriorated, Improved, Unchanged) and the monthly "On the Horizon" feature, which highlights roughly three conflict risks and one resolution opportunity expected in the next three to six months. ICG is highly selective about what it flags, so countries appearing in "On the Horizon" receive a prominent note in their RC assessment. Unlike the other structured feeds (which are stored in the database), CrisisWatch assessments are retrieved via web research at prompt time and injected directly.

All conflict forecast feeds are used exclusively for armed conflict (ACE) hazards. VIEWS, conflictforecast.org, and ACLED CAST predictions are stored in the database and injected alongside the ACLED retrospective base rates as structured quantitative anchors. The prompt rendering includes staleness warnings for each source when data is older than 45 days, and explicitly notes where sources agree or disagree on trend direction. ICG CrisisWatch is injected into the RC assessment only.

### Automated crisis monitoring

**HDX Signals (OCHA).** New in this version, HDX Signals is an automated crisis monitoring system operated by OCHA's Centre for Humanitarian Data. It continuously monitors a set of global indicators and generates alerts when anomalous or noteworthy conditions are detected. The connector downloads the `hdx_signals.csv` dataset from the HDX CKAN API and caches it locally.

Each HDX Signals indicator type is mapped to one or more Fred hazard codes:
- `acled_conflict` → ACE (armed conflict)
- `idmc_displacement_conflict` → ACE
- `idmc_displacement_disaster` → FL, TC, DR, HW (disaster-driven displacement)
- `ipc_food_insecurity` → DR (food security as a drought proxy)
- `jrc_agricultural_hotspots` → DR, HW (agricultural stress)
- `wfp_market_monitor` → DR (market disruption as a drought indicator)
- `acaps_inform_severity` → all hazard types

Alerts are filtered by country (ISO3), hazard relevance, and recency (default 180 days). The formatted output for prompt injection includes for each alert: the indicator name, alert level, summary text, date, and source URL. The connector includes a disclosure note that HDX Signals alerts are suppressed for 6 months after the last alert for a given indicator-country pair, so the absence of recent alerts does not necessarily mean the absence of concerning conditions. HDX Signals is injected into both RC and triage prompts for all hazard types, providing a broad automated screening layer that complements the more targeted individual data sources.

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
| ENSO state and forecast (IRI/CPC) | DR, FL, HW, TC | DR, FL, HW, TC | — |
| Seasonal TC forecasts (TSR/NOAA/BoM) | TC | TC | — |
| VIEWS conflict forecasts | ACE | ACE | — |
| conflictforecast.org risk scores | ACE | ACE | — |
| ACLED CAST event forecasts | ACE | ACE | — |
| ICG CrisisWatch | ACE | — | — |
| HDX Signals (OCHA) | All | All | — |
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

The Horizon Scanner now generates separate grounding calls per hazard, and the grounding calls are different for RC and triage (see Section 6 for details). The RC grounding uses signal categories tuned for novelty detection — TRIGGER, DAMPENER, BASELINE — while the triage grounding uses categories tuned for the operational picture — SITUATION, RESPONSE, FORECAST, VULNERABILITY. This separation ensures that the LLM's web search is calibrated to the specific question being asked (is the pattern changing? vs. how bad is it?) rather than trying to answer both with a single query.

Evidence packs are not used in the forecaster SPD prompt. The forecaster instead receives structured data from the connectors described in Section 4, which is more deterministic and reproducible.

## 6. Horizon Scanner (HS): per-hazard risk screening with dedicated grounding

Horizon Scanner is the system's front-end risk screening stage. Its job is not to forecast numbers. Instead, it triages which country–hazard combinations merit deeper attention and provides structured context for downstream forecasting.

HS is typically run on a schedule (for example, monthly) and writes into the canonical DuckDB. Two tables are important for understanding HS outputs. The hs_country_reports table stores the evidence pack used for a country (including sources and markdown renderings). The hs_triage table stores one row per country–hazard, including the triage score and other structured fields.

### Per-hazard pipeline architecture

The HS pipeline has been restructured in this version into a fully per-hazard architecture. Rather than producing a single assessment per country that tries to evaluate all hazards simultaneously, the pipeline now runs dedicated RC and triage assessments for each individual hazard. For a country exposed to five hazard types (ACE, DR, FL, HW, TC), the pipeline produces five separate RC assessments and five separate triage assessments, each with its own grounding call, prompt, and LLM invocation.

The pipeline runs in the following order for each country:

1. **Per-hazard RC grounding** — For each hazard, a dedicated Google Search grounding call is made using hazard-specific search queries. The RC grounding queries are designed to detect signals of structural change — new conflict dynamics, unusual climate patterns, policy shifts — rather than to describe the current situation. Each query specifies signal categories (TRIGGER, DAMPENER, BASELINE) and a hazard-appropriate recency window: 90 days for ACE, FL, and HW; 120 days for DR and TC. These grounding calls run in parallel across hazards using a thread pool.

2. **Per-hazard RC assessment** — For each hazard, the RC LLM is invoked with the grounding results plus all relevant structured data for that hazard. The RC prompt is hazard-specific (see Section 7 for details). RC runs two passes: Pass 1 uses Gemini Flash, Pass 2 uses GPT-5-mini. The results are merged by averaging likelihood and magnitude and reconciling direction labels. RC calls run sequentially (not in parallel) to manage rate limits and cost.

3. **Per-hazard triage grounding** — For each hazard, a separate grounding call is made using triage-specific queries. These queries are explicitly different from the RC grounding: they focus on the current operational picture rather than structural change. Signal categories are SITUATION, RESPONSE, FORECAST, and VULNERABILITY. Recency windows are generally longer than RC: 120 days for ACE, FL, HW, and TC; 180 days for DR. The RC result is included in the triage grounding prompt so the search can be calibrated by what RC found — for example, if RC detected escalation, the triage grounding can focus on confirming or quantifying that escalation. Triage grounding calls also run in parallel across hazards.

4. **Per-hazard triage assessment** — For each hazard, the triage LLM is invoked with the triage grounding results, the RC result from step 2, and all relevant structured data for that hazard. The triage prompt is hazard-specific (see below). Triage also runs two passes (Gemini Flash, then GPT-5-mini) with averaged results. The ACLED low-activity filter is applied before ACE triage: if a country has 0 fatalities in the 2 most recent months and fewer than 25 fatalities in the trailing 12 months, ACE triage is skipped entirely and a default quiet score is assigned.

This architecture means each hazard gets 4 LLM-adjacent operations (RC grounding, RC call, triage grounding, triage call), multiplied by 2 passes each for RC and triage, for a total of up to 8 model interactions per hazard per country. The grounding calls are parallelized while the LLM calls are sequential, balancing throughput with cost control.

### Data sources injected into HS triage

The triage step receives the following data for each country–hazard pair:

- **Resolver features** — historical base-rate data from the facts_resolved table (all hazards).
- **RC result** — the output of the prior per-hazard RC assessment step, including likelihood, magnitude, direction, window, and rationale (all hazards). The RC result is presented as structured context with interpretation guidance: the triage prompt explains that triage_score and RC are "partially independent" — a country can have a high triage score (because the current situation is severe) but low RC (because the situation is chronic, not changing), or low triage but high RC (because a structural shift is emerging from a low baseline).
- **Triage grounding results** — hazard-specific grounded evidence from the triage Google Search call (all hazards).
- **Season context** — a human-friendly description of the current season for the region (all hazards).
- **ReliefWeb situation reports** — recent humanitarian reports from OCHA (all hazards).
- **ACAPS INFORM Severity** — crisis severity scores and trends (all hazards).
- **ACAPS Risk Radar** — forward-looking risk with triggers (all hazards).
- **ACAPS Daily Monitoring** — analyst-curated situational updates (all hazards).
- **ACAPS Humanitarian Access** — access constraint scores (all hazards).
- **IPC phases** — food security phase classifications (all hazards).
- **NMME seasonal forecasts** — temperature and precipitation anomalies (climate hazards: DR, FL, HW, TC).
- **ENSO state and forecast** — current ENSO conditions, Niño 3.4 anomaly, and 9-season probabilistic outlook from IRI/CPC (climate hazards: DR, FL, HW, TC).
- **Seasonal TC forecasts** — basin-level seasonal activity predictions from TSR, NOAA CPC, and BoM, mapped to the country's basin(s) (TC only).
- **HDX Signals** — automated crisis alerts from OCHA, filtered by country and hazard (all hazards).
- **ACLED summary** — monthly aggregate conflict data (ACE only).
- **ACLED political events** — event-level political incidents (ACE only).
- **VIEWS + conflictforecast.org + ACLED CAST** — quantitative conflict predictions, including CAST's event-type disaggregation (ACE only).

### Per-hazard triage prompts and scoring anchors

Each hazard type receives a tailored triage prompt with hazard-specific scoring guidance. The prompts share a common preamble that explains the relationship between triage_score and RC — emphasizing that they are partially independent assessments — and then provide hazard-specific scoring anchors that tie numeric triage scores to concrete impact levels. For example:

- **ACE triage** anchors: 0 fatalities in window → 0.02–0.10; 1–9 fatalities → 0.10–0.30; 10–99 → 0.30–0.55; 100–999 → 0.55–0.75; 1,000–9,999 → 0.75–0.85; 10,000+ → 0.85–1.00.
- **DR triage** anchors: no food security concern → 0.02–0.10; moderate rainfall deficit → 0.15–0.35; IPC Phase 3 in multiple areas → 0.40–0.60; widespread crop failure → 0.60–0.80; famine declared or imminent → 0.80–1.00.
- **TC triage** anchors: out-of-season and no formation → 0.02–0.10; active season but no current threat → 0.15–0.30; tropical storm approaching → 0.35–0.55; major hurricane/typhoon threatening → 0.60–0.80; catastrophic landfall imminent → 0.80–1.00.

All triage prompts specify a priority ordering for scoring inputs: (1) current situation evidence, (2) RC result and direction, (3) seasonal context, (4) structural exposure and historical base rates, (5) vulnerability and capacity factors. This ordering prevents the LLM from defaulting to structural vulnerability when active situation evidence should dominate, while still giving weight to chronic risk factors when current signals are absent.

### ACLED low-activity filter

To prevent wasted LLM calls and false-positive ACE scores, the triage step applies a low-activity filter before running the ACE assessment. A country is filtered out if it meets both conditions: (a) 0 reported fatalities in the 2 most recent months of ACLED data, AND (b) fewer than 25 total fatalities in the trailing 12 months. Filtered countries receive a default quiet triage score without an LLM call. This filter saves cost and reduces noise for countries where armed conflict is structurally implausible (e.g., small island states or historically peaceful countries). The filter is conservative — 25 fatalities in 12 months is a low threshold — so it is unlikely to filter out countries with genuine emerging risk.

### Stabilization and tier assignment

To reduce run-to-run variance, HS uses a stabilization pattern: both RC and triage run a two-pass pipeline and average numeric outputs such as triage_score and RC likelihood. Pass 1 uses Gemini Flash and Pass 2 uses GPT-5-mini. This introduces model diversity at the HS level — the two models have different strengths and tendencies, and averaging their outputs produces more stable results than relying on either model alone. This is not a deep statistical ensemble; it is a practical technique that improves consistency for dashboard users.

HS computes a tier label — quiet or priority — based on a single configurable threshold (default 0.50). Scores at or above the threshold are "priority"; scores below are "quiet". Tiers are intended for interpretability and prioritization, while the numeric triage_score is the primary signal that downstream components can use.

## 7. Regime Change (RC): per-hazard out-of-pattern detection

Regime Change is Fred's mechanism for flagging when base rates may be misleading in the forecast window. The intent is not to predict black swans on demand. Rather, it is to detect credible situations in which the generating process for a hazard may shift — for example, when a conflict enters a new phase, when a border policy changes, or when seasonal forecasts indicate unusually severe hazards.

RC is represented with three conceptual dimensions: likelihood, direction, and magnitude. Likelihood captures the chance that a regime shift occurs during the next 1–6 months. Direction indicates whether the shift would push outcomes up (worse), down (improving), mixed, or unclear relative to baseline. Magnitude captures how large the shift could be if it occurs.

### Per-hazard RC prompts and calibration

RC assessment now runs per-hazard with dedicated prompts for each hazard type. All hazard prompts share a common RC calibration preamble that sets conservative defaults: "Default to likelihood 0.05 and magnitude 0.05 unless evidence says otherwise." The preamble also specifies an expected distribution across all country-hazard assessments: roughly 80% should have likelihood at or below 0.10, roughly 10% between 0.10 and 0.30, roughly 7% between 0.30 and 0.55, and roughly 3% at 0.55 or above. This calibration guidance is critical for preventing score inflation — without it, LLMs tend to assign elevated RC to too many country-hazard pairs.

Each hazard type then receives its own specialized prompt content:

- **ACE (armed conflict)**: The RC prompt defines six categories of conflict early warning signals that the model should look for: (1) escalation signals (military buildup, arms flows, mobilization), (2) political triggers (election crises, coup attempts, government collapse), (3) social cohesion breakdown (ethnic/sectarian polarization, hate speech escalation), (4) economic shocks (currency collapse, severe austerity, resource competition), (5) external intervention shifts (foreign military involvement, proxy dynamics, sanctions changes), and (6) peace process disruption (ceasefire violations, negotiation collapse, spoiler activity). The ACE prompt also includes specific anchoring rules for ACLED data: the model is instructed how to interpret different levels of recent conflict activity in terms of RC likelihood. Conflict forecast data from VIEWS, conflictforecast.org, and ACLED CAST is injected alongside ICG "On the Horizon" alerts and HDX Signals.

- **DR (drought)**: The RC prompt emphasizes slow-onset dynamics and consecutive season analysis. It instructs the model to look for consecutive rainfall failures (one failed season is concerning; two consecutive failures are a strong RC signal), ENSO phase transitions (especially transitions into El Niño in regions where El Niño suppresses rainfall), IPC phase escalation (movement from Phase 2 to Phase 3+), and crop production disruptions. The ENSO state and probabilistic forecast are injected directly into the prompt, along with IPC data and HDX Signals for agricultural stress indicators.

- **FL (flood)**: The RC prompt focuses on GloFAS (Global Flood Awareness System) signals, upstream conditions, rainfall anomaly forecasts, and infrastructure degradation. It instructs the model to weigh seasonal timing heavily — flood risk is highly seasonal in most regions — and to consider ENSO phase effects on regional rainfall patterns. ENSO context is injected, along with NMME precipitation anomaly forecasts.

- **HW (heat wave)**: The RC prompt emphasizes vulnerability amplifiers — urban heat island effects, power grid capacity, water availability — alongside temperature anomaly signals. It instructs the model to pay special attention to duration (prolonged heat is much more dangerous than brief spikes) and to compound effects (heat concurrent with drought or conflict). ENSO context is injected because El Niño conditions can amplify heat extremes in many regions.

- **TC (tropical cyclone)**: The RC prompt includes a hard out-of-season rule: if the country is outside its historical cyclone season, RC likelihood must not exceed 0.03 regardless of other signals. During the active season, the prompt instructs the model to weigh the seasonal TC forecasts (from TSR, NOAA CPC, and BoM) and ENSO state heavily, since ENSO phase is the strongest predictor of seasonal TC activity at the basin level. The seasonal TC context and ENSO context are both injected directly into the prompt.

All hazard-specific RC prompts accept a common set of new data source sections through a shared formatting function. This function builds clearly labeled text blocks for: ReliefWeb situation reports, ACLED political events (ACE/DI only), IPC food security phases, ACAPS INFORM severity, ACAPS Humanitarian Access, and HDX Signals. Each block is only included when data is available for the given country and hazard.

### Separate RC grounding prompts

The RC grounding queries that feed the Google Search web research call are now hazard-specific and separate from the triage grounding queries. RC grounding is designed to detect novelty — signals that the pattern may be changing — and uses three signal categories:

- **TRIGGER**: leading indicators of escalation or worsening (e.g., for ACE: military deployments, ceasefire violations, new armed group activity; for DR: consecutive rainfall failure, ENSO transition, crop production collapse).
- **DAMPENER**: counter-signals suggesting stabilization or de-escalation (e.g., peace talks progress, improved forecasts, humanitarian response scaling up).
- **BASELINE**: signals supporting continuation of current patterns (e.g., stable governance, normal seasonal progression, no change in structural factors).

Recency windows for RC grounding are shorter than for triage, reflecting RC's focus on recent disruptions: ACE uses 90 days, DR uses 120 days, FL uses 90 days, HW uses 90 days, and TC uses 120 days.

### Data sources injected into RC assessment

RC assessment has its own dedicated prompt context for each hazard. The data injected includes:

- **RC grounding results** — hazard-specific grounded evidence from the RC Google Search call (all hazards).
- **Resolver features** — historical base-rate data (all hazards).
- **ReliefWeb situation reports** — recent humanitarian reports (all hazards).
- **NMME seasonal forecasts** — temperature and precipitation anomalies (climate hazards: DR, FL, HW, TC).
- **ENSO state and forecast** — current ENSO conditions, Niño 3.4 anomaly, and 9-season probabilistic outlook from IRI/CPC (climate hazards: DR, FL, HW, TC).
- **Seasonal TC forecasts** — basin-level seasonal activity predictions from TSR/NOAA CPC/BoM (TC only).
- **HDX Signals** — automated crisis alerts from OCHA, filtered by hazard (all hazards).
- **ACLED summary** — monthly aggregate conflict data (ACE only).
- **ACLED political events** — event-level political incidents (ACE only).
- **ACAPS INFORM Severity** — crisis severity scores (ACE only).
- **ACAPS Risk Radar** — forward-looking risk assessments (ACE only).
- **ACAPS Daily Monitoring** — analyst-curated updates (ACE only).
- **VIEWS + conflictforecast.org + ACLED CAST** — quantitative conflict predictions (ACE only).
- **ICG CrisisWatch "On the Horizon"** — ICG's forward-looking conflict flags (ACE only). Countries flagged by ICG receive a prominent note, as ICG is highly selective about what it flags.
- **IPC phases** — food security data (DR only).
- **ACAPS Humanitarian Access** — access constraint scores (all hazards, via triage data when available).

### Scoring and levels

Fred stores RC results as numeric and categorical fields in hs_triage. A derived RC score is computed as likelihood multiplied by magnitude. This provides a single 'how much should I care' number that suppresses cases that are high-likelihood but low-impact, or high-impact but highly speculative.

RC scores are mapped into discrete levels used for UI and downstream gating. Level 0 is base-rate normal, Level 1 is a watch state, Level 2 indicates elevated regime-change plausibility, and Level 3 indicates a strong out-of-pattern hypothesis. The RC scoring was calibrated in February 2026 to reduce false positives. The Level 1 likelihood threshold is 0.45, and the Level 1 composite-score threshold is 0.25, so that mildly elevated signals do not trigger a watch flag. The per-hazard RC prompts include explicit calibration guidance: the expected distribution — roughly 80% of country–hazard pairs should have likelihood of 0.10 or below — and examples of what is and is not a regime change. A run-level distribution check warns operators when the fraction of elevated flags exceeds configurable thresholds (e.g., more than 15% at Level 2).

### Behavioral implications

Critically, RC does not merely decorate the dashboard. When RC is elevated, the system can override the 'need_full_spd' gating so that a full probabilistic forecast is produced even if the triage tier would otherwise be quiet. This is how Fred avoids an important failure mode: missing a structural break because it does not look like a high base-rate risk.

RC is surfaced to users across the dashboard: on country maps and lists as a 'highest RC' indicator, on triage tables as probability/direction/magnitude/score columns, and on question drilldown pages as summary boxes. This makes the system's out-of-pattern reasoning visible rather than hidden inside prompts.

## 8. Adversarial evidence checks: testing the RC hypothesis

A component added in March 2026, adversarial evidence checks provide a structured "devil's advocate" mechanism for high-risk forecasts. When Regime Change is elevated to Level 2 or above, the system automatically runs targeted web searches looking for counter-evidence — reasons the predicted regime shift might not materialize.

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
- **HS triage output** — triage score, tier, RC fields (likelihood, magnitude, direction, score, level), and other structured assessments. This is the forecaster's primary link to the Horizon Scanner's judgment. Because the HS now runs per-hazard, the triage output passed to the forecaster reflects the hazard-specific assessment rather than a country-level aggregate.
- **Adversarial evidence check** — counter-evidence to the RC hypothesis, including historical analogs and stabilizing factors (RC Level 2+ only).
- **Hazard tail pack** — trigger, dampener, and baseline signals from the second-stage evidence retrieval (RC-elevated hazards only, capped at 12 bullets).

**Forecaster-specific inputs:**
- **Resolver history summary** — historical distribution data (source, recent level, trend, data quality) from the Resolver.
- **Calibration advice** — per-hazard/metric guidance text generated from historical scoring performance (all hazards).
- **Hazard-specific reasoning block** — tailored forecasting instructions for the hazard type (all hazards).
- **Prediction market signals** — crowd forecasts from Metaculus, Polymarket, and Manifold (all hazards, when available).

This evidence is presented to the forecasting models as structured text blocks, each clearly labeled with its source. The forecaster sees, for example, the latest ReliefWeb reports about the country, the ACAPS severity score and trend, the IPC phase populations, the NMME climate outlook, and any adversarial counter-evidence — all formatted for readability rather than raw data.

Note that some data sources available to the Horizon Scanner are not passed through to the forecaster. ACAPS Humanitarian Access, ACLED monthly summaries, VIEWS/conflictforecast.org/ACLED CAST quantitative predictions, ICG CrisisWatch, ENSO state/forecast, seasonal TC forecasts, and HDX Signals are used during HS triage and RC assessment but are not directly injected into the SPD prompt. Their influence reaches the forecaster indirectly through the HS triage output and RC fields.

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

Two new scheduled workflows were added in this version: one for refreshing the ENSO state cache (scraping IRI/CPC), and one for refreshing the seasonal TC forecast cache (running the TSR, NOAA CPC, and BoM scrapers). These workflows run on independent schedules and write cached JSON files that the HS pipeline reads at runtime. This decouples the climate data refresh from the HS run itself, ensuring that fresh ENSO and TC data is available even between HS runs and that scraper failures do not block the main pipeline.

A signature guardrail is used to ensure the database contains required tables before being treated as canonical. This protects against partial artifacts and prevents downstream stages from running on an incomplete database.

Debugging is treated as an operational requirement. Runs log LLM calls, costs, evidence pack diagnostics, and the reasons why certain questions were included or skipped. A comprehensive debug bundle can be generated that includes triage summaries, research briefs, forecast details, and scoring results for a given run. This makes it feasible to answer "why did the model think this?" after the fact.

## 17. Practical guidance for forecasters

The system is designed to support, not replace, human judgment. In practice, forecasters should treat HS triage as a prioritization and hypothesis-generation tool. The most productive workflow is: (1) inspect HS triage and RC flags; (2) review the structured data feeds (ReliefWeb reports, ACAPS assessments, IPC phases, ENSO conditions, seasonal TC outlooks, HDX Signals alerts) and tail packs for high-RC hazards; (3) check the adversarial evidence to understand counter-arguments; (4) decide what would move the distribution; (5) ensure the SPD expresses both the baseline and the plausible tail mechanisms; and (6) write explanations that are accountable and falsifiable.
