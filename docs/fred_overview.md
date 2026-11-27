## Fred in brief

Fred is an experimental, automated system that forecasts humanitarian impact. Once a month it looks at about 120 countries and asks how many people floods and tropical cyclones will affect, how many people armed conflict will kill and newly displace, how many people will be in food crisis, and whether a major flood, drought or cyclone alert will be raised. Each forecast covers the next six months and is a probability distribution, not a single number. Later, Fred grades its forecasts against what happened and uses the grades to adjust how it forecasts. Nothing it produces is reviewed by a person; do not use it for decisions. Its code is called Pythia.

### The monthly cycle

1. **The 11th: refresh and grade.** Fred refreshes the data it grades against (conflict deaths, displacement, disaster impacts, food-crisis figures, disaster alerts), settles the outcomes of past forecasts, scores them, refits the weights and corrections it applies to its models, and publishes the results (Sections 2, 7 and 8).
2. **The 13th: forecast.** Fred scans every country and hazard for unusual risk, decides which questions to ask, and sends them to its models through the providers' cheaper batch service. The forecasts usually finish between the 13th and the 17th (Sections 3 to 5).
3. **At the end of the forecast:** Sibyl, an independent research agent, reforecasts 15 questions from the open web; Fred writes a plain-language report; the forecasts are published (Sections 6, 9 and 10).
4. **Every Sunday:** fast-moving feeds are refreshed (conflict forecasts, disaster alerts, humanitarian reports, political events, news indicators).
5. **Days 1 to 12:** a daily check fetches the newest International Crisis Group CrisisWatch bulletin.

### What gets forecast and how it is graded

| Question | What is asked | Where it is asked | What grades it |
| --- | --- | --- | --- |
| Conflict deaths | Deaths from armed conflict each month | Every country where conflict goes forward to forecasting | ACLED's count of deaths across all event types |
| Conflict displacement | People newly displaced by conflict each month | Only countries whose displacement is reported in most months | IDMC first, then figures OCHA or the UN reported, then the rise in IOM's count of displaced people as a floor |
| Flood and cyclone people affected | People affected each month | Countries where the hazard goes forward to forecasting | Fred's resolution machine, which reads storm tracks, disaster alerts, ReliefWeb reports, IFRC and IDMC |
| Drought food crisis | People in IPC Phase 3 or worse | Countries where FEWS NET or the IPC published a figure in the last nine months | FEWS NET, else the IPC |
| Yes/no disaster events | Will an orange or red flood, drought or cyclone alert be raised? | Every flood, drought and cyclone hazard that goes forward | GDACS alerts |

Details in Sections 4 and 7.

### How a forecast is made

1. **Evidence.** For each country and hazard, Fred gathers web sources on signs of change and on the current situation (Section 3).
2. **Regime change.** A model judges whether the coming months may break from the historical pattern, and how likely and how large such a break would be.
3. **Triage.** A model scores how serious the hazard is now. Hazards flagged for regime change skip this step and go straight to the full ensemble.
4. **Counter-evidence.** For flagged hazards, Fred searches for reasons the predicted change might not happen.
5. **Two tracks.** Flagged questions go to a five-model ensemble (Claude Opus 5.5, GPT-6.1 Sol, GPT-6 Luna, Gemini 4 Argon, Gemini 3.8 Flash). Serious but unflagged questions go to a single model, Gemini 3.8 Flash (Section 5).
6. **Combining.** Each ensemble model's forecast is corrected for its known lean, then the models are pooled, the better-performing ones counting for more (Section 8).
7. **Second reader.** Sibyl researches 15 questions on the open web and publishes its own forecasts beside the ensemble's (Section 6).

### Experiments running now

| Experiment | What is compared | How questions are split | When results are due |
| --- | --- | --- | --- |
| Calibration advice | Forecasts shown past-performance advice against those shown none | Half the questions, by a fixed rule on the question's name | Ten scored questions per half with advice to test |
| Regime-change wording | "Move the forecast in the flagged direction" against "widen it" | Half of full-ensemble questions, by a separate fixed rule | Ten scored questions per half |
| Counter-evidence | Forecasts shown the counter-evidence against forecasts not shown it | Half of eligible questions, from the December 2026 run | Forty resolved conflict-death questions per half |
| Regime-change second reader | GPT-6.1 Sol's judgements, stored but never used, against Gemini 3.8 Flash's | Every assessment | Owner decision after two months |
| New model versions | A new model starts from its predecessor's record and earns its own weight | All questions | Twenty scored questions per hazard and measure |
| Sibyl advice | Sibyl trials shown their own track record against trials not shown it | Half of Sibyl's questions | Ten scored questions per half |
| Sibyl data pack | A trial given Fred's structured data against one without, on the same question | Within every Sibyl question | Ten scored questions |
| Sibyl second model | Sibyl's pool with an extra GPT-6.1 Sol trial, never published | Every researched question | Twenty resolved questions; ends April 2027 |
| Sibyl prior weight | How much of Sibyl's forecast comes from its mechanical prior | Fitted on all resolved Sibyl questions | Twenty resolved questions |

### What Fred cannot see yet

- **Ground truth for people affected is thin.** Many floods and storms are never reported with a figure, so some months stay ungraded.
- **EM-DAT is not read.** The disaster database that tops Fred's ladder of impact sources needs an account Fred does not have.
- **Few questions have resolved.** Most hazards and measures have well under a hundred scored questions, so skill figures carry wide margins of error and some model weights are not yet fitted.
- **Sibyl's record is short.** By November 2026 Sibyl had 47 scored questions, 28 of them comparable with the ensemble, and no clear difference had emerged.
- **Some displacement forecasts cannot be marked.** Older forecasts for countries reported only now and then are "indicative" and kept out of learning and headline skill.

### Where to look on the dashboard

- **Run Results** (home): the latest run, the impact map and the risk table.
- **Fred's Monthly Forecast Report**: the plain-language report and its PDF.
- **Countries** and **Forecasts**: every country and question, with evidence and model transcripts.
- **HS Triage & RC**: the scan's scores and regime-change flags.
- **Resolver**: the data behind the forecasts, source by source.
- **Sibyl**: the deep-research forecasts beside the ensemble's.
- **Performance** and **Performance Analysis**: scores, benchmarks and written reviews.
- **Costs**, **Downloads** and **About**: spending, data exports, this document, the decks and the prompts.

## 1. Purpose and key terms

Fred converts noisy, scattered observations into probability statements about the near future. It assumes that base rates matter: for most humanitarian measures the best starting point is what usually happens in a country in that season. It also assumes base rates are not enough. Conflicts escalate, policies change and seasons fail, and a central aim is to notice when "business as usual" may not hold and to move probability accordingly.

Fred is also built to be checked. Every stage writes its inputs, its model calls and its outputs into one database, which is kept for each run. Any forecast can be traced back to the evidence and the instructions behind it.

Its sources are imperfect proxies for humanitarian impact. They reflect who reports, where access allows, and how each organisation defines its figures. Fred does not hide this. It states uncertainty through the width of its forecasts, through the notes it attaches to thin or stale data, and through the regime-change flags, and it leaves a month ungraded rather than guess its outcome.

Forecasts are distributions because a decision about a humanitarian response turns on the tail as much as the middle: a 10% chance of half a million people affected calls for different preparation than a certain 20,000. Each forecast is stored both as numbers and with readable band labels.

### Key terms

A **hazard** is a class of risk Fred tracks the same way in every country. Four are active: armed conflict (ACE), drought (DR), flood (FL) and tropical cyclone (TC). Several other codes exist but are switched off (Section 12).

A **question** is one country, hazard and measure, forecast for each of the next six months. Month 1 is the first month after the forecast is made.

A **forecast** is a probability distribution over a fixed set of size bands for each month, or a single probability for a yes/no question. The bands always start with a band for exactly zero.

A **fact** is a cleaned monthly figure taken from a source, recorded with where it came from and when.

A **resolution** is the graded outcome for one month of one question. Some months stay unresolved because no source can answer them (Section 7).

| Measure | Shown on the dashboard as | What it counts | Bands |
| --- | --- | --- | --- |
| People affected | `PA` | People affected by a flood or cyclone, or newly displaced by conflict | 0; 1 to 9,999; 10,000 to 49,999; 50,000 to 249,999; 250,000 to 499,999; 500,000 or more |
| Conflict deaths | `FATALITIES` | Deaths in armed conflict | 0; 1 to 4; 5 to 24; 25 to 99; 100 to 499; 500 to 999; 1,000 or more |
| Food crisis | `PHASE3PLUS_IN_NEED` | People in IPC Phase 3 (crisis) or worse | 0; 1 to 99,999; 100,000 to 999,999; 1 to 5 million; 5 to 15 million; 15 million or more |
| Event | `EVENT_OCCURRENCE` | Whether GDACS raises an orange or red alert | Yes or no |

## 2. Data: sources and how they are refreshed

Fred reads three kinds of data. **Ground-truth sources** give the figures questions are graded against. **Context feeds** describe the situation and the outlook and are shown to the models. **Web evidence** is gathered fresh by search for each country and hazard during the horizon scan (Section 3).

### Ground-truth sources

- **ACLED** (the Armed Conflict Location and Event Data project) counts conflict events and deaths. Fred stores one figure a month per country: deaths across all event types. A month is stored only once it has ended and ACLED's weekly releases have covered it.
- **IDMC** (the Internal Displacement Monitoring Centre) records displacement. Fred keeps only the figures IDMC marks as caused by conflict and recommends for totals, and sets aside any figure that spans more than a month.
- **ReliefWeb**, OCHA's humanitarian reporting site, is read by a small model for stated figures of people affected and people newly displaced by conflict. A figure is used only if its quote is found word for word in the report (Section 7).
- **IOM's Displacement Tracking Matrix (DTM)** counts people currently displaced. The rise between two close survey rounds gives a floor on new displacement.
- **IFRC GO**, the Red Cross and Red Crescent's operational database, gives people-affected figures from field reports and appeals.
- **GDACS** (the Global Disaster Alert and Coordination System) raises colour-coded alerts for floods, droughts and cyclones, and estimates how many people were exposed. Alerts grade the yes/no questions; exposure is used only as a ceiling.
- **The global storm-track archive** (IBTrACS) records every tropical cyclone's path and strength.
- **FEWS NET and the IPC** publish the number of people in Phase 3 or worse. FEWS NET covers about 48 countries; the IPC's own service covers others, and is read for a FEWS NET country once FEWS NET has stopped publishing for it. FEWS NET publishes a range, such as "1.0 to 2.49 million"; Fred stores both ends, and grades on the lower one.
- **Drought indicators**: crop-stress hotspots from the EU's Joint Research Centre (directly and through HDX Signals), the probability of a dry month from the seasonal outlook, and a three-month water-balance index (SPEI-3) computed from Copernicus data. These decide only whether a rise in food crisis can be called drought (Section 7).
- **EM-DAT**, the international disaster database, would sit at the top of the impact ladder, but Fred has no account and does not read it.

### Food-crisis figures

Food-crisis figures serve twice: as ground truth for food-crisis questions and as context in prompts. Fred reads FEWS NET first and the IPC's own service second. For each country and analysis it stores the current-situation figure, which grades questions, and the "most likely" projection, which is shown to the models as context only. Where both sources report the same month, FEWS NET is preferred. A country whose newest figure is more than nine months old is treated as no longer published, and no food-crisis question is asked for it. A separate read-only check is asking whether current Cadre Harmonisé figures for the Sahel can be found elsewhere.

### Context feeds

| Feed | What it gives | Regime change | Triage | Forecast | Sibyl pack |
| --- | --- | --- | --- | --- | --- |
| ReliefWeb reports | Recent situation reports (titles, sources, dates) | All | All | All | No |
| ACAPS INFORM Severity | Crisis severity, 0 to 10, with trend | All | All | All | Yes |
| ACAPS Risk Radar | Expected deteriorations and their triggers | All | All | All | Yes |
| ACAPS Daily Monitoring | Analyst-curated daily updates | All | All | All | Yes |
| ACAPS Humanitarian Access | How hard it is for aid to reach people | No | All | No | No |
| Food-crisis figures (FEWS NET, IPC) | Phase 3+ population, current and projected | ACE, DR | All | All; DR with projections | Yes |
| ACLED monthly summary | Recent conflict deaths and events | ACE | ACE | ACE (as the conflict base rate) | No |
| ACLED political events | Protests, riots and strategic developments | ACE | ACE | ACE | Yes |
| GDELT | News-coded conflict intensity over 7, 30 and 60 days | ACE | ACE | ACE | Yes |
| VIEWS, conflictforecast.org, ACLED CAST | Other systems' conflict forecasts | ACE | ACE | ACE | Only if the owner switches it on |
| ICG CrisisWatch | Monthly direction of each conflict and "On the Horizon" warnings | ACE | ACE | ACE | Yes |
| HDX Signals | OCHA's automated crisis alerts | All | All | All | Yes |
| NMME seasonal outlook | Temperature and rainfall outlook, chance of a dry or wet month | DR, FL, TC | DR, FL, TC | DR, FL, TC | Yes |
| ENSO state and forecast | El Niño or La Niña phase and outlook | DR, FL, TC | DR, FL, TC | DR, FL, TC | Yes |
| Seasonal cyclone outlooks | Expected basin activity | TC | TC | TC | Yes |
| GDACS alert history | Past alert months for the country | No | No | DR, FL, TC | Yes |
| Resolution machine base rates | How often and how badly the hazard has struck | No | No | FL, TC, DR people affected | No |

The rest of this section describes each feed in more detail.

### Humanitarian monitoring

**ReliefWeb.** Fred fetches up to 15 reports per country from the last 45 days, with title, source, disaster type and date.

**ACAPS** provides four products. INFORM Severity rates a crisis from 0 to 10, built from its impact, the conditions of the people affected and its complexity; Fred shows the latest score and how it compares with one and three months earlier. Risk Radar names crises expected to worsen and the events that would cause it. Daily Monitoring gives analysts' notes on recent developments. Humanitarian Access scores how far aid can reach people, and is shown only at triage.

**ACLED political events** add the detail behind the monthly counts: a protest movement growing, a coup, a ceasefire. Each event is attributed by the country ACLED itself records, never by the country Fred asked about.

**HDX Signals** is OCHA's automated monitor. It raises alerts when an indicator moves sharply. Each indicator is linked to hazards: conflict events and conflict displacement to armed conflict; disaster displacement to floods, cyclones and drought; food insecurity, crop stress and market prices to drought; crisis severity to all four. Fred filters alerts by country, hazard and the last 180 days, and notes that an indicator stays silent for six months after an alert, so silence is not proof of calm.

### Climate and seasonal outlooks

**The NMME seasonal outlook** (the North American Multi-Model Ensemble, from NOAA's Climate Prediction Center) gives temperature and rainfall anomalies for each country for the months ahead. For rainfall Fred shows the chance that the month falls in the driest third and in the wettest third of the model's record, beside the one in three an ordinary month carries. That chance means the same in a desert and a rainforest, which a rainfall shortfall in millimetres does not. Where the centre makes no forecast, as in a desert's dry season, Fred records no reading rather than "not dry". The outlook is issued around the 8th of each month.

**ENSO.** Fred works out the El Niño or La Niña phase itself, from the Oceanic Niño Index and NOAA's international definition: El Niño at +0.5 °C or above, La Niña at -0.5 °C or below, neutral between, with a strength band. It reads four numeric sources on every run and holds the index back to 1950. A phase is never stated without a number behind it, a sudden jump needs a second source to agree, and if no source can be read the last reading is carried forward with its age stated. The International Research Institute's ENSO page supplies the season-by-season outlook, the model plume and the Indian Ocean Dipole, but not the phase.

**Seasonal cyclone outlooks** come from several forecasters, each covering its own basins:

- **Tropical Storm Risk** publishes detailed outlooks for the Atlantic and the North-West Pacific: counts of named storms, hurricanes or typhoons and intense ones, the expected accumulated cyclone energy, comparisons with ten- and thirty-year norms, and the chances of an above-, near- or below-normal season.
- **NOAA** publishes ranges of storm counts and seasonal probabilities for the Atlantic (in May, updated in August) and the Eastern and Central Pacific.
- **Australia's Bureau of Meteorology** gives the chance of above-average activity for the Australian region and the South Pacific.
- **Météo-France La Réunion**, the regional centre for the South-West Indian Ocean, issues an outlook each October or November for the November to April season.
- The **North Indian Ocean** has no published count outlook, so its countries get a climatology block: average storm counts, the two seasonal peaks (April to June and October to December) and which coasts each part of the basin threatens, plus India's two-week cyclogenesis outlook when it is fresh.

Each outlook is stored once, with its issue date or a stated reason for having none. Each country is mapped to its basins; landlocked countries get nothing. The newest outlook from each source is shown first, older ones in a line each, with a warning when the outlooks are more than 120 days old.

### Conflict forecasts and assessments

**VIEWS** (Uppsala University and the Peace Research Institute Oslo) forecasts conflict deaths and the chance of at least 25 battle deaths, one to six months ahead, from historical patterns. It is strong on levels and trends and weak on sudden onset. **conflictforecast.org** gives news-based risk scores at three and twelve months and a violence-intensity outlook, which tend to catch escalation earlier. Where the two disagree, the models are asked to say why. **ACLED CAST** forecasts event counts by type: battles, explosions and remote violence, and violence against civilians. A shift between types can reveal a change in the character of violence that totals hide, and events are not deaths, which the prompt says. Only forecasts for months that have not yet passed are shown. Each states its age, and one older than two months is labelled as a historical prior rather than an outlook. ACLED CAST has served the same December 2025 vintage since January 2026, so its forecasts are out of date; where a source has nothing current, the prompt says so rather than leaving it out.

**ICG CrisisWatch** is the International Crisis Group's monthly bulletin. Fred records each country's direction (deteriorated, improved, unchanged) and the "On the Horizon" warnings. ICG's site blocks cloud servers, so Fred reads the Internet Archive's copy, asking for a fresh one each day from the 1st to the 12th. The prompt always names the newest edition held and says whether the country is listed in it, was last listed some months earlier, or is not covered by ICG at all; it warns when the edition is three or more months old. A heading that names several countries, such as "Israel/Palestine", is counted for each of them.

**GDELT** codes world news into event records. Fred sorts conflict events into three tiers (direct violence, escalating posture, political deterioration), counts them per country and day, and tracks the cooperation-conflict balance and tone of reporting over 7, 30 and 60 days, with the direction of travel. The prompt says plainly that these are counts of reporting, not of deaths.

### How the data is refreshed

The main refresh runs on the 11th of each month, two days before the forecast. By then the month just ended is complete in ACLED, the CrisisWatch edition is out and the seasonal outlook has been issued. It runs in phases: the ground-truth connectors first (a failure there turns the run red but no longer stops the rest), then the resolution sources and the resolution machine for the last three months, then the context feeds, and finally a check of every table before the database is saved. A weekly refresh on Sundays updates the fast-moving feeds; it skips the 10th to the 13th, which belong to the monthly cycle. The drought water-balance index is rebuilt on the 8th, and the resolution machine's history is extended every night (Section 11).

Each feed is checked after every refresh. A table that should have been written and was not, a source that has gone quiet, and a vintage that has stopped moving are each reported by name. A source that could not be read is recorded as unread, never as empty: "the source said nothing happened" and "Fred could not reach the source" lead to different answers. A run that finds faults still finishes and saves its work, so one blocked source never costs the others their refresh, but it ends with a report of every fault, worst first, with what it cost and who owns it. Faults already diagnosed and being chased are registered with a review date, reported once rather than each time they recur, and reported again as overdue once that date passes.

Where several reports describe the same figure, Fred keeps the most recent report from the most authoritative source, and never a report with no value over one with a value. Every stored figure carries the date it was published and the date Fred read it; a publication date is never later than the day it was read.

## 3. Horizon scan: evidence, regime change, triage and counter-evidence

The horizon scan decides which questions to ask and how hard to work on each. It does not produce numbers. For each of the 120 listed countries it works through the four hazards one at a time, leaving out cyclones out of season.

### Web evidence

For each country and hazard, Fred gathers two packs of web sources by search. The search goes to Brave Search first, then OpenAI's web search if Brave finds nothing, then Google's. A pack keeps up to ten sources, and Fred stops searching once it is full.

- The **regime-change pack** looks for novelty: signs of escalation (triggers), signs of calm (dampeners), and signs that things continue as before (baseline). It covers the last 30 days for conflict and floods and 60 days for droughts and cyclones, and favours wire services and specialists.
- The **triage pack** looks at the current picture: the situation, the response, forecasts and vulnerability. It covers 60 days (90 for drought) and favours OCHA and humanitarian sources. It is told what the regime-change step found.

If the web search budget runs out and a hazard has no evidence at all, that hazard is not forecast.

Both packs are passed on to the forecasting models as "HS grounding evidence".

### Regime change

The regime-change step asks whether the process that generates outcomes may shift in the next six months: a conflict entering a new phase, a failed season, a policy change. A model (Gemini 3.8 Flash) gives three judgements: the **likelihood** of a shift, its **direction** (up, down, mixed or unclear) and its **magnitude**. The likelihood alone sets the level: below 0.15 is level 0, 0.15 or above level 1, 0.35 or above level 2, and 0.55 or above level 3.

Each hazard has its own prompt. All share a cautious default (likelihood and magnitude 0.05 unless the evidence says otherwise) and an expected spread: about 80% of assessments at or below 0.10 and about 3% at 0.55 or above. The conflict prompt lists six kinds of warning sign (military build-up, political triggers, social breakdown, economic shocks, outside intervention and broken peace processes). The drought prompt looks for consecutive failed seasons, an ENSO shift and rising food-crisis phases. The flood prompt weighs the season, rainfall outlooks and upstream conditions. The cyclone prompt forbids a likelihood above 0.03 outside the season.

The prompt carries the regime-change pack and the feeds marked in Section 2's table, with ICG's "On the Horizon" warnings given prominence because ICG flags few countries. A run-level check warns when too many assessments are elevated, for example more than 15% at level 2. Regime change is shown across the dashboard: as the highest level per country on maps and lists, as likelihood, direction, magnitude and score on triage tables, and as a summary panel with rationale and triggers on each question page.

A second model, GPT-6.1 Sol, receives the identical prompt as a **shadow**. Its answers are stored and never change a level, a track or a question. A monthly check on the 12th asks which of the two readers caught the real breaks (Section 8). Whether to use the second reader is the owner's decision after at least two months of results.

### Triage

Triage scores how serious the hazard is now, from 0 to 1, against anchors written for each hazard. For conflict the anchor is ACLED's deaths over the last twelve months: none, with no armed groups active, scores 0.02 to 0.10; 1 to 99 deaths 0.10 to 0.25; 100 to 499 deaths 0.25 to 0.45; 500 to 1,999 deaths 0.45 to 0.65; 2,000 to 9,999 deaths 0.65 to 0.85; and 10,000 or more 0.85 to 1.00. For drought, no food insecurity scores 0.02 to 0.10 and catastrophic drought with famine risk 0.85 to 1.00, with crisis-phase populations between. For cyclones, out of season with no active storm scores 0.02 to 0.05 and an intense storm on a direct track with landfall expected 0.75 to 0.95. The prompt also explains that triage and regime change are partly independent: a chronic crisis can score high on triage and low on regime change, and an emerging shift from a quiet base can do the reverse. It weighs, in order: current evidence, the regime-change result, the season, exposure and history, and vulnerability. A score of 0.50 or more makes the hazard **priority**; below that it is **quiet**.

Triage is skipped in three cases:
- the hazard is at regime-change level 1 or above, and goes straight to the full ensemble;
- a cyclone is out of season;
- a country has had no conflict deaths in the last two months and fewer than 25 in the last twelve, so conflict is left quiet without a model call.

Both the regime-change and triage steps run once per hazard, at temperature 0. A second pass with a different model can be switched on, but is off: two passes of the same model at temperature 0 give the same answer. The web searches for different hazards run in parallel; the model assessments run one at a time to control rate limits and cost. A hazard whose answer comes back malformed is repaired by a second model before it is given up.

### Counter-evidence

For each hazard at regime-change level 1 or above (at most two per country), Fred runs two or three searches for reasons the predicted change might not happen: for an expected conflict escalation, peace talks and ceasefire progress; for drought, revised rainfall forecasts. A model writes up the counter-evidence, past situations that looked similar but did not shift, stabilising factors, and a verdict: strong, moderate or weak counter-evidence, or inconclusive.

The check is stored for every eligible question. From the December 2026 run, half the eligible questions, chosen by a fixed rule on the question's name, are forecast without it, because the November 2026 performance review found that when models said this evidence moved them, it more often moved them away from the outcome. Once each half has 40 resolved conflict-death questions, the scores decide whether it stays.

### What comes out

Each country and hazard ends with a triage score, a tier, the regime-change judgements and a **track**: Track 1 (the full ensemble) for level 1 or above, Track 2 (a single model) for priority hazards at level 0, and no forecast otherwise. Level 2 or above always forces a forecast.

## 4. Questions

Each hazard that goes forward becomes one or more questions:

| Hazard | Questions |
| --- | --- |
| Armed conflict | Conflict deaths; conflict displacement where the country qualifies |
| Drought | Food crisis (IPC Phase 3+) where the figures are current; yes/no drought alert |
| Flood | People affected; yes/no flood alert |
| Tropical cyclone | People affected; yes/no cyclone alert |

**Conflict displacement** is asked only for countries whose conflict displacement is reported in most months: an IDMC figure, or a figure OCHA or the UN reported, in at least eight of the twelve months before the forecast window. A rise in IOM's count does not count toward the eight. If that record is missing or out of date, Fred reuses the previous run's list and says so. For the 2026 windows eight countries qualified (Haiti, Indonesia, Mali, Myanmar, Mozambique, Russia, Somalia and Ukraine). Where a conflict country has no displacement question, the dashboard and the monthly report say that displacement is not forecast for it, so the gap is never read as no risk.

**Food crisis** is asked only where FEWS NET or the IPC has published a current figure for the country no more than nine months before the first forecast month. Countries left out are named in the run log.

**People affected for drought** is not asked; drought is forecast as food crisis instead.

Questions are meant to be stable and comparable from month to month: the same measure and the same bands each time, so scores can be compared and models weighted. Each question's name carries the month its window opens, for example the Ethiopia conflict-deaths question for the window opening in November 2026. A question created once keeps its wording and dates. When the same window is forecast twice, the latest production forecast is the one that counts everywhere Fred learns from or reports on forecasts; earlier forecasts stay stored.

## 5. Forecasting

### What the models see

Each question gets one prompt, the same for every model on its track. It is built with the fixed instructions first and the country's data last, so providers can reuse the shared part at a discount. It contains:

- the question, its six months and its size bands;
- the hazard-specific reasoning guidance;
- the historical record for the question's own measure, from the source that will grade it;
- a base-rate block (below);
- the regime-change result and the guidance that goes with it;
- the context feeds marked in Section 2's table;
- the horizon scan's web evidence;
- the counter-evidence check, where eligible and in the half that sees it;
- calibration advice, for the half of questions that receive it (Section 8).

**Base-rate blocks.** For conflict deaths, the prompt shows a starting distribution for months 1 and 6: where the count stands in the last complete month and how far counts like it have moved over the same number of months, in that country or in countries at a similar level. The models are told to copy it as their prior and to argue every change from it. For flood, cyclone and drought people affected, the prompt shows the resolution machine's own base rates: how often the hazard struck in each of the six months, how large it was when it did, where those figures came from, and the rule that will grade the answer, written out from the machine's rulebook. A country the machine never examined gets no block. For food crisis, the prompt shows the Phase 3+ history over a stated window and FEWS NET's projections, labelled as projections. For yes/no questions it shows how often an orange or red alert was raised in each calendar month over the full GDACS record, and says when that record is too short to give a rate.

**Hazard guidance.** Each hazard has its own reasoning notes: for conflict deaths, ceasefire dynamics and tail risk; for floods, seasonality and sudden peaks; for cyclones, that months are either a landfall near people or nothing, so probability belongs at zero and in the upper bands; for conflict displacement, the ladder of sources it is graded on; for food crisis, that gaps in FEWS NET's cycle are not improvements.

**Regime-change guidance.** For half of full-ensemble questions, chosen by a fixed rule on the question's name, a flagged regime change comes with the older instruction to widen the forecast or rebut the flag. The other half are told to move the forecast in the flagged direction, by an amount matching the flag's likelihood and size, and to stay about as concentrated as the historical starting point. The two halves are compared once each holds ten scored questions. Every model must explain in a sentence beginning "RC:" what was flagged and how it affected the forecast.

### Track 1: the ensemble

Track 1 questions go to five models: Claude Opus 5.5 (Anthropic), GPT-6.1 Sol and GPT-6 Luna (OpenAI), and Gemini 4 Argon and Gemini 3.8 Flash (Google). Each model's thinking effort is set explicitly. Before a run starts, Fred asks each provider a small question; a provider that refuses (for example because its spending limit is reached) is left out of the whole run, and the health report says so.

Each model returns a forecast and a **reasoning trace**: the starting distribution and why, each piece of evidence that moved it with the exact shift in probability, a point estimate, and whether it accepted the regime-change flag. Fred checks the trace on three counts: whether the starting distribution matches the base rate the prompt showed, whether each shift sums to about zero and adds up to the next distribution, and whether each shift's stated size (small, moderate, large) matches its actual size. It records a quality score (40% start, 40% arithmetic, 20% size) and never rejects a forecast for it. Track 1 traces must list at least two updates; Track 2 needs only the start and the regime-change verdict; yes/no answers carry no trace. The traces feed the calibration advice and the analysis package. An answer whose explanation is malformed is kept if its forecast block can be recovered.

Each member's forecast is first corrected for its family's known lean (Section 8). The corrected forecasts are then combined in two ways: a weighted average and a Bayesian pooling that treats each answer as evidence. Both use the calibration weights. The average is the main ensemble forecast. Each member's own forecast is stored and scored too. Track 1 questions also get scenario narratives (below).

### Track 2: the single model

Track 2 questions go to Gemini 3.8 Flash alone, with the same prompt minus the requirement for a full reasoning trace. They get no scenarios. The forecast is stored under a single fixed name, whichever model is behind it, so its record continues across model changes. Track 2 reflects a practical judgement: when the base rate is expected to hold, one capable model is enough.

### Yes/no event questions

A yes/no question asks for the probability, month by month, that GDACS raises an orange or red alert. Its prompt carries the historical alert rate for each calendar month, the months with recent orange or red alerts (one line per month), the current situation, the climate outlook and hazard-specific guidance. Track 1 members each answer, and their answers are averaged with the calibration weights; Track 2 uses the single model. Probabilities are kept between 0.001 and 0.999. An answer missing any month is rejected. At regime-change level 1 or above, each model must also say how it reconciled the flag with the historical rate. Yes/no forecasts are scored by the Brier score only, because the other two rules need a full distribution. On the dashboard they appear as probability bars by month.

### Batches and prompt caching

The monthly forecast runs through the providers' batch services, which answer within a day at half the price. The pipeline runs in four stages, pausing after each set of model requests and resuming when they are answered; a small watcher checks every few minutes and starts the next stage. Requests sent in a batch are identical to those sent one by one, so the forecasts do not change. A request a batch fails to answer is re-batched first, and only then asked one at a time at full price. Fred also marks the fixed part of each prompt so providers can reuse it, which cuts the price of reading it again.

### Scenarios

For Track 1 questions, a model writes short scenario narratives describing plausible paths to different parts of the forecast. They help readers see what a regime shift would look like in practice. They are not forecasts and are never scored.

## 6. Sibyl

Sibyl is a second, independent forecaster. Where the main track reasons over the data Fred has gathered, Sibyl sends a research agent (Claude Opus 5.5 with Brave Search) to the open web to build its own picture. When the two disagree, the question deserves a closer look.

### Which questions

After each horizon scan Sibyl takes 15 numeric questions: conflict deaths, flood and cyclone people affected, and food crisis. Yes/no questions are never included.

- **13 flagged questions.** Each must carry a regime-change flag of level 1 or above (level 2 or above for cyclones). The strongest flags come first, then the most volatile. Conflict gets four to seven, drought two to three, floods one to two and cyclones at most one; slots a hazard cannot fill go to conflict, then drought.
- **2 controls.** Conflict-deaths questions with no flag, chosen by a fixed rule on their names. They show what Sibyl does where nothing is expected to move.

The controls are worked first, so a budget or time cut always drops the extra picks. If too few questions qualify, Sibyl forecasts fewer.

### How a Sibyl forecast is made

1. **The reference.** Sibyl builds its own prior from the question's history. For conflict deaths it mixes the last twelve months' spread of counts (three parts) with the moves seen from the current level (one part). For floods and cyclones it uses each calendar month's own record. For food crisis it mixes "the last figure persists" with the last three years.
2. **The resolving source's latest reading.** Before the trials start, Sibyl reads the source that will grade the question once and shows every trial what it says: ACLED's count for the month so far, the newest Phase 3+ figures and FEWS NET projections, or the last six months of reported figures with recent GDACS alerts.
3. **Two research trials.** Each trial starts from the reference and researches step by step, up to twelve steps of up to three searches or page reads each. After every step it restates its whole belief: its forecast, the evidence for and against, and what it still does not know. Each trial follows a research plan (how the question resolves, the latest figures, drivers, calendar, reversion and disconfirming evidence) and must read five distinct documents before it may submit. The first trial is also shown a starting pack of Fred's structured feeds (marked in Section 2's table; never the main track's own research or forecasts); the second is not. Each trial submits two distributions, for months 1 and 6: the chance of zero, and the 5%, 25%, 50%, 75% and 95% points if not. The months between are a mix of the two.
4. **A reconciler, when the trials disagree.** If the two trials' month-1 forecasts differ by more than a set amount, a third trial is shown both trials' forecasts and evidence, must name the facts they disagree on, and must search on exactly those points before forecasting. It counts as one more trial with equal weight.
5. **Pooling.** A trial counts only if it finished with at least three successful searches and two documents read. The valid trials are averaged into one distribution, which widens when they disagree. When three trials are valid, a trial whose middle estimate is far from the others is left out; with two it cannot be judged, so the rule only acts when the reconciler ran. A forecast resting on a single valid trial is labelled as such. If no trial is valid, nothing is published.
6. **The published forecast.** For each month Sibyl publishes half its reference and half the pooled trials. Once twenty resolved questions carry both parts' scores, the reference's share is refitted each month, choosing a quarter, a half or three quarters, pulled strongly toward a half. The result goes into Fred's size bands and is stored beside the main forecasts, so the same scoring grades both.

A control runs a single trial without the pack.

### Keeping it honest

In retrospective tests every search is capped at the forecast date, later material is filtered out, and the sources Fred grades against are blocked. In live runs the outcome does not exist yet, so those sources count as evidence like any other. Every page Sibyl reads is opened only if its address resolves to the public web.

### Cost and time

Measured on a recorded trial, a flagged question costs about $1.30 and a control about half that; the stated figure the run is held to is $1.40 a question. The run stops starting new questions at $25, or at $2.50 for any one question, and after three hours; the question in progress finishes, and the rest are recorded as skipped. From the third question on, Sibyl compares what it has spent per question with that $1.40, warns above one and a half times ($2.10) and stops starting new questions above two and a half times ($3.50). A test fails if a code change would push the measured cost more than half above $1.40. Long documents are condensed by a cheaper model (Claude Haiku 4.5).

### Feedback and learning

After each scoring round, Fred checks Sibyl's record on the questions it researched: how often the outcome fell inside its 10% to 90% range, below it and above it, how it handled "exactly zero", and whether its middle estimate beat the reference's. Each question counts once, however many of its months have resolved. A finding becomes advice only when it is clear beyond chance and rests on at least 20 resolved questions for the hazard (or 20 across all hazards). The advice is built from fixed sentences, not written by a model. Each line gives the evidence with its count, then one proportionate change; for example, that the outcome fell below the 10% point seven times in 24 questions against two or three expected, so the low end should come down. It names no country and never mentions the other forecasters. It is shown to half of Sibyl's questions under "YOUR TRACK RECORD", right after the reference, framed as feedback to weigh and never as a target, so its effect can be measured once each half holds ten scored questions. It is never shown in retrospective tests, because it was learned after the forecast date.

Sibyl also writes a short post-mortem on each newly resolved question, labelled with up to three failure types from a fixed list of seventeen. Once a hazard has eight notes, Sibyl is asked for general lessons; a lesson is kept only if it rests on three notes and names no country or year. The lessons are shown to the same half of questions as the advice. This costs at most $5 a month.

### Measurement

Fred scores more than Sibyl's published forecast. It scores the reference alone, the pooled trials alone, each trial alone, and rearrangements of the same trials at no extra cost: fixed reference weights of a quarter, a half and three quarters, a weight that grows with disagreement, the pool without the reconciler, and the pool without its outlier rule. For floods and cyclones it scores separately whether any report appeared and how well the size was placed. Each comparison reports "not yet" until twenty questions have resolved.

**A second model on the side.** Each researched question gets one extra trial on GPT-6.1 Sol, run after all the normal trials and dropped first when money or time runs short. A second pool with that trial added is scored against the same outcomes and never published. The test ends after April 2027; adopting the model would be the owner's decision.

**The pack.** The trial shown Fred's data is compared with the one that was not, on the same question: how far each moved from the reference and how much it read (from five questions), and which did better (from ten scored questions).

Each run also reports how it researched: documents per trial, the share of trials that fell short of five documents, steps and searches per trial, and the share of documents from Wikipedia. Everything a trial read is stored with its address, time and a fingerprint of its text, in Fred's working database only, because the text belongs to its publishers.

### Record so far

In the November 2026 scoring round, 47 scored questions had a Sibyl forecast and 28 could be paired with the ensemble on the same forecast run. On those, Sibyl and the ensemble were level: Sibyl was sharper on conflict deaths, but no difference was clear at that sample size, and a simple "same as last month" forecast beat both on most conflict pairs. Those questions were researched under the older three-trial design.

### On the dashboard

The Sibyl page shows each run's coverage, cost and research measures; a table of questions sortable by how far Sibyl's forecast differs from the main track's; each question's two distributions overlaid, with every trial's step-by-step reasoning; and a Calibration tab with Sibyl's record, advice, failure labels and experiment results. A forecast carried over from an earlier run of the same question says so, with its date.

## 7. Grading outcomes

### General rules

Each month of each question is graded on its own against that calendar month's figure. Month 1 is graded against the first window month, month 6 against the last.

A month can end three ways: **graded with a figure**, **graded zero**, or **left ungraded**. Fred grades a month zero only when the source's silence can mean zero. For conflict deaths and yes/no events it can, because ACLED and GDACS cover the world continuously; even then, a month is defaulted to zero only if the source has data for that month somewhere, and for conflict deaths only if the country has ever appeared in ACLED. For people affected and food crisis, a missing figure means "unknown", and the month stays ungraded. Ungraded months are left out of scoring, so a question may be scored on fewer than six months.

A month is graded only once it has ended, and only if some source holds data that recent; a grade written too early by an earlier run is removed with its scores. Grades can change while sources revise: Fred keeps the first grade and later readings beside it, and scores against the latest. Any grade that a later rule no longer supports is removed, with its scores.

### Conflict deaths

Conflict deaths are graded on ACLED's monthly count of deaths across all event types: the same series the prompt shows and the reference forecasts are built from. A month counts only once ACLED's releases have covered all of it. Fred keeps the first reading and readings about two and three months later, because ACLED revises its counts.

### Conflict displacement

Conflict displacement is graded on a ladder, taking the first source that has a figure for the month:

1. **IDMC's recommended monthly figure for conflict**, once 90 days have passed since the month ended. By then most of a month's figures have arrived.
2. **A figure of people newly displaced by conflict that OCHA, a UN agency or a government stated** for that month in a ReliefWeb report, available 30 days after the month ends.
3. **The rise in IOM's count of conflict-displaced people** between two survey rounds no more than 35 days apart, credited to the later round's month. People who returned home in between reduce it, so it is a floor, and it is labelled as one wherever it appears.

A lower source never overrules a higher one. If two sources differ tenfold, the higher one stands and the month is flagged for a person to check. A month is graded zero only for a country IDMC reports in most months, and only when a later IDMC report brackets the gap; otherwise a missing month stays open.

**Reading ReliefWeb for displacement.** Each month, for every listed conflict country, Fred reads up to 15 ReliefWeb reports from international organisations and governments, those most likely to state new displacement first (situation reports, flash updates, displacement alerts). A small model (Claude Haiku 4.5) transcribes each stated figure with its exact quote, whether it is new displacement or a running total, its cause and its period. A figure is kept only if its quote is in the report, it is new displacement caused by conflict, and its period falls inside one calendar month; a figure about a different month is credited to that month. Households are converted to people with a stated multiplier. This reading has its own monthly budget of 2,000 calls (about $4). A monthly report shows where figures are lost, reason by reason.

**Scoring classes.** Displacement months are labelled as they are graded. A country IDMC alone reports in most months is **scored** like any other question. A country that qualifies only once the OCHA and UN figures are counted is forecast and graded as usual, but kept out of the model weights, advice, corrections and headline skill until six of its months have resolved. Forecasts for countries reported only now and then, made before October 2026 when they stopped being asked, are **indicative**: published and graded but never used for learning. The dashboard and monthly report say so in plain words. Fred keeps the reading of each displacement month at about six and nine months after it ends, because IDMC's figures keep arriving.

### Floods and cyclones: the resolution machine

Flood and cyclone people-affected questions are graded by Fred's **resolution machine**, which grades each country, month and hazard in two layers.

**Did anything happen?** A cyclone is detected when a storm of tropical-storm strength (34 knots) passes within 200 km of the country. A flood is detected when GDACS raises an orange or red flood alert for the country. If there is no detection, Fred searches ReliefWeb for a report of the hazard whose main country is this one. No detection and no report gives a confident zero, stored with the evidence of absence. A report without a detection proves the month was not quiet but not that the hazard struck; such a month counts as an event only if layer two finds a figure, and otherwise is left undecided. If ReliefWeb cannot be read, no zero is given.

**How many people?** The machine takes the figure from the highest rung of a fixed ladder that has one: EM-DAT (not read at present), then figures stated in ReliefWeb reports, then IFRC GO, then IDMC displacement figures as a floor. ReliefWeb figures are read by a small model and kept only if the quote is in the text. A figure must be about the cell: the report's main country must match, and the figure is dated by what its own sentence says, so a sentence about the 2022 floods cannot grade a 2026 month. A figure about another hazard or an earlier event is rejected, a stated count of people beats a household conversion of the same sentence, and one figure answers one hazard only. Every figure is checked against the country's population and, for cyclones, against three times what GDACS estimated was exposed; a figure that breaks a ceiling is kept and flagged, never rewritten. GDACS's exposure is never published as an answer.

**Timing.** Before the 60-day freeze, a cell with no figure yet is left blank, not graded "no data", because EM-DAT, IFRC and ReliefWeb figures arrive late. After the freeze a cell with no figure is "no data" and flagged. Grades written before the freeze are provisional and may be revised; after it they are final, and any later attempt to change them is logged, never applied. A month still in progress gets no answer for any hazard.

**Grading the questions.** A graded figure or zero grades the forecast, provisional or not; if the machine revises it before the freeze, the next grading run uses the revision. Where the machine has no answer, the month stays ungraded: Fred does not fall back to IFRC, which the machine has already read. A displacement figure that wins the ladder is graded as the answer, and every record of it says it is a floor.

**The rulebook.** Every threshold (the 200 km buffer, the alert level, the ladder's order, the tenfold disagreement test, the 60-day freeze) lives in one reviewable rulebook. The prompt's description of the grading rule is written from it each time, so a changed threshold reaches the models on the next run.

**History and base rates.** The same machine is replayed over the past, a little each night: cyclones from 2000, floods from 2010, droughts from 2017. Because those months are long frozen, the replay only fills gaps. From it come two base rates per country and calendar month: **how often** the hazard occurred (counting only years the machine actually graded, and published only with three years of evidence) and **how bad** it was when it did (percentiles since 2015, with the mix of sources behind them, including the share that are displacement floors). These are the blocks the forecasting models see (Section 5).

**Checks on the machine.** A one-page scorecard covers the last twelve frozen months: the share of cells graded per hazard against an 80% goal and the IFRC-only baseline, divided by every cell the machine examined. A hazard it has not examined reads "not assessed", not 0%. Flood detection is compared with the Dartmouth Flood Observatory's hand-curated archive for the years before the machine's own flood history begins; countries where the archive found far more floods are listed for a person to investigate. The archive never triggers a hazard or supplies a figure. The scorecard's worked examples are drawn with a fixed seed, so a reviewer can regenerate the same page, and where the machine falls short it states the shortfall rather than loosening a threshold. Every grade carries its provenance: what was queried, what was found, when, and which rule fired. A source that could not be read is recorded as unread, never as empty.

### Drought and food crisis

**Food-crisis questions** are graded on the number of people in IPC Phase 3 or worse in FEWS NET's current-situation analysis covering the month, else the IPC's. FEWS NET publishes a range, and Fred grades on its lower end; its ranges almost never straddle Fred's band edges, so the band is right even when the figure sits on a band's lower edge. Projections are shown as context and never grade anything.

**The machine's drought grades** are a trial and grade no question. Drought has no event day, so the machine skips the ladder and takes the rise in the Phase 3+ population since the previous analysis, floored at zero. It counts that rise as drought impact only where an independent drought indicator agrees (crop-stress hotspots, a high chance of a dry month, or the water-balance index); otherwise conflict- or price-driven hunger would be filed as drought. A zero needs at least two indicators to have answered. A country with no earlier analysis is graded "no data", because a standing total is not a monthly rise. The same rise is repeated in every month the analysis covers, and the record says these must never be added up.

### Yes/no events

A yes/no month is graded "yes" if GDACS raised an orange or red alert for the country and hazard, and "no" otherwise. Fred keeps the first reading and readings two and three months later.

## 8. Scoring, benchmarks, calibration and advice

### How scores are computed

Every graded month of every forecast is scored by three rules. The **Brier score** sums the squared gaps between the forecast probabilities and what happened (0 to 2 for banded forecasts; lower is better). The **log score** is the negative log of the probability given to the band that happened, and punishes confident misses hard. The **ranked probability score** compares the cumulative forecast with the outcome, so being one band off costs less than five (0 to 1). Yes/no forecasts get a Brier score from 0 to 1 only. Brier scores for banded and yes/no forecasts are never averaged together, because they sit on different scales.

Each question counts once, by its latest production forecast. Test runs never count.

### Benchmarks

Fred scores simple reference forecasts on the same questions every round:
- **climatology**, the historical base rate;
- a **flat** forecast;
- **"same as last month"**, for conflict deaths, conflict displacement and food crisis;
- for conflict deaths, the **level-and-spread** distribution the prompt shows, a **twelve-month record** of counts, a version using only moves from the current level, and a pool of these;
- **ViEWS**, whose point forecasts of conflict deaths are turned into distributions over Fred's bands (an indicative comparison, since the spread is Fred's assumption).

Skill against a reference is reported only on the months both scored, and with a range. References are never weighted or advised.

### Correcting each model's lean

Before the ensemble is formed, each member's forecast is corrected for its model family's known lean. For each family, hazard and measure, Fred compares the probability the family put on each band with how often outcomes landed there, shrinks that comparison heavily toward "no correction" when it rests on few questions, and caps it. Nothing is fitted below ten questions. A correction is applied only to forecasts made under the same prompt version it was learned from; otherwise, or if half of its factors sit at the cap, it is computed and stored but not used. The uncorrected forecast is kept and scored too.

### Calibration weights

Each ensemble member gets a weight per hazard and measure from its recent Brier scores, with older months fading (a twelve-month half-life). Weights need twenty scored questions; below that, members count equally. Both ensemble combinations and the yes/no average use them. A new version of a model family starts from its predecessor's record, counted as ten questions, and builds its own; a retired version drops out. A model can also run as a shadow, scored but never counted, until it earns a place. The weights are computed separately for each hazard and measure, because a model good at conflict deaths may be poor at floods, and yes/no questions form their own pool. Aggregate forecasts, Sibyl and the reference forecasts are never weighted; weights cover the ensemble's own members only.

### Calibration advice

After each scoring round, Fred writes advice on where forecasts have been miscalibrated, for each hazard and measure and for each model. Advice is written as observations, not orders ("you assigned 9%; outcomes landed there 34% of the time") and asks the models to treat them as tendencies to check. It leaves out what the prompt already settles: where conflict-death models are told to copy the starting distribution, it does not argue with that. A model gets its own note only after twenty distinct questions; until then a new version is shown its family's findings, labelled and faded. A hazard with no advice gets none, never another hazard's.

Half of all questions, chosen by a fixed rule on the question's name, get no advice at all, so its effect can be measured; the two halves are compared once each holds ten scored questions. A question meant to get advice that had none to give is recorded as such. In October 2026 no kind of question yet had advice to test.

### Regime-change recall

A check on the 12th of each month reads what happened in the six months after each scan and asks, for the production reader and its shadow, how many real breaks each flagged and how many flags became breaks. A break is a month at least two bands above the middle of the country's usual range.

### The analysis package

After each scoring round Fred builds a package for analysis. For every scored question it pairs what went into the forecast (the exact prompt, each model's answer and reasoning, the scan's rationale and evidence) with what happened and every score. It explains each score: the base rate shown, the model's declared start and its final forecast are scored side by side, so a reader can see whether error came from the start or from the adjustments. It scores each claimed adjustment, tests whether regime-change flags were followed by the moves they predicted, lists large outcomes nobody asked about, compares every running experiment, and records which feeds each forecast saw and how old they were. It includes flat tables of scores and outcomes, the calibration weights and how they moved, the advice, and case studies of the best and worst scored questions with Sibyl's research inlined. A guide written for the reader explains the layout, the score scales and the known blind spots, and a small briefing folder holds summaries sized for pasting into a conversation. Any section that cannot be computed leaves a note saying why rather than vanishing. Written reviews built from it appear on the Performance Analysis page.

## 9. The monthly report (the Interpreter)

At the end of each forecast cycle Fred writes a short report in plain language for readers with no forecasting background. It opens with the scan: how many countries were screened and how many went to each track. It carries at most five entries.

### What it covers

The report opens with three lines on what Fred cannot see, because the thinness of people-affected ground truth changes how every climate entry should be read.

**Potentially worsening situations**, split into climate hazards and conflict, holds forecasts that are both **unusual** (further from their own history than most of that month's forecasts) and **moving**: at the month of peak impact, either the middle estimate or the high-end figure sits above the historical one by at least a threshold, in people (or percentage points for a yes/no question). The threshold is the lower of an absolute figure and a share of the population, so small island states are not excluded. This list is ordered by how far the planning figure moved.

**Major impact but roughly stable** holds heavy burdens that have not moved: a real chance, in some month, of an impact large enough to act on. It is ordered by the size of the burden.

A forecast that is unusual but too small to act on goes to a **watchlist** at the back. Where a country's history is very thin, the comparison is fragile, so the entry is ranked lower and says so.

### Each entry

Each entry gives two figures in a sentence: the number to plan against in a named month and the number to hold contingency for, plus, where it matters, the chance that nothing is recorded. It names a decision and a deadline. The deadline is the month of peak impact minus the hazard's lead time (longer for drought than for a cyclone), and all deadlines are gathered on a calendar near the front.

### Other sections

- **A second reader**: where Sibyl disagrees with the main forecast and which way, whether its trials agreed, and one or two things it found that the main evidence did not contain.
- **Fred against the sector**: countries ranked by Fred and by two humanitarian severity indices, with the largest gaps. The report says the three measure different things.
- **What changed**: risks matched across months by country, hazard and measure, compared on the months two forecasts share, and labelled new or flagged for so many months running.
- **How Fred did**: once outcomes exist, skill in the latest scored round, including against the historical average. Below a set number of resolved questions the report declines to make a skill claim at all. Every printed skill figure has a range.

### How it stays honest

The model that writes the report (Claude Opus 5.5) never calculates. Every number is computed by Fred and substituted into the text afterwards. Probability words have fixed meanings, printed in the appendix. Every claim names the questions it rests on, linked to the dashboard. The system, not the writer, decides which situations appear and under which heading, and the writer is checked against that. Names and places in the text must appear in the evidence. A panel on the first page states the rules and the counts, and an appendix lists every question considered with its figures, so a reader can see why a country is or is not included.

The writing follows a house style: plain words, active voice, no unexplained jargon, a list of banned phrases, and no long dashes. Countries and hazards are named in words, never codes, and each entry describes the shape of its forecast in a sentence; the full distributions are in the appendix, with their bands written out ("1 to 9,999 people"). The printed version has a table of contents, the map on the first page, an issue line naming the forecast date and the months covered, and the limits of the data stated on page one.

Machine checks re-derive the numbers independently, check probability words against their bands, and reject stray numerals, em dashes and banned phrases. A report that fails is corrected once; if it still fails, it is stored and shown behind an "UNVALIDATED" banner naming the failed checks.

### Where to find it

The report appears on the dashboard as "Fred's Monthly Forecast Report", with a map of where forecasts sit against their usual level (warm for more people than usual, cool for fewer). A PDF is attached to the public data release each cycle.

## 10. Dashboard

The dashboard serves two kinds of reader: forecasters who want the evidence, and decision-makers who want a summary of risk.

- **Run Results** (home page) shows the latest run's coverage, models, costs and health, an impact map and a risk table. It shows when the data was last updated and warns when the data service is behind the latest release.
- **Fred's Monthly Forecast Report** shows the report described in Section 9.
- **Countries** lists every country with its latest scan and forecasts.
- **HS Triage & RC** shows the horizon scan's scores, tiers and regime-change judgements.
- **Forecasts** lists every question. Each question page shows the forecast for every month, the regime-change summary, scores once months resolve, and the full model transcripts by stage (loaded when a panel is opened). Yes/no questions show probability bars with outcomes.
- **Resolver** lets a reader browse each data source, grouped by category (resolution data, conflict forecasts, weather and climate, situation reports, other alerts). Each source shows when it was last updated and how many rows it holds, and expands into its data, filtered by the country chosen at the top.
- **Sibyl** is described in Section 6.
- **Costs** shows spending by run, stage and model, and compares Fred's own records with what each provider billed (Google publishes no bill Fred can read).
- **Performance** shows the ensemble's three scores beside the toughest benchmark on the same questions, a monthly trend, resolution coverage by hazard and measure (questions too new to resolve are shown as "awaiting first horizon"), a head-to-head with Sibyl, and detailed tables by hazard, run and model. Current models are marked "Active", a new model appears as "awaiting first scores", and retired ones are greyed. Banded and yes/no Brier scores are always in separate columns. A second tab explains each score.
- **Performance Analysis** lists the written reviews by month: after each scoring round, a skill analysis and a review of how outcomes were graded; after each forecast run, a review of the pipeline with a verdict. Each can be read on the page or downloaded.
- **Downloads** offers forecasts, scores and model rationales as files: per-question score files for each ensemble with the full six-month forecast, a per-model summary, and the written explanation each model gave, filterable by hazard. Every export says which rows are test data.
- **About** holds eight slide decks, this document with all its earlier versions, and the AI prompts with theirs.

A switch in the menu shows test data; it is off by default (Section 11).

## 11. Operations

### Schedule

| Job | When (UTC) | What it does |
| --- | --- | --- |
| CrisisWatch check | Daily 06:17, 1st to 12th | Asks the Internet Archive for a fresh copy of CrisisWatch and reads it |
| Drought water-balance feed | 8th, 05:40 | Rebuilds the SPEI-3 country figures from Copernicus data |
| Monthly refresh | 11th, 03:00 | Refreshes every source, runs the resolution machine, then grades, scores, recalibrates and publishes |
| Regime-change recall check | 12th, 04:37 | Scores both regime-change readers against real breaks |
| Forecast pipeline | 13th, 00:00 | Scan, questions, forecasts, Sibyl, report, publish |
| Batch watcher | Every 2 hours, 13th to 17th | Backstop that starts the next forecast stage when batches finish |
| Weekly refresh | Sundays, 03:00 (not 10th to 13th) | Conflict forecasts, GDACS, ReliefWeb, ACLED political events, GDELT, CrisisWatch |
| Hazard history | Nightly 20:30 (not 10th to 12th) | Extends the resolution machine's history and base rates |
| Database compaction | 4th and 20th, 04:17 | Reclaims space in the working database |
| Billing reconciliation | 1st, 05:47 | Compares recorded spending with Anthropic's and OpenAI's bills |
| Drive archive | Mondays, 05:23 | Copies any finished cycle not yet archived to Google Drive |
| Analysis report sync | Daily 09:41 | Copies the written reviews from Drive to the public download area |
| Watchdog | Daily 11:00 | Fails loudly if any job above has not succeeded within its expected interval, or if the dashboard serves stale data |

Scheduled jobs on GitHub often start hours late, which is why the monthly refresh runs two days before the forecast and the watchdog exists.

### The forecast pipeline

The forecast runs in four stages. The first scans countries and sends the regime-change requests as a batch; the second collects them and sends triage; the third collects triage, creates questions and sends the forecasts; the fourth collects the forecasts and finishes. A watcher, started by each stage, checks the batches every few minutes and starts the next stage when they are done. If a provider refuses a batch, Fred retries over a fixed budget of minutes, then re-batches at collection and waits up to an hour and a half before asking one by one at full price. A slow batch is given up to a day. Before any spending, every provider is asked whether it will answer; one that refuses is left out of the run.

### Cost tracking

Every model call and web search is logged with its tokens, price and stage, at the batch or cached rate where one applies. Prices that end on a date carry the date and the price after it. The Costs page and the run's health report count the same calls, so their totals agree. On the 1st of each month Fred compares its own totals with the bills Anthropic and OpenAI report and shows any difference above 15% in red.

### The database

Every stage reads and writes one working database, which travels between jobs. Only one job may change it at a time. Before saving, each job checks that nobody saved a newer copy while it worked, and stops rather than overwrite it; a job started by another first checks that its trigger actually produced a database. Every job checks there is room for the database before opening it and records how much it added. Each saved database must contain the tables it should, or it is not treated as the real one. Compaction twice a month keeps it from growing without bound.

### Diagnostics and archive

Each run keeps a record: every model call, its cost and its result; the reasons questions were included or skipped; and a list of anomalies, worst first. A debug bundle is built at the end of each forecast run. It opens with that list of anomalies and holds the triage summaries, every model call, the batch and cost records, connector freshness, each model's completeness, a comparison with the previous two runs, and the workflow logs. The monthly refresh builds a similar bundle for its data: which address each source was asked, what it answered, why each graded cell got no figure, and every table's row count and freshness. The publish step gathers the whole cycle's diagnostics into one file. Credentials are scanned for and removed before any of these is written. Every finished cycle is copied permanently to Google Drive with a compressed copy of the database. A register of known faults reports each problem once, with its cost, its owner and how long it has been present.

### Security and the public copy

Fred publishes a copy of its database for the dashboard. Before upload, every text column is scanned and cleaned of anything that looks like a password or key; the upload stops if anything survives. The copy leaves out data Fred may not republish (raw ACLED events, EM-DAT figures, ACAPS's analytical products and the pages Sibyl read), though the monthly totals derived from them are kept. Keys never travel inside web addresses. The data service limits how fast any one visitor can ask, caps the size of any answer, and does not publish a map of its internals. Links written by a model are shown as links only if they are ordinary web addresses. The dashboard's data service checks the release every minute and switches to a new database on its own, refusing a download that is incomplete or does not match the published fingerprint. Its status reports the database it actually serves; a check after every publish, and again daily, compares that with the release and fails loudly when they differ. Until the service catches up, the dashboard shows a banner rather than silently showing old data.

### Model configuration

Every model is named once in a central configuration, with its price per million tokens kept separately. Each model's thinking effort is set explicitly rather than left to the provider's default, which can change between versions. The ensemble and every task (triage, regime change, Track 2, scenarios, search fallbacks, the report) refer to those names, so swapping a model is a one-line change plus its price.

### Test mode

A run in test mode is identical to a real one, except that everything it writes is marked as test data. The dashboard hides test data unless the menu switch is on, which also shows an amber banner. Test data never reaches calibration, scoring or advice, a test run never changes a real question, and Sibyl's research is skipped on test runs unless deliberately enabled. A run started by mistake without test mode can be marked as test data afterwards.

## 12. Switched off or retired

- **Silenced hazards.** Displacement inflows, civil unrest, heatwaves and the old armed-conflict code are blocked at every stage, because no reliable source can grade them. Two further codes (epidemics and economic crisis) were never switched on.
- **Hazard tail packs**, a third round of searches for flagged hazards, were retired because the counter-evidence check and the scan's own evidence already covered the same ground.
- **The question-level research pipeline**, in which a model wrote a research brief for each question, was replaced by the structured feeds, which are reproducible.
- **Prediction markets** (Metaculus, Polymarket, Manifold) are switched off in production because their services refuse or time out. The system's own default would switch them on, so any new job must keep them off explicitly.
- **Model self-search**, in which a forecasting model could ask for one extra web search, is off. It needs three settings together, so a prompt can never offer a search the system would refuse.
- **People-affected questions for drought** are not asked; drought is forecast as food crisis.
- **EM-DAT** is not read, for want of an account (Section 2).

## 13. Practical guidance for forecasters

Fred supports human judgement rather than replacing it. A good way to use it:

1. Start with the horizon scan and the regime-change flags.
2. Read the structured feeds and the web evidence gathered for flagged hazards.
3. Read the counter-evidence to see the case against the flag.
4. Decide what would move the distribution, and by how much.
5. Make sure the forecast holds both the usual outcome and the plausible tail.
6. Write explanations that someone could later prove wrong.

## Changes in this Version

This section summarises the changes from July to December 2026, the period in which the system took its current shape. Every earlier version of this document can be read in full from the version selector above.

### November 2026

- **About page restructured.** This document now opens with a short summary, describes the system as it is now, and keeps the change log at the end. Several statements were corrected against the code, and the page no longer loads every archived version at once.
- **More conflict displacement figures kept.** ReliefWeb reports are now read for every listed conflict country, up to fifteen a month, most relevant first. A figure about another month is credited to that month, and a blank "new or running total" answer is read again rather than discarded.
- **A monthly report shows where displacement figures are lost.** Before any rule is loosened, Fred measures each step from report to accepted figure, with examples of every rejection.
- **A second regime-change reader, for comparison only.** GPT-6.1 Sol now receives the same regime-change prompt as Gemini 3.8 Flash. Its answers never change anything; a monthly check on the 12th scores both against real breaks.
- **December model line-up.** GPT-6.1 Sol, Gemini 4 Argon and Gemini 3.8 Flash replace GPT-6 Sol, Gemini 3.1 Pro and Gemini 3.6 Flash. Each inherits its predecessor's record. Gemini 4 Argon is a new generation, and its settings are now sent correctly.
- **Performance Analysis page.** The written skill, grading and pipeline reviews are published by month, readable on the page or as PDF and Markdown.
- **Slide decks on the About page.** Eight decks explain Fred part by part, and a check reminds contributors to update them when the system changes.
- **Counter-evidence experiment.** From the December run, half of eligible questions are forecast without the counter-evidence check, because models that cited it moved away from the outcome more often than toward it.
- **Dashboard outage fixed.** For about five days the dashboard could not read its data, written by a newer database library. Every install is now held to a compatible version, and an unreadable file is refused.
- **ReliefWeb figures dated by their own words.** A figure about an earlier event or another hazard no longer grades a month, and a stated count of people beats a household conversion.
- **Floods and cyclones give no answer for a month in progress**, as drought already did.
- **Performance page.** All three scores beside the toughest benchmark, a monthly trend, and a skill bar for each model; new models show as "awaiting first scores".
- **Displacement admission recorded.** Each run records which source made a country eligible for displacement questions, and a newly counted source is held out of learning until six of its months have resolved.

### October 2026

- **Monthly cycle moved to the 11th and 13th**, from the 28th and 1st, so the forecast sees a complete month of ACLED data, the new CrisisWatch edition and the new rainfall outlook.
- **Provider pre-flight.** Each run asks every AI provider whether it will answer before spending anything. A provider at its spending limit is left out, and Sibyl and the report do not start without Claude.
- **Billing reconciliation.** On the 1st, Fred compares its recorded costs with Anthropic's and OpenAI's bills; the Costs page shows both.
- **Sibyl's cost cut by three quarters.** Each research step now carries a compact record of earlier steps. Sibyl takes 15 questions with two trials each, a reconciler when they disagree, and a $25 cap.
- **Cheaper prompt caching.** Claude's reusable prompt section now ends where the shared text ends, so far more is read back at the cached price.
- **Fewer wasted web searches.** The scan stops searching once a source pack is full.
- **Flood and cyclone questions graded by the resolution machine.** They were graded only where IFRC published a matching report; now 47 of 66 flood months and 24 of 33 cyclone months due in October had grades.
- **Conflict displacement graded on a ladder.** IDMC first, then OCHA or UN figures from ReliefWeb, then IOM's rise in displaced people as a floor; tenfold disagreements are flagged.
- **Conflict displacement asked only where reported regularly.** Other countries cost model calls for forecasts that could never be marked.
- **Conflict displacement fixed at the source.** IDMC figures for every cause had been summed, so typhoon evacuations read as conflict. Fred now keeps only conflict, IDMC's recommended figures, and settles a month only after 90 days.
- **Food-crisis questions only where figures are current**, within nine months; the IPC is read for a FEWS NET country once FEWS NET stops publishing.
- **FEWS NET ranges shown in full**, with the question graded on the lower end.
- **Rainfall outlooks as chances.** The prompts show the chance of a dry and a wet month beside one in three, not a fixed millimetre cut-off.
- **Regime-change wording experiment.** Half of full-ensemble questions are told to move their forecast in the flagged direction rather than widen it.
- **Shared advice as observations.** Advice no longer gives orders, no longer contradicts the starting distribution, and is never borrowed from another hazard.
- **The latest forecast counts.** When a window is forecast twice, the latest production forecast is used everywhere, and test runs never count.
- **Test runs forecast every question they ask**, without touching real questions.
- **Sibyl reads the grading source first**, and its first trial is shown Fred's structured data while the second is not.
- **Sibyl's alternatives, failure labels and second model.** Rearrangements of its trials are scored at no cost; post-mortems name the kind of mistake; a GPT trial is scored on the side, never published.
- **Sibyl forecasts need evidence.** A forecast whose trials never completed a search is not published, and July's evidence-free forecasts are left out of Sibyl's record.
- **CrisisWatch said aloud.** Every conflict prompt names the newest edition and whether the country is listed; Palestine is counted under "Israel/Palestine".
- **Five prompt blocks corrected**: rainfall stored in the right unit, displacement flows never negative, INFORM severity on its 0 to 10 scale, food-crisis history counted honestly, and one base-rate block per prompt.
- **The report compares itself with the right month**, and reports that fail validation are marked UNVALIDATED.
- **The dashboard picks up new data on its own** and warns when it is behind.
- **Security before going public.** The public copy is scanned for keys and leaves out data Fred may not republish; the data service limits visitors; Sibyl opens only public pages; the report PDF cannot pull in files.
- **Database kept in check.** Re-checks of frozen answers write a line only when something differs, and compaction runs twice a month.

### September 2026

- **Conflict-death base rate as a distribution.** Each conflict-death prompt shows a starting distribution and tells the models to copy it; "last month" is always a complete month.
- **Advice follows model upgrades** and half of questions get none, so its effect can be measured; corrections for each family's lean are applied before pooling.
- **New models earn their weight.** A new version starts from its predecessor's record, retired versions drop out, and candidate models can run as shadows.
- **"Same as last month" benchmark** added; summaries count each question once; revised outcomes are kept.
- **Conflict deaths graded on all event types**, the series they are forecast from, rather than battles alone.
- **Drought-alert history counted in calendar months**, and each ensemble member's yes/no answer is stored and scored.
- **ENSO computed from the index** rather than read from a web page, after a strong El Niño was recorded as neutral.
- **Drought grading no longer depends on one feed**, needs two indicators for a zero, and gained the SPEI-3 water-balance feed reaching back to 2016.
- **Resolution machine fixes**: no ceiling of zero, a three-fold allowance over GDACS exposure, no GDACS ceiling for floods, cached sources used when a source is down, and the history replayed newest first.
- **Silence sweep scoped.** A ReliefWeb report naming a country no longer counts as a disaster there; only its main country counts.
- **Batch path rides out outages.** Refused batches are retried and re-batched before falling back to full price.
- **Model upgrades**: Claude Opus 5.5, GPT-6 Sol and GPT-6 Luna, with thinking effort set explicitly.
- **The report's worsening test rebuilt** to ask whether the planning figure moved, not only whether the burden is large.

### August 2026

- **Resolution machine runs on a schedule.** It grades the last three months on each monthly refresh and replays history nightly, with its spending on the Costs page.
- **The machine tells the models how they will be graded**, generated from its rulebook.
- **The monthly report** launched and was rebuilt: entries ordered by extra people expected, a decision and deadline for each, a second-reader section, and a sector comparison.
- **Core data refreshed before the forecast**, not two weeks after it.
- **Public release kept under the size limit** by leaving out the machine's working tables.
- **Staged batch pipeline in production**, at half the model price with identical forecasts.

### July 2026

- **Sibyl launched**, an independent forecaster researching the open web.
- **Resolution machine built** for cyclones, floods and drought.
- **Model line-up renewed** and three providers retired; thinking settings now take effect.
- **Zero bands and a 1,000+ fatality band** added to every banded measure.
- **Forecasts cover exactly the months each question asks about.**
- **Scores use the standard textbook formula**, and months with no data are no longer scored as "nothing happened".
- **Past accuracy feeds back into the ensemble** through calibration weights.
- **Regime change and triage run one pass**, since a second identical pass added nothing.
- **Counter-evidence reaches the forecasters at level 1**, where it had been generated and never shown.
- **Hazard tail packs retired.**
