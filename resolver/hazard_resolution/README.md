# Hazard resolution machine

**What this is.** For every country, month, and hazard (flood, drought,
tropical cyclone), this module produces exactly one answer to the question
*"how many people were affected?"* — either a credible number, a credible
zero, or an explicit "no data". Every answer carries full provenance: which
source said it, which records or documents it came from, when we retrieved
them, and which rule made the decision. The goal is that at least 80% of
country-month-hazard cells resolve to a number (including zero) rather than
"no data" — today's IFRC-GO-only resolution manages about 4%.

**How it decides — two layers.**

1. **Detection** — did a qualifying hazard happen at all? Physical and event
   sources answer this: IBTrACS storm tracks for cyclones, GDACS alerts for
   floods, IPC food-security data for droughts. If nothing qualifying
   occurred *and* a ReliefWeb keyword sweep confirms silence, the cell
   resolves to **zero**, and we record the evidence of absence.
2. **Impact** — if a hazard was detected, walk a fixed ladder of reporting
   sources, best first: **EM-DAT → figures extracted from ReliefWeb
   documents → IFRC GO field reports → IDMC displacement (a lower bound)**.
   The highest rung with a figure wins. The figure is sanity-checked against
   the GDACS exposed-population estimate (a ceiling — exposure is never used
   as the answer itself) and against national population. If the freeze date
   passes with no figure, the cell resolves to **"no data"** and is flagged
   for human review. Drought skips the ladder entirely: it resolves as the
   increase in IPC Phase 3+ population between consecutive analyses,
   attributed to drought only where independent drought indicators agree
   (Phase 4, below).

**The rulebook.** Every threshold and policy switch lives in
[`rulebook.yaml`](rulebook.yaml) — cyclone wind/distance thresholds, the
GDACS alert colour that counts as a flood, the ladder order, the freeze
window, sanity-check multipliers, backcast start years. Change the YAML and
the machine behaves differently on its next run; nothing needs a programmer.
The file is validated on load and refuses anything that looks like a
credential (API keys come from environment variables only).

**Hard guarantees.**

- Reconciliation is deterministic: rules only, no AI judgement calls.
- The one AI step (reading ReliefWeb documents) only transcribes figures a
  document actually states — it never estimates or infers a number.
- Resolutions freeze 60 days after month-end and are never reopened. Later
  source revisions are logged in an audit table (`haz_revisions`) but do not
  change resolved values.
- GDACS is used for detection and plausibility ceilings only, never as a
  resolution value.

**Where the data lives.** All tables sit in the resolver DuckDB alongside
the existing `facts_resolved`: thin raw caches per source (`haz_raw_*`),
detection verdicts (`haz_triggers`), candidate figures
(`haz_impact_candidates`), the document-extraction cache and cost ledger
(`haz_doc_extractions`), final answers (`haz_resolutions`), the post-freeze
audit log (`haz_revisions`), and historical base rates
(`haz_base_rates_occurrence`, `haz_base_rates_severity`). See
[`schema.py`](schema.py).

**Running it.**

```bash
# Create/refresh the haz_* tables (idempotent, safe on a live DB)
python -m resolver.hazard_resolution.migrate

# Load national population denominators into haz_raw_population
python -m resolver.hazard_resolution.population

# One month, one hazard — detection, zeros, and the impact ladder
resolve-hazards --hazard cyclone --month 2026-06
resolve-hazards --hazard flood   --month 2026-06

# Drought: IPC Phase 3+ delta, gated by drought indicators (no ladder, no spend)
resolve-hazards --hazard drought --month 2026-06

# Three months in one run (each month is independent)
resolve-hazards --hazard flood --months 2026-04 2026-05 2026-06

# Backcast an old month from the full IBTrACS archive, limited countries
resolve-hazards --hazard cyclone --month 2013-11 --scope ALL --countries PHL MDG

# Re-run on the existing stores without downloading
resolve-hazards --hazard cyclone --month 2026-06 --skip-fetch

# Detection and zeros only, no ladder
resolve-hazards --hazard flood --month 2026-06 --no-ladder

# Run the ladder but skip the one paid step (no model calls, no spend)
resolve-hazards --hazard flood --month 2026-06 --no-extract

# --- Phase 5: history, base rates, and the acceptance report ---

# Replay a hazard over its full backcast window (resumable; hours, not minutes)
python -m resolver.hazard_resolution.backcast --hazard flood
python -m resolver.hazard_resolution.backcast --hazard cyclone --to 2015-12
python -m resolver.hazard_resolution.backcast --hazard flood --no-extract  # spend nothing

# Rebuild the occurrence and severity base rates from what the backcast wrote
python -m resolver.hazard_resolution.base_rates

# The one-page acceptance report (last 12 fully frozen months)
python -m resolver.hazard_resolution.acceptance --out acceptance.md
```

## Phase 1: the cyclone path (implemented)

Per (cyclone, month) the CLI: fetches IBTrACS (idempotent) → writes a
`haz_triggers` row for **every** country in the universe (triggered and
not) → sweeps ReliefWeb for each non-triggered country-month → writes
`RESOLVED_ZERO` rows into `haz_resolutions` where the sweep confirms
silence. Non-silent sweeps flip the trigger
(`trigger_source='reliefweb_sweep'`) and leave the month for the impact
ladder; failed sweeps are *inconclusive* and never produce a zero
(fail-closed).

**Detection rule.** A cyclone triggers for a country-month when any
IBTrACS track point qualifies under `rules.cyclone_track_qualifies`:
within `cyclone.buffer_km` (default 200 km) of the country's territory
at `cyclone.min_wind_kt` or stronger (default 34 kt — tropical-storm
strength). Wind is read per `cyclone.wind_source_priority` (USA
1-minute sustained first, WMO agency wind as fallback). A storm-point's
month is its timestamp's month; a storm spanning a month boundary can
trigger both months.

**IBTrACS storage.** [`ibtracs.py`](ibtracs.py) fetches the NOAA v04r01
CSV (`last3years` rolling file by default; `--scope ALL` for backcast)
and stores **one row per storm** in the generic `haz_raw_ibtracs` cache:
`record_id` = storm id, the full track in `payload_json`,
`content_hash` over the payload — so an identical re-fetch is a no-op
while an upstream best-track revision appends a new row, and detection
always reads the newest revision.

**Geometry choice.**

- **Boundaries**: a vendored, slimmed **Natural Earth 1:50m Admin-0
  countries** layer
  ([`data/ne_50m_admin_0_countries.slim.geojson.gz`](data/), public
  domain). 1:50m rather than 1:110m because the coarser layer drops the
  small island states (Tonga, Tuvalu, Kiribati…) that matter most for
  tropical cyclones. Vendored rather than downloaded at runtime so CI
  is network-free and detection doesn't depend on a CDN. Properties are
  reduced to `{iso3, name}`; coordinates are rounded to 4 decimals
  (~11 m — negligible against a 200 km buffer); Natural Earth's `-99`
  ISO codes are resolved via `ISO_A3_EH`/`ADM0_A3` and remapped to the
  resolver universe (Kosovo→RKS, Somaliland→SOM, N. Cyprus→CYP).
  Regenerate with [`data/build_boundaries.py`](data/build_boundaries.py).
- **Distances**: the country geometry is clipped to a local window and
  re-projected into an **azimuthal equidistant (AEQD) projection
  centred on the track point**; the distance to the projection origin
  is then the true geodesic distance from the point to the nearest
  territory (0 over land). AEQD preserves distance from its centre
  exactly, so the buffer test is accurate where it matters and immune
  to the lon/lat anisotropy a degree-based buffer would suffer. The
  antimeridian is handled by splitting clip windows at ±180° (the
  projection itself is periodic in longitude).
- **Consequence of Admin-0**: territories that Natural Earth folds into
  their parent country (e.g. the French overseas departments) resolve
  to the parent ISO3; universe entries with no Admin-0 geometry are
  skipped by detection and logged.

**Zero-resolution safety.** Two gates in front of every zero:

- **Coverage gate**: zeros are only written when the newest stored
  IBTrACS point is no earlier than `month_end −
  cyclone.ibtracs.coverage_grace_days`. Months inside ingestion gaps
  stay unresolved instead of becoming false zeros (same month-gate
  principle as `compute_resolutions`' zero defaults).
- **ReliefWeb silence sweep**: two recorded queries per country-month —
  a disaster-type taxonomy filter, then (if silent) a full-text keyword
  search — over `country.iso3` and a publication window of month start
  → month end + `publication_pad_days`. Silence means
  `total hits ≤ max_hits_for_silence`. Every query, hit count, and
  sample title/URL is stored in the zero's provenance
  (`evidence_of_absence`), and a silent sweep also lands in the
  trigger row's `evidence_of_absence_json`.

**Environment.** `RESOLVER_DB_URL` (or `--db`) targets the DB;
`RELIEFWEB_APPNAME` is the ReliefWeb `appname` parameter (falls back to
`pythia-resolution-machine` with a warning). IBTrACS and the vendored
boundaries need no credentials.

## Phase 2: floods and the impact ladder (implemented)

Phase 2 adds the second detector and the whole of Layer 2 — the ladder
that turns "a hazard happened here" into a number.

**Flood detection.** A country-month triggers when a GDACS flood event
naming that country, overlapping that month, carries an alert level at or
above `flood.gdacs_trigger_level` (green &lt; orange &lt; red). Non-triggered
country-months get the same ReliefWeb silence sweep, the same fail-closed
handling and the same coverage gate as cyclones — zeros are suppressed
unless the newest stored GDACS event reaches `month_end −
flood.gdacs.coverage_grace_days`.

**Event-to-month attribution.** Two decisions, deliberately different:

| | rule | rulebook key |
|---|---|---|
| **Detection** | an event is detected in **every** month its date range overlaps | `event_attribution.detection: overlap` |
| **Figure** | its people-affected figure is attributed **wholly to the start month** | `event_attribution.figure: start_month` |

So a flood running 28 Jan – 4 Feb triggers *both* months, while its
reported figure lands entirely in January. Figures are **never split**
across months: a source states one number for the event, and
apportioning it would invent a monthly breakdown nobody reported. The
full span travels with the figure in provenance (`event_span`), so a
reader can always see that it covers more than the month it sits in.

**The ladder.** For each triggered country-month, candidate figures are
collected into `haz_impact_candidates` and the reconciler walks
`rulebook.ladder`:

1. **EM-DAT** — curated, definitionally stable, and slow (hence the freeze
   window).
2. **ReliefWeb-extracted** — figures a cheap model transcribed out of
   humanitarian reporting (Phase 3, below).
3. **IFRC GO** — a real stated `num_affected`, thin coverage.
4. **IDMC IDU** — displacement, i.e. a **lower bound**.

The first populated rung wins. Nothing below it can overturn that: a
lower rung that disagrees is a reason to **flag**, never to substitute.

**Sanity checks bound the answer; they never rewrite it.** The figure is
checked against `sanity.ceiling_multiplier` × the GDACS exposed
population and against national population (`sanity.population_cap`). A
breach is kept and flagged — `conflict_rule: ladder_with_flag`. So is a
"wild conflict", defined deterministically as adjacent populated rungs
differing by more than `conflict_detection.order_of_magnitude_factor`.

**Lower bounds are labelled at every layer.** An IDU figure resolves with
`value_type='displaced_lower_bound'`, `rule_fired='ladder:idmc_idu:lower_bound'`,
and an explicit note in provenance. Displaced people are a strict subset
of affected people, and losing that label would silently turn a floor
into an answer.

**GDACS is still never a value.** Its exposure enters
`haz_impact_candidates` only as `value_type='exposed_ceiling'`, which
`ladder_candidates()` filters out of the ladder's view. It is recorded as
a candidate so the ceiling that applied to a decision is visible in the
record.

**When no rung has a figure**, the answer depends on the clock:

- **before** the freeze deadline → **pending**: no row is written at all.
  EM-DAT alone routinely lands weeks after an event, so declaring "no
  data" the week of a flood would record our impatience as a fact about
  the world.
- **on or after** it → `NO_DATA`, flagged for human review, with the
  rungs consulted recorded so the gap is legible.

**Provisional vs. final.** The core resolver tables have no concept of
provisional data (`facts_resolved.confidence` describes source quality,
not finality), so `haz_resolutions` carries its own `provisional BOOLEAN`.
An answer written before month-end + `freeze_days` is real and usable but
may still change; from the deadline on it is written final and becomes
immutable, and a later re-run logs to `haz_revisions` instead.

**A rung that could not be read is not a rung that was empty.** Every
source connector returns an outcome rather than raising, and a failure
(missing key, API outage) is recorded on the resolution row under
`decision.sources_unavailable`. Only "the source answered and said
nothing" can justify a zero.

**Reuse.** GDACS event discovery and per-event enrichment come from
`resolver/connectors/gdacs.py`, and IFRC GO paging/country resolution from
`resolver/ingestion/ifrc_go_client.py`, rather than being re-implemented.
Consequently the GDACS endpoints live in that connector and **not** in
`rulebook.yaml` — duplicating a URL would create two sources of truth for
one address. A test asserts the borrowed helpers still exist, so an
upstream rename fails loudly in CI rather than at runtime in production.

**Environment.** `EMDAT_API_KEY` (EM-DAT), `IDMC_API_KEY` (sent as the IDU
`client_id`), `RELIEFWEB_APPNAME`. GDACS, IFRC GO, IBTrACS and the
vendored boundaries need no credentials. Keys are read from the
environment only; the rulebook loader rejects any key that looks like one.

The IDU rung accepts **either** `IDMC_API_KEY` or, when that is unset,
`IDMC_HELIX_CLIENT_ID`. They are the same IDMC credential: both are sent as
the `client_id` query parameter to `helix-tools-api.idmcdb.org`, this rung
against `/external-api/idus/all/` and `resolver/ingestion/idmc` against
`/external-api/idus/last-180-days/`. One secret therefore serves both
consumers, rather than two secrets holding one value and drifting apart.
`IDMC_API_KEY` wins when both are set, so this rung can be pointed at a
different client id deliberately.

`IDMC_API_TOKEN` is **not** accepted and must never be repurposed here: it
is a bearer token for a different host (`backend.idmcdb.org`), and its mere
presence is a feature flag — `scripts/ci/run_connectors.py` derives
`RESOLVER_SKIP_IDMC` from it, so moving it would silently switch off the
whole IDMC ingestion path. With neither client id set, the rung reports
UNAVAILABLE, which the resolution row records as unread rather than empty.

## Phase 3: reading the documents (implemented)

Rung 2 is the machine's only paid step, and the only place a model
touches a resolution. It exists because most floods are never written up
in EM-DAT and never reach an IFRC field report — but somebody, somewhere,
published a sentence with a number in it.

**What the model does.** One call per document, with a single job:
transcribe every people-affected figure the document explicitly states,
each with the sentence it came from. It is forbidden to estimate, to
convert units, to sum across areas, or to use anything it knows that the
document does not say. If the document states nothing, an empty list is
the right answer.

**Why that holds.** Not because the prompt says so. Three mechanisms:

- **Quote verification** — every figure must arrive with a quote, and the
  quote must be found in the document text that was actually sent (after
  whitespace/typography normalisation only, so no words are dropped from
  either side). A figure whose quote cannot be found is discarded with a
  reason. To fabricate a number, a model would have to fabricate a
  sentence that is already in the document.
- **A closed schema** — values must be numbers, units must be `people` or
  `households`. Anything else is rejected, never coerced. A households
  figure silently read as people understates by about five.
- **No arithmetic anywhere near the model** — everything a figure could
  be *turned into* happens afterwards, deterministically, in
  [`figures.py`](figures.py).

**Document selection** ([`reliefweb_docs.py`](reliefweb_docs.py)). For
**triggered country-months only**: one ReliefWeb query per cell filtered
on country, the hazard's disaster types (the same taxonomy the silence
sweep uses, so detection and extraction cannot disagree about what a flood
is), the content formats in `reliefweb.documents.formats`, and the month
window plus `publication_pad_days`. Hits are ranked by
`documents.source_priority` (OCHA, then governments, then UN agencies…),
ties broken by recency then document id, and the top
`max_docs_per_cell` (default 30) are cached in `haz_raw_reliefweb_docs`.
Bodies are stored **already truncated** to `body_char_limit`, because the
quote verifier checks against the text that was sent.

**Deterministic post-processing** ([`figures.py`](figures.py)), in order:

1. **Households → people**, and only when the document said households.
   The multiplier comes from `reliefweb.household_conversion`
   (per-country, with a declared default) and the whole conversion —
   stated value, multiplier, where the multiplier came from — is recorded
   on the candidate, so a reader can undo it.
2. **Ceiling rejection.** A figure above `sanity.ceiling_multiplier` ×
   GDACS exposed population is **rejected** here, with a logged reason.
   This is deliberately stricter than the reconciler, which flags a
   ceiling breach and keeps the value: there, a curated source
   disagreeing with modelled exposure is a finding worth a human's time;
   here, it is far more likely a mis-transcription — a regional total, or
   a figure about a different emergency in the same document — and
   admitting it would let a transcription error outrank an EM-DAT record.
3. **Deduplication.** The same sentence republished across five situation
   reports is one statement; the same figure for the same area worded
   differently is one figure. Both collapse, and what was dropped is
   recorded.
4. **Precedence.** `reliefweb.authority_precedence` orders attributions —
   government > UN agency > IFRC/NGO > media — and within a tier the
   latest-dated figure wins. Unrecognised attributions rank last but are
   **not** discarded: an unattributed figure in an OCHA sitrep is still
   the best evidence in the room when nothing else is available.

Survivors are written to `haz_impact_candidates` with the quote, the
document URL, the extraction model, and a `preference_rank` — 0 being the
figure the rulebook prefers. The reconciler reads that rank instead of
its usual "largest stated figure wins", because when several bodies quote
different assessments of one flood, the biggest number is not the right
answer.

**Cost control.** Three guards, because this is the only line item:

- documents are capped per cell (`documents.max_docs_per_cell`);
- calls are capped per **calendar month across every run**
  (`extraction.max_calls_per_month`), counted from the
  `haz_doc_extractions` ledger — so three CLI invocations in one month
  share one allowance rather than each spending it;
- a cell whose **higher** rung already has a figure is skipped entirely
  (`extraction.skip_when_higher_rung_populated`, on by default).
  `reliefweb_extracted` sits below EM-DAT and cannot win those cells, so
  reading them buys only adjacent-rung conflict flags — at roughly a
  quarter of a dollar per cell. Turn it off to buy them anyway.

Extractions are cached in `haz_doc_extractions` keyed by (document, model,
prompt version), so **a re-run costs nothing** and the monthly cap counts
real spend rather than repeated work. Bumping `extraction.prompt_version`
is what invalidates the cache.

A cell that ran out of budget records `budget_capped` in its provenance:
its remaining documents are **unread**, which must never be mistaken for
ReliefWeb having been silent.

**Model choice.** The rulebook names a *role* (`extraction.model_role:
hazard_extraction`), not a model id; `pythia/config.yaml` maps that role
to a Haiku-class model. The rulebook owns the policy, the model registry
owns the model — putting an id in the rulebook would fork the repo's
single source of truth for model choice.

**Budget arithmetic.** A Haiku-class call over a 24,000-character document
is roughly 6k input + 500 output tokens, about USD 0.009. At
`max_calls_per_month: 1500` that is near USD 14/month — comfortably inside
the USD 50 ceiling, with room for a backcast. Every other source in the
machine is free.

**Environment.** `RELIEFWEB_APPNAME` and the provider key for whichever
model backs `hazard_extraction` (`ANTHROPIC_API_KEY` by default).

## Phase 4: drought via IPC (implemented)

Drought is the one hazard with no event. Nothing makes landfall, no alert
goes orange, and there is no day on which a situation report counts the
affected. What there is, is a food-security analysis cycle — so drought
skips the ladder entirely and resolves as:

    people affected(country, month)
        = max(0, Phase3+(analysis covering the month)
                 − Phase3+(previous current-period analysis))

admitted as **drought** impact only where the drought indicators say the
country was in drought. No model is called anywhere on this path, so a
drought run costs nothing.

**Why the indicators are load-bearing.** IPC's structured feeds state how
many people are in Phase 3+; they do not state *why*. Driver attribution —
drought vs conflict vs economic shock — lives in narrative PDFs. Without an
independent signal, the rule would file South Sudan's conflict-driven food
insecurity as drought impact, which is the same definitional blending the
resolver's base-rate rule forbids. The indicators are what closes that gap,
and they do the same work at the other end: a drought zero rests on two
statements (no indicator signal **and** no IPC deterioration), so an
indicator that could not be *read* suppresses the zero rather than
permitting it.

**Where the figures come from** ([`ipc.py`](ipc.py)). Two acquisition
paths into one cache:

- **`ipc_api`** — the IPC public API, authenticated with the `key=` query
  parameter the repo's existing IPC connector already uses. It states the
  current-period window explicitly.
- **`facts_resolved`** — the `phase3plus_in_need` rows the repo's FEWS NET
  and IPC connectors already write, where `ym` is the window's first month
  and `as_of_date` its last. Needs no key, and covers the FEWS NET
  countries that `resolver/connectors/ipc_api.py` deliberately *excludes*
  to avoid double-writing facts_resolved — which are exactly the countries
  where drought resolution matters most.

An analysis seen by both paths is one analysis: they collapse per validity
window, preferring whatever `drought.ipc.source_priority` lists first. Only
the CURRENT period is cached; the projection is a forecast, and a forecast
is not a resolution source. Period-date parsing is *borrowed* from the
existing connector rather than re-implemented, with a guard test — two
parsers for one free-text format would eventually disagree about which
months an analysis covers.

**Where the drought signal comes from**
([`drought_indicators.py`](drought_indicators.py)). Each entry in
`drought.indicators.entries` is fetched once per run as a global feed and
cached as **values, not verdicts** — thresholds are applied at evaluation
time, so retuning one changes behaviour on the next run with no re-fetch
and no code change. Two providers:

| provider | shape | absence of a country means |
|---|---|---|
| `asap` | JRC ASAP agricultural hotspot classes, free, no key | **no warning issued** — which is what makes it usable as evidence of absence |
| `tabular` | a pre-computed `(iso3, ym, value)` CSV/JSON anomaly feed compared against a threshold | **unknown** — an anomaly feed that omits a country says nothing about it |

CHIRPS and SPEI are gridded rasters, and turning those into a country
number needs zonal statistics that have no business running inside the
resolution path; the `tabular` provider is how a maintainer points the
machine at whatever country-level anomaly product they already produce.
Both entries ship with an empty `url` (not consulted) and
`required: false`, so their absence cannot suppress zeros everywhere.

Three guards, each learned from a failure mode elsewhere in the repo:

- **0-of-N resolution refuses the feed.** If no record resolves to a
  country, the feed's shape changed — that is not evidence the world is
  drought-free, and the snapshot is rejected rather than cached.
- **Staleness is bounded** by `max_observation_age_months`. A feed with no
  date is stamped with its *retrieval* month, so a backcast month falls
  outside the window and correctly resolves to nothing instead of a false
  zero. Point an entry's `url` at a per-month archive (it may contain
  `{ym}`, `{year}`, `{month}`) to backcast properly.
- **A later snapshot is never read backwards.** A September warning is not
  evidence about March.

**Which months a delta resolves.** An analysis is valid over several
months but states ONE figure, so the delta has no monthly breakdown and the
machine may not invent one. `drought.month_attribution` decides:

- `analysis_window` (shipped) — every month the window covers resolves to
  the delta. The newly-Phase-3+ population is treated as persisting through
  the window the analysis declares it for, which is what a slow-onset
  hazard does. **These values must never be summed across months**: each
  answers "how many people were affected in this month", a stock, and the
  same people appear in every month of the window. The resolution says so
  in its own provenance note.
- `start_month` — only the window's first month carries the delta;
  the rest resolve to zero ("no NEW deterioration reported"), mirroring
  `event_attribution.figure`.

**The five outcomes** ([`drought.py`](drought.py) — `decide_drought` is a
pure function of analyses, indicator verdict, rulebook, population and
today, exactly as `reconcile.reconcile` is for the ladder):

| outcome | when |
|---|---|
| `RESOLVED_VALUE` | indicators dry, two analyses in hand, increase clears `min_delta_people` / `min_delta_pct` |
| `RESOLVED_ZERO` | *nothing happened* (no signal **and** no deterioration) — or the analyses were read and state no qualifying increase. **Two different zeros, two different `rule_fired` strings**, because "we know it was quiet" and "the number came out at zero" are different findings |
| `NO_DATA` | indicators dry but no IPC analysis covers the month by the freeze deadline; or no previous analysis to difference against; or a deterioration the indicators do not attribute to drought. All flagged |
| `PENDING` | the same situations *before* the freeze deadline. IPC publishes months late, so nothing is written |
| `INCONCLUSIVE` | a required indicator could not be read. Fail-closed, exactly as a failed ReliefWeb sweep is — nothing is written, and the trigger row keeps the evidence |

Two of those deserve their own note. A covering analysis with **no
baseline** resolves to NO_DATA rather than to its own Phase 3+ population:
that figure is a *stock*, and publishing it as people-affected-this-month
would overstate by the entire pre-existing caseload. And a **deterioration
without a drought signal** publishes no figure at all — the increase is
real, but it is not drought's.

**Same tables, same freeze logic.** Triggers land in `haz_triggers`
(carrying the indicator readings, and the evidence of absence for a zero),
the delta lands in `haz_impact_candidates` with `source='ipc'` and
`value_type='affected'` — so a drought answer is auditable through the same
table as every other hazard's — and the resolution goes through the same
writer, and therefore the same freeze guard and revision log, as the
ladder's. The rule replaces the ladder, not the bookkeeping. Every
resolution's provenance names the analysis window it rests on.

**Sanity checks.** The national-population cap applies and flags without
rewriting, as everywhere else. The GDACS exposure ceiling deliberately does
*not*: GDACS drought exposure is an agricultural-footprint quantity, and
the drought value is a *delta* rather than a stock, so the two are not
comparable and bounding one by the other would flag noise.

**Environment.** `IPC_API_KEY` (optional — without it the machine falls
back to the Phase 3+ rows already in `facts_resolved`). ASAP needs no
credentials.

**One thing to verify on the first live run.** Egress to the JRC ASAP host
was blocked from the environment this was built in, so `asap.url` in the
rulebook is a best-known endpoint and the parser is deliberately tolerant
of field naming. If the live feed differs, the failure is loud and safe —
the snapshot is refused, the indicator reads UNAVAILABLE, and no zeros are
written — and the fix is a URL in YAML plus, at worst, a key name in
`_CLASS_KEYS`. It is not a redesign.

## Phase 5: backcast, base rates, and the acceptance report (implemented)

### The backcast — the same machine, replayed over history

[`backcast.py`](backcast.py) runs each hazard from `rulebook.backcast.<hazard>`
(cyclone 2000, flood 2010, drought 2017) through the last month that has
FROZEN, one month at a time, **through exactly the functions a live run
uses**. There is no backcast-specific detection, no backcast-specific
ladder, and no second copy of any rule. The only difference is the
`run_type='backcast'` stamped on every `haz_resolutions` and `haz_triggers`
row it writes.

That sameness is the point rather than a convenience. Base rates drawn from
a history the machine did not actually produce would describe a different
machine — and if the backcast had its own path, a rulebook change could
move the live answers without moving the base rates the forecaster compares
them against, and nothing would fail.

Each hazard's start year is set by when its **detector's** record begins,
not by preference: IBTrACS best tracks run back decades, GDACS alerts only
stabilise around 2010, IPC's structured feeds later still. Backcasting past
a detector's coverage does not produce zeros — the coverage gate suppresses
them — it produces empty months that cost real time.

**A backcast fills gaps and never rewrites.** Every month it touches is
frozen by definition, so the freeze guard applies in full: a cell that
already has an answer keeps it, and the attempt is logged to
`haz_revisions`. That also makes the resume ledger (`haz_backcast_progress`)
more than a convenience — re-walking a completed month would write one
revision row per already-answered cell, burying the genuine post-freeze
revisions the audit table exists for. `--no-resume` re-walks anyway, for
when that is what you want.

**Two costs to know before starting one.** ReliefWeb silence sweeps run once
per non-triggered country-month, so a 25-year cyclone backcast over ~200
countries is on the order of half a million paced requests — the driver
prints an estimate before it begins. And ladder rung 2 spends real money,
bounded by `extraction.max_calls_per_month` counted per *calendar* month
across all runs, so a long backcast accrues extraction gradually rather
than in one bill. `--no-extract` turns it off entirely.

**A known gap, deliberately not papered over.** The drought path's
indicator feeds publish a *latest* snapshot with no observation date, so a
backcast month falls outside `max_observation_age_months`, the required
indicator reads UNAVAILABLE, and the month resolves INCONCLUSIVE rather
than to a false zero. That is the correct behaviour and it means **a
drought backcast produces almost nothing** until `drought.indicators`
entries point at a per-month archive (their URLs accept `{ym}`, `{year}`
and `{month}`). The driver's pre-flight check says so before the run
starts rather than letting it look merely disappointing.

### Base rates

[`base_rates.py`](base_rates.py) rebuilds two tables from what is already on
disk — drop them and a re-run reproduces them exactly.

| table | grain | content |
|---|---|---|
| `haz_base_rates_occurrence` | iso3 × hazard × calendar month | share of backcast years that TRIGGERED, from `haz_triggers`, over the hazard's full depth |
| `haz_base_rates_severity` | iso3 × hazard | q10/q25/q50/q75/q90 of RESOLVED_VALUE amounts from `severity_base_rate_window_start` (2015) on, with `n_events` and the provenance mix |

**The occurrence denominator counts years ASSESSED, not years in the
calendar.** A country-month with no trigger row was never looked at, and
counting it as a quiet month would manufacture a low base rate out of an
ingestion gap — the same mistake the coverage gate prevents one cell at a
time. `n_years` travels with every row, and
`base_rates.occurrence.min_years` (3) refuses to publish below a floor: one
observed November is not a base rate, and a `p_occurrence` of 1.0 from a
single year reads as certainty.

**The severity provenance mix matters more than it looks.** A distribution
assembled mostly from IDMC displacement figures is a distribution of *lower
bounds*; one from EM-DAT is a distribution of settled assessments.
`provenance_mix_json` carries the share by source **per year** (so a
composition that drifted is visible rather than averaged away) plus
`lower_bound_share`, which is what tells a consumer these quantiles sit
below the truth.

### The Dartmouth Flood Observatory cross-check

[`dfo.py`](dfo.py) answers the question the backcast cannot ask about
itself: *is the flood occurrence rate the machine derives from its own
detector plausible?* DFO's manually curated archive reaches back to 1985,
so the comparison runs against the years strictly **before** the GDACS era
— the rulebook validator rejects a `cross_check_end_year` past
`backcast.flood`, because comparing the machine against years it already
resolved would confirm nothing. Countries whose DFO rate exceeds the
machine's by more than `dfo.divergence_factor` are listed in the report as
places flood detection is probably missing events.

**It is calibration and nothing else.** DFO writes to `haz_raw_dfo` and to
the report; it is never a trigger and never a resolution value. A test
asserts it leaves `haz_triggers`, `haz_impact_candidates` and
`haz_resolutions` untouched — because an archive built from press reporting
is a different quantity from a resolved figure, and a source admitted "just
for calibration" is one refactor away from being admitted as an answer.
That is exactly how GDACS exposure once got into a people-affected series.

> **Verify on first run.** `dfo.url` could not be reached from the
> environment this was built in. A wrong URL or a changed sheet shape fails
> loud and safe — the fetch reports unavailable, the report's cross-check
> section says so, and no resolved row is affected, because nothing else
> reads this source. The fix is a URL in YAML and at worst a column name in
> `_COLUMN_ALIASES`.

### The acceptance report

[`acceptance.py`](acceptance.py) renders one page of markdown covering the
last 12 fully frozen months: the resolution rate per hazard against the
IFRC-GO-only baseline it replaces, the shortfall against the 80% target,
flag counts by reason, the provenance mix, the DFO cross-check, and a
reproducible random sample of 20 zeros and 20 values with their evidence.

**The denominator is cells assessed** — every `haz_triggers` row in the
window — not the rows the machine managed to write. Dividing resolutions by
resolutions would score 100% against a machine that answered one cell.
Cells that produced no row at all (a fail-closed INCONCLUSIVE, a
coverage-gated zero) count against the rate and are reported in their own
column so the reason stays visible. A hazard with *zero* assessed cells is
reported as **not assessed**, never as 0% — an absence of evidence is not a
failure to resolve.

The baseline column is **recomputed from the database**: the share of
assessed cells where the IFRC GO rung alone held a figure. The documented
~4% is quoted for context, but the number in the table is one that can be
audited.

**The report measures; it never tunes.** If the machine falls short of 80%
the report says so, per hazard, and says in as many words that moving a
threshold until the target is met would change the number without changing
what the machine knows.

The sample is seeded (`--seed`, default fixed) rather than
`ORDER BY random()`, so a reviewer who queries a row can regenerate the
same page — a report whose sample changes on every run cannot be cited in a
review.

## Phase 6: what the forecaster is told (implemented)

[`prompt_block.py`](prompt_block.py) closes the loop. Everything above
decides what a question resolves to; this renders that knowledge into the
SPD prompt of every PA question the machine resolves, so the forecaster
reasons about the generating process it will be scored against rather than
guessing at both the world and the target.

`forecaster/prompts.py::build_spd_prompt_v2` calls
`_load_haz_base_rate_block`, which injects the block immediately after the
Resolver history — the other prior anchor in that prompt, and the one STEP 1
tells the model to build its prior from.

```
PA RESOLUTION BASE RATES — flood people affected, Pakistan (PAK)
P(qualifying event), 2010-01..2026-05 backcast, 16y assessed:
  Jun 44%  Jul 69%  Aug 75%  Sep 50%  Oct 19%  Nov 6%
Severity | people affected in months WITH an event (n=21 events, 2015-2026):
  q10 12k  q25 48k  q50 210k  q75 900k  q90 3.1M
Figure sources: emdat 48%, ifrc_go 29%, reliefweb_extracted 14%, idmc_idu 9%
9% of the severity record is displacement counts — LOWER BOUNDS, so those quantiles sit below the truth.
HOW THIS RESOLVES: a flood month = a GDACS flood alert at orange+ over the country; no
qualifying event plus a silent ReliefWeb sweep resolves ZERO, not missing. A detected
month takes the first figure on the fixed ladder EM-DAT > ReliefWeb docs > IFRC GO >
IDMC displacement (lower bound), capped by GDACS exposure and national population;
answers freeze 60d after month end and never reopen.
```

**Every parameter in that last paragraph is read from `rulebook.yaml` at
prompt-build time** — the buffer and wind threshold for cyclones, the GDACS
alert level for floods, the Phase 3+ rule and its delta floor for drought,
the ladder order, the sanity ceiling, `freeze_days`. Change a threshold and
the prompt changes on the next run with no code edit. This is the whole
reason the text is generated rather than written: a prompt that describes
last quarter's rules is confidently wrong, which is worse than one that
describes none. Tests pin the propagation for `freeze_days`,
`flood.gdacs_trigger_level`, `ladder`, `cyclone.buffer_km` and
`cyclone.min_wind_kt`.

**Eligibility is deliberately narrow.** `is_eligible` admits the machine's
three hazards and the metric `PA`, nothing else. Pythia's drought questions
usually resolve on `PHASE3PLUS_IN_NEED` — a *stock* of people already in
Phase 3+ — while this machine's drought value is the monthly *increase* in
that stock; `EVENT_OCCURRENCE` resolves from GDACS alert levels for all
three hazards, whereas the machine detects cyclones from IBTrACS track
geometry. Same sources, different quantities, and putting one where the
other belongs is the failure the repo already paid for once when GDACS
modelled exposure reached models as IFRC reported impact.

**Three states, three renderings.** A country-hazard with occurrence rows
but no severity row was assessed across the backcast and never resolved to a
value — "no historical events in record" is true of it. One with too few
events for `base_rates.prompt.min_events_for_quantiles` (3) gets its event
count instead of quantiles, because five quantiles read as a distribution
whatever the header says. One with *no rows at all* was never assessed and
gets **no block**: claiming an empty record would state as fact something
nothing was ever checked for.

**Token discipline.** The shipped blocks render at 800–1000 characters
(~200–250 tokens) against a declared `base_rates.prompt.max_chars` budget of
1200. A block over budget is logged and fails a test; it is never
truncated, because a base rate cut off mid-table still reads as a complete
one. Empty quantile rows are omitted rather than padded, a shared occurrence
denominator is stated once instead of six times, and a rate that is small
but non-zero renders `<1%` rather than rounding to `0%`.

**Drought carries an explicit caveat**, generated from `backcast.drought`:
the record starts 2017 (~10 years) and IPC coverage is partial by country,
so a low rate is weak evidence of a low rate in the world. Under
`month_attribution: analysis_window` it adds the warning the provenance
already carries — the same deterioration repeats across every month its
analysis window covers, and these values must never be summed.

Tests: [`resolver/tests/test_hazard_resolution_prompt_block.py`](../tests/test_hazard_resolution_prompt_block.py)
(rendering, eligibility, rulebook propagation, budget) and
[`forecaster/tests/test_haz_base_rate_prompt.py`](../../forecaster/tests/test_haz_base_rate_prompt.py)
(assembled-prompt snapshots for one cyclone, one flood and one drought PA
question). The second file exists because the failure mode at this seam is
silent: a loader whose text nobody renders costs money and changes nothing,
which is how CrisisWatch and the HS grounding packs were both lost.

## Production wiring (Aug 2026 — shadow mode)

The machine runs in two workflows, and its answers grade nothing yet:

- **`resolver_update.yml` Phase 2.4 + 2.5** (28th monthly): Phase 2.4 first
  refreshes what the machine reads from the same DB — the NMME ingest and
  the HDX Signals store (the drought gate's evidence) and `haz-population`
  (the ladder's cap; until Sept 2026 nothing loaded it, so the cap never
  fired). Phase 2.5 then runs `resolve-hazards` over the trailing three
  months for flood, cyclone and drought (provisional answers are revised
  each cycle until their 60-day freeze; each step under coreutils `timeout`
  so a kill is annotated, the summary written after every month), then
  `haz-base-rates` so the 1st-of-month forecast reads fresh base rates from
  the same DB artifact, then `haz-acceptance` into the `backfill-diagnostics`
  artifact, then `retention --apply` to collapse the raw caches. Non-fatal
  end to end: a machine failure never blocks the ingest.
- **`haz_backcast.yml`** (nightly 20:30 UTC, skipping the 27th and 28th): a time-boxed chunk of the
  historical replay per hazard (`haz-backcast --time-budget-min`), resumed
  from the `haz_backcast_progress` ledger, then a base-rate refresh and a
  canonical-DB re-upload. Self-converging: once the window is covered the
  run is a near-instant no-op that rolls forward as months freeze.

**Shadow mode** means `pythia/tools/compute_resolutions.py` still resolves PA
questions from `facts_resolved` (IFRC first) — nothing reads
`haz_resolutions` outside this package except the Phase 6 prompt block. The
flip to scoring is a deliberate follow-up change, taken only after the
monthly acceptance reports show the machine meeting its ≥80% target with
credible provenance.

**Telemetry**: every real extraction call writes a rich `llm_calls` row
(`phase='hazard_extraction'`, mapped to the Costs page's "PA resolution"
bucket) in addition to the `haz_doc_extractions` cache/ledger row. The
extraction cache is scoped to the CELL (doc, model, prompt version, iso3,
hazard, month) — the prompt names all three, so one cell's answer is never
served for another's question — and error rows are retried on the next run
rather than cached forever.
