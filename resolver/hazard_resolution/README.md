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
   for human review. Drought skips the ladder entirely and resolves from IPC
   data by a fixed rule.

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

Later phases add the IPC drought rule and the backcast.
