# IFRC Montandon / GO as a resolution source — assessment

_August 2026._

Pythia's Resolver treats **IFRC Montandon as the tier-0 source for natural-hazard "people
affected" (PA)** for FL, DR and TC. PA is the worst-resolving metric in the system:
**FL/PA ≈ 4%, ACE/PA ≈ 2%, DR/PA = 0%** (`scripts/diagnose_pa_resolution_coverage.py`).

This document assesses where the weaknesses sit — conceptually and from our operational
record — and what IFRC could change that would most help Pythia and similar consumers.
[§3](#3-practical-improvements-ifrc--montandon-could-make) is written to stand alone so it
can be lifted into an issue against `IFRCGo/GCDB`; [§4](#4-what-is-ours-to-fix-not-theirs)
records what is ours to fix, because a fair share is.

---

## 0. The framing problem: we are not using Montandon

Our connector is named `ifrc_montandon`, but nothing in the production path touches
Montandon.

`resolver/ingestion/config/ifrc_go.yml` sets
`base_url: https://goadmin.ifrc.org/api/v2/`, polling `event/`, `field-report/`,
`appeal/`, `situation_report/`. The production chain is:

```
run_connectors.py → resolver.ingestion.ifrc_go_client → resolver/staging/ifrc_go.csv
  → resolver.transform.normalize (IFRCAdapter) → load_and_derive → resolve_sources
  → facts_resolved (publisher='IFRC')
```

Montandon (Monty, the Global Crisis Data Bank) is a separate STAC system
(`montandon-eoapi.ifrc.org`; the documented endpoint is still `-stage`) that aggregates
GDACS / EM-DAT / GLIDE / USGS / IBTrACS / PDC for events and hazards, and
EM-DAT / IDMC / IFRC field reports / DesInventar for impacts.

**So we consume the rawest input tier of the thing Montandon exists to harmonise, and call
it Montandon.** Several complaints below are exactly what Montandon claims to solve —
cross-source event linking, a harmonised impact taxonomy, historical depth. We have never
evaluated it.

Two further wrinkles:

- The Connector-protocol wrapper `resolver/connectors/ifrc_montandon.py` and the
  `resolve_facts_frame` precedence engine are **not on the IFRC production path**.
  `run_pipeline` runs in production only for GDACS / FEWS NET / IPC via
  `ingest_structured_data.py`. IFRC goes through `resolve_sources` and
  `source_priority.yml` instead.
- **GO is polled anonymously.** `GO_API_TOKEN` is read by the client but never set in
  `resolver_update.yml`.

---

## 1. Conceptual weaknesses

### 1.1 Response-coordination data used as a measurement series

GO records exist so IFRC can mount and fund an operation. Absence of a record means "no
IFRC operation", **not** "no impact". Pythia needs a value for every
(country, hazard, month) including a credible zero, and GO structurally cannot supply one.

That is why `compute_resolutions.py` excludes PA from `_ZERO_DEFAULT_RULES` (only
FATALITIES/ACE-ACO and EVENT_OCCURRENCE zero-default), and why every unreported PA horizon
exits via `skipped_null_resolution` — dropped, never scored.

### 1.2 Coverage is capacity-correlated — missing-not-at-random

Whether a record exists depends on National Society reporting capacity and on whether an
appeal was launched. Coverage is thinnest exactly where forecasting matters most: weak NS,
conflict-affected, access-constrained countries.

For calibration this is worse than noise. A model forecasting high PA in a low-reporting
country is neither rewarded nor penalised — the horizon never resolves. **The scoring set
is systematically biased toward places IFRC is already operating.** Our own documentation
says as much: `README.md` lists "IFRC Montandon sparsity" as a known limitation, and
`forecaster/hazard_prompts.py` instructs models to "treat a sparse base rate as a lower
bound".

### 1.3 Event-shaped data forced into a country-month grid

We resolve on `(iso3, hazard_code, ym)`; GO publishes per-record events with revisions.
Our mapping: `as_of = report_date → disaster_start_date → start_date → updated_at →
created_at`, truncated to month, one row per GO record, with `series_semantics` **hardcoded
to `"stock"`**.

Consequences: a multi-month event either lands in one month (impact appears to stop) or
re-asserts the same cumulative figure across months (double counting); successive field
reports on one disaster become separate facts distinguishable only by record id; and
cumulative-to-date is indistinguishable from new-this-month.

### 1.4 "Affected" is three different quantities in one column

`extract_all_metrics` yields an `affected` row three ways:

1. `num_affected` / `gov_num_affected` / `other_num_affected` — **max across variants**
   (systematic upward bias). `confidence='med'`.
2. Appeal `num_beneficiaries` promoted to `affected` — *people targeted for assistance*, a
   planning figure driven by IFRC capacity and funding appetite, not by the hazard.
   Stamped `confidence='high'`.
3. `derived_sum:` of fatalities + injured + displaced + missing when ≥2 are present — a
   floor estimate. `confidence='low'`.

All three are bucketed against PA thresholds of 10k / 50k / 250k / 500k
(`pythia/buckets.py`). Ground truth that oscillates between "people IFRC plans to help"
and "sum of casualties" is scoring probabilistic forecasts, and appeal beneficiaries
cluster on round planning numbers that sit near bucket boundaries.

The spread this produces is documented in `compute_resolutions.py`: **IFRC `affected` 19M
vs `displaced` 130k for the same month — 145×.** That is what forced deterministic metric
precedence on us.

**And the provenance is then thrown away**: `load_and_derive.py` writes
`confidence = None` for every Phase-1 row. Nothing downstream reads `method` or
`confidence`, so at resolution time a derived floor estimate and a real `num_affected` are
indistinguishable apart from an `event_id` suffix.

### 1.5 Multi-country records replicate the full figure to every country

`iso3_pairs_from_go` returns every listed country, and the emit loop writes the **same
value** to each. A regional appeal covering five countries writes the full regional figure
five times. GDACS, by contrast, does population-weighted allocation. The gap exists because
GO publishes no per-country impact breakdown.

### 1.6 Zero is unrepresentable

`extract_all_metrics` skips `0`, `"0"`, `""` and `None` alike, so a genuinely reported zero
is indistinguishable from an unreported field. Combined with §1.1, this means PA's
dedicated exact-zero bucket can never be confirmed by ground truth — we ask models for
`P(nothing happens)` on a metric whose zero is unobservable.

### 1.7 Hazard classification is partly heuristic, and the taxonomy doesn't cut where we need

Priority 1 is the structured `dtype` id — good, and 18 IDs are mapped. Priority 2 is
substring keyword matching over title + summary + dtype name. Beyond that, GO's taxonomy
collapses distinctions we forecast on: "Complex Emergency" (dtype 6) becomes ACE with no
ACE/ACO split, and Drought (20) vs Food Insecurity (21) is ambiguous for the drought hazard.

### 1.8 No stable cross-source event identity exposed

We key on the GO record id, so an IFRC `affected` figure cannot be joined to the GDACS
alert or the IDMC displacement figure for the same flood. Cross-source event linking is
Montandon's stated purpose; it just isn't surfaced on the operational API.

### 1.9 No revision history or point-in-time reads

GO returns current state. A figure revised upward six months later is used as if it had
been known at forecast time. Harmless for resolution, but it makes historical backtests
optimistic, base rates unstable between runs, and Sibyl's asOf leakage discipline
unenforceable against this source.

### 1.10 No incremental cursor, and schema varies by endpoint

There is no reliable "changed since". We compensate with a 45-day window, **two passes per
endpoint** (`created_at__gte`, then `updated_at__gte`), and a consecutive-older-pages
early-exit heuristic. Schema also varies by endpoint: `_dtype_id_from_record` tries four
field names, and `iso3_pairs_from_go` handles both `countries_details` (plural) and
`country_details` (singular).

---

## 2. What our operational record shows

**PA resolution rates of 4% / 2% / 0%** — with three honest caveats. The diagnostic queries
only `facts_resolved`, not `facts_deltas`, so it is a floor. DR/PA is 0 partly because we
stopped generating DR/PA questions (remapped to `PHASE3PLUS_IN_NEED`). And
`/v1/diagnostics/resolution_rates` counts a question "resolved" if **any one of six
horizons** resolved, so per-horizon coverage is worse than the dashboard tile shows. The
script runs in no workflow and its output is committed nowhere.

**Every appeal was silently dropped** until `iso3_pairs_from_go` learned the singular
`country_details` shape — the test docstring in
`resolver/tests/test_ifrc_go_dtype_and_metrics.py` records it. That is the schema
inconsistency of §1.10 turning into total data loss for one endpoint, invisibly.

**IFRC was silently demoted out of tier 0.** `precedence_config.yml` listed
`ifrc_montandon` while the connector writes `IFRC` — no match, so the documented tier-0 PA
source fell to the default (worst) tier, with within-tier winners chosen *alphabetically*
by `(source, run_id)` (audit F2, HIGH). Worth noting the fix's real scope: that engine is
not on the IFRC production path anyway (§0), so IFRC precedence in production is governed
by `source_priority.yml`, and the F2 fix did not change it.

**Within a country-month, the earliest and least-informed report wins.** `IFRCAdapter`
snaps `as_of_date` to month-end; `resolve_sources` then dedupes on
`(iso3, hazard_code, metric, unit, as_of_date, series_semantics)` — **`ym` and `event_id`
are not in the key** — sorting ascending by `provenance_rank, publication_date, source` and
keeping `first`. So all IFRC rows for a country-hazard-metric-month collapse to one, and
the tiebreak keeps the **earliest published** figure. Multiple distinct GO events in one
country-month are discarded (no max, no sum), and a later upward revision loses to the
initial field report. This is ours to fix, but it is a direct consequence of event-shaped
input meeting a month-shaped store.

**Failures look like absence.** `IfrcMontandonConnector.fetch_and_normalize` swallows any
exception to an empty frame with a WARNING; `req_json` returns `{}` on HTTP 202, which
reads as an empty page and breaks pagination; `main()` writes a header-only CSV on error.
Per-pass drop counters (`no_country`, `no_hazard`, `no_metric`) exist but only print under
`RESOLVER_DEBUG=1`, which CI never sets. A total GO outage is indistinguishable from a
quiet month.

**Ingest ran two weeks after the forecast it fed.** Until 2026-08-03, `resolver_update.yml`
ran on the 15th while the pipeline forecast on the 1st — the 2026-08-01 run carried an
IFRC/ACLED/IDMC vintage of 2026-07-15. Now on the 28th.

**The PA cascade is two steps, not three.** `facts_resolved` → `facts_deltas` → `emdat_pa`,
and the EM-DAT step is dead (no connector since the Feb 2026 refactor; the table is created
and never written). PA depends on IFRC + IDMC alone.

**IFRC has effectively been on probation.** `tools/compare_gdacs_ifrc.py` exists
specifically to evaluate replacing IFRC with GDACS for DR/FL/TC PA.

---

## 3. Practical improvements IFRC / Montandon could make

Ranked by value to a downstream forecasting consumer.

1. **Publish a country-month impact series, not only events** — a derived collection keyed
   `(iso3, hazard, month)` with explicit `no_record` vs `zero` semantics, and a stated rule
   for aggregating multiple events within a month. Single highest-value change.
2. **Distinguish "no impact" from "no report."** A per-country-month reporting-activity
   signal would let consumers safely treat silence as zero where appropriate. Consumers can
   build this for sources with declared global coverage; for IFRC it is impossible from
   outside.
3. **Typed, non-overlapping impact fields with published definitions.** Never let
   "people targeted" (appeal beneficiaries) share a column with "people affected", and make
   a reported zero distinguishable from a missing field.
4. **Per-country breakdown on multi-country records** — or at minimum a
   `figure_scope: regional | national` flag, so consumers stop multiplying regional totals
   across every listed country.
5. **Provenance and revision history on every figure**: `reported_by`
   (NS / government / other), `as_of`, `revision`, `superseded_by`, and ideally a
   point-in-time query (`?as_of=`). This is what makes honest backtesting possible.
6. **Expose stable cross-source event IDs** (GLIDE / Montandon links) on every GO record,
   so an IFRC figure can be joined to the GDACS alert and IDMC displacement for the same
   event.
7. **A real incremental cursor** — `?modified_since=` with monotonic ordering and stable
   pagination, removing the need for window + dual-pass + early-exit heuristics.
8. **Schema consistency across endpoints** — one country shape and one disaster-type shape
   across `event`, `field-report`, `appeal` and `situation_report`. The silently-dropped
   appeals incident above is what inconsistency costs downstream.
9. **Data-quality flags** — estimate vs confirmed, order-of-magnitude confidence, inclusion
   notes. Consumers currently infer confidence from which field happened to be populated.
10. **Make Montandon production-grade for programmatic consumers** — a stable production
    STAC endpoint, versioned collection names, a changelog, and documentation reachable by
    non-browser clients. As of this writing the documented endpoint is a staging host and
    the Monty documentation pages refuse non-browser requests.

---

## 4. What is ours to fix, not theirs

- **Evaluate consuming Montandon STAC instead of raw GO.** Most of §1.4, §1.8 and §1.9 are
  what Montandon claims to solve. We have never tried it. Note the trap: Montandon also
  ingests GDACS, so any connector must filter by upstream source and admit only
  reported-impact records — otherwise we import modelled exposure into the same column as
  reported impact.
- **Fix the month collapse.** *Half done, August 2026.* `resolve_sources` no longer lets
  the earliest report beat the latest revision — the tiebreak is now source priority, then
  non-null value, then most-recent publication, matching the policy
  `precedence_config.yml` already declared. **Still open:** the collapse yields one row per
  country-month, so a month holding genuinely distinct events keeps only one rather than
  summing them. Deciding between sum and max needs evidence on how often multi-event months
  occur versus revision chains of a single event — measure before choosing.
- **Keep modelled exposure and reported impact apart.** GDACS `in_need` is population
  inside a hazard footprint; IFRC `affected` is reported impact. They differ by orders of
  magnitude and in kind, and the switch between them is severity-correlated — a larger
  event is likelier to attract an IFRC report — so blending them would change scale as a
  function of the thing being forecast. They must be separate metrics, never a single
  metric with a precedence rule spanning both.
- **Stop discarding `method` / `confidence`.** Still open, and larger than it looks: the
  loss happens at the **adapter**, not at `load_and_derive`. `IFRCAdapter.map` emits only
  the 12 required canonical columns, so `method`, `confidence`, `publisher`, `source_url`
  and `doc_title` are gone before `load_and_derive` sets `confidence = None`. Carrying them
  means extending the canonical contract (as `publication_date` was in August 2026) and
  then tiering the three `affected` provenance branches — real figure, appeal beneficiaries,
  derived sum — inside `PA_REPORTED` precedence. Record which branch won via
  `resolutions.source_desc`.
- **Stop replicating the full value across countries** on multi-country GO records.
- **Make soft failures visible** — surface the connector's drop counters and row counts in
  the Phase 1 CI summary rather than behind `RESOLVER_DEBUG`.
- ~~**Stop GDACS exposure entering the PA base rate.**~~ **Fixed, August 2026.**
  `_build_natural_hazard_seasonal_profile` (`forecaster/cli.py`) — the live builder for
  natural-hazard PA base rates — matched `('affected', 'in_need', 'pa')` with no publisher
  filter and labelled the result `"source": "IFRC"`. GDACS writes `in_need` for FL/DR/TC,
  so modelled exposure figures were presented to models as IFRC reported-affected history
  on exactly the hazards where IFRC coverage is weakest, while `compute_resolutions`
  excluded `in_need` and would never score against them. The PA metric list is now
  single-sourced from `compute_resolutions.PA_FACTS_RESOLVED_METRICS`, the profile reports
  the publishers actually behind its rows, and
  `forecaster/tests/test_base_rate_matches_resolution_source.py` pins the base rate to a
  subset of what the resolver scores. The same leak was fixed in the latent
  `_load_ifrc_pa_history` (`forecaster/history_loaders.py`), which additionally used the
  raw read-only `duckdb.connect` the codebase forbids and whose `LIMIT months` counted rows
  rather than months.
- **Wire `diagnose_pa_resolution_coverage.py` into CI** and widen it to `facts_deltas`.
- **Set `GO_API_TOKEN`** if authenticated access lifts rate or field limits.
