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
(`haz_impact_candidates`), final answers (`haz_resolutions`), the post-freeze
audit log (`haz_revisions`), and historical base rates
(`haz_base_rates_occurrence`, `haz_base_rates_severity`). See
[`schema.py`](schema.py).

**Running it.**

```bash
# Create/refresh the haz_* tables (idempotent, safe on a live DB)
python -m resolver.hazard_resolution.migrate

# Load national population denominators into haz_raw_population
python -m resolver.hazard_resolution.population

# Phase 1 — cyclone detection + zero resolutions for one month
python -m resolver.hazard_resolution.cli --hazard cyclone --month 2026-06
resolve-hazards --hazard cyclone --month 2026-06            # poetry script form

# Backcast an old month from the full IBTrACS archive, limited countries
resolve-hazards --hazard cyclone --month 2013-11 --scope ALL --countries PHL MDG

# Re-run on the existing store without downloading
resolve-hazards --hazard cyclone --month 2026-06 --skip-fetch
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

Later phases add the remaining detection connectors (GDACS floods, IPC
droughts), the ReliefWeb extraction step, the reconciliation engine,
and the backcast. Costs stay well under USD 50/month: every data source
is free; the only spend is the cheap-model document extraction.
