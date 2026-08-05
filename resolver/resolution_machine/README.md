# People-affected resolution machine

For every country-month-hazard, produce either a **credible number of
people affected**, a **credible zero**, or an explicit **"no data"** —
always with full provenance. Built to replace the IFRC-GO-only PA
resolution path (which resolves ~4% of PA questions per month); target
is ≥80% of country-month-hazard rows resolving to a number (including
zero) rather than "no data".

## Architecture

Two layers:

1. **Detection (Layer 1)** — decide from physical/event sources whether
   any qualifying hazard occurred in a country-month. If nothing
   occurred *and* a ReliefWeb keyword sweep confirms silence, resolve
   **zero** and record the evidence of absence.
2. **Impact (Layer 2, later phases)** — when a hazard was detected,
   walk a fixed precedence ladder of reporting sources for a
   people-affected figure: (1) EM-DAT, (2) LLM-extracted figures from
   ReliefWeb documents, (3) IFRC GO structured field reports, (4) IDMC
   IDU displacement as a lower bound. Highest rung with a figure wins;
   sanity-checked against the GDACS exposed-population estimate
   (ceiling only) and national population. Drought skips the ladder and
   resolves against IPC data.

**Phase 1 (this code)** implements the cyclone path of Layer 1:
IBTrACS triggers, ReliefWeb silence checks, and `RESOLVED_ZERO` rows.

## Hard rules

1. Every resolution row carries provenance: source, source record IDs or
   document URLs, retrieval timestamp, and which rule fired.
2. Reconciliation is deterministic — rules only, no LLM calls.
3. LLM extraction (later phases) extracts *stated* figures only; it
   never estimates or infers a number not written in a document.
4. Resolutions freeze at month-end + 60 days (`freeze.days_after_month_end`)
   and are never reopened; later revisions are logged to
   `pa_resolution_revisions` but do not alter resolved values.
5. All thresholds and parameters come from [`rulebook.yaml`](rulebook.yaml),
   never hard-coded.
6. GDACS is detection + plausibility ceiling only, never a resolution value.
7. All API keys come from environment variables.
8. Recurring cost stays well under USD 50/month: all data sources are
   free; the only spend (later phases) is Haiku-class LLM extraction.

## Usage

```bash
# Module form
python -m resolver.resolution_machine.cli --hazard cyclone --month 2026-06

# Poetry script form
resolve-hazards --hazard cyclone --month 2026-06

# Backcast an old month from the full IBTrACS archive, limited countries
resolve-hazards --hazard cyclone --month 2013-11 --scope ALL --countries PHL MDG

# Re-run on the existing store without downloading
resolve-hazards --hazard cyclone --month 2026-06 --skip-fetch
```

The DB target is the resolver DuckDB (`--db` or `$RESOLVER_DB_URL`,
via `resolver.db.duckdb_io.get_db`).

Per month the CLI: fetches IBTrACS (idempotent upsert) → writes a
`haz_triggers` row for **every** country in the universe (triggered and
not) → sweeps ReliefWeb for each non-triggered country-month → writes
`RESOLVED_ZERO` rows where the sweep confirms silence. Non-silent
sweeps flip the trigger (`trigger_source='reliefweb_sweep'`) and leave
the month for the impact ladder; failed sweeps are *inconclusive* and
never produce a zero (fail-closed).

## Cyclone detection rule

A cyclone triggers for a country-month when any IBTrACS track point at
`cyclone.min_wind_kt` or stronger (default 34 kt — tropical-storm
strength) lies within `cyclone.buffer_km` (default 200 km) of the
country's territory. Wind is read per
`cyclone.wind_source_priority` (USA 1-minute sustained first, WMO
agency wind as fallback). A storm-point's month is its `iso_time`
month; a storm spanning a month boundary can trigger both months.

## Geometry choice

- **Boundaries**: a vendored, slimmed **Natural Earth 1:50m Admin-0
  countries** layer
  (`data/ne_50m_admin_0_countries.slim.geojson.gz`, public domain).
  1:50m rather than 1:110m because the coarser layer drops the small
  island states (Tonga, Tuvalu, Kiribati…) that matter most for
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

## Zero-resolution safety

Two gates in front of every zero:

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
  sample title/URL is stored in the zero's `evidence_json`.

## Tables (resolver DuckDB)

| table | contents |
|---|---|
| `haz_raw_ibtracs` | raw IBTrACS track points (PK `sid, iso_time`; idempotent re-fetch) |
| `haz_triggers` | detection verdict per (hazard, iso3, ym) — triggered AND non-triggered, with params + per-storm evidence in `detail_json` |
| `pa_resolutions` | resolution rows (PK `iso3, hazard_code, metric, ym`) with status `RESOLVED_ZERO` / `RESOLVED_VALUE` / `NO_DATA`, full provenance, `freeze_at` |
| `pa_resolution_revisions` | append-only post-freeze observation log |

The machine owns its DDL (`schema.py`, `CREATE TABLE IF NOT EXISTS`,
same self-contained pattern as `horizon_scanner/reliefweb.py`).

## Environment

| variable | purpose |
|---|---|
| `RESOLVER_DB_URL` | DuckDB target (or `--db`) |
| `RELIEFWEB_APPNAME` | ReliefWeb API `appname` parameter (falls back to `pythia-resolution-machine` with a warning) |
| `EMDAT_API_KEY`, `IDMC_API_KEY` | later phases (impact ladder) |

IBTrACS, GDACS, IFRC GO, and the country boundaries need no credentials.

## Tests

`resolver/tests/test_resolution_machine_*.py` (network-free; run by
resolver-ci-fast, which executes all of `resolver/tests`). The
acceptance suite pins the three design months: Haiyan Nov-2013 must
trigger PHL; a quiet month must produce zeros with complete
`evidence_of_absence`; a near-miss track at ~250 km must NOT trigger at
the 200 km rulebook buffer (and must at 300 km — the buffer is honoured
from the rulebook, not code).
