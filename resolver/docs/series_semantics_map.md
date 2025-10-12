# Series Semantics Canonicalisation Map

## Database write path (DuckDB)
- `resolver/db/duckdb_io.py`:
  - `_canonicalize_semantics(frame, table_name, default_target)` — enforces canonical set during writes. [Public APIs index] :contentReference[oaicite:0]{index=0}
  - `_canonicalise_series_semantics(series)` — maps into `{"", "new", "stock", "stock_estimate"}`. [Public APIs index] :contentReference[oaicite:1]{index=1}
  - `_assert_semantics_required(frame, table)` — guards required semantics fields. [Public APIs index] :contentReference[oaicite:2]{index=2}
  - `write_snapshot(...)` — transactional write that calls the canonicalizer before upsert. [Public APIs index] :contentReference[oaicite:3]{index=3}

## Common canonical logic
- `resolver/common/series_semantics.py`:
  - `_load_config()` — loads mapping from YAML. [Public APIs index] :contentReference[oaicite:4]{index=4}
  - `compute_series_semantics(metric, existing)` — returns the canonical semantics for a record. [Public APIs index] :contentReference[oaicite:5]{index=5}
- YAML mapping: `resolver/config/series_semantics.yml` (see repo tree). :contentReference[oaicite:6]{index=6}

## Helper normalizer (DataFrame-level)
- `resolver/helpers/series_semantics.py`:
  - `_iter_normalised(values)`; `normalize_series_semantics(frame, *, column)` — batch normalizer used by tools. [Public APIs index] :contentReference[oaicite:7]{index=7}

## Export path (CSV/Parquet → DB) hooks
- `resolver/tools/export_facts.py`:
  - `_apply_series_semantics(frame)` and `_warn_on_non_canonical_semantics(...)` — applies & warns before writing. [Public APIs index] :contentReference[oaicite:8]{index=8}
  - `_prepare_resolved_for_db(...)`, `_prepare_deltas_for_db(...)`, `_maybe_write_to_db(...)` — staging prior to DB write. [Public APIs index] :contentReference[oaicite:9]{index=9}

## Snapshot path hooks
- `resolver/tools/freeze_snapshot.py`:
  - `_prepare_resolved_frame_for_db(...)`, `_prepare_deltas_frame_for_db(...)`, `_maybe_write_db(...)` — snapshot → DB path. [Public APIs index] :contentReference[oaicite:10]{index=10}

## Where to search quickly
- Rooted under `resolver/` (see `REPO_TREE.txt`). Useful keywords: `series_semantics`, `canonicalise`, `canonicalize`, `stock_estimate`, `compute_series_semantics`. :contentReference[oaicite:11]{index=11}
