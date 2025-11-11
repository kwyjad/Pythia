# Resolver CLI

Answer questions like:
**“By `<DATE>`, how many people `<METRIC>` due to `<HAZARD>` in `<COUNTRY>`?”**

## Usage

```bash
# name/label inputs
python resolver/cli/resolver_cli.py \
  --country "Philippines" \
  --hazard "Tropical Cyclone" \
  --cutoff 2025-09-30

# code inputs
python resolver/cli/resolver_cli.py --iso3 PHL --hazard_code TC --cutoff 2025-09-30

# request stock totals instead of monthly new deltas
python resolver/cli/resolver_cli.py --iso3 PHL --hazard_code TC --cutoff 2025-09-30 --series stock

# JSON-only output for automation
python resolver/cli/resolver_cli.py --iso3 ETH --hazard_code DR --cutoff 2025-08-31 --json_only
```

### Monthly snapshots

```bash
# end-to-end export → validate → freeze for January 2025
python -m resolver.cli.snapshot_cli make-monthly --ym 2025-01

# list available snapshot folders
python -m resolver.cli.snapshot_cli list-snapshots
```

The monthly command performs the following steps:

1. Runs `resolver/tools/export_facts.py` with the configured staging inputs.
2. Validates the export via `resolver/tools/validate_facts.py`.
3. Resolves precedence and builds monthly deltas.
4. Calls `resolver/tools/freeze_snapshot.py` to write `resolver/snapshots/YYYY-MM/` artifacts and, when `--write-db 1` is provided, persists them into DuckDB.

### Data selection rules

- **Past months** → uses `snapshots/YYYY-MM/facts.parquet` (preferred)
- **Current month** → prefers `exports/resolved_reviewed.csv`, else `exports/resolved.csv`
- **Series selection** → defaults to monthly `new` deltas; use `--series stock` to return totals. Missing deltas emit a note and fall back to stock data.
- Returns one record per `(iso3, hazard_code)` at the cutoff (PIN preferred, else PA) following upstream policy.

### Dependencies

```bash
pip install pandas pyarrow
```

### Notes

- Keep `resolver/data/countries.csv` and `resolver/data/shocks.csv` up to date.
- If you have not frozen the month yet, the CLI will use current `exports/` outputs.
