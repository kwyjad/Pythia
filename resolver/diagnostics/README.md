# Resolver Diagnostics Artifacts

The ingestion pipeline now captures a richer set of diagnostics when running
the DTM connector. Three subdirectories are populated on every run:

- `raw/` – snapshots of responses from discovery endpoints, including
  `dtm_countries.json` with the payload returned by `get_all_countries()`.
- `metrics/` – JSONL files that summarize activity, such as
  `dtm_per_country.jsonl` which records per-country row counts and elapsed
  timings.
- `samples/` – lightweight CSV extracts like `dtm_sample.csv` containing up to
  200 representative rows across all requests.

These artifacts are uploaded as GitHub Action artifacts by the initial
backfill workflow and can be inspected to understand why a particular run
returned zero or unusually few rows.
