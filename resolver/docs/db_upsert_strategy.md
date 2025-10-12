# DuckDB Upsert Strategy

We prefer to write via DuckDB's `MERGE` statement when the target table exposes an explicit primary key or unique constraint. `MERGE` lets us update matching rows and insert new ones in a single transactional statement, keeping delete-logging and idempotent rewrites fast.

In CI and some local environments DuckDB may not recognise the `MERGE` syntax yet. When `MERGE` raises a parser/feature error—or when you opt out by setting `RESOLVER_DUCKDB_DISABLE_MERGE=1`—we automatically fall back to an equivalent `DELETE` + `INSERT ... SELECT` sequence. The fallback runs in the same transaction, reuses the natural-key filter, and logs the row counts deleted and inserted so you can confirm idempotency.

Both strategies leave the schema unchanged and guarantee that repeated runs produce the same results, so database tests succeed even on engines that do not support `MERGE`.

All DuckDB instances are bootstrapped from [`resolver/db/schema.sql`](../db/schema.sql) via
`duckdb_io.init_schema()`. The schema file defines the canonical tables,
primary keys, and supporting indexes for `facts_resolved`, `facts_deltas`,
`manifests`, and `meta_runs`. When an upsert call specifies natural keys, we
validate that the target table exposes a matching primary key or unique
constraint and raise immediately if it does not. This fail-fast check keeps the
schema and upsert expectations aligned as the database evolves.

## Series Semantics Enforcement

Writes normalise incoming `series_semantics` tokens (via
`resolver.helpers.series_semantics.normalize_series_semantics`) and then apply a
table-specific collapse in `_canonicalize_semantics`. The final DuckDB records
always store `stock` for `facts_resolved` rows and `new` for `facts_deltas`
rows, preventing `stock_estimate` or other variants from leaking into the
database while retaining diagnostics about any non-canonical inputs upstream.

## Date Columns Policy

Resolver exports compare DuckDB rows against CSV files during parity tests. To
avoid type mismatches when pandas merges on date fields, user-facing columns in
DuckDB that are also emitted to CSV are stored as `VARCHAR` values containing
ISO `YYYY-MM-DD` strings. The exporter and database writer normalise
`as_of_date`, `publication_date`, and delta `as_of` fields to the same string
format before inserting. Internal audit columns (such as `created_at`) remain
`TIMESTAMP` because they are not merged against CSV outputs.

## Function Call Signatures

`duckdb_io.upsert_dataframe` historically accepted both `(conn, table, df,
keys=...)` and `(conn, df, table, keys=...)` argument orders. The resolver
codebase now standardises on the former `(conn, table: str, df: pandas.DataFrame,
keys=...)` ordering, but the function still recognises the legacy signature and
swaps the arguments when it detects them. Any other order results in a
`TypeError`, helping older tests continue to run while flagging genuinely
invalid calls.
