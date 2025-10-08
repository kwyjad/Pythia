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
