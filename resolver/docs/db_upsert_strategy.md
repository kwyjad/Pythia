# DuckDB Upsert Strategy

We prefer to write via DuckDB's `MERGE` statement when the target table exposes an explicit primary key or unique constraint. `MERGE` lets us update matching rows and insert new ones in a single transactional statement, keeping delete-logging and idempotent rewrites fast.

In CI and some local environments DuckDB may not recognise the `MERGE` syntax yet. When `MERGE` raises a parser/feature error—or when you opt out by setting `RESOLVER_DUCKDB_DISABLE_MERGE=1`—we automatically fall back to an equivalent `DELETE` + `INSERT ... SELECT` sequence. The fallback runs in the same transaction, reuses the natural-key filter, and logs the row counts deleted and inserted so you can confirm idempotency.

Both strategies leave the schema unchanged and guarantee that repeated runs produce the same results, so database tests succeed even on engines that do not support `MERGE`.
