-- Resolver DuckDB canonical schema (v0.9 freeze)
BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS facts_resolved (
    ym TEXT NOT NULL,
    iso3 TEXT NOT NULL,
    hazard_code TEXT NOT NULL,
    hazard_label TEXT,
    hazard_class TEXT,
    metric TEXT NOT NULL,
    series_semantics TEXT NOT NULL DEFAULT '',
    value DOUBLE,
    unit TEXT,
    as_of DATE,
    as_of_date DATE,
    publication_date DATE,
    publisher TEXT,
    source_id TEXT,
    source_type TEXT,
    source_url TEXT,
    doc_title TEXT,
    definition_text TEXT,
    precedence_tier TEXT,
    event_id TEXT,
    proxy_for TEXT,
    confidence TEXT,
    series TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT facts_resolved_unique UNIQUE (ym, iso3, hazard_code, metric, series_semantics)
);

CREATE INDEX IF NOT EXISTS idx_facts_resolved_lookup
    ON facts_resolved (iso3, hazard_code, ym);

CREATE TABLE IF NOT EXISTS facts_deltas (
    ym TEXT NOT NULL,
    iso3 TEXT NOT NULL,
    hazard_code TEXT NOT NULL,
    metric TEXT NOT NULL,
    value_new DOUBLE,
    value_stock DOUBLE,
    series_semantics TEXT NOT NULL DEFAULT 'new',
    as_of DATE,
    source_id TEXT,
    series TEXT,
    rebase_flag INTEGER,
    first_observation INTEGER,
    delta_negative_clamped INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT facts_deltas_unique UNIQUE (ym, iso3, hazard_code, metric)
);

CREATE INDEX IF NOT EXISTS idx_facts_deltas_lookup
    ON facts_deltas (iso3, hazard_code, ym);

CREATE TABLE IF NOT EXISTS manifests (
    path TEXT PRIMARY KEY,
    sha256 TEXT,
    row_count INTEGER,
    schema_version TEXT,
    source_id TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS meta_runs (
    run_id TEXT PRIMARY KEY,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    status TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS snapshots (
    ym TEXT PRIMARY KEY,
    created_at TIMESTAMP,
    git_sha TEXT,
    export_version TEXT,
    facts_rows INTEGER,
    deltas_rows INTEGER,
    meta TEXT
);

COMMIT;
