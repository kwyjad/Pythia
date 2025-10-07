CREATE TABLE IF NOT EXISTS facts_raw (
    event_id TEXT,
    country_name TEXT,
    iso3 TEXT,
    hazard_code TEXT,
    hazard_label TEXT,
    hazard_class TEXT,
    metric TEXT,
    series_semantics TEXT,
    value DOUBLE,
    unit TEXT,
    as_of_date TEXT,
    publication_date TEXT,
    publisher TEXT,
    source_type TEXT,
    source_url TEXT,
    doc_title TEXT,
    definition_text TEXT,
    method TEXT,
    confidence TEXT,
    revision INTEGER,
    ingested_at TEXT,
    value_new DOUBLE,
    value_stock DOUBLE,
    rebase_flag INTEGER,
    first_observation INTEGER,
    delta_negative_clamped INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS facts_resolved (
    ym TEXT,
    iso3 TEXT,
    hazard_code TEXT,
    hazard_label TEXT,
    hazard_class TEXT,
    metric TEXT,
    series_semantics TEXT DEFAULT '',
    value DOUBLE,
    unit TEXT,
    as_of_date TEXT,
    publication_date TEXT,
    publisher TEXT,
    source_type TEXT,
    source_url TEXT,
    doc_title TEXT,
    definition_text TEXT,
    precedence_tier TEXT,
    event_id TEXT,
    proxy_for TEXT,
    confidence TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ym, iso3, hazard_code, metric, series_semantics)
);

CREATE TABLE IF NOT EXISTS facts_deltas (
    ym TEXT,
    iso3 TEXT,
    hazard_code TEXT,
    metric TEXT,
    value_new DOUBLE,
    value_stock DOUBLE,
    series_semantics TEXT DEFAULT '',
    rebase_flag INTEGER,
    first_observation INTEGER,
    delta_negative_clamped INTEGER,
    as_of TEXT,
    source_name TEXT,
    source_url TEXT,
    definition_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ym, iso3, hazard_code, metric)
);

CREATE TABLE IF NOT EXISTS snapshots (
    ym TEXT PRIMARY KEY,
    created_at TEXT,
    git_sha TEXT,
    export_version TEXT,
    facts_rows INTEGER,
    deltas_rows INTEGER,
    meta TEXT
);

CREATE TABLE IF NOT EXISTS manifests (
    ym TEXT,
    name TEXT,
    path TEXT,
    rows INTEGER,
    checksum TEXT,
    payload TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ym, name)
);
