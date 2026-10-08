# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""FEWS NET publishes a range and ``value`` is its lower bound.

The probe (run 37315217164) found ``value == low_value`` in 975 of 999
Current Situation rows. The connector dropped ``high_value``, so every reader
saw one number with no sign it was a floor. The upper bound now travels as
``value_high`` from the connector into ``facts_resolved``.
"""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd

from resolver.connectors.fewsnet_ipc import FewsnetIpcConnector, _range_high

CSV = textwrap.dedent("""\
    country_code,phase,scenario_name,projection_start,projection_end,value,low_value,high_value,phase_name,population_range,fnid,reporting_date
    ET,3+,Current Situation,2025-06-01,2025-06-30,1000000.0,1000000.0,2490000.0,Phase 3 and above,1.0 - 2.49 million,ET,2025-05-15
    SO,3+,Current Situation,2025-06-01,2025-06-30,4500000.0,4500000.0,4500000.0,Phase 3 and above,4.5 million,SO,2025-05-15
""")


def _fetch(csv: str) -> pd.DataFrame:
    response = MagicMock(status_code=200, text=csv, content=csv.encode())
    response.raise_for_status = lambda: None
    session = MagicMock()
    session.get.return_value = response
    with patch("resolver.connectors.fewsnet_ipc._build_session", return_value=session), \
            patch("resolver.connectors.fewsnet_ipc._write_country_list"), \
            patch.dict("os.environ", {"FEWSNET_MONTHS": "240", "FEWSNET_REQUEST_DELAY": "0"}):
        return FewsnetIpcConnector().fetch_and_normalize()


def test_the_connector_carries_the_upper_bound_of_the_range():
    df = _fetch(CSV).set_index("iso3")
    assert df.loc["ETH", "value"] == 1_000_000
    assert df.loc["ETH", "value_high"] == 2_490_000
    # A point figure (low == high) has no range to report.
    assert pd.isna(df.loc["SOM", "value_high"])


def test_a_feed_without_high_value_writes_null_not_a_guess():
    csv = "\n".join(
        ",".join(c for i, c in enumerate(line.split(",")) if i != 7)
        for line in CSV.strip().splitlines()
    ) + "\n"
    df = _fetch(csv)
    assert df["value_high"].isna().all()
    assert _range_high({"value": 5, "high_value": "x"}, True) is None


def test_the_upper_bound_survives_precedence_into_facts_resolved(tmp_path):
    """Through the path production runs: enrich, precedence, the DB writer."""
    from resolver.tools import run_pipeline as rp
    from resolver.tools.enrich import derive_ym, enrich

    combined = derive_ym(enrich(_fetch(CSV)))
    resolved = rp._run_precedence(combined)
    assert "value_high" in resolved.columns
    db = tmp_path / "r.duckdb"
    rp._write_to_db(f"duckdb:///{db}", resolved, rp._run_deltas(resolved))
    con = duckdb.connect(str(db))
    types = {r[1]: r[2] for r in con.execute("PRAGMA table_info('facts_resolved')").fetchall()}
    assert types["value_high"] == "DOUBLE"
    got = dict(con.execute(
        "SELECT iso3, value_high FROM facts_resolved WHERE metric = 'phase3plus_in_need'"
    ).fetchall())
    assert got == {"ETH": 2_490_000.0, "SOM": None}


#: facts_resolved exactly as the canonical DB held it on 8 Oct 2026
#: (backcast run 37707136171): no value_high, both indexes present.
CANONICAL_2026_10_08_DDL = (
    "CREATE TABLE facts_resolved(ym VARCHAR NOT NULL, iso3 VARCHAR NOT NULL, hazard_code VARCHAR NOT NULL, "
    "hazard_label VARCHAR, hazard_class VARCHAR, metric VARCHAR NOT NULL, series_semantics VARCHAR "
    "DEFAULT('') NOT NULL, \"value\" DOUBLE, unit VARCHAR, as_of DATE, as_of_date VARCHAR, "
    "publication_date VARCHAR, publisher VARCHAR, source_id VARCHAR, source_type VARCHAR, source_url VARCHAR, "
    "doc_title VARCHAR, definition_text VARCHAR, precedence_tier VARCHAR, event_id VARCHAR, proxy_for VARCHAR, "
    "confidence VARCHAR, provenance_source VARCHAR, provenance_rank INTEGER, series VARCHAR, alertlevel VARCHAR, "
    "created_at TIMESTAMP DEFAULT(CURRENT_TIMESTAMP) NOT NULL, updated_at TIMESTAMP)",
    "CREATE INDEX idx_facts_resolved_lookup ON facts_resolved(iso3, hazard_code, ym)",
    "CREATE UNIQUE INDEX ux_facts_resolved_series ON facts_resolved(event_id, iso3, hazard_code, metric, "
    "as_of_date, publication_date, source_id, series_semantics, ym)",
)


def test_the_next_ingest_adds_value_high_to_the_canonical_schema(tmp_path):
    """The 11 Oct 2026 Resolver Update writes FEWS NET through run_pipeline's
    _write_to_db, which runs init_schema first. Against a copy of the canonical
    schema of 8 Oct 2026 that adds the column, keeps the existing rows, and
    lands the upper bound, with no manual step."""
    from resolver.tools import run_pipeline as rp
    from resolver.tools.enrich import derive_ym, enrich

    db = tmp_path / "canonical.duckdb"
    con = duckdb.connect(str(db))
    for ddl in CANONICAL_2026_10_08_DDL:
        con.execute(ddl)
    con.execute(
        "INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, publisher, event_id, source_id) "
        "VALUES ('2026-08', 'KEN', 'ACE', 'fatalities', 12, 'ACLED', 'e1', 'acled')"
    )
    con.close()

    combined = derive_ym(enrich(_fetch(CSV)))
    resolved = rp._run_precedence(combined)
    rp._write_to_db(f"duckdb:///{db}", resolved, rp._run_deltas(resolved))

    con = duckdb.connect(str(db))
    cols = [r[1] for r in con.execute("PRAGMA table_info('facts_resolved')").fetchall()]
    assert "value_high" in cols
    assert con.execute("SELECT value FROM facts_resolved WHERE iso3 = 'KEN'").fetchone() == (12.0,)
    got = dict(con.execute(
        "SELECT iso3, value_high FROM facts_resolved WHERE metric = 'phase3plus_in_need'"
    ).fetchall())
    assert got == {"ETH": 2_490_000.0, "SOM": None}
