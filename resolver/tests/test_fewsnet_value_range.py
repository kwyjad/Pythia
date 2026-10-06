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
