import pytest

duckdb = pytest.importorskip("duckdb")

import pythia.db.schema as schema_mod
from horizon_scanner import horizon_scanner as hs_mod


def test_build_resolver_features_handles_acled_month(monkeypatch, tmp_path):
    db_path = tmp_path / "resolver.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month TEXT, fatalities INTEGER);"
    )
    con.execute(
        "INSERT INTO acled_monthly_fatalities VALUES ('ETH','2025-01',10),('ETH','2025-02',20);"
    )
    con.close()

    def fake_connect(read_only=False):
        return duckdb.connect(str(db_path), read_only=read_only)

    monkeypatch.setattr(schema_mod, "connect", fake_connect)

    feats = hs_mod._build_resolver_features_for_country("ETH")
    assert "conflict" in feats
    conf = feats["conflict"]
    assert conf["source"] == "ACLED"
    assert conf["history_length"] == 2
    assert conf["recent_max"] == 20
