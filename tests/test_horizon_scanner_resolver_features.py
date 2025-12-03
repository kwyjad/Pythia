import pytest

try:
    import duckdb
except ModuleNotFoundError:  # pragma: no cover - dependency may be absent in CI smoke runs
    pytest.skip("duckdb not installed", allow_module_level=True)

from horizon_scanner.horizon_scanner import _build_resolver_features_for_country


def test_build_resolver_features_handles_month_column(monkeypatch):
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month TEXT, fatalities INTEGER);"
    )
    con.execute(
        "INSERT INTO acled_monthly_fatalities VALUES ('ETH','2025-01', 10), ('ETH','2025-02', 20);"
    )

    import pythia.db.schema as schema_mod

    monkeypatch.setattr(schema_mod, "connect", lambda read_only=False: con)

    features = _build_resolver_features_for_country("ETH")

    assert "conflict" in features
    conflict = features["conflict"]
    assert conflict["history_length"] == 2
    assert conflict["recent_max"] == 20
