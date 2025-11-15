import pandas as pd
import pytest

from resolver.cli import acled_to_duckdb
from resolver.db import duckdb_io
from resolver.ingestion import acled_client as acled_client_module
from resolver.ingestion.acled_client import ACLEDClient


def test_monthly_fatalities_converts_event_date(monkeypatch):
    monkeypatch.setattr(
        acled_client_module.acled_auth,
        "get_access_token",
        lambda: "token",
    )
    client = ACLEDClient()

    source = pd.DataFrame(
        {
            "iso3": ["afg", "AFG", "ALB"],
            "event_date": ["2024-01-05", "2024-01-20", "2024-01-10"],
            "fatalities": [2, 4, 1],
        }
    )

    def fake_fetch_events(self, *_args, **_kwargs):
        return source.copy()

    monkeypatch.setattr(
        client,
        "fetch_events",
        fake_fetch_events.__get__(client, ACLEDClient),
    )

    frame = client.monthly_fatalities("2024-01-01", "2024-01-31")
    assert set(frame.columns) == {"iso3", "month", "fatalities", "source", "updated_at"}
    assert len(frame) == 2

    afg = frame[frame["iso3"] == "AFG"].iloc[0]
    alb = frame[frame["iso3"] == "ALB"].iloc[0]
    assert afg["fatalities"] == 6
    assert alb["fatalities"] == 1
    assert afg["month"].strftime("%Y-%m-%d") == "2024-01-01"


@pytest.mark.duckdb
def test_acled_cli_appends_summary_on_error(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")

    if not duckdb_io.DUCKDB_AVAILABLE:
        pytest.skip("DuckDB module not available")

    class _BoomClient:
        def __init__(self, *_, **__):
            pass

        def monthly_fatalities(self, *_args, **_kwargs):
            raise RuntimeError("explode")

    monkeypatch.setattr(acled_to_duckdb, "ACLEDClient", _BoomClient)

    called = {}

    def fake_append(section, error_type, message, context):
        called.update(
            {
                "section": section,
                "error_type": error_type,
                "message": message,
                "context": context,
            }
        )

    monkeypatch.setattr(acled_to_duckdb.append_error_to_summary, "append_error", fake_append)

    args = [
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-31",
        "--db",
        str(tmp_path / "acled.duckdb"),
    ]

    with pytest.raises(RuntimeError):
        acled_to_duckdb.run(args)

    assert called["section"] == "ACLED CLI — monthly_fatalities error"
    assert called["error_type"] == "RuntimeError"
    assert called["message"] == "explode"
    assert called["context"]["start"] == "2024-01-01"
    assert called["context"]["end"] == "2024-01-31"
