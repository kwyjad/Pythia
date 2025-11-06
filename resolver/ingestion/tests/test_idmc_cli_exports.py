from pathlib import Path

import pandas as pd
import pytest

from resolver.ingestion.idmc import cli
from resolver.ingestion.idmc.export import FACT_COLUMNS, FLOW_EXPORT_COLUMNS


@pytest.fixture
def stubbed_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.chdir(tmp_path)

    config_dir = tmp_path / "resolver" / "ingestion" / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "idmc.yml").write_text(
        """\
enabled: true
api:
  countries: []
  series: ["flow"]
""",
        encoding="utf-8",
    )

    iso_dir = tmp_path / "resolver" / "ingestion" / "static"
    iso_dir.mkdir(parents=True, exist_ok=True)
    (iso_dir / "iso3_master.csv").write_text("iso3\nAFG\n", encoding="utf-8")

    empty_normalized = pd.DataFrame(
        {
            "iso3": pd.Series(dtype="string"),
            "as_of_date": pd.Series(dtype="datetime64[ns]"),
            "metric": pd.Series(dtype="string"),
            "value": pd.Series(dtype=pd.Int64Dtype()),
            "series_semantics": pd.Series(dtype="string"),
            "source": pd.Series(dtype="string"),
        }
    )

    def fake_fetch(*args, **kwargs):
        return {"monthly_flow": pd.DataFrame()}, {"filters": {"rows_before": 0, "rows_after": 0}}

    def fake_normalize(*args, **kwargs):
        drops = {
            "date_parse_failed": 0,
            "no_iso3": 0,
            "no_value_col": 0,
            "date_out_of_window": 0,
            "negative_value": 0,
            "dup_event": 0,
        }
        return empty_normalized.copy(), drops

    def fake_map_hazards(frame, enabled):
        return frame, pd.DataFrame()

    def fake_build_resolution_ready(frame):
        return pd.DataFrame(columns=FACT_COLUMNS)

    def fake_to_facts(frame):
        return pd.DataFrame(columns=FACT_COLUMNS)

    def fake_write_csv(frame, out_dir):
        dest = Path(out_dir) / "facts.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("", encoding="utf-8")
        return dest.as_posix()

    def fake_write_parquet(frame, out_dir):
        return None

    monkeypatch.setattr(cli, "fetch", fake_fetch)
    monkeypatch.setattr(cli, "normalize_all", fake_normalize)
    monkeypatch.setattr(cli, "maybe_map_hazards", fake_map_hazards)
    monkeypatch.setattr(cli, "build_resolution_ready_facts", fake_build_resolution_ready)
    monkeypatch.setattr(cli, "to_facts", fake_to_facts)
    monkeypatch.setattr(cli, "write_facts_csv", fake_write_csv)
    monkeypatch.setattr(cli, "write_facts_parquet", fake_write_parquet)
    monkeypatch.setattr(cli, "probe_reachability", lambda *args, **kwargs: {})

    return tmp_path


def test_cli_writes_header_when_empty(stubbed_cli: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESOLVER_OUTPUT_DIR", raising=False)
    exit_code = cli.main(
        [
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--enable-export",
            "--series",
            "flow",
            "--skip-network",
        ]
    )
    assert exit_code == 0
    flow_path = Path("resolver/staging/idmc/flow.csv")
    assert flow_path.exists()
    header = flow_path.read_text(encoding="utf-8").splitlines()
    assert header
    assert header[0].split(",") == FLOW_EXPORT_COLUMNS


def test_cli_populates_series_semantics(monkeypatch: pytest.MonkeyPatch, stubbed_cli: Path) -> None:
    def fake_build_resolution_ready(_frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "iso3": "AFG",
                    "as_of_date": "2024-01-31",
                    "metric": "new_displacements",
                    "value": 5,
                    "source": "idmc_idu",
                    "series_semantics": pd.NA,
                }
            ]
        )

    monkeypatch.setattr(cli, "build_resolution_ready_facts", fake_build_resolution_ready)
    monkeypatch.delenv("RESOLVER_OUTPUT_DIR", raising=False)

    exit_code = cli.main(
        [
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--enable-export",
            "--series",
            "flow",
            "--skip-network",
        ]
    )
    assert exit_code == 0

    flow_path = Path("resolver/staging/idmc/flow.csv")
    frame = pd.read_csv(flow_path)
    assert list(frame.columns) == FLOW_EXPORT_COLUMNS
    assert frame["series_semantics"].tolist() == ["new"]


def test_cli_accepts_debug_flag(stubbed_cli: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESOLVER_OUTPUT_DIR", raising=False)
    exit_code = cli.main(
        [
            "--debug",
            "--skip-network",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-02",
            "--series",
            "flow",
        ]
    )
    assert exit_code == 0
