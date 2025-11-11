"""Lightweight exit-code assertions for ``idmc_to_duckdb``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from resolver.cli import idmc_to_duckdb


def _write_csv(dest: Path, rows: list[dict[str, object]]) -> Path:
    frame = pd.DataFrame(rows)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    return dest


def _base_args(facts_path: Path, db_path: Path, out_dir: Path) -> list[str]:
    return [
        "--facts-csv",
        str(facts_path),
        "--db-url",
        str(db_path),
        "--out",
        str(out_dir),
    ]


@pytest.mark.duckdb
def test_exit_code_ok_for_dry_run(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    facts_path = _write_csv(
        tmp_path / "facts.csv",
        [
            {
                "iso3": "COL",
                "as_of_date": "2024-02-29",
                "metric": "new_displacements",
                "value": 150,
                "series_semantics": "new",
                "source": "IDMC",
            }
        ],
    )
    db_path = tmp_path / "resolver.duckdb"
    out_dir = tmp_path / "out"

    exit_code = idmc_to_duckdb.run(_base_args(facts_path, db_path, out_dir))

    assert exit_code == idmc_to_duckdb.EXIT_OK
    stdout = capfd.readouterr().out
    assert "✅ Wrote 0 rows to DuckDB (dry-run)" in stdout


@pytest.mark.duckdb
def test_exit_code_strict_warns(tmp_path: Path, capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    facts_path = _write_csv(
        tmp_path / "facts.csv",
        [
            {
                "iso3": "COL",
                "as_of_date": "2024-02-29",
                "metric": "new_displacements",
                "value": 150,
                "series_semantics": "new",
                "source": "IDMC",
            }
        ],
    )
    db_path = tmp_path / "resolver_warn.duckdb"
    out_dir = tmp_path / "warn_out"

    forced_warnings = ["stock.csv: not present"]

    monkeypatch.setattr(
        idmc_to_duckdb,
        "_gather_warnings",
        lambda *sources, forced=tuple(forced_warnings): list(forced),
    )

    argv = [
        *_base_args(facts_path, db_path, out_dir),
        "--write-db",
        "--strict",
    ]

    exit_code = idmc_to_duckdb.run(argv)

    assert exit_code == idmc_to_duckdb.EXIT_STRICT_WARNINGS
    stdout = capfd.readouterr().out
    assert "Warnings:" in stdout
    for message in forced_warnings:
        assert message in stdout


@pytest.mark.duckdb
def test_exit_code_empty_facts(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    facts_path = _write_csv(
        tmp_path / "facts_empty.csv",
        [
            {
                "iso3": "COL",
                "as_of_date": "2024-02-29",
                "metric": "new_displacements",
                "value": 150,
                "series_semantics": "new",
                "source": "IDMC",
            }
        ][:0],
    )
    db_path = tmp_path / "resolver_empty.duckdb"
    out_dir = tmp_path / "empty_out"

    argv = [
        *_base_args(facts_path, db_path, out_dir),
        "--write-db",
    ]

    exit_code = idmc_to_duckdb.run(argv)

    assert exit_code == idmc_to_duckdb.EXIT_EMPTY_FACTS
    stdout = capfd.readouterr().out
    assert "✅ Wrote 0 rows to DuckDB" in stdout
    assert "(dry-run)" not in stdout
    assert "Warnings:" in stdout
    assert "No canonical facts rows available for DuckDB write" in stdout
