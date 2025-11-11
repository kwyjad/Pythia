from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from resolver.cli import idmc_to_duckdb


_ROWS = (
    {
        "iso3": "COL",
        "as_of_date": "2024-02-29",
        "metric": "new_displacements",
        "value": 150,
        "series_semantics": "new",
        "source": "IDMC",
    },
    {
        "iso3": "COL",
        "as_of_date": "2024-02-29",
        "metric": "idp_displacement_stock_idmc",
        "value": 250,
        "series_semantics": "stock",
        "source": "IDMC",
    },
)


def _write_facts_csv(dest: Path) -> Path:
    frame = pd.DataFrame(_ROWS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    return dest


@pytest.mark.duckdb
@pytest.mark.parametrize(
    "extra_cli, expected_exit, expect_dry_run, forced_warnings",
    [
        pytest.param(
            (),
            idmc_to_duckdb.EXIT_OK,
            True,
            (),
            id="dry_run_success",
        ),
        pytest.param(
            ("--write-db",),
            idmc_to_duckdb.EXIT_OK,
            False,
            ("stock.csv: not present",),
            id="write_warn_non_strict",
        ),
        pytest.param(
            ("--write-db", "--strict"),
            idmc_to_duckdb.EXIT_STRICT_WARNINGS,
            False,
            ("stock.csv: not present",),
            id="write_warn_strict",
        ),
    ],
)
def test_idmc_to_duckdb_exit_policy_regressions(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    extra_cli: tuple[str, ...],
    expected_exit: int,
    expect_dry_run: bool,
    forced_warnings: tuple[str, ...],
) -> None:
    facts_path = _write_facts_csv(tmp_path / "facts.csv")
    db_path = tmp_path / "resolver.duckdb"
    out_dir = tmp_path / "out"

    if forced_warnings:
        monkeypatch.setattr(
            idmc_to_duckdb,
            "_gather_warnings",
            lambda *sources, forced=tuple(forced_warnings): list(forced),
        )

    argv = [
        "--facts-csv",
        str(facts_path),
        "--db-url",
        str(db_path),
        "--out",
        str(out_dir),
        *extra_cli,
    ]

    exit_code = idmc_to_duckdb.run(argv)
    assert exit_code == expected_exit

    stdout = capfd.readouterr().out
    banner_lines = [line for line in stdout.splitlines() if line.startswith("✅ Wrote ")]
    assert banner_lines, stdout
    banner = banner_lines[0]

    if expect_dry_run:
        assert banner.endswith("(dry-run)")
    else:
        assert "(dry-run)" not in banner

    if forced_warnings:
        assert "Warnings:" in stdout
        for message in forced_warnings:
            assert message in stdout

    assert f"exit={expected_exit}" in stdout
