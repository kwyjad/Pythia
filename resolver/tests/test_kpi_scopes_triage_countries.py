from __future__ import annotations

import string

import pytest

duckdb = pytest.importorskip("duckdb")

from resolver.query.kpi_scopes import compute_countries_triaged_for_month


def _build_iso3_codes(count: int) -> list[str]:
    iso3s: list[str] = []
    for a in string.ascii_uppercase:
        for b in string.ascii_uppercase:
            for c in string.ascii_uppercase:
                iso3s.append(f"{a}{b}{c}")
                if len(iso3s) == count:
                    return iso3s
    return iso3s


def test_compute_countries_triaged_for_month_from_llm_calls() -> None:
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE llm_calls (
            iso3 TEXT,
            phase TEXT,
            created_at TIMESTAMP
        );
        """
    )
    iso3s = _build_iso3_codes(120)
    rows = [(iso3, "hs_triage", "2026-01-15 00:00:00") for iso3 in iso3s]
    con.executemany("INSERT INTO llm_calls VALUES (?, ?, ?)", rows)

    result = compute_countries_triaged_for_month(con, "2026-01")

    assert result == 120
    con.close()


def test_compute_countries_triaged_for_month_fallback_to_hs_triage() -> None:
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE llm_calls (
            iso3 TEXT,
            phase TEXT,
            created_at TIMESTAMP
        );
        """
    )
    con.execute(
        """
        CREATE TABLE hs_triage (
            iso3 TEXT,
            created_at TIMESTAMP
        );
        """
    )
    con.execute(
        """
        INSERT INTO llm_calls VALUES ('USA', 'forecast', '2026-01-10 00:00:00');
        """
    )
    con.execute(
        """
        INSERT INTO hs_triage VALUES
            ('USA', '2026-01-02 00:00:00'),
            ('CAN', '2026-01-03 00:00:00'),
            ('MEX', '2026-01-04 00:00:00');
        """
    )

    result = compute_countries_triaged_for_month(con, "2026-01")

    assert result == 3
    con.close()
