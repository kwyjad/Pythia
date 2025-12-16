# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from pathlib import Path

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from resolver.tools import export_facts


def _acled_row(
    iso3: str,
    country: str,
    metric: str,
    value: int,
    hazard_code: str,
    hazard_label: str,
    unit: str,
) -> dict[str, object]:
    return {
        "event_id": f"{iso3}-{hazard_code}-{metric}-2025-11",
        "country_name": country,
        "iso3": iso3,
        "hazard_code": hazard_code,
        "hazard_label": hazard_label,
        "hazard_class": "human-induced",
        "metric": metric,
        "series_semantics": "new",
        "value": value,
        "unit": unit,
        "as_of_date": "2025-11",
        "publication_date": "2025-11-30",
        "publisher": "ACLED",
        "source_type": "other",
        "source_url": "https://example.org/acled/nov-2025",
        "doc_title": "ACLED monthly aggregation",
        "definition_text": f"ACLED monthly {metric} for {hazard_label}",
        "method": "ACLED monthly aggregation",
        "confidence": "med",
        "revision": 1,
        "ingested_at": "2025-12-05T00:00:00Z",
        "source": "ACLED",
    }


def test_acled_ingestion_export_and_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "resolver" / "tools" / "export_config.yml"

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RESOLVER_DB_URL", raising=False)
    monkeypatch.delenv("RESOLVER_WRITE_DB", raising=False)

    staging_dir = tmp_path / "resolver" / "staging"
    staging_dir.mkdir(parents=True)
    rows = [
        _acled_row("AFG", "Afghanistan", "events", 2, "CU", "Civil Unrest", "events"),
        _acled_row(
            "AFG",
            "Afghanistan",
            "fatalities_battle_month",
            6,
            "ACE",
            "Armed Conflict - Escalation",
            "persons",
        ),
        _acled_row("AGO", "Angola", "events", 1, "CU", "Civil Unrest", "events"),
        _acled_row(
            "AGO",
            "Angola",
            "fatalities_battle_month",
            1,
            "ACE",
            "Armed Conflict - Escalation",
            "persons",
        ),
    ]
    pd.DataFrame(rows).to_csv(staging_dir / "acled.csv", index=False)

    facts_out_dir = tmp_path / "resolver" / "exports"
    facts_out_dir.mkdir(parents=True)
    db_path = tmp_path / "resolver.duckdb"

    export_facts.export_facts(
        inp=staging_dir,
        config_path=config_path,
        out_dir=facts_out_dir,
        write_db=True,
        db_url=f"duckdb:///{db_path.as_posix()}",
    )

    facts_csv = facts_out_dir / "facts.csv"
    assert facts_csv.exists(), "export_facts should create facts.csv"
    facts = pd.read_csv(facts_csv)
    assert len(facts) == 4
    nov_rows = facts[facts["ym"] == "2025-11"]
    assert not nov_rows.empty
    assert set(nov_rows["iso3"]) == {"AFG", "AGO"}
    assert set(nov_rows["series_semantics"].str.lower()) == {"new"}
    assert set(nov_rows["as_of_date"]) == {"2025-11-01"}

    def _value_for(iso: str, metric: str) -> int:
        subset = nov_rows[(nov_rows["iso3"] == iso) & (nov_rows["metric"] == metric)]
        assert not subset.empty, f"missing {iso} {metric} row"
        return int(pd.to_numeric(subset["value"]).iloc[0])

    assert _value_for("AFG", "events") == 2
    assert _value_for("AFG", "fatalities_battle_month") == 6
    assert _value_for("AGO", "events") == 1
    assert _value_for("AGO", "fatalities_battle_month") == 1

    with duckdb.connect(db_path.as_posix()) as conn:
        rows = conn.execute(
            """
            SELECT iso3, ym, metric, SUM(value_new) AS total, MIN(series_semantics) AS semantics
            FROM facts_deltas
            GROUP BY 1, 2, 3
            ORDER BY iso3, metric
            """
        ).fetchall()

    assert rows, "facts_deltas should receive ACLED rows"
    assert all(row[1] == "2025-11" for row in rows)
    assert all(row[4] == "new" for row in rows)
    db_summary = {(row[0], row[2]): row[3] for row in rows}
    assert db_summary[("AFG", "events")] == 2
    assert db_summary[("AFG", "fatalities_battle_month")] == 6
    assert db_summary[("AGO", "events")] == 1
    assert db_summary[("AGO", "fatalities_battle_month")] == 1
