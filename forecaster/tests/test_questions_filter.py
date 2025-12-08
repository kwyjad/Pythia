from __future__ import annotations

import pytest

import forecaster.cli as fc_cli
from pythia.db import schema as db_schema  # type: ignore

duckdb = pytest.importorskip("duckdb")


@pytest.mark.db
def test_load_pythia_questions_prefers_latest_hs_and_excludes_aco(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path.mktemp("db") / "questions_filter.duckdb"

    con = duckdb.connect(str(db_path))
    try:
        db_schema.ensure_schema(con)

        con.execute(
            """
            INSERT INTO questions (
                question_id, hs_run_id, scenario_ids_json, iso3, hazard_code, metric,
                target_month, window_start_date, window_end_date, wording, status, pythia_metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "ETH_ACE_FATALITIES_LEGACY",
                None,
                "[]",
                "ETH",
                "ACE",
                "FATALITIES",
                "2026-01",
                None,
                None,
                "Legacy ETH ACE fatalities",
                "active",
                None,
            ],
        )

        con.execute(
            """
            INSERT INTO questions (
                question_id, hs_run_id, scenario_ids_json, iso3, hazard_code, metric,
                target_month, window_start_date, window_end_date, wording, status, pythia_metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "ETH_ACE_FATALITIES_OLD",
                "hs_20250101T000000",
                "[]",
                "ETH",
                "ACE",
                "FATALITIES",
                "2026-01",
                None,
                None,
                "Old HS ETH ACE fatalities",
                "active",
                '{"source": "hs_triage"}',
            ],
        )
        con.execute(
            """
            INSERT INTO questions (
                question_id, hs_run_id, scenario_ids_json, iso3, hazard_code, metric,
                target_month, window_start_date, window_end_date, wording, status, pythia_metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "ETH_ACE_FATALITIES_NEW",
                "hs_20250201T000000",
                "[]",
                "ETH",
                "ACE",
                "FATALITIES",
                "2026-01",
                None,
                None,
                "New HS ETH ACE fatalities",
                "active",
                '{"source": "hs_triage"}',
            ],
        )

        con.execute(
            """
            INSERT INTO questions (
                question_id, hs_run_id, scenario_ids_json, iso3, hazard_code, metric,
                target_month, window_start_date, window_end_date, wording, status, pythia_metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "ETH_ACO_FATALITIES",
                None,
                "[]",
                "ETH",
                "ACO",
                "FATALITIES",
                "2026-01",
                None,
                None,
                "ETH ACO fatalities (should be excluded)",
                "active",
                None,
            ],
        )
    finally:
        con.close()

    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")

    questions = fc_cli._load_pythia_questions()

    ids = {q.question_id for q in questions}
    assert "ETH_ACE_FATALITIES_NEW" in ids
    assert "ETH_ACE_FATALITIES_OLD" not in ids
    assert "ETH_ACE_FATALITIES_LEGACY" not in ids
    assert "ETH_ACO_FATALITIES" not in ids


@pytest.mark.db
def test_load_pythia_questions_uses_legacy_when_no_hs(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path.mktemp("db") / "questions_filter_legacy.duckdb"

    con = duckdb.connect(str(db_path))
    try:
        db_schema.ensure_schema(con)

        con.execute(
            """
            INSERT INTO questions (
                question_id, hs_run_id, scenario_ids_json, iso3, hazard_code, metric,
                target_month, window_start_date, window_end_date, wording, status, pythia_metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "SOM_FL_PA_LEGACY",
                None,
                "[]",
                "SOM",
                "FL",
                "PA",
                "2026-02",
                None,
                None,
                "Legacy SOM FL PA",
                "active",
                None,
            ],
        )
    finally:
        con.close()

    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")

    questions = fc_cli._load_pythia_questions()

    ids = {q.question_id for q in questions}
    assert ids == {"SOM_FL_PA_LEGACY"}
