from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

import forecaster.cli as fc_cli
from pythia.db import schema as db_schema  # type: ignore


@pytest.mark.db
def test_epoch_loader_prefers_latest_hs_and_scopes_iso3(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "epoch_loader.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        db_schema.ensure_schema(con)

        con.execute(
            """
            INSERT INTO hs_triage (run_id, iso3, hazard_code, tier, triage_score,
                                   drivers_json, regime_shifts_json, data_quality_json, scenario_stub)
            VALUES
                ('hs_20250101T000000', 'SOM', 'DR', 'watchlist', 0.5, '[]', '[]', '{}', ''),
                ('hs_20250201T000000', 'SOM', 'DR', 'priority',  0.8, '[]', '[]', '{}', ''),
                ('hs_20250115T000000', 'ETH', 'DR', 'watchlist', 0.4, '[]', '[]', '{}', '')
            """
        )

        con.execute(
            """
            INSERT INTO questions (
                question_id, hs_run_id, scenario_ids_json, iso3, hazard_code, metric,
                target_month, window_start_date, window_end_date, wording, status, pythia_metadata_json
            ) VALUES
                ('SOM_DR_PA_OLD', 'hs_20250101T000000', '[]', 'SOM', 'DR', 'PA',
                 '2026-01', NULL, NULL, 'Old HS SOM DR PA', 'active', '{"source":"hs_triage"}'),
                ('SOM_DR_PA_NEW', 'hs_20250201T000000', '[]', 'SOM', 'DR', 'PA',
                 '2026-01', NULL, NULL, 'New HS SOM DR PA', 'active', '{"source":"hs_triage"}')
            """
        )

        con.execute(
            """
            INSERT INTO questions (
                question_id, hs_run_id, scenario_ids_json, iso3, hazard_code, metric,
                target_month, window_start_date, window_end_date, wording, status, pythia_metadata_json
            ) VALUES
                ('ETH_DR_PA_LEGACY', NULL, '[]', 'ETH', 'DR', 'PA',
                 '2026-01', NULL, NULL, 'Legacy ETH DR PA', 'active', NULL)
            """
        )
    finally:
        con.close()

    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")

    questions = fc_cli._load_pythia_questions(limit=None, iso3_filter={"SOM"})
    ids = {q.question_id for q in questions}

    assert ids == {"SOM_DR_PA_NEW"}


@pytest.mark.db
def test_loader_defaults_to_hs_iso3s_and_excludes_aco(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "iso3_default.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        db_schema.ensure_schema(con)

        con.execute(
            """
            INSERT INTO hs_triage (run_id, iso3, hazard_code, tier, triage_score,
                                   drivers_json, regime_shifts_json, data_quality_json, scenario_stub)
            VALUES
                ('hs_20250101T000000', 'SOM', 'DR', 'priority', 0.7, '[]', '[]', '{}', '')
            """
        )

        con.execute(
            """
            INSERT INTO questions (
                question_id, hs_run_id, scenario_ids_json, iso3, hazard_code, metric,
                target_month, window_start_date, window_end_date, wording, status, pythia_metadata_json
            ) VALUES
                ('SOM_DR_PA_HS', 'hs_20250101T000000', '[]', 'SOM', 'DR', 'PA',
                 '2026-02', NULL, NULL, 'HS SOM DR PA', 'active', '{"source":"hs_triage"}'),
                ('ETH_DR_PA_LEGACY', NULL, '[]', 'ETH', 'DR', 'PA',
                 '2026-01', NULL, NULL, 'Legacy ETH DR PA', 'active', NULL),
                ('SOM_ACO_FATALITIES', NULL, '[]', 'SOM', 'ACO', 'FATALITIES',
                 '2026-03', NULL, NULL, 'ACO question should be excluded', 'active', NULL)
            """
        )
    finally:
        con.close()

    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")

    questions = fc_cli._load_pythia_questions(limit=None, iso3_filter=None)
    ids = {q.question_id for q in questions}

    assert "SOM_DR_PA_HS" in ids
    assert "ETH_DR_PA_LEGACY" not in ids
    assert "SOM_ACO_FATALITIES" not in ids


@pytest.mark.db
def test_load_pythia_questions_uses_legacy_when_no_hs(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "questions_filter_legacy.duckdb"

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

    questions = fc_cli._load_pythia_questions(limit=None, iso3_filter={"SOM"})

    ids = {q.question_id for q in questions}
    assert ids == {"SOM_FL_PA_LEGACY"}
