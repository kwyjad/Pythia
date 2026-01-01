from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
import yaml
from fastapi.testclient import TestClient

duckdb = pytest.importorskip("duckdb")

from pythia import config as pythia_config
from pythia.api.app import app


def _write_config(tmp_path: Path, db_path: Path, token: str) -> Path:
    cfg = {
        "app": {"db_url": f"duckdb:///{db_path}"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[dict[str, str], None, None]:
    db_path = tmp_path / "bundle.duckdb"
    con = duckdb.connect(str(db_path), read_only=False)
    con.execute(
        """
        CREATE TABLE hs_runs (hs_run_id TEXT, generated_at TIMESTAMP, created_at TIMESTAMP);
        INSERT INTO hs_runs VALUES
          ('hs-old', TIMESTAMP '2024-01-01', TIMESTAMP '2024-01-01'),
          ('hs-new', TIMESTAMP '2024-02-01', TIMESTAMP '2024-02-01');
        CREATE TABLE hs_triage (
            run_id TEXT, iso3 TEXT, hazard_code TEXT, tier TEXT, triage_score DOUBLE, created_at TIMESTAMP,
            drivers_json TEXT, regime_shifts_json TEXT, data_quality_json TEXT
        );
        INSERT INTO hs_triage VALUES (
            'hs-new', 'KEN', 'DR', 'priority', 0.8, TIMESTAMP '2024-02-02',
            '{"drivers":["dry"]}', '{"regime":["shift"]}', '{"data":"ok"}'
        );
        CREATE TABLE hs_country_reports (
            hs_run_id TEXT, iso3 TEXT, report_markdown TEXT, sources_json TEXT,
            grounded BOOLEAN, grounding_debug_json TEXT, structural_context TEXT, recent_signals_json TEXT
        );
        INSERT INTO hs_country_reports VALUES ('hs-new', 'KEN', '# report', '["source"]', TRUE, '{"groundingSupports_count":1}', 'struct', '["sig"]');
        CREATE TABLE hs_scenarios (
            hs_run_id TEXT, scenario_id TEXT, iso3 TEXT, hazard_code TEXT, scenario_title TEXT,
            scenario_markdown TEXT, scenario_json TEXT
        );
        INSERT INTO hs_scenarios VALUES ('hs-new', 'S-NEW', 'KEN', 'DR', 'Title', 'md', '{"key":"value"}');
        CREATE TABLE questions (
            question_id TEXT, hs_run_id TEXT, scenario_ids_json TEXT, iso3 TEXT, hazard_code TEXT,
            metric TEXT, target_month TEXT, wording TEXT
        );
        INSERT INTO questions VALUES
          ('Q1', 'hs-old', '["S-OLD"]', 'KEN', 'DR', 'PA', '2025-01', 'Old wording'),
          ('Q1', 'hs-new', '["S-NEW"]', 'KEN', 'DR', 'PA', '2025-01', 'New wording'),
          ('Q2', 'hs-new', '[]', 'UGA', 'ACE', 'FATALITIES', '2025-02', 'Fatalities wording');
        CREATE TABLE forecasts_ensemble (
            run_id TEXT, question_id TEXT, month_index INTEGER, bucket_index INTEGER,
            probability DOUBLE, model_name TEXT, created_at TIMESTAMP
        );
        INSERT INTO forecasts_ensemble VALUES
          ('f-old', 'Q1', 1, 1, 0.1, 'm', TIMESTAMP '2024-02-01'),
          ('f-new', 'Q1', 1, 2, 0.2, 'm', TIMESTAMP '2024-03-01');
        CREATE TABLE forecasts_raw (
            run_id TEXT, question_id TEXT, model_name TEXT, month_index INTEGER,
            bucket_index INTEGER, probability DOUBLE, spd_json TEXT
        );
        INSERT INTO forecasts_raw VALUES ('f-new', 'Q1', 'm', 1, 1, 0.3, '{"spd":1}');
        CREATE TABLE question_research (
            run_id TEXT, question_id TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, research_json TEXT,
            hs_evidence_json TEXT, question_evidence_json TEXT, merged_evidence_json TEXT,
            created_at TIMESTAMP
        );
        INSERT INTO question_research VALUES (
            'f-new', 'Q1', 'KEN', 'DR', 'PA', '{"base_rate":{"note":"test"}}',
            '{}', '{}', '{}', TIMESTAMP '2024-03-02'
        );
        CREATE TABLE scenarios (
            run_id TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, scenario_type TEXT,
            bucket_label TEXT, probability DOUBLE, text TEXT, created_at TIMESTAMP
        );
        INSERT INTO scenarios VALUES ('f-new', 'KEN', 'DR', 'PA', 'baseline', 'baseline', 0.4, 'story', now());
        CREATE TABLE question_context (
            run_id TEXT, question_id TEXT, snapshot_end_month TEXT, context_json TEXT, pa_history_json TEXT
        );
        INSERT INTO question_context VALUES
          ('f-old', 'Q1', '2025-05', '{"ctx":"old"}', '{"pa":1}'),
          ('f-new', 'Q1', '2025-06', '{"ctx":"new"}', '{"pa":2}');
        CREATE TABLE resolutions (question_id TEXT, observed_month TEXT, value DOUBLE);
        INSERT INTO resolutions VALUES ('Q1', '2025-01', 12.0);
        CREATE TABLE scores (
            question_id TEXT, horizon_m INTEGER, model_name TEXT, score_type TEXT, value DOUBLE, created_at TIMESTAMP
        );
        INSERT INTO scores VALUES ('Q1', 1, 'model-a', 'brier', 0.12, TIMESTAMP '2025-02-01');
        CREATE TABLE llm_calls (
            call_id TEXT, run_id TEXT, hs_run_id TEXT, question_id TEXT, phase TEXT, prompt_text TEXT,
            response_text TEXT, parsed_json TEXT, usage_json TEXT, timestamp TIMESTAMP, iso3 TEXT,
            hazard_code TEXT, model_id TEXT, model_name TEXT
        );
        INSERT INTO llm_calls VALUES
          ('c1', 'f-new', NULL, 'Q1', 'research_v2', 'prompt', 'response', '{"foo":1}', '{"bar":2}', now(), NULL, NULL, 'gemini-3-pro-preview', 'gemini'),
          ('c2', NULL, 'hs-new', NULL, 'hs_triage', 'hs prompt', 'hs response', '{"note":"triage"}', '{"usage":1}', now(), 'KEN', 'DR', 'gpt-5.1', 'gpt'),
          ('c3', NULL, 'hs-new', NULL, 'hs_web_research', 'hs web', 'hs web response', '{"note":"web"}', '{"usage":2}', now(), 'KEN', 'DR', 'gemini-3-flash-preview', 'gemini'),
          ('c4', 'f-new', NULL, 'Q1', 'forecast_web_research', 'web prompt', 'web response', '{"note":"forecast web"}', '{"usage":3}', now(), NULL, NULL, 'claude-opus-4-5-20240229', 'claude'),
          ('c_lower', NULL, 'hs-new', NULL, 'hs_triage', 'hs prompt 2', 'hs response 2', '{"note":"triage-2"}', '{"usage":4}', now(), 'ken', 'dr', 'gemini-3-flash-preview', 'gemini');
        CREATE TABLE bucket_definitions (
            metric TEXT, bucket_index INTEGER, label TEXT, lower_bound DOUBLE, upper_bound DOUBLE
        );
        INSERT INTO bucket_definitions VALUES
          ('PA', 1, '<10k', 0, 10000),
          ('PA', 2, '10k-<50k', 10000, 50000),
          ('PA', 3, '50k-<250k', 50000, 250000),
          ('PA', 4, '250k-<500k', 250000, 500000),
          ('PA', 5, '>=500k', 500000, NULL),
          ('FATALITIES', 1, '<5', 0, 5),
          ('FATALITIES', 2, '5-<25', 5, 25),
          ('FATALITIES', 3, '25-<100', 25, 100),
          ('FATALITIES', 4, '100-<500', 100, 500),
          ('FATALITIES', 5, '>=500', 500, NULL);
        CREATE TABLE bucket_centroids (
            hazard_code TEXT, metric TEXT, bucket_index INTEGER, centroid DOUBLE
        );
        INSERT INTO bucket_centroids VALUES
          ('*', 'PA', 1, 0),
          ('*', 'PA', 2, 30000),
          ('*', 'PA', 3, 150000),
          ('*', 'PA', 4, 375000),
          ('*', 'PA', 5, 700000),
          ('*', 'FATALITIES', 1, 0),
          ('*', 'FATALITIES', 2, 15),
          ('*', 'FATALITIES', 3, 62),
          ('*', 'FATALITIES', 4, 300),
          ('*', 'FATALITIES', 5, 700);
        """
    )
    con.close()

    token = "secret-token"
    config_path = _write_config(tmp_path, db_path, token)
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("PYTHIA_API_TOKEN", token)
    pythia_config.load.cache_clear()

    try:
        yield {"token": token}
    finally:
        pythia_config.load.cache_clear()
        monkeypatch.delenv("PYTHIA_API_TOKEN", raising=False)


@pytest.fixture()
def client(api_env: dict[str, str]) -> TestClient:
    return TestClient(app, headers={"Authorization": f"Bearer {api_env['token']}"})


@pytest.fixture()
def unauthorized_client(api_env: dict[str, str]) -> TestClient:
    return TestClient(app)


def test_question_bundle_returns_expected_payload(client: TestClient) -> None:
    resp = client.get("/v1/question_bundle", params={"question_id": "Q1"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["question"]["hs_run_id"] == "hs-new"
    assert data["hs"]["hs_run"]["hs_run_id"] == "hs-new"
    assert data["hs"]["scenario_ids"] == ["S-NEW"]
    assert data["forecast"]["forecaster_run_id"] == "f-new"
    assert data["forecast"]["research"]["run_id"] == "f-new"
    assert data["context"]["question_context"]["run_id"] == "f-new"
    assert isinstance(data["context"]["scores"], list)
    assert data["forecast"]["bucket_labels"] == [
        "<10k",
        "10k-<50k",
        "50k-<250k",
        "250k-<500k",
        ">=500k",
    ]
    assert data["forecast"]["bucket_centroids"] == [0.0, 30000.0, 150000.0, 375000.0, 700000.0]
    assert any(
        row["score_type"] == "brier" and row["value"] == pytest.approx(0.12)
        for row in data["context"]["scores"]
    )
    assert data["llm_calls"]["included"] is False


def test_question_bundle_is_public(unauthorized_client: TestClient) -> None:
    resp = unauthorized_client.get("/v1/question_bundle", params={"question_id": "Q1"})
    assert resp.status_code == 200


def test_question_bundle_fatalities_bucket_specs(client: TestClient) -> None:
    resp = client.get("/v1/question_bundle", params={"question_id": "Q2"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["forecast"]["bucket_labels"] == ["<5", "5-<25", "25-<100", "100-<500", ">=500"]
    assert data["forecast"]["bucket_centroids"] == [0.0, 15.0, 62.0, 300.0, 700.0]


def test_question_bundle_accepts_legacy_header(api_env: dict[str, str]) -> None:
    client = TestClient(app, headers={"X-Pythia-Token": api_env["token"]})
    resp = client.get("/v1/question_bundle", params={"question_id": "Q1"})
    assert resp.status_code == 200


def test_question_bundle_llm_calls_toggle(client: TestClient) -> None:
    resp = client.get(
        "/v1/question_bundle",
        params={"question_id": "Q1", "include_llm_calls": True},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["llm_calls"]["included"] is True
    assert data["llm_calls"]["transcripts_included"] is False
    assert len(data["llm_calls"]["rows"]) == 5
    assert all("prompt_text" not in row for row in data["llm_calls"]["rows"])
    assert "hs_triage" in data["llm_calls"]["by_phase"]
    assert "hs_web_research" in data["llm_calls"]["by_phase"]
    assert "forecast_web_research" in data["llm_calls"]["by_phase"]
    assert any(
        row.get("model_id") == "gemini-3-flash-preview"
        for row in data["llm_calls"]["by_phase"]["hs_web_research"]
    )
    assert any(
        row.get("model_id") == "gemini-3-flash-preview"
        for row in data["llm_calls"]["by_phase"]["hs_triage"]
    )
    assert data["llm_calls"]["debug"]["counts_by_phase"]["hs_triage"] >= 1


def test_question_bundle_llm_calls_with_transcripts(client: TestClient) -> None:
    resp = client.get(
        "/v1/question_bundle",
        params={"question_id": "Q1", "include_llm_calls": True, "include_transcripts": True},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["llm_calls"]["transcripts_included"] is True
    assert any("prompt_text" in row for row in data["llm_calls"]["rows"])
