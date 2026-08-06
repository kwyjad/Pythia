# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""End-to-end runner tests: model mocked, temp DuckDB, bundle-dir fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

import interpreter.run as run_mod
from interpreter.run import parse_model_json, run_interpreter

QID = "ETH_ACE_FATALITIES_2026-08"
FC_RUN = "fc_2000"


# ---------------------------------------------------------------------------
# Bundle fixtures (load_pack accepts a directory)
# ---------------------------------------------------------------------------


@pytest.fixture
def current_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "current_bundle"
    (root / "questions").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({
        "bundle_kind": "current_run_analysis",
        "run_id": FC_RUN,
        "hs_run_id": "hs_2",
        "n_questions": 1,
    }), encoding="utf-8")
    (root / "ANALYST_GUIDE.md").write_text("# guide", encoding="utf-8")
    (root / "attention_index.csv").write_text(
        "question_id,iso3,hazard_code,metric,score_family,js_vs_baserate,"
        "log_ev_ratio,eiv_nominal,eiv_per_100k,rc_level,rc_score,"
        "attention_rank,baserate_source,record_path\n"
        f"{QID},ETH,ACE,FATALITIES,spd,0.3466,1.2,200000,180,2,0.24,1.0,"
        f"acled_monthly_fatalities:6m,questions/{QID}.json\n",
        encoding="utf-8",
    )
    (root / "deltas.json").write_text('{"previous_run_id": null, "note": "first cycle"}',
                                      encoding="utf-8")
    (root / "blind_spots.json").write_text('{"standing_caveats": []}', encoding="utf-8")
    (root / "questions" / f"{QID}.json").write_text(json.dumps({
        "question": {"question_id": QID},
        "bucket_labels": {"1": "0", "2": "1-<5", "3": "5-<25", "4": "25-<100",
                          "5": "100-<500", "6": "500-<1000", "7": ">=1000"},
        "ensemble": {
            "ensemble_mean_v2": {
                "months": {"1": {"1": 0.1, "2": 0.1, "3": 0.1, "4": 0.4,
                                 "5": 0.2, "6": 0.05, "7": 0.05}},
            },
        },
    }), encoding="utf-8")
    return root


@pytest.fixture
def scored_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "scored_bundle"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({
        "bundle_kind": "scored_forecast_analysis",
        "generated_at": "2026-08-08T00:00:00+00:00",
        "n_scored_questions": 4,
    }), encoding="utf-8")
    (root / "digest.md").write_text("# digest", encoding="utf-8")
    (root / "rollups.csv").write_text(
        "hazard_code,metric,score_family,model_name,score_type,n_samples,"
        "n_questions,mean_value,median_value,climatology_mean,skill_vs_climatology\n"
        "ACE,FATALITIES,spd,ensemble_bayesmc_v2,brier,10,5,0.4,0.35,0.8,0.5\n"
        "ACE,FATALITIES,spd,__ext_climatology,brier,10,5,0.8,0.8,0.8,0.0\n",
        encoding="utf-8",
    )
    return root


def _content(kind: str) -> dict:
    content = {
        "schema_version": "1",
        "template_version": "v1",
        "kind": kind,
        "headline": "One risk stands out this month.",
        "attention": [{
            "rank": 1,
            "reason_code": "base_rate_deviation",
            "iso3": "ETH",
            "hazard_code": "ACE",
            "metric": "FATALITIES",
            "question_ids": [QID],
            "why_it_stands_out": "The ensemble moved {{fig:js_vs_baserate}} from its base rate, with a modal outcome of {{fig:modal_bucket_label}} fatalities.",
            "lead_time_months": 2,
        }],
        "performance": {
            "plain_summary": "The system beat its base rate.",
            "skill_statement": "Skill against climatology was {{fig:skill_brier_spd}}.",
        },
        "blind_spots": ["Impact reporting is thin."],
    }
    if kind == "scored":
        content.pop("attention")
    if kind == "current":
        content.pop("performance")
    return content


@pytest.fixture
def mock_model(monkeypatch):
    """Patch the model seam; returns a recorder with .calls and .response."""
    class Recorder:
        def __init__(self):
            self.calls: list[str] = []
            self.response = json.dumps(_content("combined"))

    rec = Recorder()

    def _fake_call(prompt: str, model_ref: str):
        rec.calls.append(prompt)
        return rec.response, {"prompt_tokens": 1000, "completion_tokens": 200}, ""

    monkeypatch.setattr(run_mod, "_call_model", _fake_call)
    monkeypatch.setattr(run_mod, "_log_llm_call", lambda **kw: None)
    monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
    return rec


def _rows(db: Path):
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.execute(
            "SELECT kind, version, status, content_md, figures_json, "
            "model_name, thinking_level FROM interpretations "
            "ORDER BY created_at, version"
        ).fetchall()
    finally:
        con.close()


class TestParseModelJson:
    def test_plain_fenced_and_prefixed(self):
        obj = {"a": 1}
        assert parse_model_json(json.dumps(obj)) == obj
        assert parse_model_json(f"```json\n{json.dumps(obj)}\n```") == obj
        assert parse_model_json(f"Here you go:\n{json.dumps(obj)} thanks") == obj
        assert parse_model_json("no json at all") is None
        assert parse_model_json("") is None


class TestRunner:
    def test_scored_then_combined(self, tmp_path, current_bundle, scored_bundle, mock_model):
        db = tmp_path / "p.duckdb"

        mock_model.response = json.dumps(_content("scored"))
        result = run_interpreter(db=str(db), kind="scored", pack_path=str(scored_bundle))
        assert result["status"] == "ok"

        mock_model.response = json.dumps(_content("combined"))
        result2 = run_interpreter(
            db=str(db), kind="combined", pack_path=str(current_bundle)
        )
        assert result2["status"] == "ok"

        rows = _rows(db)
        assert [(r[0], r[1], r[2]) for r in rows] == [
            ("scored", 1, "ok"), ("combined", 1, "ok"),
        ]
        # The combined prompt carried the stored scored interpretation.
        assert "scored-run interpretation already" in mock_model.calls[-1].replace("\n", " ") \
            or "already exists" in mock_model.calls[-1].replace("\n", " ")

        combined_md = rows[1][3]
        assert "{{fig:" not in combined_md
        # Per-question figure from the pack...
        assert "50% of maximum" in combined_md
        # ...modal bucket from the ensemble grid...
        assert "25-<100" in combined_md
        # ...and the performance figure from the SCORED interpretation's
        # stored figure map (the scored pack is not loaded here).
        assert "+50%" in combined_md
        assert rows[1][5] == "claude-opus-5"
        assert rows[1][6] == "high"

    def test_combined_without_scored_says_so(self, tmp_path, current_bundle, mock_model):
        db = tmp_path / "p.duckdb"
        result = run_interpreter(db=str(db), kind="combined", pack_path=str(current_bundle))
        assert result["status"] == "ok"
        assert "NO scored run exists yet" in mock_model.calls[-1]

    def test_skip_then_force_new_version(self, tmp_path, current_bundle, mock_model):
        db = tmp_path / "p.duckdb"
        assert run_interpreter(db=str(db), kind="combined",
                               pack_path=str(current_bundle))["status"] == "ok"
        skipped = run_interpreter(db=str(db), kind="combined",
                                  pack_path=str(current_bundle))
        assert skipped["status"] == "skipped" and skipped["reason"] == "exists"
        assert len(mock_model.calls) == 1  # no second model call
        forced = run_interpreter(db=str(db), kind="combined",
                                 pack_path=str(current_bundle), force=True)
        assert forced["status"] == "ok" and forced["version"] == 2

    def test_dry_run_calls_nothing_stores_nothing(self, tmp_path, current_bundle, mock_model):
        db = tmp_path / "p.duckdb"
        out = tmp_path / "out"
        result = run_interpreter(
            db=str(db), kind="combined", pack_path=str(current_bundle),
            dry_run=True, out_dir=str(out),
        )
        assert result["reason"] == "dry_run"
        assert mock_model.calls == []
        assert (out / "prompt_combined.txt").exists()
        con = duckdb.connect(str(db), read_only=True)
        try:
            n = con.execute("SELECT COUNT(*) FROM interpretations").fetchone()[0]
        finally:
            con.close()
        assert n == 0

    def test_failed_generation_stored_and_inspectable(self, tmp_path, current_bundle, mock_model):
        db = tmp_path / "p.duckdb"
        mock_model.response = "I refuse to emit JSON."
        result = run_interpreter(db=str(db), kind="combined", pack_path=str(current_bundle))
        assert result["status"] == "failed_generation"
        rows = _rows(db)
        assert rows[0][2] == "failed_generation"

    def test_failed_validation_still_renders(self, tmp_path, current_bundle, mock_model):
        db = tmp_path / "p.duckdb"
        bad = _content("combined")
        del bad["headline"]
        mock_model.response = json.dumps(bad)
        result = run_interpreter(db=str(db), kind="combined", pack_path=str(current_bundle))
        assert result["status"] == "failed_validation"
        rows = _rows(db)
        assert rows[0][2] == "failed_validation"
        assert rows[0][3]  # content_md written so the failure is inspectable

    def test_kill_switch(self, tmp_path, current_bundle, mock_model, monkeypatch):
        monkeypatch.setenv("PYTHIA_INTERPRETER_ENABLED", "0")
        result = run_interpreter(db=str(tmp_path / "p.duckdb"), kind="combined",
                                 pack_path=str(current_bundle))
        assert result["reason"] == "disabled"
        assert mock_model.calls == []

    def test_wrong_pack_kind_raises_but_main_returns_zero(
        self, tmp_path, current_bundle, scored_bundle, mock_model
    ):
        with pytest.raises(ValueError):
            run_interpreter(db=str(tmp_path / "p.duckdb"), kind="scored",
                            pack_path=str(current_bundle))
        rc = run_mod.main([
            "--db", str(tmp_path / "p.duckdb"), "--kind", "scored",
            "--pack", str(current_bundle),
        ])
        assert rc == 0  # never fail the pipeline
