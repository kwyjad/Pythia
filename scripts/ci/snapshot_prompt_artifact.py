# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Generate a markdown artifact showing the full rendered prompts seen by LLMs.

For each hazard type (ACE, DR, FL, HW, TC) this script renders one complete
example of each prompt stage:
  1. Regime Change (RC) prompt
  2. Triage prompt
  3. SPD Forecast prompt
  4. Scenario prompt

All injects (resolver features, evidence packs, calibration advice, etc.) are
included so that the artifact shows exactly what the LLM sees. This is used
for prompt review and improvement.

Usage:
    python -m scripts.ci.snapshot_prompt_artifact --db <db_url> --out prompts_artifact.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _get_db_url() -> str:
    return (
        os.getenv("PYTHIA_DB_URL")
        or os.getenv("RESOLVER_DB_URL")
        or "duckdb:///data/resolver.duckdb"
    )


def _connect(db_url: str):
    from resolver.db import duckdb_io
    return duckdb_io.get_db(db_url)


def _close(con):
    from resolver.db import duckdb_io
    duckdb_io.close_db(con)


def _load_sample_country_for_hazard(
    con, hazard_code: str, run_id: str | None = None,
) -> Optional[Dict[str, Any]]:
    """Pick the best sample country for a hazard from the latest HS run.

    Prefers a priority-tier country with a moderate triage_score so the
    rendered prompt is interesting.
    """
    try:
        # Resolve run_id: use explicit if provided, otherwise latest
        if not run_id:
            latest_row = con.execute(
                "SELECT run_id FROM hs_triage ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if not latest_row:
                return None
            run_id = latest_row[0]

        row = con.execute(
            """
            SELECT iso3, triage_score, tier, run_id,
                   regime_change_likelihood, regime_change_magnitude,
                   regime_change_direction, regime_change_window,
                   regime_change_level, regime_change_score,
                   drivers_json, regime_shifts_json, data_quality_json,
                   scenario_stub, regime_change_json
            FROM hs_triage
            WHERE hazard_code = ? AND run_id = ?
            ORDER BY
                CASE WHEN tier = 'priority' THEN 0 ELSE 1 END,
                triage_score DESC
            LIMIT 1
            """,
            [hazard_code, run_id],
        ).fetchone()
    except Exception:
        return None

    if not row:
        return None

    return {
        "iso3": row[0],
        "triage_score": row[1],
        "tier": row[2],
        "run_id": row[3],
        "regime_change_likelihood": row[4],
        "regime_change_magnitude": row[5],
        "regime_change_direction": row[6],
        "regime_change_window": row[7],
        "regime_change_level": row[8],
        "regime_change_score": row[9],
        "drivers_json": row[10],
        "regime_shifts_json": row[11],
        "data_quality_json": row[12],
        "scenario_stub": row[13],
        "regime_change_json": row[14],
    }


def _load_resolver_features(con, iso3: str, hazard_code: str) -> Dict[str, Any]:
    """Load resolver features for a country-hazard pair."""
    features: Dict[str, Any] = {}
    try:
        rows = con.execute(
            """
            SELECT source, metric, hazard_code, ym, value
            FROM facts_deltas
            WHERE iso3 = ? AND (hazard_code = ? OR hazard_code = '*')
            ORDER BY ym DESC
            LIMIT 50
            """,
            [iso3, hazard_code],
        ).fetchall()
        entries = []
        for source, metric, hz, ym, value in rows:
            entries.append({
                "source": source, "metric": metric,
                "hazard_code": hz, "ym": str(ym), "value": value,
            })
        if entries:
            features["facts_deltas"] = entries[:20]
    except Exception:
        features["facts_deltas"] = "(unavailable)"

    try:
        rows = con.execute(
            """
            SELECT ym, fatalities
            FROM acled_monthly_fatalities
            WHERE iso3 = ?
            ORDER BY ym DESC
            LIMIT 24
            """,
            [iso3],
        ).fetchall()
        if rows:
            features["acled_monthly_fatalities"] = [
                {"ym": str(r[0]), "fatalities": r[1]} for r in rows[:12]
            ]
    except Exception:
        pass

    try:
        rows = con.execute(
            """
            SELECT shock_type, year, total_affected, total_deaths
            FROM emdat_pa
            WHERE iso3 = ?
            ORDER BY year DESC
            LIMIT 10
            """,
            [iso3],
        ).fetchall()
        if rows:
            features["emdat_pa"] = [
                {"shock_type": r[0], "year": r[1],
                 "total_affected": r[2], "total_deaths": r[3]}
                for r in rows[:5]
            ]
    except Exception:
        pass

    return features


def _load_evidence_pack(con, iso3: str, hazard_code: str, run_id: str) -> Optional[Dict[str, Any]]:
    """Try to load an evidence pack from question_research or hs_country_reports."""
    try:
        row = con.execute(
            """
            SELECT merged_evidence_json
            FROM question_research
            WHERE iso3 = ? AND hazard_code = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [iso3, hazard_code],
        ).fetchone()
        if row and row[0]:
            data = json.loads(row[0])
            if isinstance(data, dict) and data.get("markdown"):
                return data
    except Exception:
        pass

    try:
        row = con.execute(
            """
            SELECT report_markdown
            FROM hs_country_reports
            WHERE iso3 = ? AND hs_run_id = ?
            LIMIT 1
            """,
            [iso3, run_id],
        ).fetchone()
        if row and row[0]:
            return {"markdown": row[0]}
    except Exception:
        pass

    return {"markdown": "(Sample evidence pack text — actual web research results appear here during live runs)"}


def _load_calibration_advice(con, hazard_code: str, metric: str) -> str:
    """Load calibration advice from the DB."""
    try:
        row = con.execute(
            """
            SELECT advice
            FROM calibration_advice
            WHERE hazard_code = ? AND metric = ?
            ORDER BY as_of_month DESC
            LIMIT 1
            """,
            [hazard_code, metric],
        ).fetchone()
        if row and row[0]:
            return str(row[0])

        row = con.execute(
            """
            SELECT advice
            FROM calibration_advice
            ORDER BY as_of_month DESC
            LIMIT 1
            """,
        ).fetchone()
        if row and row[0]:
            return str(row[0])
    except Exception:
        pass
    return "(no calibration advice available)"


def _load_question_for_hazard(con, iso3: str, hazard_code: str) -> Optional[Dict[str, Any]]:
    """Load a sample question for SPD/scenario prompt rendering."""
    try:
        row = con.execute(
            """
            SELECT question_id, hs_run_id, iso3, hazard_code, metric,
                   target_month, window_start_date, window_end_date,
                   wording, track
            FROM questions
            WHERE iso3 = ? AND hazard_code = ? AND status = 'active'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [iso3, hazard_code],
        ).fetchone()
    except Exception:
        return None

    if not row:
        return None

    return {
        "question_id": row[0],
        "hs_run_id": row[1],
        "iso3": row[2],
        "hazard_code": row[3],
        "metric": row[4],
        "target_month": str(row[5]) if row[5] else None,
        "window_start_date": str(row[6]) if row[6] else None,
        "window_end_date": str(row[7]) if row[7] else None,
        "wording": row[8],
        "track": row[9],
    }


def _iso3_to_country_name(iso3: str) -> str:
    """Best-effort ISO3 to country name."""
    try:
        from horizon_scanner.hs_countries import iso3_to_name
        return iso3_to_name(iso3)
    except Exception:
        pass
    return iso3


# ---------------------------------------------------------------------------
# Hazard catalog (same as used in prompts)
# ---------------------------------------------------------------------------

HAZARD_CATALOG = {
    "ACE": "Armed Conflict Events — fatalities and violence from battles, explosions, attacks on civilians",
    "DI": "Displacement Internally — new internal displacements from conflict, violence, or disaster",
    "DR": "Drought — sustained rainfall deficit affecting agriculture, water supply, livelihoods",
    "FL": "Flood — riverine, flash, or coastal flooding causing damage, displacement, deaths",
    "HW": "Heatwave — prolonged extreme heat causing health impacts, crop damage, infrastructure stress",
    "TC": "Tropical Cyclone — hurricanes, typhoons, cyclones causing wind/storm surge/flooding damage",
}

ACTIVE_HAZARDS = ["ACE", "DR", "FL", "HW", "TC"]


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def _render_rc_prompt(hazard_code: str, country_name: str, iso3: str,
                      resolver_features: Dict[str, Any],
                      evidence_pack: Optional[Dict[str, Any]]) -> str:
    """Render the RC prompt for a hazard."""
    try:
        from horizon_scanner.rc_prompts import build_rc_prompt
        return build_rc_prompt(
            hazard_code=hazard_code,
            country_name=country_name,
            iso3=iso3,
            resolver_features=resolver_features,
            evidence_pack=evidence_pack,
        )
    except Exception as e:
        return f"(RC prompt rendering failed: {e})"


def _render_triage_prompt(hazard_code: str, country_name: str, iso3: str,
                          resolver_features: Dict[str, Any],
                          evidence_pack: Optional[Dict[str, Any]],
                          rc_result: Optional[Dict[str, Any]] = None) -> str:
    """Render the triage prompt for a hazard."""
    try:
        from horizon_scanner.hs_triage_prompts import build_triage_prompt
        return build_triage_prompt(
            hazard_code=hazard_code,
            country_name=country_name,
            iso3=iso3,
            resolver_features=resolver_features,
            rc_result=rc_result,
            evidence_pack=evidence_pack,
        )
    except Exception as e:
        return f"(Triage prompt rendering failed: {e})"


def _render_spd_prompt(question: Dict[str, Any],
                       history_summary: Dict[str, Any],
                       hs_triage_entry: Dict[str, Any],
                       research_json: Dict[str, Any]) -> str:
    """Render the SPD forecast prompt."""
    try:
        from forecaster.prompts import build_spd_prompt_v2
        return build_spd_prompt_v2(
            question=question,
            history_summary=history_summary,
            hs_triage_entry=hs_triage_entry,
            research_json=research_json,
        )
    except Exception as e:
        return f"(SPD prompt rendering failed: {e})"


def _render_scenario_prompt(run_id: str, question: Dict[str, Any],
                            ensemble_spd: Dict[str, Any],
                            hs_triage_entry: Dict[str, Any]) -> str:
    """Render the scenario prompt."""
    try:
        from forecaster.prompts import build_scenario_prompt
        return build_scenario_prompt(
            run_id=run_id,
            question=question,
            ensemble_spd=ensemble_spd,
            hs_triage_entry=hs_triage_entry,
        )
    except Exception as e:
        return f"(Scenario prompt rendering failed: {e})"


# ---------------------------------------------------------------------------
# Main artifact builder
# ---------------------------------------------------------------------------

def build_artifact(db_url: str, run_id: str | None = None) -> str:
    """Build the full markdown artifact."""

    con = _connect(db_url)
    lines: list[str] = []

    lines.append("# Pythia LLM Prompt Artifact")
    lines.append("")
    lines.append("This artifact shows the **full rendered prompts** sent to LLMs during the ")
    lines.append("Pythia pipeline. One example per hazard type for each of the four prompt stages.")
    lines.append("Use this to review and improve the prompts.")
    lines.append("")
    lines.append("---")
    lines.append("")

    try:
        for hazard_code in ACTIVE_HAZARDS:
            lines.append(f"# Hazard: {hazard_code} — {HAZARD_CATALOG.get(hazard_code, '')}")
            lines.append("")

            # Find a sample country
            sample = _load_sample_country_for_hazard(con, hazard_code, run_id=run_id)
            if not sample:
                lines.append(f"_No HS triage data found for {hazard_code}; skipping._")
                lines.append("")
                continue

            iso3 = sample["iso3"]
            run_id = sample["run_id"]
            country_name = _iso3_to_country_name(iso3)

            lines.append(f"**Sample country:** {country_name} ({iso3})")
            lines.append(f"**Triage score:** {sample.get('triage_score', 'n/a')} | "
                         f"**Tier:** {sample.get('tier', 'n/a')} | "
                         f"**RC likelihood:** {sample.get('regime_change_likelihood', 'n/a')} | "
                         f"**RC level:** {sample.get('regime_change_level', 'n/a')}")
            lines.append("")

            # Load shared data
            resolver_features = _load_resolver_features(con, iso3, hazard_code)
            evidence_pack = _load_evidence_pack(con, iso3, hazard_code, run_id)

            # Build RC result dict for triage prompt injection
            rc_result = {
                "likelihood": sample.get("regime_change_likelihood") or 0.05,
                "magnitude": sample.get("regime_change_magnitude") or 0.05,
                "direction": sample.get("regime_change_direction") or "unclear",
                "window": sample.get("regime_change_window") or "month_1-2",
                "rationale_bullets": ["(from prior RC assessment)"],
            }

            # ── 1. Regime Change prompt ──
            lines.append("## 1. Regime Change (RC) Prompt")
            lines.append("")
            lines.append("<details>")
            lines.append(f"<summary>Full RC prompt for {hazard_code} — {country_name} ({iso3})</summary>")
            lines.append("")
            lines.append("```")
            rc_prompt = _render_rc_prompt(hazard_code, country_name, iso3,
                                         resolver_features, evidence_pack)
            lines.append(rc_prompt)
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")

            # ── 2. Triage prompt ──
            lines.append("## 2. Triage Prompt")
            lines.append("")
            lines.append("<details>")
            lines.append(f"<summary>Full Triage prompt for {hazard_code} — {country_name} ({iso3})</summary>")
            lines.append("")
            lines.append("```")
            triage_prompt = _render_triage_prompt(hazard_code, country_name, iso3,
                                                 resolver_features, evidence_pack,
                                                 rc_result=rc_result)
            lines.append(triage_prompt)
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")

            # ── 3. SPD Forecast prompt ──
            lines.append("## 3. SPD Forecast Prompt")
            lines.append("")
            question = _load_question_for_hazard(con, iso3, hazard_code)
            if question:
                metric = question.get("metric", "PA")
                cal_advice = _load_calibration_advice(con, hazard_code, metric)
                history_summary = {
                    "source": "resolver",
                    "summary": resolver_features,
                }
                research_json = {
                    "prediction_market_signals": None,
                    "nmme_seasonal_outlook": None,
                }
                hs_triage_entry = dict(sample)

                lines.append("<details>")
                lines.append(f"<summary>Full SPD Forecast prompt for {hazard_code}/{metric} — "
                             f"{country_name} ({iso3})</summary>")
                lines.append("")
                lines.append("```")
                spd_prompt = _render_spd_prompt(question, history_summary,
                                               hs_triage_entry, research_json)
                lines.append(spd_prompt)
                lines.append("```")
                lines.append("")
                lines.append("</details>")
            else:
                lines.append("_No active question found for this hazard-country pair; "
                             "SPD prompt not rendered._")
            lines.append("")

            # ── 4. Scenario prompt ──
            lines.append("## 4. Scenario Prompt")
            lines.append("")
            if question:
                # Build a synthetic ensemble SPD summary
                sample_ensemble = {
                    "bucket_labels": ["<10k", "10k-<50k", "50k-<250k", "250k-<500k", ">=500k"],
                    "per_month": {
                        "1": {"probs": [0.3, 0.35, 0.2, 0.1, 0.05], "bucket_label_max": "10k-<50k", "prob_max": 0.35},
                    },
                    "bucket_max": {"bucket_label": "10k-<50k", "probability": 0.35},
                    "bucket_alt": {"bucket_label": "<10k", "probability": 0.30},
                }

                lines.append("<details>")
                lines.append(f"<summary>Full Scenario prompt for {hazard_code} — "
                             f"{country_name} ({iso3})</summary>")
                lines.append("")
                lines.append("```")
                scenario_question = dict(question)
                scenario_question["forecaster_rationale"] = "(sample forecaster rationale text)"
                scenario_prompt = _render_scenario_prompt(
                    run_id=run_id,
                    question=scenario_question,
                    ensemble_spd=sample_ensemble,
                    hs_triage_entry=hs_triage_entry,
                )
                lines.append(scenario_prompt)
                lines.append("```")
                lines.append("")
                lines.append("</details>")
            else:
                lines.append("_No active question; Scenario prompt not rendered._")
            lines.append("")

            lines.append("---")
            lines.append("")

    finally:
        _close(con)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM prompt snapshot artifact")
    parser.add_argument("--db", default=None, help="DuckDB URL (or uses PYTHIA_DB_URL)")
    parser.add_argument("--out", default="diagnostics/prompt_artifact.md",
                        help="Output markdown file path")
    parser.add_argument("--run-id", default=None, help="HS run ID to use (defaults to latest)")
    args = parser.parse_args()

    db_url = args.db or _get_db_url()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    LOG.info("Generating prompt artifact from %s → %s", db_url, out_path)

    md = build_artifact(db_url, run_id=args.run_id)
    out_path.write_text(md, encoding="utf-8")

    size_kb = out_path.stat().st_size / 1024
    LOG.info("Wrote %s (%.1f KB)", out_path, size_kb)


if __name__ == "__main__":
    main()
