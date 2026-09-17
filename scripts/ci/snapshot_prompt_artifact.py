# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Generate a markdown artifact showing the full rendered prompts seen by LLMs.

For each active hazard type (ACE, DR, FL, TC) this script renders one complete
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


def _ensure_live(con, db_url: str):
    """Return a usable connection, reopening if the shared handle was closed.

    Anything that calls into the forecaster/pythia loaders can close AND evict
    the shared ``duckdb_io`` connection out from under us — see
    :func:`_load_structured_data_for_artifact`. The hazard loop must therefore
    never assume the handle it started an iteration with is still open: on the
    2026-08-01 production run the ACE section rendered fully and then DR, FL
    and TC all reported "No HS triage data found" against a database that had
    122 rows for each of them, because the first hazard's structured-data load
    closed the connection and every later query raised into a bare
    ``except Exception: return None``.
    """
    try:
        con.execute("SELECT 1").fetchone()
        return con
    except Exception as exc:  # noqa: BLE001
        LOG.warning(
            "shared DB connection is no longer usable (%s); reopening", exc
        )
        return _connect(db_url)


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
    except Exception as exc:  # noqa: BLE001
        # Never silent: a dead connection used to be indistinguishable from
        # "this hazard has no triage rows", which hid three of four hazards.
        LOG.warning(
            "sample-country lookup failed for %s (run_id=%s): %s",
            hazard_code, run_id, exc,
        )
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


#: Metric order for the artifact. PA leads because the PA base-rate block
#: the machine generates appears in no other metric's prompt, and this
#: diagnostic exists to show what the model was actually sent.
_METRIC_PRIORITY = ("PA", "FATALITIES", "PHASE3PLUS_IN_NEED", "EVENT_OCCURRENCE")


def _metric_rank(metric: str) -> int:
    try:
        return _METRIC_PRIORITY.index(str(metric or "").upper())
    except ValueError:
        return len(_METRIC_PRIORITY)


def _load_questions_for_hazard(
    con, iso3: str, hazard_code: str
) -> list[Dict[str, Any]]:
    """One sample question per METRIC, newest epoch each, PA first.

    Until Sept 2026 this loaded ONE question ordered by question_id, and
    ``..._EVENT_OCCURRENCE_...`` sorts before ``..._PA_...`` — so the
    artifact rendered a binary prompt for every hazard that had one and the
    PA base-rate block appeared nowhere in the diagnostic, on a run where
    it was in 96 of 97 production prompts. A diagnostic that cannot show
    the thing it is read for is worse than none.
    """
    try:
        # NOTE: the ``questions`` table has no ``created_at`` column — ordering
        # by it raises a DuckDB BinderException that a bare except would swallow,
        # blanking the SPD/Scenario sections for every hazard. Order by the epoch
        # (window_start_date) with question_id as a deterministic tiebreak.
        rows = con.execute(
            """
            SELECT question_id, hs_run_id, iso3, hazard_code, metric,
                   target_month, window_start_date, window_end_date,
                   wording, track
            FROM (
                SELECT q.*, ROW_NUMBER() OVER (
                    PARTITION BY metric
                    ORDER BY window_start_date DESC, question_id
                ) AS rn
                FROM questions AS q
                WHERE iso3 = ? AND hazard_code = ? AND status = 'active'
            )
            WHERE rn = 1
            """,
            [iso3, hazard_code],
        ).fetchall()
    except Exception as exc:
        LOG.warning(
            "snapshot_prompt_artifact: failed to load sample questions for "
            "%s/%s — SPD/Scenario prompts will not render: %s",
            iso3, hazard_code, exc,
        )
        return []

    out = [_question_dict(r) for r in rows]
    out.sort(key=lambda q: (_metric_rank(q.get("metric") or ""), q.get("metric") or ""))
    return out


def _question_dict(row) -> Dict[str, Any]:

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

# HW is in BLOCKED_HAZARDS (db_writer.py): it never enters RC, triage or
# question generation, so it could only ever render a skip line here.
ACTIVE_HAZARDS = ["ACE", "DR", "FL", "TC"]


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def _ace_conflict_kwargs(iso3: str) -> Dict[str, Any]:
    """The ACE-only injects production passes to RC and triage.

    The artifact rendered neither, so an ACE prompt here was missing its
    conflict forecasts, its CrisisWatch note and its ACLED summary while
    production sent all three — and the discrepancy is worse than an
    unrendered section, because a reader compares the artifact against what
    the model saw and concludes the wrong thing. Best effort per inject: a
    dead loader costs that inject and nothing else.
    """
    kwargs: Dict[str, Any] = {}
    try:
        from horizon_scanner.horizon_scanner import _build_acled_summary_for_country
        acled = _build_acled_summary_for_country(iso3)
        if acled:
            kwargs["acled_summary"] = acled
    except Exception as exc:
        LOG.warning("artifact: ACLED summary failed for %s: %s", iso3, exc)
    try:
        from horizon_scanner.conflict_forecasts import load_conflict_forecasts
        forecasts = load_conflict_forecasts(iso3)
        if forecasts:
            kwargs["conflict_forecasts"] = forecasts
    except Exception as exc:
        LOG.warning("artifact: conflict forecasts failed for %s: %s", iso3, exc)
    try:
        from horizon_scanner.crisiswatch import format_crisiswatch_for_prompt
        cw = format_crisiswatch_for_prompt(iso3)
        if cw:
            kwargs["crisiswatch_context"] = cw
            kwargs["icg_on_the_horizon"] = cw
    except Exception as exc:
        LOG.warning("artifact: CrisisWatch failed for %s: %s", iso3, exc)
    return kwargs


def _render_rc_prompt(hazard_code: str, country_name: str, iso3: str,
                      resolver_features: Dict[str, Any],
                      evidence_pack: Optional[Dict[str, Any]],
                      extra_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Render the RC prompt for a hazard."""
    try:
        from horizon_scanner.rc_prompts import build_rc_prompt
        return build_rc_prompt(
            hazard_code=hazard_code,
            country_name=country_name,
            iso3=iso3,
            resolver_features=resolver_features,
            evidence_pack=evidence_pack,
            **(extra_kwargs or {}),
        )
    except Exception as e:
        return f"(RC prompt rendering failed: {e})"


def _render_triage_prompt(hazard_code: str, country_name: str, iso3: str,
                          resolver_features: Dict[str, Any],
                          evidence_pack: Optional[Dict[str, Any]],
                          rc_result: Optional[Dict[str, Any]] = None,
                          extra_kwargs: Optional[Dict[str, Any]] = None) -> str:
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
            **(extra_kwargs or {}),
        )
    except Exception as e:
        return f"(Triage prompt rendering failed: {e})"


def _load_structured_data_for_artifact(
    iso3: str,
    hazard_code: str,
    run_id: str | None,
    rc_level: int | None,
) -> Optional[Dict[str, Any]]:
    """Load the SPD structured-data injects exactly as the pipeline does.

    Reuses the forecaster's ``_load_structured_data`` so the rendered artifact
    contains the same conflict-forecast / adversarial / HS-grounding / GDACS /
    CrisisWatch / food-security context the live SPD prompt receives.

    IMPORTANT: several of these sub-loaders (``pythia/food_security.py``,
    ``pythia/acaps.py``) call ``resolver.db.duckdb_io.get_db()`` — a cache HIT
    returns the artifact's own shared connection — and then ``close_db()`` it,
    which closes AND evicts that shared handle. The caller must therefore
    reopen its connection after calling this function (``get_db`` returns a
    fresh connection once the cached one is closed/evicted). Returns None on
    failure so the SPD prompt still renders (without injects) rather than being
    blanked.
    """
    try:
        from forecaster.cli import _load_structured_data
        return _load_structured_data(iso3, hazard_code, run_id, rc_level)
    except Exception as exc:
        LOG.warning(
            "snapshot_prompt_artifact: structured-data load failed for %s/%s "
            "— SPD prompt will render without injects: %s",
            iso3, hazard_code, exc,
        )
        return None


def _render_spd_prompt(question: Dict[str, Any],
                       history_summary: Dict[str, Any],
                       hs_triage_entry: Dict[str, Any],
                       research_json: Dict[str, Any],
                       structured_data: Optional[Dict[str, Any]] = None) -> str:
    """Render the SPD forecast prompt with the full structured-data injects."""
    try:
        from forecaster.prompts import build_spd_prompt_v2
        return build_spd_prompt_v2(
            question=question,
            history_summary=history_summary,
            hs_triage_entry=hs_triage_entry,
            research_json=research_json,
            structured_data=structured_data,
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

            # The previous iteration called into _load_structured_data, whose
            # sub-loaders close+evict the shared connection. Re-acquire before
            # the first query of every hazard rather than trusting that the
            # single reopen further down was the only one needed.
            con = _ensure_live(con, db_url)

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
            # ACE carries injects no other hazard does; the artifact must
            # send what production sends or it describes a different prompt.
            ace_kwargs = (
                _ace_conflict_kwargs(iso3) if hazard_code == "ACE" else {}
            )
            con = _ensure_live(con, db_url)  # the loaders above may close it
            rc_prompt = _render_rc_prompt(hazard_code, country_name, iso3,
                                         resolver_features, evidence_pack,
                                         extra_kwargs=ace_kwargs)
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
                                                 rc_result=rc_result,
                                                 extra_kwargs=ace_kwargs)
            lines.append(triage_prompt)
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")

            # ── 3. SPD Forecast prompt ──
            lines.append("## 3. SPD Forecast Prompt")
            lines.append("")
            # One question per METRIC, PA first. A single question ordered
            # by question_id always chose EVENT_OCCURRENCE, so the PA
            # base-rate block never appeared in this artifact.
            questions = _load_questions_for_hazard(con, iso3, hazard_code)
            question = questions[0] if questions else None
            hs_triage_entry = dict(sample)
            if questions:
                history_summary = {
                    "source": "resolver",
                    "summary": resolver_features,
                }
                research_json = {
                    "prediction_market_signals": None,
                    "nmme_seasonal_outlook": None,
                }

                # Load the full structured-data injects the pipeline feeds the
                # SPD prompt (conflict forecasts, adversarial checks, HS
                # grounding evidence, GDACS, CrisisWatch, food security, …) so
                # the artifact shows exactly what the LLM sees. Its sub-loaders
                # close+evict the shared duckdb_io connection (see
                # _load_structured_data_for_artifact), so reopen ``con``
                # immediately afterwards — this MUST run after all con-based
                # reads for this hazard (sample/features/evidence/question/
                # calibration) and before the scenario block / next iteration.
                # It does not vary by metric, so it is loaded once.
                try:
                    rc_level_raw = sample.get("regime_change_level")
                    rc_level = int(rc_level_raw) if rc_level_raw is not None else None
                except (TypeError, ValueError):
                    rc_level = None
                structured_data = _load_structured_data_for_artifact(
                    iso3, hazard_code, run_id, rc_level
                )
                con = _ensure_live(con, db_url)  # loaders closed the shared con

                metrics_rendered = ", ".join(
                    str(q.get("metric") or "?") for q in questions
                )
                lines.append(f"_Metrics rendered: {metrics_rendered}._")
                lines.append("")
                for q in questions:
                    metric = q.get("metric", "PA")
                    lines.append("<details>")
                    lines.append(
                        f"<summary>Full SPD Forecast prompt for "
                        f"{hazard_code}/{metric} — {country_name} ({iso3})</summary>"
                    )
                    lines.append("")
                    lines.append("```")
                    spd_prompt = _render_spd_prompt(
                        q, history_summary, hs_triage_entry, research_json,
                        structured_data=structured_data,
                    )
                    lines.append(spd_prompt)
                    lines.append("```")
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")
            else:
                lines.append("_No active question found for this hazard-country pair; "
                             "SPD prompt not rendered._")
            lines.append("")

            # ── 4. Scenario prompt ──
            lines.append("## 4. Scenario Prompt")
            lines.append("")
            if question:
                # Build a synthetic ensemble SPD summary from canonical specs
                from pythia.buckets import labels_for as _labels_for

                _pa_labels = _labels_for("PA")
                _pa_probs = [0.1, 0.2, 0.35, 0.2, 0.1, 0.05][: len(_pa_labels)]
                sample_ensemble = {
                    "bucket_labels": _pa_labels,
                    "per_month": {
                        "1": {
                            "probs": _pa_probs,
                            "bucket_label_max": _pa_labels[2],
                            "prob_max": max(_pa_probs),
                        },
                    },
                    "bucket_max": {"bucket_label": _pa_labels[2], "probability": max(_pa_probs)},
                    "bucket_alt": {"bucket_label": _pa_labels[3], "probability": 0.20},
                }

                lines.append(
                    f"_Rendered for the {question.get('metric')} question; the "
                    "scenario prompt does not vary by metric._"
                )
                lines.append("")
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
