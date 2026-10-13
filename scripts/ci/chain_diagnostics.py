# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""One diagnostics directory per run of a workflow in the post-ingest chain.

On 11 October 2026 the resolutions chain was refused at its canonical upload,
and Scores, Calibration and Publish never ran. Working out what had happened
needed the Actions API and job logs, because no artifact said so: Compute
Resolutions and Compute SPD Scores uploaded nothing but the DB, and the
outcome table compute_resolutions builds lived only in a step summary. A
diagnostic that exists only in a job log does not exist.

So every workflow in the chain calls this at the end of its job (``if:
always()``, ``continue-on-error``) and uploads the directory it writes as
``chain-diagnostics-<slug>-<run_id>``. ``scripts/ci/build_cycle_diagnostics.py``
then gathers every run's directory into one zip on the publish run.

Two subcommands::

    python -m scripts.ci.chain_diagnostics before --workflow compute_scores \\
        --db data/resolver.duckdb
    python -m scripts.ci.chain_diagnostics write --workflow compute_scores \\
        --db data/resolver.duckdb --started-at "${STAGE_STARTED_AT}"

``write`` produces ``diagnostics/<slug>/``:

``run.json``         who ran, when, what triggered it, which DB it downloaded
``lineage.json``     this run, its trigger, the DB downloaded and uploaded;
                     the publish run walks the chain backwards through it
``table_counts.csv`` before and after for every table
``freshness.csv``    per table and per source (the resolver bundle's logic)
``step_timings.csv`` from the jobs API
``issues.json/.md``  a short register, worst first, in the Resolver Update format
``summary.md``       one page a person reads first (also the step summary)
``MISSING.json``     every section that could not be built, and why

plus a section of its own for each workflow. Every section is guarded: a
failure writes a line in ``MISSING.json`` and the next section runs. The
command always exits 0. The run's own inputs arrive as ``CHAIN_*``
environment variables, never as expressions pasted into the script.

Stdlib and duckdb, with the resolver bundle's freshness, reconciliation and
step-timing code imported lazily (reused, not copied); nothing here imports
the forecaster.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import convenience
    sys.path.insert(0, str(REPO_ROOT))

#: Workflow display name (``GITHUB_WORKFLOW``) -> slug -> workflow file.
#: build_cycle_diagnostics reads the same table, so the two cannot disagree
#: about what a workflow is called.
WORKFLOWS: dict[str, tuple[str, str]] = {
    "Resolver Update": ("resolver_update", "resolver_update.yml"),
    "Pythia — Compute Resolutions": ("compute_resolutions", "compute_resolutions.yml"),
    "Pythia — Compute SPD Scores": ("compute_scores", "compute_scores.yml"),
    "Pythia — Compute Calibration Weights & Advice": (
        "compute_calibration", "compute_calibration_pythia.yml",
    ),
    "Ingest Structured Data": ("ingest_structured_data", "ingest-structured-data.yml"),
    "Refresh CrisisWatch Data": ("refresh_crisiswatch", "refresh-crisiswatch.yml"),
    "Publish Latest Data (Release)": ("publish_latest_data", "publish_latest_data.yml"),
    "Sibyl Deep Research": ("sibyl", "run_sibyl.yml"),
    "Pythia Pipeline Stage": ("pipeline_stage", "pythia_pipeline_stage.yml"),
    "Horizon Scanner Triage": ("horizon_scanner", "run_horizon_scanner.yml"),
}
SLUG_TO_NAME = {slug: name for name, (slug, _file) in WORKFLOWS.items()}
ARTIFACT_PREFIX = "chain-diagnostics"


def artifact_name(slug: str, run_id: str) -> str:
    return f"{ARTIFACT_PREFIX}-{slug}-{run_id}"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _db_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    text = str(raw)
    if text.startswith("duckdb:///"):
        text = text[len("duckdb:///"):]
    return Path(text)


def _chain_env(env: Mapping[str, str]) -> dict[str, str]:
    """Every ``CHAIN_*`` variable the step was given, prefix dropped, lowercased."""
    return {k[len("CHAIN_"):].lower(): v for k, v in env.items() if k.startswith("CHAIN_")}


def _outcome_verdict(outcome: str) -> str:
    """A step's ``outcome`` turned into a word a reader can use."""
    return {
        "success": "ok", "failure": "refused", "skipped": "not run",
        "cancelled": "cancelled", "": "unknown",
    }.get((outcome or "").strip(), outcome)


class Writer:
    """Collects sections into one directory. Nothing it does raises."""

    def __init__(self, *, slug: str, out_dir: Path, db_path: Path | None,
                 diagnostics_dir: Path, started_at: str,
                 env: Mapping[str, str] | None = None,
                 connect: Callable[[Path], Any] | None = None) -> None:
        self.slug = slug
        self.out = out_dir
        self.db_path = db_path
        self.diagnostics_dir = diagnostics_dir
        self.started_at = (started_at or "").strip()
        self.env = dict(os.environ if env is None else env)
        self.chain = _chain_env(self.env)
        self.missing: dict[str, str] = {}
        self.issues: list[dict[str, Any]] = []
        self.summary: list[str] = []
        self._connect = connect
        self._con = None
        self._con_tried = False
        self._builder = None
        self._builder_staging: Path | None = None

    # -- infrastructure ------------------------------------------------------

    def section(self, name: str, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a section never fails the step
            self.missing[name] = f"{type(exc).__name__}: {exc}"
            sys.stderr.write(f"[chain_diagnostics] section {name} failed:\n{traceback.format_exc()}")

    def issue(self, issue_id: str, severity: str, title: str, *, evidence: str = "",
              cost: float | None = None, cost_unit: str = "", owner: str = "pythia",
              recovers_on_rerun: bool = True) -> None:
        self.issues.append({
            "id": issue_id, "severity": severity, "title": title, "evidence": evidence,
            "cost": cost, "cost_unit": cost_unit, "owner": owner,
            "recovers_on_rerun": recovers_on_rerun,
            "source": f"chain_diagnostics:{self.slug}",
        })

    @property
    def con(self):
        if self._con is not None or self._con_tried:
            return self._con
        self._con_tried = True
        if self.db_path is None or not self.db_path.is_file():
            self.missing.setdefault(
                "database", f"no readable database at {self.db_path or '(none given)'}"
            )
            return None
        try:
            if self._connect is not None:
                self._con = self._connect(self.db_path)
            else:
                try:
                    from resolver.db import duckdb_io  # shares the bundle builder's cache

                    self._con = duckdb_io.get_db(str(self.db_path))
                except Exception:  # noqa: BLE001 - no pandas on this runner
                    import duckdb

                    self._con = duckdb.connect(str(self.db_path))
        except Exception as exc:  # noqa: BLE001
            self.missing["database"] = f"could not open {self.db_path}: {exc}"
            self._con = None
        return self._con

    def q(self, sql: str, params: list | None = None) -> list[tuple]:
        con = self.con
        if con is None:
            return []
        return con.execute(sql, params or []).fetchall()

    def tables(self) -> set[str]:
        try:
            return {str(r[0]) for r in self.q(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            )}
        except Exception:  # noqa: BLE001
            return set()

    def columns(self, table: str) -> set[str]:
        try:
            return {str(r[1]).lower() for r in self.q(f'PRAGMA table_info("{table}")')}
        except Exception:  # noqa: BLE001
            return set()

    def since_clause(self, table: str, column: str) -> tuple[str, list]:
        """``WHERE column >= started_at`` when both exist, else everything (and says so)."""
        if self.started_at and column in self.columns(table):
            return f' WHERE TRY_CAST("{column}" AS TIMESTAMP) >= TRY_CAST(? AS TIMESTAMP)', [self.started_at]
        return "", []

    def builder(self):
        """The resolver bundle's BundleBuilder, for its freshness, reconciliation
        and step-timing code. Imported lazily and reused rather than copied."""
        if self._builder is not None:
            return self._builder
        from scripts.build_resolver_debug_bundle import BundleBuilder

        self._builder_staging = Path(tempfile.mkdtemp(prefix="chain_diag_"))
        env = dict(self.env)
        if self.started_at:
            env.setdefault("PYTHIA_RUN_STARTED_AT", self.started_at)
        self._builder = BundleBuilder(
            out_path=self._builder_staging / "unused.zip",
            db_path=self.db_path,
            diagnostics_dir=self.diagnostics_dir,
            run_log_dir=None,
            staging=self._builder_staging,
            environ=env,
        )
        return self._builder

    def close(self) -> None:
        if self._builder_staging is not None:
            shutil.rmtree(self._builder_staging, ignore_errors=True)
        try:
            from resolver.db import duckdb_io

            if self._con is not None:
                duckdb_io.close_db(self._con)
        except Exception:  # noqa: BLE001
            try:
                if self._con is not None:
                    self._con.close()
            except Exception:  # noqa: BLE001
                pass

    # -- common sections -----------------------------------------------------

    def run_json(self) -> None:
        env = self.env
        name = env.get("GITHUB_WORKFLOW") or SLUG_TO_NAME.get(self.slug, self.slug)
        payload = {
            "workflow": name,
            "slug": self.slug,
            "run_id": env.get("GITHUB_RUN_ID", ""),
            "run_attempt": env.get("GITHUB_RUN_ATTEMPT", ""),
            "event": env.get("GITHUB_EVENT_NAME", ""),
            "actor": env.get("GITHUB_ACTOR", ""),
            "repository": env.get("GITHUB_REPOSITORY", ""),
            "commit_sha": env.get("GITHUB_SHA", ""),
            "ref": env.get("GITHUB_REF", ""),
            "trigger_run_id": self.chain.get("trigger_run_id", "") or None,
            "canonical_db_downloaded": {
                "run_id": self.chain.get("downloaded_run_id", "") or None,
                "source": self.chain.get("download_source", "") or None,
            },
            "trigger_did_work": self.chain.get("did_work", "") or None,
            "job_status": self.chain.get("job_status", "") or None,
            "started_at": self.started_at or None,
            "ended_at": _now(),
            "inputs": {k: v for k, v in sorted(self.chain.items())},
        }
        write_json(self.out / "run.json", payload)

    def lineage_json(self) -> None:
        run_id = self.env.get("GITHUB_RUN_ID", "")
        uploaded = (self.chain.get("upload_outcome", "") or "").strip() == "success"
        payload = {
            "workflow": self.env.get("GITHUB_WORKFLOW") or SLUG_TO_NAME.get(self.slug, self.slug),
            "slug": self.slug,
            "this_run": run_id or None,
            "trigger_run": self.chain.get("trigger_run_id", "") or None,
            "canonical_db_run_downloaded": self.chain.get("downloaded_run_id", "") or None,
            "canonical_db_download_source": self.chain.get("download_source", "") or None,
            "canonical_db_run_uploaded": (run_id or None) if uploaded else None,
            "upload_guard_verdict": _outcome_verdict(self.chain.get("guard_outcome", "")),
            "recorded_at": _now(),
        }
        write_json(self.out / "lineage.json", payload)
        if (self.chain.get("guard_outcome", "") or "") == "failure":
            self.issue(
                "canonical_upload_refused", "blocking",
                "The upload guard refused this run's canonical DB: another run uploaded "
                "a newer one after the DB this run downloaded, so nothing it wrote was kept",
                evidence=f"downloaded run {payload['canonical_db_run_downloaded']}",
                recovers_on_rerun=True,
            )
        did = (self.chain.get("did_work", "") or "").lower()
        if did == "false":
            self.issue("trigger_did_no_work", "info",
                       "The triggering run uploaded no canonical DB; this run did nothing")

    def before(self) -> None:
        sig = signature(self.db_path)
        write_json(self.out / "db_signature_before.json", sig)

    def table_counts(self) -> None:
        before = read_json(self.out / "db_signature_before.json") or {}
        before_counts = before.get("all_counts") or {}
        if self.db_path is None or not self.db_path.is_file():
            raise RuntimeError("no database: no table counts")
        after = signature(self.db_path, con=self.con).get("all_counts") or {}
        names = sorted(set(before_counts) | set(after))
        rows = []
        for t in names:
            b, a = before_counts.get(t), after.get(t)
            delta = (a - b) if isinstance(a, int) and isinstance(b, int) else ""
            rows.append([t, "" if b is None else b, "" if a is None else a, delta])
        write_csv(self.out / "table_counts.csv", ["table", "before", "after", "delta"], rows)
        if not before_counts:
            self.missing["table_counts.before"] = (
                "no before signature (run `chain_diagnostics before` at job start); "
                "after counts only"
            )
        lost = [r for r in rows if isinstance(r[3], int) and r[3] < 0]
        if lost:
            self.issue("tables_lost_rows", "info",
                       f"{len(lost)} table(s) hold fewer rows than when the job started",
                       evidence=", ".join(f"{r[0]} {r[3]}" for r in lost[:10]),
                       cost=sum(-r[3] for r in lost), cost_unit="rows")
        self.summary.append(
            f"- Tables: {len(rows)}; {sum(1 for r in rows if isinstance(r[3], int) and r[3] > 0)} "
            f"gained rows, {len(lost)} lost rows."
        )

    def freshness(self) -> None:
        if self.con is None:
            raise RuntimeError("no database: no freshness")
        b = self.builder()
        b._freshness(self.out)  # noqa: SLF001 - the resolver bundle's own logic, reused
        stale = []
        with (self.out / "freshness.csv").open(encoding="utf-8") as fh:
            lines = [ln for ln in fh if not ln.startswith("#")]
        for rec in csv.DictReader(lines):
            if rec.get("verdict") == "stale" and rec.get("scope") == "table":
                stale.append(f"{rec.get('table')}.{rec.get('date_column')} ({rec.get('age_days')}d)")
        if stale:
            self.issue("stale_tables", "info", f"{len(stale)} table column(s) past their staleness limit",
                       evidence="; ".join(stale[:10]), cost=len(stale), cost_unit="columns")

    def step_timings(self) -> None:
        b = self.builder()
        payload, source = b._step_timings_from_api()  # noqa: SLF001
        rows = []
        if payload is not None:
            for st in payload.get("steps") or []:
                rows.append([st.get("number"), st.get("name"), st.get("status"),
                             st.get("conclusion"), st.get("started_at"),
                             st.get("completed_at"), st.get("duration_sec")])
            write_json(self.out / "step_timings.json", payload)
        else:
            self.missing["step_timings"] = source
        write_csv(self.out / "step_timings.csv",
                  ["number", "step", "status", "conclusion", "started_at", "completed_at",
                   "duration_sec"], rows)
        failed = [r[1] for r in rows if r[3] == "failure"]
        if failed:
            self.issue("steps_failed", "degraded", f"{len(failed)} step(s) failed in this job",
                       evidence="; ".join(str(f) for f in failed), cost=len(failed), cost_unit="steps")

    def include(self, paths: list[str]) -> None:
        dest = self.out / "included"
        for raw in paths:
            src = Path(raw)
            if src.is_dir():
                shutil.copytree(src, dest / src.name, dirs_exist_ok=True)
            elif src.is_file():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / src.name)
            else:
                self.missing[f"include:{raw}"] = "not present"

    def reports(self) -> None:
        """Copy the tools' own run reports (``--report-json``) into ``reports/``."""
        src = self.out / "reports"
        if not src.is_dir():
            return
        names = sorted(p.name for p in src.glob("*.json"))
        if names:
            self.summary.append(f"- Tool reports: {', '.join(names)}.")

    # -- per-workflow sections -------------------------------------------------

    def compute_resolutions(self) -> None:
        rep = read_json(self.out / "reports" / "compute_resolutions.json")
        if not rep:
            raise RuntimeError("reports/compute_resolutions.json absent (compute_resolutions "
                               "did not run, or ran without --report-json)")
        outcomes = rep.get("outcome_counts") or {}
        rows = []
        for group, counts in sorted(outcomes.items()):
            lag = int(counts.get("unresolved_for_lag", 0))
            other = sum(int(v) for k, v in counts.items()
                        if k.startswith("unresolved") and k != "unresolved_for_lag")
            rows.append([group, int(counts.get("sourced", 0)), int(counts.get("zero_default", 0)),
                         lag, other])
        write_csv(self.out / "resolution_outcomes.csv",
                  ["hazard_metric", "sourced", "zero_default", "unresolved_lag", "unresolved_other"], rows)
        md = ["# Resolution outcomes", "",
              "| hazard/metric | sourced | zero-default | unresolved (lag) | unresolved (other) |",
              "|---|---:|---:|---:|---:|"]
        md += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in rows]
        (self.out / "resolution_outcomes.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        write_csv(self.out / "resolutions_by_source_desc.csv", ["source_desc", "rows"],
                  [[k, v] for k, v in (rep.get("by_source_desc") or {}).items()])
        before, after = rep.get("resolutions_before"), rep.get("resolutions_after")
        deleted = (int(rep.get("purged_stale_beyond_cutoff") or 0)
                   + int(rep.get("purged_non_series_fatalities") or 0)
                   + int(rep.get("machine_retracted") or 0))
        self.summary.append(
            f"- Resolutions: {rep.get('rows_written', 0)} rows written "
            f"({rep.get('resolved_from_source', 0)} sourced, {rep.get('resolved_as_zero', 0)} "
            f"zero-default); table {before} -> {after}; {deleted} withdrawn or purged; "
            f"{rep.get('vintages_written', 0)} vintages recorded."
        )
        for group in rep.get("mostly_zero_default_groups") or []:
            self.issue(f"resolutions_mostly_zero_default:{group}", "degraded",
                       f"{group}: more than half of this run's resolutions are zero-defaults",
                       evidence="resolution_outcomes.csv")
        if not int(rep.get("rows_written") or 0):
            self.issue("no_resolutions_written", "degraded",
                       "compute_resolutions wrote no resolution rows",
                       evidence=f"{rep.get('questions_considered', 0)} questions considered")

    def compute_scores(self) -> None:
        reports = self.out / "reports"
        rep = read_json(reports / "compute_scores.json") or {}
        views = read_json(reports / "score_views.json") or {}
        base = read_json(reports / "score_baselines.json") or {}
        if not rep and not views and not base:
            self.missing["score_reports"] = "no compute_scores/score_views/score_baselines report"
        rows: list[list[Any]] = []
        if "scores" in self.tables():
            where, params = self.since_clause("scores", "created_at")
            rows = [list(r) for r in self.q(
                "SELECT model_name, score_type, "
                "CASE WHEN upper(metric) = 'EVENT_OCCURRENCE' THEN 'binary' ELSE 'spd' END, "
                f"COUNT(*) FROM scores{where} GROUP BY 1, 2, 3 ORDER BY 1, 2, 3", params)]
            if not where:
                self.missing["scores_written.window"] = "no start time: counts are the whole table"
        write_csv(self.out / "scores_written.csv",
                  ["model_name", "score_type", "score_family", "rows"], rows)
        orphans = rep.get("orphans_purged") or {}
        self.summary.append(
            f"- Scores: {rep.get('score_rows_written', 0)} score rows and "
            f"{rep.get('eiv_rows_written', 0)} EIV rows written; {sum(int(v or 0) for v in orphans.values())} orphan rows purged; "
            f"ViEWS {views.get('views_rows_filtered')} rows filtered, {views.get('matched_pairs', 0)} "
            f"matched, {views.get('scored', 0)} scored; baselines {base or '—'}."
        )
        if views and views.get("views_rows_filtered") == 0:
            self.issue("views_rows_filtered_zero", "degraded",
                       "No ViEWS rows passed score_views' source/metric filter: a literal no "
                       "longer matches the connector", evidence="reports/score_views.json")
        if any(int(v or 0) for v in orphans.values()):
            self.issue("orphan_scores_purged", "info",
                       "Scores whose resolution was withdrawn were deleted",
                       cost=sum(int(v or 0) for v in orphans.values()), cost_unit="rows")

    def compute_calibration(self) -> None:
        tables = self.tables()
        out = self.out
        # Weights per (hazard, metric), against the floor.
        try:
            from scripts.ai_bundle.provenance import calibration_status

            status = calibration_status(self.con) if self.con is not None else []
        except Exception as exc:  # noqa: BLE001
            status = []
            self.missing["calibration_status"] = f"{type(exc).__name__}: {exc}"
        written: dict[tuple[str, str], int] = {}
        if "calibration_weights" in tables:
            where, params = self.since_clause("calibration_weights", "created_at")
            for h, m, n in self.q("SELECT upper(hazard_code), upper(metric), COUNT(*) FROM "
                                  f"calibration_weights{where} GROUP BY 1, 2", params):
                written[(h, m)] = int(n)
        rows = [[s["hazard_code"], s["metric"], s["n_questions_with_member_scores"], s["floor"],
                 written.get((s["hazard_code"], s["metric"]), 0), s["status"]] for s in status]
        write_csv(out / "calibration_weights.csv",
                  ["hazard_code", "metric", "questions_with_member_scores", "floor",
                   "weight_rows_written_this_run", "status"], rows)
        # Family recalibration: factors and how many sit at a clip limit.
        fam_rows: list[list[Any]] = []
        if "family_recalibration" in tables:
            where, params = self.since_clause("family_recalibration", "fitted_at")
            fam_rows = [list(r) for r in self.q(
                "SELECT family, hazard_code, metric, score_family, "
                "COALESCE(base_rate_block_version, ''), COALESCE(rc_guidance, ''), "
                "COUNT(*), SUM(CASE WHEN score_family = 'spd' AND (factor <= 0.5 OR factor >= 2.0) "
                "THEN 1 ELSE 0 END), MAX(n_questions) FROM family_recalibration"
                f"{where} GROUP BY 1, 2, 3, 4, 5, 6 ORDER BY 1, 2, 3", params)]
            for r in fam_rows:
                n, at_clip = int(r[6] or 0), int(r[7] or 0)
                share = at_clip / n if n else 0.0
                r.append(round(share, 3))
                r.append("auto_shadow: factor set at a clip limit" if r[3] == "spd" and share >= 0.5
                         else "")
        write_csv(out / "family_recalibration.csv",
                  ["family", "hazard_code", "metric", "score_family", "block_version", "rc_guidance",
                   "factors", "at_clip_limit", "n_questions", "clip_share", "shadow_reason"], fam_rows)
        # Advice written this run, with its length.
        adv_rows: list[list[Any]] = []
        if "calibration_advice" in tables:
            where, params = self.since_clause("calibration_advice", "created_at")
            adv_rows = [list(r) for r in self.q(
                "SELECT as_of_month, hazard_code, metric, model_name, LENGTH(COALESCE(advice, '')) "
                f"FROM calibration_advice{where} ORDER BY 2, 3, 4", params)]
        write_csv(out / "calibration_advice.csv",
                  ["as_of_month", "hazard_code", "metric", "model_name", "advice_chars"], adv_rows)
        # Centroid EMA moves (reports/ema_*.json written by compute_bucket_centroids).
        ema: list[list[Any]] = []
        for path in sorted((out / "reports").glob("ema_*.json")):
            for u in (read_json(path) or {}).get("updates") or []:
                ema.append([u.get("hazard_code"), u.get("metric"), u.get("bucket_index"),
                            u.get("old"), u.get("new"), u.get("empirical"), u.get("n_resolutions")])
        write_csv(out / "centroid_ema_updates.csv",
                  ["hazard_code", "metric", "bucket_index", "old", "new", "empirical", "n_resolutions"], ema)
        # Sibyl's calibration-time steps.
        sib: dict[str, Any] = {}
        for table, col in (("sibyl_variant_scores", "created_at"),
                           ("sibyl_calibration_advice", "created_at"),
                           ("sibyl_postmortem_notes", "created_at"),
                           ("sibyl_lessons", "created_at"),
                           ("sibyl_pool_weights", "created_at")):
            if table in tables:
                where, params = self.since_clause(table, col)
                sib[f"{table}_rows_this_run"] = int(self.q(f"SELECT COUNT(*) FROM {table}{where}", params)[0][0])
        if "sibyl_pool_weights" in tables:
            r = self.q("SELECT as_of_month, weight, status, n_questions FROM sibyl_pool_weights "
                       "ORDER BY as_of_month DESC LIMIT 1")
            sib["newest_pool_weight"] = list(r[0]) if r else None
        if "llm_calls" in tables:
            ts_col = "timestamp" if "timestamp" in self.columns("llm_calls") else "created_at"
            where, params = self.since_clause("llm_calls", ts_col)
            clause = (where + " AND" if where else " WHERE") + " phase = 'sibyl'"
            r = self.q(f"SELECT COUNT(*), COALESCE(SUM(cost_usd), 0) FROM llm_calls{clause}", params)
            sib["sibyl_llm_calls_this_run"] = int(r[0][0]) if r else 0
            sib["sibyl_spend_usd_this_run"] = round(float(r[0][1] or 0), 4) if r else 0.0
        write_json(out / "sibyl_calibration_steps.json", sib)
        dispatch = _outcome_verdict(self.chain.get("publish_dispatch_outcome", ""))
        write_json(out / "publish_dispatch.json", {"outcome": dispatch})
        weighted = sum(1 for s in status if s.get("has_weights"))
        self.summary.append(
            f"- Calibration: {weighted} of {len(status)} hazard/metric groups weighted; "
            f"{sum(written.values())} weight rows, {len(adv_rows)} advice rows, {len(fam_rows)} "
            f"recalibration groups, {len(ema)} centroids moved this run; publish dispatch {dispatch}."
        )
        if dispatch == "refused":
            self.issue("publish_dispatch_failed", "blocking",
                       "Calibration could not dispatch Publish: this cycle's scores are not released")

    def crisiswatch(self) -> None:
        cw = crisiswatch_state(REPO_ROOT, self.diagnostics_dir, env=self.env)
        if self.db_path is not None and self.con is not None and "crisiswatch_entries" in self.tables():
            where, params = self.since_clause("crisiswatch_entries", "fetched_at")
            cw["db_rows_written_this_run"] = int(self.q(
                f"SELECT COUNT(*) FROM crisiswatch_entries{where}", params)[0][0])
            editions = self.q("SELECT DISTINCT year, month FROM crisiswatch_entries ORDER BY 1, 2")
            cw["db_editions_held"] = [f"{int(y)}-{int(m):02d}" for y, m in editions]
            cw["db_newest_edition"] = cw["db_editions_held"][-1] if editions else None
            want = cw.get("expected_edition")
            if want:
                cw["gate_would_pass_on_db"] = want in cw["db_editions_held"]
        write_json(self.out / "crisiswatch.json", cw)
        self.summary.append(
            f"- CrisisWatch: edition {cw.get('edition_before')} -> {cw.get('edition_after')}; "
            f"expected {cw.get('expected_edition')}; gate would pass: "
            f"{cw.get('gate_would_pass_on_db', cw.get('gate_would_pass'))}."
        )
        passes = cw.get("gate_would_pass_on_db", cw.get("gate_would_pass"))
        if passes is False:
            self.issue("crisiswatch_expected_edition_missing", "degraded",
                       f"The expected CrisisWatch edition {cw.get('expected_edition')} is not held",
                       evidence="crisiswatch.json", owner="external")
        unresolved = (cw.get("parse_accounting") or {}).get("unresolved_names") or []
        if unresolved:
            self.issue("crisiswatch_unmapped_headings", "degraded",
                       f"{len(unresolved)} CrisisWatch heading(s) resolved to no country",
                       evidence=", ".join(unresolved[:10]), cost=len(unresolved), cost_unit="headings")

    def ingest(self) -> None:
        report = self.diagnostics_dir / "ingestion" / "connectors_report.jsonl"
        for src in (report, self.diagnostics_dir / "ingestion" / "duckdb_summary.md",
                    self.diagnostics_dir / "acled_cast_diagnostic.json"):
            if src.is_file():
                shutil.copy2(src, self.out / src.name)
        if not report.is_file():
            self.missing["connectors_report"] = f"{report} absent"
        if self.con is not None:
            b = self.builder()
            b._write_reconciliation(self.out / "reconciliation.md")  # noqa: SLF001
            bad = [r for r in b.reconciliation_rows if str(r.get("agrees", "")).startswith("**NO")]
            if bad:
                self.issue("connector_claims_unproven", "degraded",
                           f"{len(bad)} connector(s) claimed rows and stamped none of their own",
                           evidence=", ".join(r["connector"] for r in bad), cost=len(bad),
                           cost_unit="connectors")
        gate = {k: v for k, v in self.chain.items() if k.startswith("gate")}
        write_json(self.out / "pipeline_gate.json", gate)
        failed = []
        if report.is_file():
            for line in report.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if str(rec.get("status", "")).lower() in {"error", "failed", "failure"}:
                    failed.append(str(rec.get("connector_id") or rec.get("connector") or "?"))
        if failed:
            self.issue("connectors_failed", "degraded", f"{len(failed)} connector(s) failed",
                       evidence=", ".join(failed), cost=len(failed), cost_unit="connectors")
        self.summary.append(f"- Ingest: pipeline gate {gate or '—'}; {len(failed)} connector(s) failed.")
        self.crisiswatch()

    def publish(self) -> None:
        verdict = {
            "source_run": self.chain.get("trigger_run_id") or None,
            "resolve_source": self.chain.get("resolve_source") or None,
            "regression_guard_outcome": _outcome_verdict(self.chain.get("regression_outcome", "")),
            "skip_publish": self.chain.get("regression_skip_publish") or None,
            "release_upload": _outcome_verdict(self.chain.get("release_upload_outcome", "")),
        }
        write_json(self.out / "regression_guard.json", verdict)
        self.summary.append(
            f"- Publish: source run {verdict['source_run']} ({verdict['resolve_source']}); "
            f"regression guard {verdict['regression_guard_outcome']}, skip_publish="
            f"{verdict['skip_publish']}; release upload {verdict['release_upload']}."
        )
        if verdict["regression_guard_outcome"] == "refused":
            self.issue("publish_regression_guard_refused", "blocking",
                       "The regression guard refused this DB: the release was not updated")
        elif str(verdict["skip_publish"]).lower() == "true":
            self.issue("publish_skipped", "degraded",
                       "Publish skipped: the release already holds more post-pipeline rows")

    # -- outputs ---------------------------------------------------------------

    def write_issues(self) -> None:
        try:
            from resolver.diagnostics import issues as issues_mod

            try:
                register = issues_mod.IssueRegister(issues_mod.KnownIssues.load())
            except Exception:  # noqa: BLE001
                register = issues_mod.IssueRegister()
            for raw in self.issues:
                register.add(issues_mod.Issue(**raw))
            label = f"{SLUG_TO_NAME.get(self.slug, self.slug)} run {self.env.get('GITHUB_RUN_ID', '')}"
            issues_mod.write_outputs(register, json_path=self.out / "issues.json",
                                     markdown_path=self.out / "issues.md",
                                     run={"run_id": self.env.get("GITHUB_RUN_ID", ""),
                                          "workflow": self.slug}, run_label=label)
        except Exception as exc:  # noqa: BLE001 - fall back to a plain register
            self.missing["issues_register"] = f"{type(exc).__name__}: {exc}"
            order = {"blocking": 0, "degraded": 1, "known": 2, "info": 3}
            ranked = sorted(self.issues, key=lambda i: order.get(i["severity"], 9))
            write_json(self.out / "issues.json", {"issues": ranked})
            lines = ["# Issues", ""] + [f"- **{i['severity']}** {i['title']}" for i in ranked]
            (self.out / "issues.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_summary(self) -> str:
        name = SLUG_TO_NAME.get(self.slug, self.slug)
        run_id = self.env.get("GITHUB_RUN_ID", "")
        order = {"blocking": 0, "degraded": 1, "known": 2, "info": 3}
        ranked = sorted(self.issues, key=lambda i: order.get(i["severity"], 9))
        lines = [f"## Chain diagnostics: {name} (run {run_id or 'local'})", ""]
        lines += self.summary or ["- (no sections produced a summary line)"]
        lines += ["", "**Issues, worst first:**", ""]
        lines += [f"- `{i['severity']}` {i['title']}" for i in ranked] or ["- none"]
        if self.missing:
            lines += ["", "**Sections not built:**", ""]
            lines += [f"- {k}: {v}" for k, v in sorted(self.missing.items())]
        text = "\n".join(lines) + "\n"
        (self.out / "summary.md").write_text(text, encoding="utf-8")
        return text


def signature(db_path: Path | None, con: Any = None) -> dict:
    """Row counts for every table: db_signature's code where it imports, plain SQL otherwise."""
    if db_path is None or not Path(db_path).is_file():
        return {"error": f"no database at {db_path}", "all_counts": {}}
    if con is None:
        try:
            from scripts.ci.db_signature import compute_signature

            return compute_signature(db_path, [], [], allow_missing_required=True, all_tables=True)
        except Exception:  # noqa: BLE001 - fall through to plain duckdb
            pass
    close = False
    if con is None:
        import duckdb

        con = duckdb.connect(str(db_path), read_only=True)
        close = True
    try:
        names = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' "
            "ORDER BY 1").fetchall()]
        counts = {n: int(con.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]) for n in names}
        return {"db_path": str(db_path), "tables": names, "all_counts": counts}
    finally:
        if close:
            con.close()


def _edition_of(payload: Mapping[str, Any] | None) -> str | None:
    """``YYYY-MM`` of a crisiswatch_latest.json payload, whatever form its month takes."""
    if not payload:
        return None
    month, year = str(payload.get("month") or ""), payload.get("year")
    import re

    m = re.match(r"^(\d{4})-(\d{2})", month)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    names = ["january", "february", "march", "april", "may", "june", "july", "august",
             "september", "october", "november", "december"]
    low = month.lower()
    for i, name in enumerate(names, start=1):
        if name in low:
            y = re.search(r"(20\d{2})", month)
            yy = int(y.group(1)) if y else int(year or 0)
            return f"{yy}-{i:02d}" if yy else None
    return None


def crisiswatch_state(repo_root: Path, diagnostics_dir: Path, *, env: Mapping[str, str],
                      today: dt.date | None = None) -> dict[str, Any]:
    """What the CrisisWatch refresh changed, measured from the JSON and git, no DB."""
    path = repo_root / "horizon_scanner" / "data" / "crisiswatch_latest.json"
    after = read_json(path) if path.is_file() else None
    before = None
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "show", "HEAD:horizon_scanner/data/crisiswatch_latest.json"],
            capture_output=True, text=True, check=True, timeout=30).stdout
        before = json.loads(out)
    except Exception:  # noqa: BLE001
        before = None
    today = today or dt.date.today()
    expected = None
    try:
        from horizon_scanner.crisiswatch import expected_edition

        y, m = expected_edition(today)
        expected = f"{int(y)}-{int(m):02d}"
    except Exception as exc:  # noqa: BLE001
        expected = None
        expected_err = f"{type(exc).__name__}: {exc}"
    else:
        expected_err = None
    entries = (after or {}).get("entries") or []
    reasons: dict[str, int] = {}
    for e in entries:
        key = str(e.get("iso3_reason") or ("resolved" if e.get("iso3") else "unresolved"))
        reasons[key] = reasons.get(key, 0) + 1
    state: dict[str, Any] = {
        "edition_before": _edition_of(before),
        "edition_after": _edition_of(after),
        "content_hash_after": (after or {}).get("content_hash"),
        "changed": (before or {}).get("content_hash") != (after or {}).get("content_hash"),
        "expected_edition": expected,
        "expected_edition_error": expected_err,
        "parse_accounting": (after or {}).get("parse_accounting") or {},
        "entries_by_iso3_reason": reasons,
        "source": (after or {}).get("source"),
    }
    after_ed = state["edition_after"]
    if expected and after_ed:
        state["gate_would_pass"] = after_ed >= expected
    log = diagnostics_dir / "crisiswatch_refresh.log"
    if log.is_file():
        text = log.read_text(encoding="utf-8", errors="replace").splitlines()
        attempts = [ln for ln in text if any(k in ln.lower() for k in
                                             ("save page", "spn", "cdx", "snapshot", "wayback"))]
        state["wayback_lines"] = len(attempts)
        state["wayback_tail"] = attempts[-40:]
    pub = diagnostics_dir / "crisiswatch_publication_days.json"
    if pub.is_file():
        state["publication_days"] = read_json(pub)
    return state


SECTIONS: dict[str, list[str]] = {
    "compute_resolutions": ["compute_resolutions"],
    "compute_scores": ["compute_scores"],
    "compute_calibration": ["compute_calibration"],
    "ingest_structured_data": ["ingest"],
    "refresh_crisiswatch": ["crisiswatch"],
    "publish_latest_data": ["publish"],
}


def write(args: argparse.Namespace, env: Mapping[str, str] | None = None) -> Path:
    out = Path(args.out_dir or (Path(args.diagnostics_dir) / args.workflow))
    out.mkdir(parents=True, exist_ok=True)
    w = Writer(slug=args.workflow, out_dir=out, db_path=_db_path(args.db),
               diagnostics_dir=Path(args.diagnostics_dir), started_at=args.started_at or "",
               env=env)
    try:
        w.section("run.json", w.run_json)
        w.section("lineage.json", w.lineage_json)
        w.section("reports", w.reports)
        for name in SECTIONS.get(args.workflow, []):
            w.section(name, getattr(w, name))
        if w.db_path is not None:
            w.section("table_counts", w.table_counts)
            w.section("freshness", w.freshness)
        if not args.no_api:
            w.section("step_timings", w.step_timings)
        if args.include:
            w.section("include", lambda: w.include(args.include))
        w.section("issues", w.write_issues)
        text = w.write_summary()
        write_json(out / "MISSING.json", w.missing)
        summary = (env or os.environ).get("GITHUB_STEP_SUMMARY")
        if summary:
            try:
                with open(summary, "a", encoding="utf-8") as fh:
                    fh.write(text + "\n")
            except OSError:
                pass
        print(text)
    finally:
        w.close()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("before", "write"):
        p = sub.add_parser(name)
        p.add_argument("--workflow", required=True, help="slug, e.g. compute_scores")
        p.add_argument("--db", default="", help="DuckDB path or duckdb:/// URL")
        p.add_argument("--diagnostics-dir", default="diagnostics")
        p.add_argument("--out-dir", default="", help="default: <diagnostics-dir>/<workflow>")
        if name == "write":
            p.add_argument("--started-at", default=os.environ.get("STAGE_STARTED_AT", ""))
            p.add_argument("--include", action="append", default=[],
                           help="a file or directory to copy into included/ (repeatable)")
            p.add_argument("--no-api", action="store_true", help="skip the jobs API")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "before":
            out = Path(args.out_dir or (Path(args.diagnostics_dir) / args.workflow))
            out.mkdir(parents=True, exist_ok=True)
            write_json(out / "db_signature_before.json", signature(_db_path(args.db)))
            print(f"before signature written to {out}")
        else:
            write(args)
    except Exception:  # noqa: BLE001 - a diagnostic never fails the job
        sys.stderr.write(traceback.format_exc())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
