# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Load and validate the input bundles (the Phase 2 zips) for the interpreter.

Two pack shapes:
- current-run bundle (``current_run_analysis__*.zip`` from
  scripts/ai_bundle/build_current_run_bundle.py): manifest, guide,
  attention_index.csv, deltas.json, blind_spots.json, questions/*.json.
- scored bundle (``scored_forecast_analysis__*.zip``): manifest, digest,
  rollups.csv (with the skill columns), questions_index.csv, briefing/.

Both load from a zip or an unzipped directory. The figure maps built here are
what the renderer resolves ``{{fig:...}}`` placeholders against — the model's
own numbers are never printed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from interpreter import selection

LOGGER = logging.getLogger(__name__)


@dataclass
class Pack:
    kind: str  # "current" | "scored"
    path: Path
    manifest: dict[str, Any]
    files: dict[str, str]  # name -> text (selected files only)
    attention_rows: list[dict[str, Any]] = field(default_factory=list)
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    rollups: list[dict[str, Any]] = field(default_factory=list)
    pack_hash: str = ""


def _read_all(path: Path) -> dict[str, bytes]:
    """{relative name: bytes} from a zip or a directory."""
    out: dict[str, bytes] = {}
    if path.is_dir():
        for f in sorted(path.rglob("*")):
            if f.is_file():
                out[f.relative_to(path).as_posix()] = f.read_bytes()
    else:
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if not name.endswith("/"):
                    out[name] = zf.read(name)
    return out


def _csv_rows(text: str) -> list[dict[str, Any]]:
    return list(csv.DictReader(io.StringIO(text)))


def load_pack(path: str | Path) -> Pack:
    """Load a bundle and classify it by its manifest's bundle_kind."""
    p = Path(path)
    raw = _read_all(p)
    if "manifest.json" not in raw:
        raise ValueError(f"{p} is not a bundle: no manifest.json")
    manifest = json.loads(raw["manifest.json"].decode("utf-8"))
    bundle_kind = str(manifest.get("bundle_kind") or "")
    if bundle_kind == "current_run_analysis":
        kind = "current"
    elif bundle_kind == "scored_forecast_analysis":
        kind = "scored"
    else:
        raise ValueError(f"unrecognized bundle_kind {bundle_kind!r} in {p}")

    digest = hashlib.sha256()
    for name in sorted(raw):
        digest.update(name.encode("utf-8"))
        digest.update(raw[name])

    pack = Pack(
        kind=kind,
        path=p,
        manifest=manifest,
        files={},
        pack_hash=digest.hexdigest()[:16],
    )

    def _text(name: str) -> str | None:
        if name in raw:
            return raw[name].decode("utf-8", errors="replace")
        return None

    for name in (
        "ANALYST_GUIDE.md", "deltas.json", "blind_spots.json", "digest.md",
        "attention_index.csv", "rollups.csv", "questions_index.csv",
        "calibration_advice.md", "briefing/02_case_studies.md",
    ):
        text = _text(name)
        if text is not None:
            pack.files[name] = text

    if kind == "current":
        if "attention_index.csv" in pack.files:
            pack.attention_rows = _csv_rows(pack.files["attention_index.csv"])
        for name, data in raw.items():
            if name.startswith("questions/") and name.endswith(".json"):
                qid = name[len("questions/"):-len(".json")]
                try:
                    pack.records[qid] = json.loads(data.decode("utf-8"))
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Unreadable record %s: %s", name, exc)
    else:
        if "rollups.csv" in pack.files:
            pack.rollups = _csv_rows(pack.files["rollups.csv"])

    return pack


def pack_question_ids(pack: Pack) -> set[str]:
    """Every question_id the pack knows about (the referential check's
    universe): attention rows + records for the current pack, the
    questions_index for the scored pack."""
    out: set[str] = set()
    for row in pack.attention_rows:
        qid = str(row.get("question_id") or "")
        if qid:
            out.add(qid)
    out.update(pack.records.keys())
    if "questions_index.csv" in pack.files:
        for row in _csv_rows(pack.files["questions_index.csv"]):
            qid = str(row.get("question_id") or "")
            if qid:
                out.add(qid)
    return out


def pack_identity(pack: Pack) -> dict[str, dict[str, str]]:
    """{question_id: {iso3, hazard_code, metric}} straight from the pack.

    These are facts about a question, not judgements, so the model's copy of
    them is repaired rather than trusted: a mis-copied hazard_code printed
    "Mali, fatalities: deaths" in a published report.
    """
    out: dict[str, dict[str, str]] = {}
    for row in pack.attention_rows:
        qid = str(row.get("question_id") or "")
        if not qid:
            continue
        fields = {
            name: str(row.get(name) or "")
            for name in ("iso3", "hazard_code", "metric")
        }
        if all(fields.values()):
            out[qid] = fields
    return out


def pack_categories(pack: Pack) -> dict[str, tuple[str | None, str | None]]:
    """{question_id: (category, hazard_family)} for the rows the pack put in a
    section. The validator holds the report to exactly this list."""
    out: dict[str, tuple[str | None, str | None]] = {}
    for row in pack.attention_rows:
        qid = str(row.get("question_id") or "")
        category = str(row.get("category") or "") or None
        if not qid or not category:
            continue
        out[qid] = (category, str(row.get("hazard_family") or "") or None)
    return out


# ---------------------------------------------------------------------------
# Figures — the values {{fig:...}} placeholders resolve against
# ---------------------------------------------------------------------------

# Every key here must be a column the attention index actually carries, and
# every {{fig:...}} key the templates instruct the model to use must be here.
# test_template_figure_keys_are_all_produced pins the second half: a template
# naming a key nothing produces renders [figure unavailable] in the report and
# fails the referential check, which is how ev_multiple shipped broken.
_ATTENTION_FIG_KEYS = (
    "js_vs_baserate", "log_ev_ratio", "ev_multiple",
    "eiv_nominal", "eiv_per_100k",
    "rc_level", "rc_score", "triage_score", "attention_rank",
    "baserate_source",
)


def _preferred_grid(record: dict[str, Any]) -> dict[str, Any] | None:
    grids = record.get("ensemble") or {}
    for pref in ("ensemble_bayesmc_v2", "ensemble_mean_v2", "track2_flash"):
        if pref in grids:
            return grids[pref]
    return next(iter(grids.values()), None)


def _mean_vector(grid: dict[str, Any]) -> list[float] | None:
    months = (grid or {}).get("months") or {}
    if not months:
        return None
    k = 0
    for buckets in months.values():
        k = max(k, max((int(b) for b in buckets), default=0))
    if k < 2:
        return None
    acc = [0.0] * k
    n = 0
    for buckets in months.values():
        vec = [float(buckets.get(str(i + 1), 0.0) or 0.0) for i in range(k)]
        total = sum(vec)
        if total <= 0:
            continue
        for i, v in enumerate(vec):
            acc[i] += v / total
        n += 1
    if n == 0:
        return None
    return [v / n for v in acc]


def question_figures(pack: Pack) -> dict[str, dict[str, Any]]:
    """{question_id: {figure key: raw value}} for the current-run pack."""
    figures: dict[str, dict[str, Any]] = {}
    for row in pack.attention_rows:
        qid = str(row.get("question_id") or "")
        if not qid:
            continue
        figs: dict[str, Any] = {}
        for key in _ATTENTION_FIG_KEYS:
            value = row.get(key)
            if value not in (None, ""):
                figs[key] = value
        record = pack.records.get(qid) or {}
        grid = _preferred_grid(record)
        vec = _mean_vector(grid) if grid else None
        if vec:
            if str(row.get("score_family") or "") == "binary":
                figs["p_event_mean"] = vec[0]
            else:
                labels = record.get("bucket_labels") or {}
                modal = max(range(len(vec)), key=lambda i: vec[i])
                figs["modal_bucket_label"] = labels.get(str(modal + 1), str(modal + 1))
                figs["p_modal_bucket"] = vec[modal]
                figs["p_top_two_buckets"] = sum(sorted(vec, reverse=True)[:2])
        figures[qid] = figs
    return figures


def _weighted_skill(rows: list[dict[str, Any]], family: str) -> dict[str, float]:
    """Sample-weighted mean brier + skill for the preferred aggregate."""
    out: dict[str, float] = {}
    for model in ("ensemble_bayesmc_v2", "ensemble_mean_v2", "track2_flash"):
        pool = [
            r for r in rows
            if r.get("model_name") == model and r.get("score_type") == "brier"
            and r.get("score_family") == family and r.get("mean_value") not in (None, "")
        ]
        if not pool:
            continue
        n = sum(int(float(r.get("n_samples") or 0)) for r in pool)
        if n <= 0:
            continue
        out[f"mean_brier_{family}"] = sum(
            float(r["mean_value"]) * int(float(r.get("n_samples") or 0)) for r in pool
        ) / n
        skill_pool = [r for r in pool if r.get("skill_vs_climatology") not in (None, "")]
        sn = sum(int(float(r.get("n_samples") or 0)) for r in skill_pool)
        if sn > 0:
            out[f"skill_brier_{family}"] = sum(
                float(r["skill_vs_climatology"]) * int(float(r.get("n_samples") or 0))
                for r in skill_pool
            ) / sn
        clim = [
            r for r in rows
            if r.get("model_name") == "__ext_climatology"
            and r.get("score_type") == "brier" and r.get("score_family") == family
            and r.get("mean_value") not in (None, "")
        ]
        cn = sum(int(float(r.get("n_samples") or 0)) for r in clim)
        if cn > 0:
            out[f"climatology_brier_{family}"] = sum(
                float(r["mean_value"]) * int(float(r.get("n_samples") or 0)) for r in clim
            ) / cn
        break  # first preferred model with data wins
    return out


# Run-level figure keys (they belong to the run, not to a question). Same
# contract as _ATTENTION_FIG_KEYS: a template may only name keys listed here.
_RUN_SUMMARY_KEYS = (
    "countries_scanned",
    "countries_with_questions",
    "countries_track1",
    "countries_track2",
    "n_questions",
    "n_above_base_rate",
    "n_below_base_rate",
)


def run_summary_figures(pack: Pack) -> dict[str, Any]:
    """Scan-scope counts for the report's opening paragraph.

    These are GLOBAL figures (they belong to the run, not to a question), so
    the model cites them the same way it cites anything else: by placeholder.
    It never counts rows itself.
    """
    manifest = pack.manifest or {}
    summary = manifest.get("run_summary") or {}
    out: dict[str, Any] = {}
    for key in _RUN_SUMMARY_KEYS:
        value = summary.get(key)
        if value is not None:
            out[key] = value
    threshold = manifest.get("worsening_multiple")
    if threshold is not None:
        out["worsening_multiple"] = threshold
    return out


def question_distributions(pack: Pack) -> dict[str, dict[str, Any]]:
    """{question_id: {"spd": [...], "bucket_labels": {...}, "binary": bool}}.

    Reuses _preferred_grid/_mean_vector, the same pair question_figures uses,
    so the chart and the modal-bucket figure printed beside it are computed
    from one aggregation rather than two that can drift.

    Charts are drawn from THIS, never from model output: the schema forbids
    extra properties on an entry, and a picture built from prose could
    disagree with the numbers next to it.
    """
    out: dict[str, dict[str, Any]] = {}
    for row in pack.attention_rows:
        qid = str(row.get("question_id") or "")
        if not qid:
            continue
        record = pack.records.get(qid) or {}
        grid = _preferred_grid(record)
        vec = _mean_vector(grid) if grid else None
        if not vec:
            continue
        out[qid] = {
            "spd": vec,
            "bucket_labels": record.get("bucket_labels"),
            "binary": str(row.get("score_family") or "") == "binary",
        }
    return out


# Performance figure keys (scored pack only). Derived from the family loop in
# _weighted_skill so the declaration cannot drift from what it emits.
_PERFORMANCE_FAMILIES = ("spd", "binary")
_PERFORMANCE_KEYS = tuple(
    f"{stem}_{family}"
    for family in _PERFORMANCE_FAMILIES
    for stem in ("mean_brier", "skill_brier", "climatology_brier")
) + ("n_scored_questions",)


def performance_figures(pack: Pack) -> dict[str, Any]:
    """Pack-level figures for the performance prose (scored pack only)."""
    figs: dict[str, Any] = {}
    for family in _PERFORMANCE_FAMILIES:
        figs.update(_weighted_skill(pack.rollups, family))
    n = pack.manifest.get("n_scored_questions")
    if n is not None:
        figs["n_scored_questions"] = n
    return figs


# ---------------------------------------------------------------------------
# Model-input assembly under the token budget
# ---------------------------------------------------------------------------


def assemble_input_text(
    pack: Pack,
    *,
    budget_tokens: int,
    estimate_tokens,
) -> tuple[str, dict[str, Any]]:
    """The pack rendered as one model-input string, under the budget.

    Fixed sections always ride; question records (current pack) are appended
    in attention-rank order until the budget is spent — same contiguous-tail
    truncation contract as the bundle builder, recorded in the returned
    stats, never silent.
    """
    parts: list[str] = []

    def _section(title: str, text: str) -> str:
        return f"\n\n===== {title} =====\n\n{text.strip()}\n"

    parts.append(_section("MANIFEST", json.dumps(pack.manifest, indent=1, default=str)))
    if "ANALYST_GUIDE.md" in pack.files:
        parts.append(_section("ANALYST GUIDE", pack.files["ANALYST_GUIDE.md"]))

    if pack.kind == "current":
        for name, title in (
            ("attention_index.csv", "ATTENTION INDEX (csv)"),
            ("deltas.json", "DELTAS VS PREVIOUS RUN"),
            ("blind_spots.json", "BLIND SPOTS"),
        ):
            if name in pack.files:
                parts.append(_section(title, pack.files[name]))
    else:
        for name, title in (
            ("digest.md", "DIGEST"),
            ("rollups.csv", "ROLLUPS incl. skill (csv)"),
            ("questions_index.csv", "QUESTIONS INDEX (csv)"),
            ("calibration_advice.md", "CALIBRATION ADVICE"),
            ("briefing/02_case_studies.md", "CASE STUDIES"),
        ):
            if name in pack.files:
                parts.append(_section(title, pack.files[name]))
        perf = performance_figures(pack)
        if perf:
            parts.append(_section(
                "PERFORMANCE FIGURES (placeholder keys for prose)",
                json.dumps(perf, indent=1, default=str),
            ))

    fixed_text = "".join(parts)
    used = estimate_tokens(fixed_text)
    kept: list[str] = []
    truncated: list[str] = []

    if pack.kind == "current":
        # Report order, not attention order. The rows the pack put in a
        # section are the ones the report is required to cover, so they are
        # the last thing a tight budget gives up. Ordering by attention rank
        # here would drop a categorised record in favour of a question the
        # report never mentions.
        categorised = [
            str(r.get("question_id") or "")
            for r in selection.selected_rows(pack.attention_rows)
            if str(r.get("question_id") or "") in pack.records
        ]
        seen = set(categorised)
        ordered = categorised + [
            str(r.get("question_id") or "")
            for r in pack.attention_rows
            if str(r.get("question_id") or "") in pack.records
            and str(r.get("question_id") or "") not in seen
        ]
        for qid in ordered:
            if truncated:
                truncated.append(qid)
                continue
            text = _section(
                f"QUESTION RECORD {qid}",
                json.dumps(pack.records[qid], indent=1, default=str),
            )
            tokens = estimate_tokens(text)
            if kept and used + tokens > budget_tokens:
                truncated.append(qid)
                continue
            parts.append(text)
            used += tokens
            kept.append(qid)
        if truncated:
            dropped_covered = [q for q in truncated if q in seen]
            notice = (
                "The following question records were omitted for the token "
                "budget (their attention_index rows above still describe "
                f"them): {', '.join(truncated)}"
            )
            if dropped_covered:
                # The model must not invent detail for a section entry whose
                # record it cannot see. Say so, by name, in the prompt.
                notice += (
                    "\n\nSome of these carry a category, so the report is "
                    "expected to cover them: "
                    f"{', '.join(dropped_covered)}. Write those entries from "
                    "their attention_index row alone. Do not invent evidence "
                    "or detail you cannot see."
                )
            parts.append(_section("TRUNCATION NOTICE", notice))

    stats = {
        "input_tokens_estimate": used,
        "budget_tokens": budget_tokens,
        "records_kept": kept,
        "records_truncated": truncated,
    }
    return "".join(parts), stats
