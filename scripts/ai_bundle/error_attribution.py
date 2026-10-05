# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Where a scored forecast's error came from: the scored bundle's error files.

The scored bundle could say how badly a forecast scored, not why. This module
adds the files that split a score into its parts, each answering one question:

* ``trace_stages.csv`` (+ ``_summary``): did the error come from the starting
  point the member chose, or from its adjustments? The base rate the prompt
  showed, the member's declared prior and its final SPD, each scored against
  the outcome.
* ``update_value.csv`` (+ ``_summary``): did each adjustment move toward the
  outcome? CLAIMED attribution: the update is the model's own account.
* ``rc_outcomes.csv`` (+ ``_summary``): does a regime-change flag predict that
  the outcome moves away from the last value shown, and in its direction?
* ``unasked_outcomes.csv``: large outcomes in cells nobody asked about.
* ``experiments.csv`` and the split columns on ``rollups.csv``: did a prompt
  version, an advice arm, a recalibration or a lineup change help?
* ``skill_history.csv``: paired skill per observed month, across the whole DB.
* ``tail_outcomes.csv`` + ``binary_reliability.csv``: what every forecaster
  gave the large outcomes and the events, and whether binary probabilities
  meant what they said.
* ``inject_health.csv``: per question, whether each inject was present and how
  old it was.
* ``headline.json``: the figures a report quotes, with intervals; the
  digest's first table is generated from it.

Every forecast also carries ``input_partial_month``: the conflict prompt's
"last month" was written before its month ended (the 1 August and
1 September 2026 runs). Rollups and history are split on it.

Rules kept throughout: binary and SPD scores are never blended; skill is
computed only on (question, horizon) pairs both sides scored, within one
track; one run per question (the latest). Every section degrades: a missing
table or column writes a stub naming the reason, and the builder exits 0.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from scripts.ai_bundle import provenance as _prov
from scripts.ai_bundle.common import (
    column_exists,
    rows_as_dicts,
    safe_json_loads,
    table_exists,
    write_csv,
    write_json,
)

LOGGER = logging.getLogger(__name__)

SEED = 20261002
#: Resamples for every interval quoted in a headline, rollup or experiment.
BOOTSTRAP = 4000
#: Resamples for the per-cell intervals of skill_history.csv, which has
#: thousands of cells; the headline figures use BOOTSTRAP.
HISTORY_BOOTSTRAP = 1000
#: Below this many questions an arm or a group is "too few".
MIN_N = 10

PRIMARY_AGGREGATES = ("ensemble_bayesmc_v2", "ensemble_mean_v2", "track2_flash")
AGGREGATES = frozenset({"ensemble_mean_v2", "ensemble_bayesmc_v2", "track2_flash", "sibyl"})
REFERENCES = (
    "__ext_climatology",
    "__ext_persistence",
    "__ext_level_volatility",
    "__ext_level_transition",
    "__ext_conflictology12",
    "__ext_ref_pool",
    "__ext_uniform",
)
#: The references skill is quoted against (history, headline).
SKILL_REFERENCES = ("__ext_climatology", "__ext_persistence", "__ext_level_volatility")
SPD_METRICS = ("FATALITIES", "PA", "PHASE3PLUS_IN_NEED")

#: The columns rollups.csv and the paired skill computation are split on.
ROLLUP_SPLIT_KEYS = (
    "lineup_id",
    "base_rate_block_version",
    "rc_guidance",
    "advice_arm",
    "recalibration_mode",
    "input_partial_month",
)
#: The flags experiments.csv compares, where the data holds more than one value.
EXPERIMENT_FLAGS = ROLLUP_SPLIT_KEYS

# ---------------------------------------------------------------------------
# input_partial_month: the conflict prompt's "last month" was a partial count
# ---------------------------------------------------------------------------

#: From this date the ACLED writer skips the month in progress and every
#: reader takes complete months only (see base_rate_spd.ACLED_COMPLETE_MONTH_SQL).
ACLED_COMPLETE_MONTH_FIX = date(2026, 9, 30)
#: (effective date, day of month) of the scheduled Resolver Update that wrote
#: acled_monthly_fatalities: the 15th until the cron moved on 2026-08-03,
#: the 28th until it moved again on 2026-10-05, the 11th since.
ACLED_INGEST_CALENDAR = (
    (date(2000, 1, 1), 15), (date(2026, 8, 3), 28), (date(2026, 10, 5), 11),
)


def _ingest_day_on(d: date) -> int:
    day = ACLED_INGEST_CALENDAR[0][1]
    for start, dd in ACLED_INGEST_CALENDAR:
        if d >= start:
            day = dd
    return day


def last_acled_ingest_on_or_before(d: date) -> date | None:
    """The scheduled ACLED ingest most recently run on or before ``d``."""
    best: date | None = None
    y, m = d.year, d.month
    for _ in range(3):
        for day in sorted({dd for _s, dd in ACLED_INGEST_CALENDAR}):
            try:
                c = date(y, m, day)
            except ValueError:
                continue
            if c <= d and _ingest_day_on(c) == day and (best is None or c > best):
                best = c
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return best


def input_partial_month(hazard_code: str | None, forecast_date: date | None) -> tuple[bool | None, str]:
    """(flag, basis): was the "last month" a conflict prompt showed a partial count?

    Only ACE prompts carry the ACLED trajectory. Before the complete-month fix
    the writer stored the month in progress, so the month before a forecast
    was partial whenever the last scheduled ingest fell inside it: the 1 August
    2026 run read a July row written on 15 July, the 1 September run an August
    row written on 28 August. Reconstructed from the ingest calendar, because
    a rewritten row keeps no history; a manual ingest between the two is not
    seen.
    """
    if str(hazard_code or "").upper() != "ACE":
        return False, "no ACLED trajectory in this prompt"
    if forecast_date is None:
        return None, "forecast date unknown"
    if forecast_date >= ACLED_COMPLETE_MONTH_FIX:
        return False, "complete-month rule in force"
    first_of_month = forecast_date.replace(day=1)
    last_month = (first_of_month - timedelta(days=1)).strftime("%Y-%m")
    ingest = last_acled_ingest_on_or_before(forecast_date)
    if ingest is None:
        return None, f"last month shown {last_month}; no scheduled ingest found"
    partial = ingest < first_of_month
    return partial, f"last month shown {last_month}; last ACLED ingest {ingest.isoformat()}"


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _scorers() -> tuple[Callable, Callable, Callable]:
    """(brier, log, rps) on a probability list and a 0-based outcome index.

    The functions compute_scores and score_baselines score with, so a stage
    score here is the number a stored score would be.
    """
    from pythia.tools.compute_scores import _brier, _log_score, _rps

    return _brier, _log_score, _rps


def _norm(vec: Sequence[float] | None, k: int | None = None) -> list[float] | None:
    if not vec:
        return None
    try:
        vals = [max(0.0, float(x or 0.0)) for x in vec]
    except (TypeError, ValueError):
        return None
    if k is not None and len(vals) != k:
        return None
    s = sum(vals)
    if s <= 0 or not math.isfinite(s):
        return None
    return [v / s for v in vals]


def entropy_bits(p: Sequence[float]) -> float:
    return -sum(x * math.log2(x) for x in p if x > 0)


def expected_bucket(p: Sequence[float]) -> float:
    """Expected bucket, 1-based (bucket 1 is the zero bucket)."""
    return 1.0 + sum(i * x for i, x in enumerate(p))


def js_distance(p: Sequence[float] | None, q: Sequence[float] | None) -> float | None:
    """sqrt of the Jensen-Shannon divergence (natural log), as the attribution
    bundle's prior_anchoring.csv reports it."""
    if not p or not q or len(p) != len(q):
        return None
    m = [(a + b) / 2 for a, b in zip(p, q)]

    def kl(a: Sequence[float]) -> float:
        return sum(x * math.log(x / y) for x, y in zip(a, m) if x > 0 and y > 0)

    jsd = 0.5 * kl(p) + 0.5 * kl(q)
    return math.sqrt(max(jsd, 0.0))


def _r(v: Any, nd: int = 4) -> Any:
    if v is None or not isinstance(v, (int, float)) or isinstance(v, bool):
        return v
    if not math.isfinite(float(v)):
        return None
    return round(float(v), nd)


def _mean(vals: Sequence[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _vjson(vec: Sequence[float] | None) -> str | None:
    return json.dumps([round(x, 4) for x in vec]) if vec else None


def _cluster_mean_ci(
    by_cluster: Mapping[Any, Sequence[float]], n_boot: int = BOOTSTRAP, seed: int = SEED
) -> tuple[float | None, float | None, float | None]:
    """(mean, 90% low, 90% high) of the values, resampling CLUSTERS (questions).

    Horizons of one question share an outcome history, so resampling rows
    would overstate the certainty; the question is the unit resampled.
    """
    sums = [(sum(v), len(v)) for v in by_cluster.values() if v]
    if not sums:
        return None, None, None
    total = sum(s for s, _ in sums) / sum(n for _, n in sums)
    if len(sums) < 2 or n_boot <= 0:
        return total, None, None
    rng = random.Random(seed)
    n = len(sums)
    stats = []
    for _ in range(n_boot):
        s = c = 0.0
        for _j in range(n):
            a, b = sums[rng.randrange(n)]
            s += a
            c += b
        stats.append(s / c)
    stats.sort()
    return total, stats[int(0.05 * n_boot)], stats[int(0.95 * n_boot) - 1]


def _cluster_skill_ci(
    by_cluster: Mapping[Any, Sequence[tuple[float, float]]],
    n_boot: int = BOOTSTRAP,
    seed: int = SEED,
) -> tuple[float | None, float | None, float | None]:
    """(skill, 90% low, 90% high), skill = 1 - sum(model) / sum(reference)
    over the paired scores, resampling questions."""
    sums = [
        (sum(m for m, _ in v), sum(r for _, r in v)) for v in by_cluster.values() if v
    ]
    if not sums:
        return None, None, None
    sm = sum(a for a, _ in sums)
    sr = sum(b for _, b in sums)
    skill = 1.0 - sm / sr if sr > 0 else None
    if len(sums) < 2 or skill is None:
        return skill, None, None
    rng = random.Random(seed)
    n = len(sums)
    stats = []
    for _ in range(n_boot):
        a = b = 0.0
        for _j in range(n):
            x, y = sums[rng.randrange(n)]
            a += x
            b += y
        if b > 0:
            stats.append(1.0 - a / b)
    if len(stats) < n_boot // 2:
        return skill, None, None
    stats.sort()
    k = len(stats)
    return skill, stats[int(0.05 * k)], stats[int(0.95 * k) - 1]


def _diff_ci(
    a: Mapping[Any, Sequence[float]], b: Mapping[Any, Sequence[float]], n_boot: int = BOOTSTRAP
) -> tuple[float | None, float | None, float | None]:
    """mean(a) - mean(b) over two independent sets of question clusters."""
    ca = [(sum(v), len(v)) for v in a.values() if v]
    cb = [(sum(v), len(v)) for v in b.values() if v]
    if not ca or not cb:
        return None, None, None
    diff = sum(s for s, _ in ca) / sum(n for _, n in ca) - sum(s for s, _ in cb) / sum(n for _, n in cb)
    if len(ca) < 2 or len(cb) < 2:
        return diff, None, None
    rng = random.Random(SEED)
    stats = []
    for _ in range(n_boot):
        sa = na = sb = nb = 0.0
        for _j in range(len(ca)):
            s, n = ca[rng.randrange(len(ca))]
            sa += s
            na += n
        for _j in range(len(cb)):
            s, n = cb[rng.randrange(len(cb))]
            sb += s
            nb += n
        stats.append(sa / na - sb / nb)
    stats.sort()
    return diff, stats[int(0.05 * n_boot)], stats[int(0.95 * n_boot) - 1]


def verdict(n_a: int, n_b: int, lo: float | None, hi: float | None, a: str, b: str) -> str:
    """Plain verdict on a lower-is-better difference ``a - b``."""
    if n_a < MIN_N or n_b < MIN_N or lo is None or hi is None:
        return "too few"
    if lo <= 0 <= hi:
        return "no clear difference"
    return f"{a} better" if hi < 0 else f"{b} better"


# ---------------------------------------------------------------------------
# Section plumbing: every section degrades to a stub naming its reason
# ---------------------------------------------------------------------------

STUB_COLUMNS = ["stub_reason"]


def write_stub(path: Path, reason: str) -> None:
    """A stub file: a CSV whose only column is ``stub_reason``, or a JSON
    object ``{"stub": true, "reason": ...}``. An absent file and a file with
    nothing to say are different statements."""
    if path.suffix == ".json":
        write_json(path, {"stub": True, "reason": reason})
    else:
        write_csv(path, STUB_COLUMNS, [{"stub_reason": reason}])


@dataclass
class SectionResult:
    files: dict[str, dict[str, Any]] = field(default_factory=dict)

    def ok(self, name: str, rows: int, **extra: Any) -> None:
        self.files[name] = {"status": "ok", "rows": rows, **extra}

    def stub(self, out_dir: Path, name: str, reason: str) -> None:
        write_stub(out_dir / name, reason)
        self.files[name] = {"status": "stub", "rows": 0, "reason": reason}


# ---------------------------------------------------------------------------
# Context: one pass over the DB that every section reads
# ---------------------------------------------------------------------------


@dataclass
class Context:
    con: Any
    include_test: bool
    #: Every scored question in the DB (not only the bundle's window).
    qmeta: dict[str, dict[str, Any]]
    #: {(question_id, horizon_m): {"value", "bucket0", "event", "observed_month"}}
    outcomes: dict[tuple[str, int], dict[str, Any]]
    bundle_qids: list[str]
    #: {(question_id, model_name, horizon_m, score_type): value}, latest run.
    scores: dict[tuple[str, str, int, str], float] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    #: {question_id: primary aggregate}, filled by index_scores().
    primary: dict[str, str] = field(default_factory=dict)
    #: {question_id: {member with a __raw sibling}}, filled by index_scores().
    raw_siblings: dict[str, set[str]] = field(default_factory=dict)

    def index_scores(self) -> None:
        models: dict[str, set[str]] = defaultdict(set)
        for (qid, model, _h, _st) in self.scores:
            models[qid].add(model)
        self.primary = {
            q: next(m for m in PRIMARY_AGGREGATES if m in ms)
            for q, ms in models.items() if any(m in ms for m in PRIMARY_AGGREGATES)
        }
        self.raw_siblings = {
            q: {m[: -len("__raw")] for m in ms if m.endswith("__raw")} for q, ms in models.items()
        }

    def sample_attrs(self, qid: str, model: str) -> dict[str, Any]:
        """The ROLLUP_SPLIT_KEYS values for a question, plus ``correction``."""
        meta = self.qmeta.get(str(qid)) or {}
        out = {k: meta.get(k) for k in ROLLUP_SPLIT_KEYS}
        out["correction"] = correction_label(model, self.has_raw(qid, model))
        return out

    def has_raw(self, qid: str, model: str) -> bool:
        return str(model) in self.raw_siblings.get(str(qid), set())


def correction_label(model: str, has_raw_sibling: bool) -> str:
    """raw / shadow_corrected / corrected / none for a scored model row."""
    m = str(model or "")
    if m.endswith("__raw"):
        return "raw"
    if m.endswith("__recal"):
        return "shadow_corrected"
    if has_raw_sibling:
        return "corrected"
    return "none"


def _label(value: Any, default: str) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    return str(value)


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)[:19]).date()
    except ValueError:
        try:
            return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
        except ValueError:
            return None


def _ym(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value)
    return s[:7] if len(s) >= 7 else None


def _bucket0(metric: str, value: Any) -> int | None:
    if value is None:
        return None
    try:
        from pythia.tools.base_rate_spd import _bucket_index_for_value

        return _bucket_index_for_value(float(value), metric)
    except Exception:  # noqa: BLE001
        return None


def _n_buckets(metric: str) -> int | None:
    try:
        from pythia.buckets import n_buckets_for

        return int(n_buckets_for(metric)) or None
    except Exception:  # noqa: BLE001
        return None


def _family(metric: str | None) -> str:
    return "binary" if str(metric or "").upper() == "EVENT_OCCURRENCE" else "spd"


def _test_sql(con, table: str, alias: str, include_test: bool) -> str:
    if include_test or not column_exists(con, table, "is_test"):
        return ""
    return f" AND COALESCE({alias}.is_test, FALSE) = FALSE"


def _latest_runs(con) -> dict[str, str]:
    """{question_id: latest run_id}: from scores, else forecasts_ensemble."""
    out: dict[str, str] = {}
    for table in ("forecasts_ensemble", "scores"):
        if table_exists(con, table) and column_exists(con, table, "run_id"):
            try:
                for qid, run in con.execute(
                    f"SELECT question_id, MAX(run_id) FROM {table} "
                    "WHERE run_id IS NOT NULL AND run_id <> '' GROUP BY 1"
                ).fetchall():
                    out[str(qid)] = str(run)
            except Exception:  # noqa: BLE001
                continue
    return out


def _mode_by_run_question(con, expr: str, where: str = "") -> dict[tuple[str, str], Any]:
    try:
        rows = con.execute(
            f"SELECT run_id, question_id, mode({expr}) FROM forecasts_raw "
            f"WHERE {expr} IS NOT NULL{where} GROUP BY 1, 2"
        ).fetchall()
    except Exception:  # noqa: BLE001
        return {}
    return {(str(r), str(q)): v for r, q, v in rows}


def build_context(con, bundle_qids: Sequence[str], include_test: bool = False) -> Context:
    """Everything the sections share, read once."""
    problems: list[str] = []
    track = "q.track" if column_exists(con, "questions", "track") else "NULL"
    hs = "q.hs_run_id" if column_exists(con, "questions", "hs_run_id") else "NULL"
    ws = "q.window_start_date" if column_exists(con, "questions", "window_start_date") else "NULL"
    questions = rows_as_dicts(
        con,
        f"SELECT DISTINCT q.question_id, {hs} AS hs_run_id, upper(q.iso3) AS iso3, "
        f"upper(q.hazard_code) AS hazard_code, upper(q.metric) AS metric, "
        f"{track} AS track, {ws} AS window_start_date "
        "FROM questions q WHERE q.question_id IN (SELECT question_id FROM scores)"
        + _test_sql(con, "questions", "q", include_test),
    )
    latest = _latest_runs(con)

    created: dict[tuple[str, str], Any] = {}
    if table_exists(con, "forecasts_ensemble") and column_exists(con, "forecasts_ensemble", "created_at"):
        try:
            for run, qid, ts in con.execute(
                "SELECT run_id, question_id, MIN(created_at) FROM forecasts_ensemble GROUP BY 1, 2"
            ).fetchall():
                created[(str(run), str(qid))] = ts
        except Exception:  # noqa: BLE001
            pass

    attrs: dict[str, dict[tuple[str, str], Any]] = {}
    fr = table_exists(con, "forecasts_raw")
    member_where = (
        " AND model_name NOT LIKE '%\\_\\_raw' ESCAPE '\\' "
        "AND model_name NOT LIKE '%\\_\\_recal' ESCAPE '\\'"
    )
    if fr:
        for col in ("base_rate_block_version", "rc_guidance", "advice_arm"):
            if column_exists(con, "forecasts_raw", col):
                attrs[col] = _mode_by_run_question(con, col, member_where)
        if column_exists(con, "forecasts_raw", "recalibration_json"):
            attrs["recalibration_mode"] = _mode_by_run_question(
                con, "json_extract_string(recalibration_json, '$.mode')", member_where
            )
    if table_exists(con, "forecasts_ensemble") and column_exists(con, "forecasts_ensemble", "advice_arm"):
        try:
            fe_arm = {
                (str(r), str(q)): v
                for r, q, v in con.execute(
                    "SELECT run_id, question_id, mode(advice_arm) FROM forecasts_ensemble "
                    "WHERE advice_arm IS NOT NULL GROUP BY 1, 2"
                ).fetchall()
            }
            base = attrs.setdefault("advice_arm", {})
            for k, v in fe_arm.items():
                base.setdefault(k, v)
        except Exception:  # noqa: BLE001
            pass

    triage: dict[tuple[str, str, str], dict[str, Any]] = {}
    if table_exists(con, "hs_triage"):
        try:
            for r in rows_as_dicts(
                con,
                "SELECT run_id, upper(iso3) AS iso3, upper(hazard_code) AS hazard_code, "
                "regime_change_level, regime_change_direction, regime_change_score, "
                "tier, triage_score FROM hs_triage",
            ):
                triage[(str(r["run_id"]), r["iso3"], r["hazard_code"])] = r
        except Exception as exc:  # noqa: BLE001
            problems.append(f"hs_triage unreadable: {type(exc).__name__}")

    lineups = _prov.lineup_ids_bulk(con)
    hs_dates: dict[str, date | None] = {}
    qmeta: dict[str, dict[str, Any]] = {}
    for q in questions:
        qid = str(q["question_id"])
        run = latest.get(qid)
        key = (str(run), qid)
        fdate = _as_date(created.get(key))
        hs_run = q.get("hs_run_id")
        if fdate is None and hs_run:
            if hs_run not in hs_dates:
                hs_dates[hs_run] = _prov.run_date(con, hs_run)
            fdate = hs_dates[hs_run]
        partial, basis = input_partial_month(q.get("hazard_code"), fdate)
        t = triage.get((str(hs_run), q.get("iso3"), q.get("hazard_code"))) or {}
        qmeta[qid] = {
            **q,
            "track": q.get("track"),
            "run_id": run,
            "forecast_date": fdate,
            "window_start_ym": _ym(q.get("window_start_date")),
            "input_partial_month": partial,
            "input_partial_month_basis": basis,
            "lineup_id": lineups.get(key),
            "base_rate_block_version": _label((attrs.get("base_rate_block_version") or {}).get(key), "none"),
            "rc_guidance": _label((attrs.get("rc_guidance") or {}).get(key), "legacy"),
            "advice_arm": _label((attrs.get("advice_arm") or {}).get(key), "none"),
            "recalibration_mode": _label((attrs.get("recalibration_mode") or {}).get(key), "off"),
            "rc_level": t.get("regime_change_level"),
            "rc_direction": t.get("regime_change_direction"),
            "rc_score": t.get("regime_change_score"),
            "tier": t.get("tier"),
        }

    outcomes: dict[tuple[str, int], dict[str, Any]] = {}
    if table_exists(con, "resolutions"):
        om = "observed_month" if column_exists(con, "resolutions", "observed_month") else "NULL"
        try:
            for qid, h, v, obs in con.execute(
                f"SELECT question_id, horizon_m, value, {om} FROM resolutions"
            ).fetchall():
                meta = qmeta.get(str(qid))
                if meta is None or h is None:
                    continue
                metric = meta["metric"]
                rec: dict[str, Any] = {"value": v, "observed_month": _ym(obs)}
                if _family(metric) == "binary":
                    rec["event"] = None if v is None else (1 if float(v) >= 0.5 else 0)
                    rec["bucket0"] = None if v is None else (0 if float(v) >= 0.5 else 1)
                else:
                    rec["bucket0"] = _bucket0(metric, v)
                outcomes[(str(qid), int(h))] = rec
        except Exception as exc:  # noqa: BLE001
            problems.append(f"resolutions unreadable: {type(exc).__name__}")
    else:
        problems.append("resolutions table absent")

    ctx = Context(con=con, include_test=include_test, qmeta=qmeta, outcomes=outcomes,
                  bundle_qids=[str(x) for x in bundle_qids if str(x) in qmeta], problems=problems)
    ctx.scores = _load_scores(con, latest)
    ctx.index_scores()
    return ctx


def _load_scores(con, latest: Mapping[str, str]) -> dict[tuple[str, str, int, str], float]:
    """Latest-run scores, plus the run-less reference rows."""
    if not table_exists(con, "scores"):
        return {}
    run = "run_id" if column_exists(con, "scores", "run_id") else "NULL"
    out: dict[tuple[str, str, int, str], float] = {}
    for qid, h, model, st, v, rid in con.execute(
        f"SELECT question_id, horizon_m, model_name, score_type, value, {run} FROM scores "
        "WHERE value IS NOT NULL AND model_name IS NOT NULL"
    ).fetchall():
        qid = str(qid)
        if rid is not None and latest.get(qid) is not None and str(rid) != latest[qid]:
            continue
        out[(qid, str(model), int(h), str(st))] = float(v)
    return out


def primary_aggregate(ctx: Context, qid: str) -> str | None:
    return ctx.primary.get(str(qid))


# ---------------------------------------------------------------------------
# Vector loaders (the bundle's questions only)
# ---------------------------------------------------------------------------


def _latest_pairs_sql(ctx: Context, qids: Sequence[str]) -> tuple[str, list[Any]]:
    """A join keeping only each question's latest run, pushed into SQL."""
    pairs = [(q, str(ctx.qmeta[q]["run_id"])) for q in qids if ctx.qmeta.get(q, {}).get("run_id")]
    sql = (
        " JOIN (SELECT UNNEST(?::VARCHAR[]) AS _q, UNNEST(?::VARCHAR[]) AS _r) _l "
        "ON _l._q = t.question_id AND _l._r = t.run_id "
    )
    return sql, [[p[0] for p in pairs], [p[1] for p in pairs]]


def _vectors_from(ctx: Context, table: str, qids: Sequence[str]) -> dict[tuple[str, str, int], dict[int, float]]:
    """{(qid, model, month_index): {bucket_index: p}} from ``table``, latest run."""
    memo_key = (table, tuple(qids))
    memo = ctx.__dict__.setdefault("_vec_memo", {})
    if memo_key in memo:
        return memo[memo_key]
    out: dict[tuple[str, str, int], dict[int, float]] = defaultdict(dict)
    if table_exists(ctx.con, table) and qids:
        join, params = _latest_pairs_sql(ctx, qids)
        for qid, model, h, b, p in ctx.con.execute(
            f"SELECT t.question_id, t.model_name, t.month_index, t.bucket_index, t.probability "
            f"FROM {table} t{join}WHERE t.probability IS NOT NULL AND t.bucket_index IS NOT NULL "
            "AND t.month_index IS NOT NULL",
            params,
        ).fetchall():
            out[(str(qid), str(model), int(h))][int(b)] = float(p)
    memo[memo_key] = out
    return out


def _raw_vectors(con, ctx: Context, qids: Sequence[str]) -> dict[tuple[str, str, int], dict[int, float]]:
    return _vectors_from(ctx, "forecasts_raw", qids)


def _ensemble_vectors(con, ctx: Context, qids: Sequence[str]) -> dict[tuple[str, str, int], dict[int, float]]:
    return _vectors_from(ctx, "forecasts_ensemble", qids)


def _dense(by_bucket: Mapping[int, float] | None, k: int | None) -> list[float] | None:
    if not by_bucket:
        return None
    k = k or max(by_bucket)
    return [float(by_bucket.get(i, 0.0)) for i in range(1, k + 1)]


def _traces(con, ctx: Context, qids: Sequence[str]) -> dict[tuple[str, str], dict[str, Any]]:
    """{(qid, model): parsed reasoning trace} for the latest run."""
    if not table_exists(con, "forecasts_raw") or not column_exists(con, "forecasts_raw", "reasoning_trace_json"):
        return {}
    memo_key = ("traces", tuple(qids))
    memo = ctx.__dict__.setdefault("_vec_memo", {})
    if memo_key in memo:
        return memo[memo_key]
    join, params = _latest_pairs_sql(ctx, qids)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for qid, model, tj in con.execute(
        "SELECT DISTINCT t.question_id, t.model_name, t.reasoning_trace_json FROM forecasts_raw t"
        f"{join}WHERE t.reasoning_trace_json IS NOT NULL",
        params,
    ).fetchall():
        trace = safe_json_loads(tj)
        if isinstance(trace, dict):
            out.setdefault((str(qid), str(model)), trace)
    memo[memo_key] = out
    return out


def _is_member(model: str) -> bool:
    m = str(model or "")
    if m.startswith("__ext_") or m.endswith(("__raw", "__recal")):
        return False
    return m == "track2_flash" or m not in AGGREGATES


# ---------------------------------------------------------------------------
# 1. trace_stages.csv: base rate shown, declared prior, final SPD
# ---------------------------------------------------------------------------

TRACE_STAGE_COLUMNS = [
    "question_id", "run_id", "iso3", "hazard_code", "metric", "track", "horizon_m",
    "model_name", "outcome_value", "realized_bucket",
    "shown_source", "shown_spd", "prior_spd", "final_spd",
    *[f"{s}_{st}" for st in ("shown", "prior", "final") for s in ("brier", "log", "rps")],
    *[f"{s}_{st}" for st in ("shown", "prior", "final")
      for s in ("max_prob", "entropy_bits", "expected_bucket")],
    "js_distance_prior_vs_shown", "expected_bucket_shift_prior_to_final",
    "entropy_change_prior_to_final",
    "rc_level", "rc_direction", "rc_assessment", "rc_guidance",
    "base_rate_block_version", "advice_arm", "recalibration_mode",
    "input_partial_month", "lineup_id",
]

TRACE_STAGE_SUMMARY_COLUMNS = [
    "hazard_code", "metric", "track", "base_rate_block_version", "rc_guidance",
    "input_partial_month", "score_type", "n", "n_questions", "n_with_shown",
    "mean_shown", "mean_prior", "mean_final",
    "prior_minus_shown", "prior_minus_shown_ci90_low", "prior_minus_shown_ci90_high",
    "final_minus_prior", "final_minus_prior_ci90_low", "final_minus_prior_ci90_high",
    "start_verdict", "adjustment_verdict",
]


def _stage_stats(prefix: str, p: list[float] | None, j: int | None, scorers) -> dict[str, Any]:
    brier, logs, rps = scorers
    out: dict[str, Any] = {}
    if p is None:
        return out
    if j is not None:
        out[f"brier_{prefix}"] = _r(brier(p, j))
        out[f"log_{prefix}"] = _r(logs(p, j))
        out[f"rps_{prefix}"] = _r(rps(p, j))
    out[f"max_prob_{prefix}"] = _r(max(p))
    out[f"entropy_bits_{prefix}"] = _r(entropy_bits(p))
    out[f"expected_bucket_{prefix}"] = _r(expected_bucket(p))
    return out


def build_trace_stages(ctx: Context) -> list[dict[str, Any]]:
    from scripts.ai_bundle.build_forecast_attribution_bundle import _normalise_rc

    con = ctx.con
    qids = [
        q for q in ctx.bundle_qids
        if _family(ctx.qmeta[q]["metric"]) == "spd" and ctx.qmeta[q].get("track") in (1, 2)
    ]
    if not qids:
        return []
    scorers = _scorers()
    vectors = _raw_vectors(con, ctx, qids)
    traces = _traces(con, ctx, qids)
    refs = _prov.reference_vectors(con, qids)
    rows: list[dict[str, Any]] = []
    for (qid, model), trace in sorted(traces.items()):
        if not _is_member(model):
            continue
        meta = ctx.qmeta[qid]
        metric = meta["metric"]
        k = _n_buckets(metric)
        prior = _norm((trace.get("prior") or {}).get("spd") if isinstance(trace.get("prior"), dict) else None, k)
        anchored = meta["base_rate_block_version"] not in ("none", None)
        for h in range(1, 7):
            out = ctx.outcomes.get((qid, h))
            if not out or out.get("bucket0") is None:
                continue
            j = out["bucket0"]
            final = _norm(_dense(vectors.get((qid, model, h)), k), k)
            src = "__ext_level_volatility" if anchored else "__ext_climatology"
            shown = _norm(refs.get((qid, src, h)), k)
            row: dict[str, Any] = {
                "question_id": qid, "run_id": meta["run_id"], "iso3": meta["iso3"],
                "hazard_code": meta["hazard_code"], "metric": metric, "track": meta["track"],
                "horizon_m": h, "model_name": model, "outcome_value": out["value"],
                "realized_bucket": j + 1,
                "shown_source": ("level_volatility" if anchored else "climatology") if shown else "absent",
                "shown_spd": _vjson(shown), "prior_spd": _vjson(prior), "final_spd": _vjson(final),
                "rc_level": meta.get("rc_level"), "rc_direction": meta.get("rc_direction"),
                "rc_assessment": _normalise_rc(trace.get("rc_assessment")),
                **{c: meta.get(c) for c in ("rc_guidance", "base_rate_block_version", "advice_arm",
                                             "recalibration_mode", "input_partial_month", "lineup_id")},
            }
            row.update(_stage_stats("shown", shown, j, scorers))
            row.update(_stage_stats("prior", prior, j, scorers))
            row.update(_stage_stats("final", final, j, scorers))
            row["js_distance_prior_vs_shown"] = _r(js_distance(prior, shown))
            if prior and final:
                row["expected_bucket_shift_prior_to_final"] = _r(expected_bucket(final) - expected_bucket(prior))
                row["entropy_change_prior_to_final"] = _r(entropy_bits(final) - entropy_bits(prior))
            rows.append(row)
    return rows


def summarise_trace_stages(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["hazard_code"], r["metric"], r["track"], r["base_rate_block_version"],
                r["rc_guidance"], r["input_partial_month"])].append(r)
    out: list[dict[str, Any]] = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        for st in ("brier", "rps", "log"):
            usable = [r for r in rs if r.get(f"{st}_prior") is not None and r.get(f"{st}_final") is not None]
            if not usable:
                continue
            start: dict[str, list[float]] = defaultdict(list)
            adjust: dict[str, list[float]] = defaultdict(list)
            for r in usable:
                adjust[r["question_id"]].append(float(r[f"{st}_final"]) - float(r[f"{st}_prior"]))
                if r.get(f"{st}_shown") is not None:
                    start[r["question_id"]].append(float(r[f"{st}_prior"]) - float(r[f"{st}_shown"]))
            sm, slo, shi = _cluster_mean_ci(start)
            am, alo, ahi = _cluster_mean_ci(adjust)
            with_shown = [r for r in usable if r.get(f"{st}_shown") is not None]
            n_q = len({r["question_id"] for r in usable})
            out.append({
                "hazard_code": key[0], "metric": key[1], "track": key[2],
                "base_rate_block_version": key[3], "rc_guidance": key[4],
                "input_partial_month": key[5], "score_type": st,
                "n": len(usable), "n_questions": n_q, "n_with_shown": len(with_shown),
                "mean_shown": _r(_mean([float(r[f"{st}_shown"]) for r in with_shown])),
                "mean_prior": _r(_mean([float(r[f"{st}_prior"]) for r in usable])),
                "mean_final": _r(_mean([float(r[f"{st}_final"]) for r in usable])),
                "prior_minus_shown": _r(sm), "prior_minus_shown_ci90_low": _r(slo),
                "prior_minus_shown_ci90_high": _r(shi),
                "final_minus_prior": _r(am), "final_minus_prior_ci90_low": _r(alo),
                "final_minus_prior_ci90_high": _r(ahi),
                "start_verdict": _stage_verdict(len(start), slo, shi, "prior worse than the base rate shown",
                                                "prior better than the base rate shown"),
                "adjustment_verdict": _stage_verdict(n_q, alo, ahi, "adjustments hurt", "adjustments helped"),
            })
    return out


def _stage_verdict(n: int, lo: float | None, hi: float | None, worse: str, better: str) -> str:
    if n < MIN_N or lo is None or hi is None:
        return "too few"
    if lo <= 0 <= hi:
        return "no clear difference"
    return worse if lo > 0 else better


# ---------------------------------------------------------------------------
# 2. update_value.csv: did each claimed adjustment move toward the outcome?
# ---------------------------------------------------------------------------

UPDATE_VALUE_COLUMNS = [
    "attribution_id", "question_id", "run_id", "iso3", "hazard_code", "metric", "track",
    "model_name", "update_index", "signal_class", "signal_text", "claimed_magnitude",
    "months_affected", "horizon_m", "realized_bucket", "mass_moved_l1", "direction",
    "p_realized_before", "p_realized_after", "delta_p_realized", "rps_before", "rps_after",
    "delta_rps", "moved_toward_outcome",
]
UPDATE_SUMMARY_COLUMNS = [
    "signal_class", "hazard_code", "metric", "n_updates", "n_rows", "n_questions",
    "share_toward_outcome", "mean_delta_p_realized", "mean_delta_rps",
    "delta_rps_ci90_low", "delta_rps_ci90_high", "verdict",
]


def months_affected(value: Any) -> set[int]:
    """The horizons a trace update says it touched; all six when unreadable."""
    allm = set(range(1, 7))
    if value is None:
        return allm
    if isinstance(value, (list, tuple)):
        got = {int(x) for x in value if isinstance(x, (int, float)) and 1 <= int(x) <= 6}
        return got or allm
    s = str(value).strip().lower()
    if not s or "all" in s:
        return allm
    got: set[int] = set()
    for part in s.replace(";", ",").replace(" and ", ",").split(","):
        part = part.strip().replace("months", "").replace("month", "").strip()
        if "-" in part or "–" in part:
            a, _, b = part.replace("–", "-").partition("-")
            if a.strip().isdigit() and b.strip().isdigit():
                got.update(range(int(a), int(b) + 1))
        elif part.isdigit():
            got.add(int(part))
    got = {m for m in got if 1 <= m <= 6}
    return got or allm


def build_update_value(ctx: Context) -> list[dict[str, Any]]:
    from scripts.ai_bundle.build_forecast_attribution_bundle import (
        ledger_rows_for_member,
        load_taxonomy,
    )

    con = ctx.con
    qids = [q for q in ctx.bundle_qids if _family(ctx.qmeta[q]["metric"]) == "spd"]
    if not qids:
        return []
    _brier, _log, rps = _scorers()
    taxonomy = load_taxonomy()
    traces = _traces(con, ctx, qids)
    rows: list[dict[str, Any]] = []
    for (qid, model), trace in sorted(traces.items()):
        if not _is_member(model):
            continue
        meta = ctx.qmeta[qid]
        k = _n_buckets(meta["metric"])
        updates = trace.get("updates") if isinstance(trace.get("updates"), list) else []
        ledger = ledger_rows_for_member(
            run_id=str(meta["run_id"]), hs_run_id=meta.get("hs_run_id"), q=meta,
            model_name=model, trace=trace, taxonomy=taxonomy, created_at=None, n_buckets=k,
        )
        for lr in ledger:
            if lr.get("is_prior_row"):
                continue
            idx = int(lr["update_index"])
            pre = _norm(safe_json_loads(lr.get("pre_spd_json")), k)
            post = _norm(safe_json_loads(lr.get("post_spd_json")), k)
            if pre is None or post is None:
                continue
            upd = updates[idx] if idx < len(updates) and isinstance(updates[idx], dict) else {}
            ma = upd.get("months_affected")
            for h in sorted(months_affected(ma)):
                out = ctx.outcomes.get((qid, h))
                if not out or out.get("bucket0") is None:
                    continue
                j = out["bucket0"]
                rb, ra = rps(pre, j), rps(post, j)
                rows.append({
                    "attribution_id": lr["attribution_id"], "question_id": qid,
                    "run_id": meta["run_id"], "iso3": meta["iso3"],
                    "hazard_code": meta["hazard_code"], "metric": meta["metric"],
                    "track": meta["track"], "model_name": model, "update_index": idx,
                    "signal_class": lr.get("signal_class"),
                    "signal_text": (str(lr.get("signal_text") or "")[:300]) or None,
                    "claimed_magnitude": lr.get("claimed_magnitude"),
                    "months_affected": ma if isinstance(ma, str) else json.dumps(ma) if ma else None,
                    "horizon_m": h, "realized_bucket": j + 1,
                    "mass_moved_l1": lr.get("mass_moved_l1"), "direction": lr.get("direction"),
                    "p_realized_before": _r(pre[j]), "p_realized_after": _r(post[j]),
                    "delta_p_realized": _r(post[j] - pre[j]),
                    "rps_before": _r(rb), "rps_after": _r(ra), "delta_rps": _r(ra - rb),
                    "moved_toward_outcome": post[j] > pre[j],
                })
    return rows


def summarise_update_value(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["signal_class"], r["hazard_code"], r["metric"])].append(r)
    out: list[dict[str, Any]] = []
    for (klass, hz, metric), rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        by_q: dict[str, list[float]] = defaultdict(list)
        for r in rs:
            by_q[r["question_id"]].append(float(r["delta_rps"]))
        m, lo, hi = _cluster_mean_ci(by_q)
        n_q = len(by_q)
        out.append({
            "signal_class": klass, "hazard_code": hz, "metric": metric,
            "n_updates": len({r["attribution_id"] for r in rs}), "n_rows": len(rs),
            "n_questions": n_q,
            "share_toward_outcome": _r(sum(1 for r in rs if r["moved_toward_outcome"]) / len(rs)),
            "mean_delta_p_realized": _r(_mean([float(r["delta_p_realized"]) for r in rs])),
            "mean_delta_rps": _r(m), "delta_rps_ci90_low": _r(lo), "delta_rps_ci90_high": _r(hi),
            "verdict": _stage_verdict(n_q, lo, hi, "updates of this class hurt", "updates of this class helped"),
        })
    return out


# ---------------------------------------------------------------------------
# 3. rc_outcomes.csv: did the outcome move as the regime-change flag said?
# ---------------------------------------------------------------------------

RC_OUTCOME_COLUMNS = [
    "question_id", "iso3", "hazard_code", "metric", "track", "horizon_m",
    "rc_level", "rc_score", "rc_direction", "last_value", "last_value_month",
    "last_value_source", "last_bucket", "outcome_value", "outcome_bucket",
    "bucket_move", "abs_bucket_move", "direction_matched", "input_partial_month",
]
RC_SUMMARY_COLUMNS = [
    "metric", "rc_level", "rc_direction", "n", "n_questions", "mean_abs_bucket_move",
    "share_moved_2plus", "n_directional", "share_matching_direction",
]


def _direction_group(direction: Any) -> str:
    d = str(direction or "").strip().lower()
    return d if d in {"up", "down", "mixed"} else "unclear"


def build_rc_outcomes(ctx: Context) -> list[dict[str, Any]]:
    from pythia.tools.base_rate_spd import last_observed_value

    rows: list[dict[str, Any]] = []
    cache: dict[str, Any] = {}
    for qid in ctx.bundle_qids:
        meta = ctx.qmeta[qid]
        metric = meta["metric"]
        if metric not in SPD_METRICS:
            continue
        if qid not in cache:
            try:
                cache[qid] = last_observed_value(
                    ctx.con, meta["iso3"], meta["hazard_code"], metric, meta.get("window_start_ym")
                )
            except Exception:  # noqa: BLE001
                cache[qid] = None
        last = cache[qid]
        if not last:
            continue
        value, ym, source = last
        lb = _bucket0(metric, value)
        dgroup = _direction_group(meta.get("rc_direction"))
        for h in range(1, 7):
            out = ctx.outcomes.get((qid, h))
            if not out or out.get("bucket0") is None or lb is None:
                continue
            move = out["bucket0"] - lb
            matched = None
            if dgroup == "up":
                matched = move > 0
            elif dgroup == "down":
                matched = move < 0
            rows.append({
                "question_id": qid, "iso3": meta["iso3"], "hazard_code": meta["hazard_code"],
                "metric": metric, "track": meta["track"], "horizon_m": h,
                "rc_level": meta.get("rc_level"), "rc_score": meta.get("rc_score"),
                "rc_direction": dgroup, "last_value": value, "last_value_month": ym,
                "last_value_source": source, "last_bucket": lb + 1,
                "outcome_value": out["value"], "outcome_bucket": out["bucket0"] + 1,
                "bucket_move": move, "abs_bucket_move": abs(move), "direction_matched": matched,
                "input_partial_month": meta.get("input_partial_month"),
            })
    return rows


def summarise_rc_outcomes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        level = r.get("rc_level")
        groups[(r["metric"], 0 if level is None else int(level), r["rc_direction"])].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        directional = [r for r in rs if r["direction_matched"] is not None]
        out.append({
            "metric": key[0], "rc_level": key[1], "rc_direction": key[2], "n": len(rs),
            "n_questions": len({r["question_id"] for r in rs}),
            "mean_abs_bucket_move": _r(_mean([r["abs_bucket_move"] for r in rs])),
            "share_moved_2plus": _r(sum(1 for r in rs if r["abs_bucket_move"] >= 2) / len(rs)),
            "n_directional": len(directional),
            "share_matching_direction": (
                _r(sum(1 for r in directional if r["direction_matched"]) / len(directional))
                if directional else None
            ),
        })
    return out


# ---------------------------------------------------------------------------
# 4. unasked_outcomes.csv: large outcomes in cells with no question
# ---------------------------------------------------------------------------

UNASKED_COLUMNS = [
    "iso3", "hazard_code", "month", "trigger", "value", "baseline_month", "baseline_value",
    "bucket", "baseline_bucket", "source", "hs_run_id", "triage_tier", "triage_score", "rc_level",
]
FORECAST_HAZARDS = ("ACE", "DR", "FL", "TC")


def hs_country_iso3s() -> list[str]:
    """The ISO3s of horizon_scanner/hs_country_list.txt, resolved as HS resolves them."""
    root = Path(__file__).resolve().parents[2]
    path = root / "horizon_scanner" / "hs_country_list.txt"
    if not path.exists():
        return []
    raw = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
           if ln.strip() and not ln.startswith("#")]
    try:
        from horizon_scanner.horizon_scanner import _load_country_registry, _resolve_country

        iso3_to_name, name_to_iso3 = _load_country_registry()
        return sorted({iso for _n, iso in (_resolve_country(r, iso3_to_name, name_to_iso3) for r in raw) if iso})
    except Exception:  # noqa: BLE001 - fall back to the registry file alone
        import csv

        names: dict[str, str] = {}
        with (root / "resolver" / "data" / "countries.csv").open(encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                names[(row.get("country_name") or "").strip().lower()] = (row.get("iso3") or "").upper()
        out = set()
        for r in raw:
            if r.lower() in names:
                out.add(names[r.lower()])
            elif len(r) == 3 and r.isalpha():
                out.add(r.upper())
        return sorted(out)


def _add_months(ym: str, n: int) -> str:
    y, m = int(ym[:4]), int(ym[5:7])
    m += n
    while m <= 0:
        m += 12
        y -= 1
    while m > 12:
        m -= 12
        y += 1
    return f"{y:04d}-{m:02d}"


def build_unasked_outcomes(ctx: Context, countries: Sequence[str] | None = None) -> list[dict[str, Any]]:
    con = ctx.con
    bundle = [ctx.qmeta[q] for q in ctx.bundle_qids]
    in_bundle = set(ctx.bundle_qids)
    scored_months = sorted({
        o["observed_month"] for (qid, _h), o in ctx.outcomes.items()
        if qid in in_bundle and o.get("observed_month")
    })
    if not scored_months:
        return []
    countries = list(countries) if countries is not None else hs_country_iso3s()
    # Which epoch covered a month: the earliest window start whose six months hold it.
    epochs = sorted({(m["window_start_ym"], m.get("hs_run_id")) for m in bundle if m.get("window_start_ym")})

    def epoch_for(month: str) -> tuple[str | None, str | None]:
        for ws, hs in epochs:
            if ws <= month <= _add_months(ws, 5):
                return ws, hs
        return None, None

    asked: set[tuple[str, str, str]] = set()
    if table_exists(con, "questions"):
        ws_col = "window_start_date" if column_exists(con, "questions", "window_start_date") else None
        if ws_col:
            for iso, hz, ws in con.execute(
                f"SELECT DISTINCT upper(iso3), upper(hazard_code), {ws_col} FROM questions"
            ).fetchall():
                w = _ym(ws)
                if not w:
                    continue
                for i in range(6):
                    asked.add((str(iso), str(hz), _add_months(w, i)))

    triage: dict[tuple[str, str, str], dict[str, Any]] = {}
    if table_exists(con, "hs_triage"):
        for r in rows_as_dicts(
            con,
            "SELECT run_id, upper(iso3) AS iso3, upper(hazard_code) AS hazard_code, tier, "
            "triage_score, regime_change_level FROM hs_triage",
        ):
            triage[(str(r["run_id"]), r["iso3"], r["hazard_code"])] = r

    rows: list[dict[str, Any]] = []
    cset = set(countries)
    # facts_resolved can hold several rows for one country-month; one cell, one row.
    seen: set[tuple[str, str, str, str]] = set()

    def add(iso: str, hz: str, month: str, trigger: str, value: Any, source: str,
            base_month: str | None = None, base_value: Any = None,
            bucket: int | None = None, base_bucket: int | None = None) -> None:
        if iso not in cset or (iso, hz, month) in asked or (iso, hz, month, trigger) in seen:
            return
        seen.add((iso, hz, month, trigger))
        _ws, hs = epoch_for(month)
        t = triage.get((str(hs), iso, hz))
        rows.append({
            "iso3": iso, "hazard_code": hz, "month": month, "trigger": trigger, "value": value,
            "baseline_month": base_month, "baseline_value": base_value,
            "bucket": None if bucket is None else bucket + 1,
            "baseline_bucket": None if base_bucket is None else base_bucket + 1,
            "source": source, "hs_run_id": hs,
            "triage_tier": (t or {}).get("tier") if t else "not assessed",
            "triage_score": (t or {}).get("triage_score") if t else None,
            "rc_level": (t or {}).get("regime_change_level") if t else None,
        })

    # ACE: all-types deaths in bucket 5+ (100+), or two buckets above the last
    # complete month before the epoch.
    if table_exists(con, "acled_monthly_fatalities"):
        try:
            from pythia.tools.base_rate_spd import acled_complete_month_clause

            complete = acled_complete_month_clause(con)
        except Exception:  # noqa: BLE001
            complete = "TRUE"
        series: dict[tuple[str, str], float] = {}
        for iso, ym, v in con.execute(
            "SELECT upper(iso3), substr(CAST(month AS VARCHAR), 1, 7), SUM(fatalities) "
            f"FROM acled_monthly_fatalities WHERE {complete} GROUP BY 1, 2"
        ).fetchall():
            series[(str(iso), str(ym))] = float(v or 0.0)
        live = {ym for (_i, ym) in series}
        for month in scored_months:
            if month not in live:
                continue
            ws, _hs = epoch_for(month)
            base_month = _add_months(ws, -1) if ws else None
            for iso in countries:
                v = series.get((iso, month), 0.0)
                b = _bucket0("FATALITIES", v)
                bv = series.get((iso, base_month), 0.0) if base_month and base_month in live else None
                bb = _bucket0("FATALITIES", bv) if bv is not None else None
                if b is not None and b >= 4:
                    add(iso, "ACE", month, "deaths_bucket_5_plus", v, "acled_monthly_fatalities",
                        base_month, bv, b, bb)
                elif b is not None and bb is not None and b - bb >= 2:
                    add(iso, "ACE", month, "deaths_rose_2_buckets", v, "acled_monthly_fatalities",
                        base_month, bv, b, bb)
    if table_exists(con, "facts_resolved"):
        alert = "upper(alertlevel)" if column_exists(con, "facts_resolved", "alertlevel") else "NULL"
        for iso, hz, ym, v, lvl in con.execute(
            f"SELECT upper(iso3), upper(hazard_code), substr(CAST(ym AS VARCHAR), 1, 7), value, {alert} "
            "FROM facts_resolved WHERE lower(metric) = 'event_occurrence' "
            "AND upper(hazard_code) IN ('FL', 'DR', 'TC') AND value >= 1"
        ).fetchall():
            if str(ym) in scored_months and (lvl is None or str(lvl) in ("ORANGE", "RED")):
                add(str(iso), str(hz), str(ym), f"gdacs_{str(lvl or 'alert').lower()}", v, "facts_resolved:GDACS")
        p3: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for iso, ym, v in con.execute(
            "SELECT upper(iso3), substr(CAST(ym AS VARCHAR), 1, 7), MAX(value) FROM facts_resolved "
            "WHERE lower(metric) = 'phase3plus_in_need' AND value IS NOT NULL GROUP BY 1, 2 ORDER BY 2"
        ).fetchall():
            p3[str(iso)].append((str(ym), float(v)))
        for iso, series3 in p3.items():
            for i, (ym, v) in enumerate(series3):
                if ym not in scored_months or i == 0:
                    continue
                pym, pv = series3[i - 1]
                b, pb = _bucket0("PHASE3PLUS_IN_NEED", v), _bucket0("PHASE3PLUS_IN_NEED", pv)
                if b is not None and pb is not None and b > pb:
                    add(iso, "DR", ym, "ipc_phase3_rose_a_bucket", v,
                        "facts_resolved:phase3plus_in_need", pym, pv, b, pb)
    rows.sort(key=lambda r: (r["month"], r["hazard_code"], r["iso3"], r["trigger"]))
    return rows


# ---------------------------------------------------------------------------
# 5. experiments.csv: did a flag's arm do better, on comparable questions?
# ---------------------------------------------------------------------------

EXPERIMENT_COLUMNS = [
    "flag", "hazard_code", "metric", "score_family", "track", "score_type", "comparison",
    "arm", "reference_arm", "n_arm", "n_reference", "mean_arm", "mean_reference",
    "difference", "ci90_low", "ci90_high", "verdict",
]


def build_experiments(ctx: Context) -> list[dict[str, Any]]:
    """Per flag with more than one value in the data, within (hazard, metric, track).

    Arms of a flag hold DIFFERENT questions, so each question is first paired
    with climatology on its own horizons (primary aggregate minus
    ``__ext_climatology``, the excess score): the arms are compared on how
    far each fell from the base rate, which takes question difficulty out.
    Each arm is compared with the most common value. The ``correction`` flag
    is genuinely paired: a member's corrected forecast against its own
    uncorrected one on the same (question, horizon).
    """
    rows: list[dict[str, Any]] = []
    excess: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    bundle = set(ctx.bundle_qids)
    for (qid, model, h, st), v in ctx.scores.items():
        if qid not in bundle or st not in ("brier", "crps"):
            continue
        if model != primary_aggregate(ctx, qid):
            continue
        c = ctx.scores.get((qid, "__ext_climatology", h, st))
        if c is None:
            continue
        meta = ctx.qmeta[qid]
        excess[(meta["hazard_code"], meta["metric"], meta["track"], st)][qid].append(v - c)
    for (hz, metric, track, st), by_q in sorted(excess.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        for flag in EXPERIMENT_FLAGS:
            arms: dict[str, dict[str, list[float]]] = defaultdict(dict)
            for qid, vals in by_q.items():
                arms[str(ctx.qmeta[qid].get(flag))][qid] = vals
            if len(arms) < 2:
                continue
            ref = max(arms, key=lambda a: (len(arms[a]), a))
            for arm in sorted(a for a in arms if a != ref):
                d, lo, hi = _diff_ci(arms[arm], arms[ref])
                ma = _cluster_mean_ci(arms[arm], n_boot=0)[0]
                mr = _cluster_mean_ci(arms[ref], n_boot=0)[0]
                rows.append({
                    "flag": flag, "hazard_code": hz, "metric": metric, "score_family": _family(metric),
                    "track": track, "score_type": st,
                    "comparison": "excess over climatology, different questions per arm",
                    "arm": arm, "reference_arm": ref, "n_arm": len(arms[arm]),
                    "n_reference": len(arms[ref]), "mean_arm": _r(ma), "mean_reference": _r(mr),
                    "difference": _r(d), "ci90_low": _r(lo), "ci90_high": _r(hi),
                    "verdict": verdict(len(arms[arm]), len(arms[ref]), lo, hi, f"arm {arm}", f"arm {ref}"),
                })
    # Correction: corrected member against its own raw forecast, paired.
    paired: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (qid, model, h, st), v in ctx.scores.items():
        if qid not in bundle or not model.endswith("__raw") or st not in ("brier", "crps"):
            continue
        corrected = ctx.scores.get((qid, model[: -len("__raw")], h, st))
        if corrected is None:
            continue
        meta = ctx.qmeta[qid]
        paired[(meta["hazard_code"], meta["metric"], meta["track"], st)][qid].append(corrected - v)
    for (hz, metric, track, st), by_q in sorted(paired.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        m, lo, hi = _cluster_mean_ci(by_q)
        n = len(by_q)
        rows.append({
            "flag": "correction", "hazard_code": hz, "metric": metric, "score_family": _family(metric),
            "track": track, "score_type": st, "comparison": "paired: same member, question and horizon",
            "arm": "corrected", "reference_arm": "raw", "n_arm": n, "n_reference": n,
            "mean_arm": None, "mean_reference": None, "difference": _r(m),
            "ci90_low": _r(lo), "ci90_high": _r(hi),
            "verdict": verdict(n, n, lo, hi, "arm corrected", "arm raw"),
        })
    return rows


# ---------------------------------------------------------------------------
# 6. skill_history.csv: paired skill per observed month, across the DB
# ---------------------------------------------------------------------------

SKILL_HISTORY_COLUMNS = [
    "observed_month", "hazard_code", "metric", "score_family", "track", "lineup_id",
    "base_rate_block_version", "rc_guidance", "input_partial_month", "forecaster",
    "forecaster_model", "reference", "score_type", "n_pairs", "n_questions",
    "mean_forecaster", "mean_reference", "skill", "ci90_low", "ci90_high",
]


def _skill_pairs(ctx: Context, qids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Every paired (forecaster, reference) score, latest run, one row each.

    ``forecaster`` is ``primary`` for the question's primary aggregate, else
    the member's model name.
    """
    want = set(qids) if qids is not None else None
    out: list[dict[str, Any]] = []
    for (qid, model, h, st), v in ctx.scores.items():
        if want is not None and qid not in want:
            continue
        if model.startswith("__ext_") or model == "sibyl":
            continue
        primary = primary_aggregate(ctx, qid)
        if model == primary:
            role = "primary"
        elif _is_member(model) and model != "track2_flash":
            role = model
        else:
            continue
        meta = ctx.qmeta.get(qid)
        out_m = ctx.outcomes.get((qid, h)) or {}
        if meta is None:
            continue
        for ref in SKILL_REFERENCES:
            r = ctx.scores.get((qid, ref, h, st))
            if r is None:
                continue
            out.append({
                "question_id": qid, "horizon_m": h, "score_type": st, "forecaster": role,
                "forecaster_model": model, "reference": ref, "model_value": v, "reference_value": r,
                "observed_month": out_m.get("observed_month"),
                "hazard_code": meta["hazard_code"], "metric": meta["metric"],
                "score_family": _family(meta["metric"]), "track": meta["track"],
                "lineup_id": meta.get("lineup_id"),
                "base_rate_block_version": meta.get("base_rate_block_version"),
                "rc_guidance": meta.get("rc_guidance"),
                "input_partial_month": meta.get("input_partial_month"),
            })
    return out


def build_skill_history(ctx: Context) -> list[dict[str, Any]]:
    groups: dict[tuple, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    models: dict[tuple, set[str]] = defaultdict(set)
    for p in _skill_pairs(ctx):
        key = (p["observed_month"], p["hazard_code"], p["metric"], p["score_family"], p["track"],
               p["lineup_id"], p["base_rate_block_version"], p["rc_guidance"],
               p["input_partial_month"], p["forecaster"], p["reference"], p["score_type"])
        groups[key][p["question_id"]].append((p["model_value"], p["reference_value"]))
        models[key].add(p["forecaster_model"])
    rows = []
    for key, by_q in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        skill, lo, hi = _cluster_skill_ci(by_q, n_boot=HISTORY_BOOTSTRAP)
        pairs = [p for v in by_q.values() for p in v]
        rows.append({
            "observed_month": key[0], "hazard_code": key[1], "metric": key[2], "score_family": key[3],
            "track": key[4], "lineup_id": key[5], "base_rate_block_version": key[6],
            "rc_guidance": key[7], "input_partial_month": key[8], "forecaster": key[9],
            "forecaster_model": ",".join(sorted(models[key])), "reference": key[10],
            "score_type": key[11], "n_pairs": len(pairs), "n_questions": len(by_q),
            "mean_forecaster": _r(_mean([m for m, _ in pairs])),
            "mean_reference": _r(_mean([r for _, r in pairs])),
            "skill": _r(skill), "ci90_low": _r(lo), "ci90_high": _r(hi),
        })
    return rows


def history_digest_lines(ctx: Context, months: int = 6) -> list[str]:
    """The last six observed months of primary-vs-climatology Brier skill per group."""
    groups: dict[tuple, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for p in _skill_pairs(ctx):
        if p["forecaster"] != "primary" or p["reference"] != "__ext_climatology" or p["score_type"] != "brier":
            continue
        groups[(p["hazard_code"], p["metric"], p["track"], p["observed_month"])][p["question_id"]].append(
            (p["model_value"], p["reference_value"])
        )
    if not groups:
        return []
    keep = sorted({k[3] for k in groups if k[3]})[-months:]
    lines = [
        "", "## Skill history (last six observed months)", "",
        "_Primary aggregate against `__ext_climatology`, Brier, paired, latest run per "
        "question, pooled across lineups and prompt versions (`skill_history.csv` keeps "
        "them apart). 90% interval resamples questions._", "",
        "| hazard | metric | track | month | n questions | skill [90% CI] |", "|---|---|---|---|---|---|",
    ]
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        if key[3] not in keep:
            continue
        s, lo, hi = _cluster_skill_ci(groups[key], n_boot=HISTORY_BOOTSTRAP)
        ci = f"[{lo:+.2f}, {hi:+.2f}]" if lo is not None else "[—]"
        lines.append(
            f"| {key[0]} | {key[1]} | T{key[2]} | {key[3]} | {len(groups[key])} | "
            f"{'—' if s is None else f'{s:+.3f}'} {ci} |"
        )
    return lines


# ---------------------------------------------------------------------------
# 7. tail_outcomes.csv + binary_reliability.csv
# ---------------------------------------------------------------------------

TAIL_COLUMNS = [
    "question_id", "iso3", "hazard_code", "metric", "score_family", "track", "horizon_m",
    "observed_month", "outcome_value", "realized_bucket", "forecaster", "forecaster_kind",
    "p_realized",
]
RELIABILITY_COLUMNS = [
    "hazard_code", "track", "forecaster", "bin", "n", "mean_forecast", "observed_rate",
]
RELIABILITY_BINS = ((0.0, 0.05, "0-5%"), (0.05, 0.20, "5-20%"), (0.20, 0.50, "20-50%"),
                    (0.50, 0.80, "50-80%"), (0.80, 1.0000001, "80-100%"))


def _forecaster_kind(model: str) -> str:
    if model.startswith("__ext_"):
        return "reference"
    if model == "sibyl":
        return "sibyl"
    if model in AGGREGATES:
        return "aggregate"
    if model.endswith(("__raw", "__recal")):
        return "member_copy"
    return "member"


def _all_vectors(ctx: Context, qids: Sequence[str]) -> dict[tuple[str, int], dict[str, list[float]]]:
    """{(qid, h): {forecaster: dense vector}} for members, aggregates, Sibyl and references."""
    out: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(dict)
    for src in (_raw_vectors(ctx.con, ctx, qids), _ensemble_vectors(ctx.con, ctx, qids)):
        for (qid, model, h), by_b in src.items():
            metric = ctx.qmeta[qid]["metric"]
            k = 2 if _family(metric) == "binary" else _n_buckets(metric)
            vec = _dense(by_b, k)
            if vec:
                out[(qid, h)].setdefault(model, vec)
    for (qid, model, h), vec in _prov.reference_vectors(ctx.con, list(qids)).items():
        out[(qid, h)].setdefault(model, vec)
    return out


def build_tails(ctx: Context) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    qids = list(ctx.bundle_qids)
    vectors = _all_vectors(ctx, qids)
    tails: list[dict[str, Any]] = []
    rel: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: {"p": [], "y": []})
    for (qid, h), by_model in sorted(vectors.items()):
        meta = ctx.qmeta[qid]
        out = ctx.outcomes.get((qid, h))
        if not out or out.get("bucket0") is None:
            continue
        fam = _family(meta["metric"])
        if fam == "binary":
            for model, vec in by_model.items():
                p = vec[0]
                for lo, hi, label in RELIABILITY_BINS:
                    if lo <= p < hi:
                        cell = rel[(meta["hazard_code"], meta["track"], model, label)]
                        cell["p"].append(p)
                        cell["y"].append(float(out["event"]))
                        break
            is_tail = out.get("event") == 1
        else:
            k = _n_buckets(meta["metric"]) or 0
            is_tail = k > 0 and out["bucket0"] >= k - 2
        if not is_tail:
            continue
        j = out["bucket0"]
        for model, vec in sorted(by_model.items()):
            if j >= len(vec):
                continue
            tails.append({
                "question_id": qid, "iso3": meta["iso3"], "hazard_code": meta["hazard_code"],
                "metric": meta["metric"], "score_family": fam, "track": meta["track"],
                "horizon_m": h, "observed_month": out.get("observed_month"),
                "outcome_value": out["value"], "realized_bucket": j + 1,
                "forecaster": model, "forecaster_kind": _forecaster_kind(model),
                "p_realized": _r(vec[j]),
            })
    reliability = []
    for (hz, track, model, label), cell in sorted(rel.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        reliability.append({
            "hazard_code": hz, "track": track, "forecaster": model, "bin": label,
            "n": len(cell["p"]), "mean_forecast": _r(_mean(cell["p"])),
            "observed_rate": _r(_mean(cell["y"])),
        })
    return tails, reliability


# ---------------------------------------------------------------------------
# 8. base_rate_shown + inject_health.csv
# ---------------------------------------------------------------------------

INJECT_HEALTH_COLUMNS = [
    "question_id", "iso3", "hazard_code", "metric", "track", "inject", "present",
    "observed", "age_days", "age_months", "stale", "reason",
]


def base_rate_shown(con, meta: Mapping[str, Any], record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The structured figures behind the prompt's base-rate block.

    Conflict questions: each of the six months the trajectory printed (value,
    month, the row's ``updated_at`` and whether the row was rewritten after
    the forecast, so the value shown may differ from the value read here), the
    three-month means and trend computed by the same function the prompt
    uses, and the level-and-volatility vector when the prompt showed it.
    Other questions: the anchor ``forecast_deviation`` reconstructs.
    """
    from pythia.tools.base_rate_spd import conflict_trajectory

    hz = str(meta.get("hazard_code") or "").upper()
    fdate = meta.get("forecast_date")
    out: dict[str, Any] = {
        "forecast_date": str(fdate) if fdate else None,
        "input_partial_month": meta.get("input_partial_month"),
        "input_partial_month_basis": meta.get("input_partial_month_basis"),
    }
    if hz == "ACE":
        if not table_exists(con, "acled_monthly_fatalities"):
            out["acled"] = {"available": False, "reason": "acled_monthly_fatalities absent"}
        elif fdate is None:
            out["acled"] = {"available": False, "reason": "forecast date unknown"}
        else:
            cur = fdate.strftime("%Y-%m")
            upd = "MAX(updated_at)" if column_exists(con, "acled_monthly_fatalities", "updated_at") else "NULL"
            rows = con.execute(
                f"SELECT substr(CAST(month AS VARCHAR), 1, 7) AS ym, SUM(fatalities), {upd} "
                "FROM acled_monthly_fatalities WHERE upper(iso3) = ? "
                "AND substr(CAST(month AS VARCHAR), 1, 7) < ? GROUP BY 1 ORDER BY 1 DESC LIMIT 6",
                [str(meta.get("iso3") or "").upper(), cur],
            ).fetchall()
            rows = list(reversed(rows))
            traj = conflict_trajectory([(r[0], r[1]) for r in rows], "ACLED")
            out["acled"] = {
                "available": bool(rows),
                "source_table": "acled_monthly_fatalities",
                "months": [
                    {
                        "month": r[0], "value": r[1],
                        "updated_at": str(r[2]) if r[2] is not None else None,
                        "updated_after_forecast": (
                            _as_date(r[2]) is not None and _as_date(r[2]) > fdate
                        ) if r[2] is not None else None,
                    }
                    for r in rows
                ],
                "trailing_3m_avg": traj.get("trailing_3m_avg"),
                "prior_3m_avg": traj.get("prior_3m_avg"),
                "trend_pct": traj.get("trend_pct"),
                "trend_direction": traj.get("trend_direction"),
                "reconstructed_at_bundle_time": True,
            }
        if meta.get("base_rate_block_version") not in (None, "none"):
            refs = _prov.reference_vectors(con, [str(meta["question_id"])])
            vecs = {h: refs.get((str(meta["question_id"]), "__ext_level_volatility", h)) for h in range(1, 7)}
            out["level_volatility"] = {
                "shown": True, "block_version": meta.get("base_rate_block_version"),
                "source_table": "baseline_scored_forecasts",
                "shown_horizons": [1, 6],
                "spd_by_horizon": {str(h): v for h, v in vecs.items() if v},
            }
        else:
            out["level_volatility"] = {"shown": False}
    if table_exists(con, "forecast_deviation"):
        try:
            dev = rows_as_dicts(
                con,
                "SELECT model_name, baserate_source, baserate_json FROM forecast_deviation "
                "WHERE question_id = ? AND run_id = ? ORDER BY CASE model_name "
                "WHEN 'ensemble_bayesmc_v2' THEN 0 WHEN 'ensemble_mean_v2' THEN 1 "
                "WHEN 'track2_flash' THEN 2 ELSE 3 END LIMIT 1",
                [str(meta.get("question_id")), str(meta.get("run_id"))],
            )
        except Exception:  # noqa: BLE001
            dev = []
        if dev:
            payload = safe_json_loads(dev[0].get("baserate_json")) or {}
            out["anchor"] = {
                "available": True, "source": dev[0].get("baserate_source"),
                "source_table": "forecast_deviation",
                "probs": payload.get("probs") if isinstance(payload, dict) else None,
                "detail": payload.get("detail") if isinstance(payload, dict) else None,
            }
        else:
            out["anchor"] = {"available": False, "reason": "no deviation row for this run"}
    else:
        out["anchor"] = {"available": False, "reason": "forecast_deviation absent"}
    return out


def inject_health_rows(qid: str, meta: Mapping[str, Any], inject: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per inject the question's prompt could carry."""
    base = {"question_id": qid, "iso3": meta.get("iso3"), "hazard_code": meta.get("hazard_code"),
            "metric": meta.get("metric"), "track": meta.get("track")}
    rows: list[dict[str, Any]] = []

    def add(name: str, present: bool, observed: Any = None, age_days: Any = None,
            age_months: Any = None, stale: Any = None, reason: Any = None) -> None:
        rows.append({**base, "inject": name, "present": present, "observed": observed,
                     "age_days": age_days, "age_months": age_months, "stale": stale, "reason": reason})

    enso = inject.get("enso") or {}
    if enso.get("available"):
        age = enso.get("age_days")
        add("enso", True, enso.get("observation_date") or enso.get("fetch_date"), age_days=age,
            stale=(enso.get("status") == "carried_forward") or (isinstance(age, (int, float)) and age > 100))
    else:
        add("enso", False, reason=enso.get("reason") or "absent")
    gd = inject.get("gdacs_history") or {}
    if gd.get("applicable"):
        add("gdacs_history", bool(gd.get("available")), gd.get("window_end"),
            age_months=gd.get("total_months"), stale=False if gd.get("available") else None,
            reason=gd.get("reason"))
    cw = inject.get("crisiswatch") or {}
    if cw.get("applicable"):
        add("crisiswatch", bool(cw.get("available")), cw.get("edition"),
            age_months=cw.get("edition_age_months"), stale=cw.get("stale"),
            reason=cw.get("reason") or (f"arrow={cw.get('arrow')} alert={cw.get('alert')}"
                                        if cw.get("available") else None))
    for label in _prov.CONFLICT_FORECAST_SOURCES:
        cf = inject.get(label) or {}
        if cf.get("applicable"):
            add(label, bool(cf.get("available")), cf.get("vintage"), age_days=cf.get("age_days"),
                stale=cf.get("stale"), reason=cf.get("reason"))
    br = inject.get("base_rate") or {}
    add("base_rate", bool(br.get("available")), br.get("source"), reason=br.get("reason"))
    if str(meta.get("hazard_code") or "").upper() == "ACE":
        add("acled_trajectory", True, None, stale=bool(meta.get("input_partial_month")),
            reason=meta.get("input_partial_month_basis"))
    return rows


def inject_digest_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return []
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in rows:
        c = counts[(str(r["hazard_code"]), str(r["inject"]))]
        c["n"] += 1
        if not r["present"]:
            c["absent"] += 1
        elif r.get("stale"):
            c["stale"] += 1
    lines = [
        "", "## Inject health", "",
        "_Per hazard and inject: questions, how many prompts carried the inject absent, and "
        "how many carried it stale (`inject_health.csv` per question)._", "",
        "| hazard | inject | questions | absent | stale |", "|---|---|---|---|---|",
    ]
    for (hz, inj), c in sorted(counts.items()):
        lines.append(f"| {hz} | {inj} | {c['n']} | {c['absent']} | {c['stale']} |")
    cw = [r for r in rows if r["inject"] == "crisiswatch"]
    if cw:
        status = Counter(
            "absent" if not r["present"] else ("stale" if r.get("stale") else "current") for r in cw
        )
        lines += ["", f"**CrisisWatch for ACE questions:** {status['current']} current, "
                      f"{status['stale']} stale (three or more editions old), {status['absent']} absent."]
        absent = sorted({str(r["iso3"]) for r in cw if not r["present"]})
        if absent:
            lines.append(f"Absent for: {', '.join(absent)}.")
    return lines


# ---------------------------------------------------------------------------
# 9. headline.json
# ---------------------------------------------------------------------------


def build_headline(ctx: Context) -> dict[str, Any]:
    """Per (hazard, metric, track): the primary aggregate against each reference.

    Paired on the (question, horizon) pairs both scored, latest run, within
    one track. Skill against climatology and against level-and-volatility
    carries a seeded 90% bootstrap interval (4000 resamples of questions);
    a win is a question whose primary mean score is below the reference's on
    the paired horizons. ``warning`` is set below ten paired questions.
    """
    bundle = set(ctx.bundle_qids)
    groups: dict[tuple, dict[str, Any]] = {}
    for (qid, model, h, st), v in ctx.scores.items():
        if qid not in bundle or model != primary_aggregate(ctx, qid):
            continue
        meta = ctx.qmeta[qid]
        if st not in ("brier", "crps") or (st == "crps" and _family(meta["metric"]) == "binary"):
            continue
        g = groups.setdefault((meta["hazard_code"], meta["metric"], meta["track"]), {})
        g.setdefault(st, []).append((qid, h, model, v))
    out_groups = []
    for (hz, metric, track), by_st in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        entry: dict[str, Any] = {"hazard_code": hz, "metric": metric, "score_family": _family(metric),
                                 "track": track, "scores": {}}
        for st, items in by_st.items():
            clim_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for qid, h, _m, v in items:
                c = ctx.scores.get((qid, "__ext_climatology", h, st))
                if c is not None:
                    clim_pairs[qid].append((v, c))
            n_q = len(clim_pairs)
            block: dict[str, Any] = {
                "n_paired_questions": n_q,
                "n_paired_pairs": sum(len(v) for v in clim_pairs.values()),
                "primary_models": sorted({m for _q, _h, m, _v in items}),
                "primary_mean": _r(_mean([v for vals in clim_pairs.values() for v, _ in vals]), 6),
                "references": {},
                "warning": "fewer than 10 paired questions" if n_q < MIN_N else None,
            }
            for ref in REFERENCES:
                pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
                for qid, h, _m, v in items:
                    r = ctx.scores.get((qid, ref, h, st))
                    if r is not None:
                        pairs[qid].append((v, r))
                if not pairs:
                    continue
                ref_block: dict[str, Any] = {
                    "n_paired_questions": len(pairs),
                    "reference_mean": _r(_mean([r for vals in pairs.values() for _, r in vals]), 6),
                    "primary_mean_on_pairs": _r(_mean([m for vals in pairs.values() for m, _ in vals]), 6),
                    "wins": sum(1 for vals in pairs.values()
                                if _mean([m for m, _ in vals]) < _mean([r for _, r in vals])),
                    "losses": sum(1 for vals in pairs.values()
                                  if _mean([m for m, _ in vals]) > _mean([r for _, r in vals])),
                }
                if ref in ("__ext_climatology", "__ext_level_volatility"):
                    s, lo, hi = _cluster_skill_ci(pairs)
                    ref_block.update({"skill": _r(s), "skill_ci90_low": _r(lo), "skill_ci90_high": _r(hi)})
                block["references"][ref] = ref_block
            entry["scores"][st] = block
        out_groups.append(entry)
    by_h: Counter = Counter()
    for (qid, h), o in ctx.outcomes.items():
        if qid in bundle and o.get("value") is not None:
            by_h[h] += 1
    return {
        "method": (
            "paired (question, horizon) scores, latest run per question, within one track; "
            "skill = 1 - sum(primary)/sum(reference); 90% interval from 4000 seeded "
            "resamples of questions; primary = ensemble_bayesmc_v2, else ensemble_mean_v2, "
            "else track2_flash"
        ),
        "seed": SEED, "bootstrap": BOOTSTRAP, "min_n": MIN_N,
        "questions_resolved_per_horizon": {str(h): n for h, n in sorted(by_h.items())},
        "groups": out_groups,
    }


def headline_digest_lines(headline: Mapping[str, Any]) -> list[str]:
    """The digest's first table, generated from headline.json and nothing else."""
    if not headline or headline.get("stub"):
        return ["## Headline", "", f"_headline.json is a stub: {headline.get('reason') if headline else 'absent'}_"]
    lines = [
        "## Headline (from `headline.json`)", "",
        "_Primary aggregate (bayesmc, else mean, else track2_flash), paired with each "
        "reference on the same (question, horizon), latest run, per track. Skill is "
        "1 − primary/reference; brackets are the 90% interval over 4000 resamples of "
        "questions. ⚠ marks fewer than 10 paired questions._", "",
        "| hazard | metric | track | score | n q | primary | climatology | skill vs clim [90%] "
        "| level-vol | skill vs level-vol [90%] | wins vs clim |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def num(v: Any) -> str:
        return "—" if v is None else f"{v:.4f}"

    def ci(b: Mapping[str, Any]) -> str:
        if b.get("skill") is None:
            return "—"
        if b.get("skill_ci90_low") is None:
            return f"{b['skill']:+.3f} [—]"
        return f"{b['skill']:+.3f} [{b['skill_ci90_low']:+.2f}, {b['skill_ci90_high']:+.2f}]"

    for g in headline.get("groups") or []:
        for st, b in sorted((g.get("scores") or {}).items()):
            refs = b.get("references") or {}
            c = refs.get("__ext_climatology") or {}
            lv = refs.get("__ext_level_volatility") or {}
            warn = " ⚠" if b.get("warning") else ""
            lines.append(
                f"| {g['hazard_code']} | {g['metric']} | T{g['track']} | {'RPS' if st == 'crps' else 'Brier'} "
                f"| {b['n_paired_questions']}{warn} | {num(b.get('primary_mean'))} "
                f"| {num(c.get('reference_mean'))} | {ci(c)} | {num(lv.get('reference_mean'))} | {ci(lv)} "
                f"| {c.get('wins', '—')}/{c.get('n_paired_questions', '—')} |"
            )
    per_h = headline.get("questions_resolved_per_horizon") or {}
    if per_h:
        lines += ["", "Questions resolved per horizon: "
                  + ", ".join(f"h{h} {n}" for h, n in sorted(per_h.items(), key=lambda kv: int(kv[0]))) + "."]
    return lines


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _section(result: SectionResult, out_dir: Path, name: str, fn: Callable[[], int]) -> None:
    try:
        n = fn()
        result.ok(name, n)
    except Exception as exc:  # noqa: BLE001 - a section never costs the bundle
        LOGGER.warning("%s skipped: %s", name, exc)
        result.stub(out_dir, name, f"{type(exc).__name__}: {exc}")


def emit_all(ctx: Context | None, out_dir: Path, *, ctx_error: str | None = None) -> tuple[SectionResult, dict[str, Any]]:
    """Write every error-attribution file into ``out_dir``.

    Returns the section statuses (for the manifest) and the pieces the digest
    renders. A missing context stubs every file with the reason.
    """
    result = SectionResult()
    digest: dict[str, Any] = {"headline": None, "history": [], "inject": [], "unasked": []}
    names = [
        "trace_stages.csv", "trace_stages_summary.csv", "update_value.csv",
        "update_value_summary.csv", "rc_outcomes.csv", "rc_outcomes_summary.csv",
        "unasked_outcomes.csv", "experiments.csv", "skill_history.csv",
        "tail_outcomes.csv", "binary_reliability.csv", "headline.json",
    ]
    if ctx is None:
        for n in names:
            result.stub(out_dir, n, ctx_error or "context unavailable")
        digest["headline"] = {"stub": True, "reason": ctx_error}
        return result, digest

    def need(*tables: str) -> None:
        missing = [t for t in tables if not table_exists(ctx.con, t)]
        if missing:
            raise LookupError(f"table(s) absent: {', '.join(missing)}")

    def trace_stages() -> int:
        need("forecasts_raw", "resolutions")
        rows = build_trace_stages(ctx)
        summary = summarise_trace_stages(rows)
        write_csv(out_dir / "trace_stages_summary.csv", TRACE_STAGE_SUMMARY_COLUMNS, summary)
        result.ok("trace_stages_summary.csv", len(summary))
        digest["trace_summary"] = summary
        return write_csv(out_dir / "trace_stages.csv", TRACE_STAGE_COLUMNS, rows)

    def update_value() -> int:
        need("forecasts_raw", "resolutions")
        rows = build_update_value(ctx)
        summary = summarise_update_value(rows)
        write_csv(out_dir / "update_value_summary.csv", UPDATE_SUMMARY_COLUMNS, summary)
        result.ok("update_value_summary.csv", len(summary))
        digest["update_summary"] = summary
        return write_csv(out_dir / "update_value.csv", UPDATE_VALUE_COLUMNS, rows)

    def rc_outcomes() -> int:
        need("resolutions", "hs_triage")
        rows = build_rc_outcomes(ctx)
        summary = summarise_rc_outcomes(rows)
        write_csv(out_dir / "rc_outcomes_summary.csv", RC_SUMMARY_COLUMNS, summary)
        result.ok("rc_outcomes_summary.csv", len(summary))
        return write_csv(out_dir / "rc_outcomes.csv", RC_OUTCOME_COLUMNS, rows)

    def unasked() -> int:
        need("resolutions")
        rows = build_unasked_outcomes(ctx)
        digest["unasked"] = rows
        return write_csv(out_dir / "unasked_outcomes.csv", UNASKED_COLUMNS, rows)

    def experiments() -> int:
        return write_csv(out_dir / "experiments.csv", EXPERIMENT_COLUMNS, build_experiments(ctx))

    def history() -> int:
        need("scores")
        digest["history"] = history_digest_lines(ctx)
        return write_csv(out_dir / "skill_history.csv", SKILL_HISTORY_COLUMNS, build_skill_history(ctx))

    def tails() -> int:
        need("resolutions")
        tail_rows, reliability = build_tails(ctx)
        write_csv(out_dir / "binary_reliability.csv", RELIABILITY_COLUMNS, reliability)
        result.ok("binary_reliability.csv", len(reliability))
        return write_csv(out_dir / "tail_outcomes.csv", TAIL_COLUMNS, tail_rows)

    def headline() -> int:
        need("scores")
        h = build_headline(ctx)
        write_json(out_dir / "headline.json", h)
        digest["headline"] = h
        return len(h.get("groups") or [])

    for name, fn, companion in (
        ("trace_stages.csv", trace_stages, "trace_stages_summary.csv"),
        ("update_value.csv", update_value, "update_value_summary.csv"),
        ("rc_outcomes.csv", rc_outcomes, "rc_outcomes_summary.csv"),
        ("unasked_outcomes.csv", unasked, None),
        ("experiments.csv", experiments, None),
        ("skill_history.csv", history, None),
        ("tail_outcomes.csv", tails, "binary_reliability.csv"),
        ("headline.json", headline, None),
    ):
        _section(result, out_dir, name, fn)
        if companion and result.files.get(name, {}).get("status") == "stub":
            result.stub(out_dir, companion, result.files[name]["reason"])
    if digest["headline"] is None:
        digest["headline"] = {"stub": True, "reason": (result.files.get("headline.json") or {}).get("reason")}
    return result, digest


__all__ = [
    "ACLED_COMPLETE_MONTH_FIX",
    "Context",
    "EXPERIMENT_FLAGS",
    "ROLLUP_SPLIT_KEYS",
    "SectionResult",
    "base_rate_shown",
    "build_context",
    "build_experiments",
    "build_headline",
    "build_rc_outcomes",
    "build_skill_history",
    "build_tails",
    "build_trace_stages",
    "build_unasked_outcomes",
    "build_update_value",
    "correction_label",
    "emit_all",
    "headline_digest_lines",
    "history_digest_lines",
    "inject_digest_lines",
    "inject_health_rows",
    "input_partial_month",
    "last_acled_ingest_on_or_before",
    "months_affected",
    "write_stub",
]
