# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Mechanical backtests of Sibyl's references (Oct 2026, review Part 7).

Sibyl's reference is half of what it publishes, and only the conflict one
rests on a backtest (Brier 0.390 / 0.384 / 0.470 for the 12-month shares, the
75/25 pool and level-and-volatility, on 8,371 country-forecasts; the script
that produced them was a one-off and is not in the repository). The drought
weight (0.5) and the flood and cyclone choice (per calendar month) rest on
nothing. This module replays each candidate over history at production
timing and scores it, read-only, so the choices can be made on evidence.

Production timing: a forecast is made on the 13th of month M for the six
months M+1 .. M+6. A candidate sees only what was knowable on that day:

* conflict: the ACLED level month and history as ``base_rate_spd`` reads
  them with ``known_at`` = the 13th (complete months, settled
  ``ACLED_SETTLE_DAYS`` before);
* drought: Phase 3+ rows for months up to M - L, for an assumed publication
  lag L of 1, 2 and 3 months, every result reported for each L;
* flood and cyclone: the history ``base_rate_spd`` reads before the window.

A target month with no record is unresolved and left out (a quiet ACLED
month that ACLED was live for counts as zero, as the resolver reads it).

Candidates:

* conflict: the 12-month shares, level-transition, level-and-volatility, and
  pools of the 12-month shares with level-transition at 0.5, 0.75 (the
  production reference) and 0.9;
* drought: persistence weights 0, 0.25, 0.5 (production), 0.75, 0.9 and 1.0,
  the same at every horizon; and a schedule fitted per horizon on forecasts
  made up to ``fit_end`` (default 2023-12), forced non-increasing across
  horizons, and scored on the forecasts made after it;
* flood and cyclone: the per-calendar-month vector (production), the pooled
  vector and a uniform one; "too few" below 100 forecasts.

Every vector is floored at ``SIBYL_BUCKET_FLOOR`` and renormalised before it
is scored, so the log score is finite and every candidate is treated alike.
Brier, RPS and log scores per horizon and over all horizons; 90% intervals
from 2,000 country resamples with a fixed seed, for each mean and for each
candidate's paired difference from the production reference.

CLI: ``python -m sibyl.reference_backtest --db <duckdb> --out-dir <dir>``
writes ``reference_backtest.md``, ``reference_backtest.csv`` and
``manifest.json``. It changes no production weight.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sibyl import config as _cfg

logger = logging.getLogger(__name__)

HORIZONS = (1, 2, 3, 4, 5, 6)
SCORE_TYPES = ("brier", "rps", "log")
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261009
INTERVAL = (0.05, 0.95)
DR_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
CONFLICT_POOL_WEIGHTS = (0.5, 0.75, 0.9)
PA_MIN_FORECASTS = 100
FORECAST_DAY = 13
LAGS = (1, 2, 3)

# The reference the paired differences are taken from, per class.
PRODUCTION = {"ACE": "pool_0.75", "DR": "persistence_w0.5", "FL": "per_month", "TC": "per_month"}

# The figures quoted for the conflict reference (12-month shares, 75/25 pool,
# level-and-volatility; Brier over all horizons; 8,371 country-forecasts).
QUOTED_CONFLICT = {"conflictology12": 0.390, "pool_0.75": 0.384, "level_volatility": 0.470}
QUOTED_N_FORECASTS = 8371


# --- small helpers ---------------------------------------------------------------

def _ym_add(ym: str, k: int) -> str:
    y, m = int(ym[:4]), int(ym[5:7])
    i = y * 12 + (m - 1) + k
    return f"{i // 12:04d}-{i % 12 + 1:02d}"


def months_between(start: str, end: str) -> List[str]:
    out, ym = [], start
    while ym <= end:
        out.append(ym)
        ym = _ym_add(ym, 1)
    return out


def _floor(vec: Sequence[float]) -> List[float]:
    f = float(_cfg.BUCKET_FLOOR)
    v = [max(float(x), f) for x in vec]
    z = sum(v)
    return [x / z for x in v]


def _norm(vec: Sequence[float]) -> List[float]:
    z = sum(float(x) for x in vec)
    return [float(x) / z for x in vec] if z > 0 else list(vec)


def score_vector(vec: Sequence[float], j: int) -> Dict[str, float]:
    from pythia.tools.compute_scores import _brier, _log_score, _rps  # noqa: PLC0415

    p = _floor(vec)
    return {"brier": _brier(p, j), "rps": _rps(p, j), "log": _log_score(p, j)}


def bucket_of(value: float, metric: str) -> Optional[int]:
    from pythia.tools.base_rate_spd import _bucket_index_for_value  # noqa: PLC0415

    return _bucket_index_for_value(float(value), metric)


@contextmanager
def _cached_acled_series():
    """Memoise the all-country ACLED read the conflict references repeat for
    every country: the same (first, last) window is read once."""
    from pythia.tools import base_rate_spd as brs  # noqa: PLC0415

    original = brs._acled_series_all
    cache: Dict[Tuple[str, str], Any] = {}

    def cached(con, first_ym, last_ym):
        key = (first_ym, last_ym)
        if key not in cache:
            cache[key] = original(con, first_ym, last_ym)
        return cache[key]

    brs._acled_series_all = cached
    try:
        yield
    finally:
        brs._acled_series_all = original


# --- the score store ---------------------------------------------------------------

@dataclass
class Scores:
    """Per (class, lag, candidate): every scored forecast-horizon."""

    rows: Dict[Tuple[str, Optional[int], str], List[Dict[str, Any]]] = field(default_factory=dict)

    def add(self, cls: str, lag: Optional[int], candidate: str, *, iso3: str, month: str,
            horizon: int, scores: Dict[str, float]) -> None:
        self.rows.setdefault((cls, lag, candidate), []).append(
            {"iso3": iso3, "month": month, "h": horizon, **scores})


def _cluster_mean(values: np.ndarray, groups: np.ndarray, rng: np.random.Generator,
                  draws: int) -> Tuple[float, float]:
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq)).astype(float)
    idx = rng.integers(0, len(uniq), size=(draws, len(uniq)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(boot, INTERVAL)
    return float(lo), float(hi)


def summarise(scores: Scores, *, draws: int = BOOTSTRAP_DRAWS,
              seed: int = BOOTSTRAP_SEED) -> List[Dict[str, Any]]:
    """One row per (class, lag, candidate, horizon or 'all', score type)."""
    out: List[Dict[str, Any]] = []
    keys = sorted(scores.rows, key=lambda k: (k[0], k[1] or 0, k[2]))
    for k_i, (cls, lag, cand) in enumerate(keys):
        rows = scores.rows[(cls, lag, cand)]
        prod_rows = scores.rows.get((cls, lag, PRODUCTION.get(cls, "")), [])
        prod_by = {(r["iso3"], r["month"], r["h"]): r for r in prod_rows}
        for h in list(HORIZONS) + ["all"]:
            sel = [r for r in rows if h == "all" or r["h"] == h]
            for s_i, st in enumerate(SCORE_TYPES):
                row: Dict[str, Any] = {
                    "class": cls, "lag": lag, "candidate": cand, "horizon": h,
                    "score_type": st, "n_forecasts": len(sel),
                    "n_country_forecasts": len({(r["iso3"], r["month"]) for r in sel}),
                    "n_countries": len({r["iso3"] for r in sel}),
                }
                too_few = cls in ("FL", "TC") and len(sel) < PA_MIN_FORECASTS
                if not sel or too_few:
                    row["status"] = "too_few" if too_few else "none"
                    out.append(row)
                    continue
                rng = np.random.default_rng(seed + 97 * k_i + 7 * s_i
                                            + (0 if h == "all" else int(h)))
                vals = np.array([r[st] for r in sel], dtype=float)
                groups = np.array([r["iso3"] for r in sel])
                lo, hi = _cluster_mean(vals, groups, rng, draws)
                row.update({"status": "ok", "mean": float(vals.mean()), "lo": lo, "hi": hi})
                paired = [(r[st] - prod_by[(r["iso3"], r["month"], r["h"])][st], r["iso3"])
                          for r in sel if (r["iso3"], r["month"], r["h"]) in prod_by]
                if paired and cand != PRODUCTION.get(cls):
                    d = np.array([p for p, _ in paired])
                    g = np.array([c for _, c in paired])
                    dlo, dhi = _cluster_mean(d, g, rng, draws)
                    row.update({"diff_vs_production": float(d.mean()), "diff_lo": dlo,
                                "diff_hi": dhi, "n_paired": len(paired)})
                out.append(row)
    return out


# --- conflict -------------------------------------------------------------------

def _acled_outcomes(con, first: str, last: str) -> Callable[[str, str], Optional[float]]:
    from pythia.tools.base_rate_spd import _acled_series_all  # noqa: PLC0415

    by_iso, live = _acled_series_all(con, first, last)

    def outcome(iso3: str, ym: str) -> Optional[float]:
        v = by_iso.get(iso3, {}).get(ym)
        if v is not None:
            return float(v)
        return 0.0 if (ym in live and iso3 in by_iso) else None

    return outcome


def backtest_conflict(con, forecast_months: Sequence[str], scores: Scores,
                      countries: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    from pythia.tools import base_rate_spd as brs  # noqa: PLC0415

    if not brs._table_exists(con, brs.CONFLICT_FATALITIES_TABLE):
        return {"status": "no_table"}
    isos = list(countries or [r[0] for r in con.execute(
        "SELECT DISTINCT upper(iso3) FROM acled_monthly_fatalities WHERE iso3 IS NOT NULL "
        "ORDER BY 1").fetchall()])
    outcome = _acled_outcomes(con, forecast_months[0], _ym_add(forecast_months[-1], 6))
    n = 0
    with _cached_acled_series():
        for m in forecast_months:
            known = date(int(m[:4]), int(m[5:7]), FORECAST_DAY)
            window = _ym_add(m, 1)
            for iso in isos:
                c12, _s, _d = brs.conflictology_spds(con, iso, window, HORIZONS, known)
                if not c12:
                    continue
                lt, _s, _d = brs.level_transition_spds(con, iso, window, HORIZONS, known)
                lv, _s, _d = brs.level_volatility_spds(con, iso, window, HORIZONS, known)
                any_scored = False
                for h in HORIZONS:
                    y = outcome(iso, _ym_add(m, h))
                    if y is None:
                        continue
                    j = bucket_of(y, "FATALITIES")
                    if j is None:
                        continue
                    any_scored = True
                    cands = {"conflictology12": c12[h]}
                    if lt.get(h):
                        cands["level_transition"] = lt[h]
                    if lv.get(h):
                        cands["level_volatility"] = lv[h]
                    for w in CONFLICT_POOL_WEIGHTS:
                        b = lt.get(h)
                        vec = (_norm([w * x + (1 - w) * z for x, z in zip(c12[h], b)])
                               if b and len(b) == len(c12[h]) else c12[h])
                        cands[f"pool_{w:g}"] = vec
                    for name, vec in cands.items():
                        scores.add("ACE", None, name, iso3=iso, month=m, horizon=h,
                                   scores=score_vector(vec, j))
                n += int(any_scored)
    return {"status": "ok", "n_country_forecasts": n, "n_countries": len(isos)}


# --- drought --------------------------------------------------------------------

def _phase3_outcomes(con) -> Dict[Tuple[str, str], float]:
    rows = con.execute(
        "SELECT upper(iso3), substr(CAST(ym AS VARCHAR), 1, 7), MAX(value) FROM facts_resolved "
        "WHERE hazard_code = 'DR' AND lower(metric) = 'phase3plus_in_need' AND value IS NOT NULL "
        "GROUP BY 1, 2").fetchall()
    return {(str(i), str(ym)): float(v) for i, ym, v in rows}


def _isotonic_non_increasing(values: Sequence[float]) -> List[float]:
    """Pool adjacent violators: the closest non-increasing sequence."""
    blocks: List[List[float]] = []  # [mean, count]
    for v in values:
        blocks.append([float(v), 1.0])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            m2, c2 = blocks.pop()
            m1, c1 = blocks.pop()
            blocks.append([(m1 * c1 + m2 * c2) / (c1 + c2), c1 + c2])
    out: List[float] = []
    for m, c in blocks:
        out.extend([m] * int(c))
    return out


def backtest_drought(con, forecast_months: Sequence[str], scores: Scores, *,
                     lags: Sequence[int] = LAGS, fit_end: str = "2023-12",
                     countries: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    from pythia.tools import base_rate_spd as brs  # noqa: PLC0415
    from pythia.tools.score_baselines import persistence_spd  # noqa: PLC0415

    if not brs._table_exists(con, "facts_resolved"):
        return {"status": "no_table"}
    outcomes = _phase3_outcomes(con)
    isos = list(countries or sorted({i for i, _ in outcomes}))
    report: Dict[str, Any] = {"status": "ok", "lags": {}}
    for lag in lags:
        # (iso, month, h, j, pers, hist) for the fit and the test.
        cases: List[Tuple[str, str, int, int, List[float], List[float]]] = []
        for m in forecast_months:
            before = _ym_add(m, 1 - int(lag))  # rows for months up to M - L
            for iso in isos:
                hist, _s, _d = brs._phase3_history(con, iso, before)
                last = brs.last_observed_value(con, iso, "DR", "PHASE3PLUS_IN_NEED", before)
                pers = persistence_spd(last[0], "PHASE3PLUS_IN_NEED") if last else None
                if not hist or not pers:
                    continue  # the weight only matters where both exist
                for h in HORIZONS:
                    y = outcomes.get((iso, _ym_add(m, h)))
                    if y is None:
                        continue
                    j = bucket_of(y, "PHASE3PLUS_IN_NEED")
                    if j is None:
                        continue
                    cases.append((iso, m, h, j, list(pers), list(hist)))
        for iso, m, h, j, pers, hist in cases:
            for w in DR_WEIGHTS:
                vec = _norm([w * a + (1 - w) * b for a, b in zip(pers, hist)])
                scores.add("DR", lag, f"persistence_w{w:g}", iso3=iso, month=m, horizon=h,
                           scores=score_vector(vec, j))
        # The fitted schedule: per horizon the grid weight with the lowest
        # mean Brier on forecasts made up to fit_end, forced non-increasing.
        fit = [c for c in cases if c[1] <= fit_end]
        test = [c for c in cases if c[1] > fit_end]
        chosen: List[float] = []
        for h in HORIZONS:
            sel = [c for c in fit if c[2] == h]
            if not sel:
                chosen.append(float(_cfg.DR_PERSISTENCE_WEIGHT))
                continue
            best = min(DR_WEIGHTS, key=lambda w: np.mean([
                score_vector(_norm([w * a + (1 - w) * b for a, b in zip(c[4], c[5])]), c[3])["brier"]
                for c in sel]))
            chosen.append(float(best))
        schedule = _isotonic_non_increasing(chosen)
        for iso, m, h, j, pers, hist in test:
            w = schedule[h - 1]
            vec = _norm([w * a + (1 - w) * b for a, b in zip(pers, hist)])
            scores.add("DR", lag, "fitted_schedule", iso3=iso, month=m, horizon=h,
                       scores=score_vector(vec, j))
            for w0 in DR_WEIGHTS:
                v0 = _norm([w0 * a + (1 - w0) * b for a, b in zip(pers, hist)])
                scores.add("DR", lag, f"test_period_w{w0:g}", iso3=iso, month=m, horizon=h,
                           scores=score_vector(v0, j))
        report["lags"][str(lag)] = {
            "n_forecast_horizons": len(cases), "n_fit": len(fit), "n_test": len(test),
            "per_horizon_best_on_fit": chosen, "schedule": [round(x, 4) for x in schedule],
            "fit_end": fit_end,
        }
    return report


# --- flood and cyclone ------------------------------------------------------------

def _pa_outcomes(con, hazard: str) -> Dict[Tuple[str, str], float]:
    from pythia.tools.base_rate_spd import _pa_metric_in_clause  # noqa: PLC0415

    rows = con.execute(
        f"SELECT upper(iso3), substr(CAST(ym AS VARCHAR), 1, 7), MAX(value) FROM facts_resolved "
        f"WHERE hazard_code = ? AND {_pa_metric_in_clause()} AND value IS NOT NULL "
        "GROUP BY 1, 2", [hazard]).fetchall()
    return {(str(i), str(ym)): float(v) for i, ym, v in rows}


def backtest_pa(con, forecast_months: Sequence[str], scores: Scores, hazard: str,
                countries: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    from pythia.buckets import n_buckets_for  # noqa: PLC0415
    from pythia.tools import base_rate_spd as brs  # noqa: PLC0415

    if not brs._table_exists(con, "facts_resolved"):
        return {"status": "no_table"}
    outcomes = _pa_outcomes(con, hazard)
    isos = list(countries or sorted({i for i, _ in outcomes}))
    k = n_buckets_for("PA")
    n = 0
    for m in forecast_months:
        window = _ym_add(m, 1)
        for iso in isos:
            probs, _src, detail = brs.base_rate_spd(con, iso, hazard, "PA", window)
            if not probs:
                continue
            pbm = (detail or {}).get("probs_by_month") or {}
            any_scored = False
            for h in HORIZONS:
                target = _ym_add(m, h)
                y = outcomes.get((iso, target))
                if y is None:
                    continue  # no record: unresolved
                j = bucket_of(y, "PA")
                if j is None:
                    continue
                any_scored = True
                cands = {"per_month": pbm.get(target) or probs, "pooled": probs,
                         "uniform": [1.0 / k] * k}
                for name, vec in cands.items():
                    scores.add(hazard, None, name, iso3=iso, month=m, horizon=h,
                               scores=score_vector(vec, j))
            n += int(any_scored)
    return {"status": "ok", "n_country_forecasts": n, "n_countries": len(isos)}


# --- the report -------------------------------------------------------------------

def _fmt(x: Any) -> str:
    return "—" if x is None else f"{x:.4f}"


def render_markdown(rows: Sequence[Dict[str, Any]], manifest: Dict[str, Any]) -> str:
    lines = ["# Sibyl reference backtest", "",
             f"Built {manifest['built_at']} from `{manifest['db']}`. Forecasts on the "
             f"{FORECAST_DAY}th of each month from {manifest['start']} to {manifest['end']} for "
             "the six months after; a target month with no record is left out. Every vector "
             f"floored at {_cfg.BUCKET_FLOOR} before scoring. Intervals: 90%, "
             f"{BOOTSTRAP_DRAWS} country resamples, seed {BOOTSTRAP_SEED}. "
             "Differences are candidate minus the production reference: negative is better.", ""]
    rc = manifest.get("reproduction") or {}
    if rc:
        lines += ["## Reproducing the quoted conflict figures", "",
                  "| candidate | quoted Brier | this run | country-forecasts |",
                  "|---|---|---|---|"]
        for cand, quoted in QUOTED_CONFLICT.items():
            got = rc.get(cand)
            lines.append(f"| {cand} | {quoted:.3f} | {_fmt(got)} | "
                         f"{rc.get('n_country_forecasts', '—')} (quoted {QUOTED_N_FORECASTS:,}) |")
        lines.append("")
    groups: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
    for r in rows:
        if r["score_type"] == "brier":
            groups.setdefault((r["class"], r["lag"]), []).append(r)
    for (cls, lag), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        title = f"## {cls}" + (f", assumed publication lag {lag} month(s)" if lag else "")
        lines += [title, "", f"Brier by horizon (production: `{PRODUCTION.get(cls)}`).", "",
                  "| candidate | h | n | mean [90%] | minus production [90%] |",
                  "|---|---|---|---|---|"]
        for r in sorted(rs, key=lambda r: (r["candidate"], str(r["horizon"]))):
            if r.get("status") != "ok":
                lines.append(f"| {r['candidate']} | {r['horizon']} | {r['n_forecasts']} | "
                             f"{r.get('status')} | |")
                continue
            diff = (f"{_fmt(r.get('diff_vs_production'))} [{_fmt(r.get('diff_lo'))}, "
                    f"{_fmt(r.get('diff_hi'))}]" if "diff_vs_production" in r else "")
            lines.append(f"| {r['candidate']} | {r['horizon']} | {r['n_forecasts']} | "
                         f"{_fmt(r['mean'])} [{_fmt(r['lo'])}, {_fmt(r['hi'])}] | {diff} |")
        lines.append("")
    dr = (manifest.get("sections") or {}).get("DR") or {}
    if dr.get("lags"):
        lines += ["## Drought: the fitted persistence schedule", "",
                  "Fitted per horizon on forecasts made up to the fit date, forced "
                  "non-increasing, scored on the forecasts after it (`fitted_schedule` against "
                  "`test_period_w*` on the same forecasts).", "",
                  "| lag | fit / test | best on fit, h1..h6 | schedule |", "|---|---|---|---|"]
        for lag, info in dr["lags"].items():
            lines.append(f"| {lag} | {info['n_fit']} / {info['n_test']} | "
                         f"{info['per_horizon_best_on_fit']} | {info['schedule']} |")
        lines.append("")
    lines.append("Nothing here changes a production weight.")
    return "\n".join(lines) + "\n"


def run_backtest(con, *, start: str = "2021-03", end: str = "2025-06",
                 fit_end: str = "2023-12", classes: Sequence[str] = ("ACE", "DR", "FL", "TC"),
                 lags: Sequence[int] = LAGS, db_label: str = "",
                 countries: Optional[Sequence[str]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run every class asked for; return (rows, manifest). Never writes the DB."""
    months = months_between(start, end)
    scores = Scores()
    sections: Dict[str, Any] = {}
    timings: Dict[str, float] = {}
    for cls in classes:
        t0 = time.monotonic()
        try:
            if cls == "ACE":
                sections[cls] = backtest_conflict(con, months, scores, countries)
            elif cls == "DR":
                sections[cls] = backtest_drought(con, months, scores, lags=lags,
                                                 fit_end=fit_end, countries=countries)
            elif cls in ("FL", "TC"):
                sections[cls] = backtest_pa(con, months, scores, cls, countries)
        except Exception as exc:  # noqa: BLE001 - one class must not sink the others
            logger.exception("reference_backtest: %s failed", cls)
            sections[cls] = {"status": "failed", "error": str(exc)[:300]}
        timings[cls] = round(time.monotonic() - t0, 1)
    rows = summarise(scores)
    reproduction: Dict[str, Any] = {}
    for r in rows:
        if r["class"] == "ACE" and r["horizon"] == "all" and r["score_type"] == "brier" \
                and r.get("status") == "ok" and r["candidate"] in QUOTED_CONFLICT:
            reproduction[r["candidate"]] = r["mean"]
            reproduction["n_country_forecasts"] = r["n_country_forecasts"]
    manifest = {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "db": db_label, "start": start, "end": end, "fit_end": fit_end,
        "classes": list(classes), "lags": list(lags), "forecast_day": FORECAST_DAY,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "interval": list(INTERVAL)},
        "bucket_floor": float(_cfg.BUCKET_FLOOR), "production": PRODUCTION,
        "sections": sections, "timings_sec": timings, "reproduction": reproduction,
        "n_rows": len(rows),
    }
    return rows, manifest


def write_outputs(rows: Sequence[Dict[str, Any]], manifest: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reference_backtest.md").write_text(render_markdown(rows, manifest), encoding="utf-8")
    cols = ["class", "lag", "candidate", "horizon", "score_type", "status", "n_forecasts",
            "n_country_forecasts", "n_countries", "mean", "lo", "hi", "diff_vs_production",
            "diff_lo", "diff_hi", "n_paired"]
    with (out_dir / "reference_backtest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str),
                                           encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    import duckdb  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", required=True, help="path to the canonical DuckDB file")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start", default="2021-03")
    ap.add_argument("--end", default="2025-06")
    ap.add_argument("--fit-end", default="2023-12")
    ap.add_argument("--classes", default="ACE,DR,FL,TC")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    db = str(args.db).replace("duckdb:///", "")
    con = duckdb.connect(db, read_only=True)
    try:
        rows, manifest = run_backtest(
            con, start=args.start, end=args.end, fit_end=args.fit_end,
            classes=[c.strip().upper() for c in args.classes.split(",") if c.strip()],
            db_label=Path(db).name)
    finally:
        con.close()
    write_outputs(rows, manifest, Path(args.out_dir))
    logger.info("reference_backtest: %d rows written to %s; timings %s",
                len(rows), args.out_dir, manifest["timings_sec"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
