# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The RC shadow arm beside production, for one HS run.

Production levels come from ``hs_triage.regime_change_level`` (what the run
acted on); shadow levels from ``hs_rc_shadow`` (stored, never used; see
``horizon_scanner/rc_shadow.py``). Read by the debug bundle
(``rc_shadow_comparison__<run>.json/.md``, which the forecast pipeline review
lifts) and by ``scripts/ci/stage_health.py``. Pure reads; never raises.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

LEVELS = (0, 1, 2, 3)
#: docs/hs_regime_change.md: about 80% of assessments should sit at
#: likelihood <= 0.10 (base rate holds).
EXPECTED_SHARE_LOW = 0.80
LOW_LIKELIHOOD = 0.10


def _has_table(con, name: str) -> bool:
    try:
        return bool(con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [name]
        ).fetchone()[0])
    except Exception:  # noqa: BLE001
        return False


def _level_counts(levels: List[Optional[int]]) -> Dict[str, int]:
    out = {str(lv): 0 for lv in LEVELS}
    for lv in levels:
        if lv is not None and int(lv) in LEVELS:
            out[str(int(lv))] += 1
    return out


def shadow_comparison(con, hs_run_id: str) -> Dict[str, Any]:
    """Distribution, agreement matrix, the two disagreement lists, and spend."""

    out: Dict[str, Any] = {"hs_run_id": hs_run_id, "available": False}
    if not hs_run_id or not _has_table(con, "hs_rc_shadow"):
        out["reason"] = "no hs_rc_shadow table" if hs_run_id else "no hs_run_id"
        return out
    try:
        rows = con.execute(
            """
            SELECT iso3, hazard_code, status, likelihood, level, rationale_json, cost_usd, model_id
            FROM hs_rc_shadow
            WHERE hs_run_id = ? AND COALESCE(is_replay, FALSE) = FALSE
            """,
            [hs_run_id],
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"read failed: {type(exc).__name__}"
        return out
    if not rows:
        out["reason"] = "the shadow arm wrote nothing for this run"
        return out

    prod: Dict[tuple, Dict[str, Any]] = {}
    if _has_table(con, "hs_triage"):
        try:
            for iso3, hz, lik, lvl in con.execute(
                """
                SELECT iso3, hazard_code, regime_change_likelihood, regime_change_level
                FROM hs_triage WHERE run_id = ?
                """,
                [hs_run_id],
            ).fetchall():
                prod[((iso3 or "").upper(), (hz or "").upper())] = {"likelihood": lik, "level": lvl}
        except Exception:  # noqa: BLE001
            pass

    ok = [r for r in rows if r[2] == "ok"]
    statuses: Dict[str, int] = {}
    for r in rows:
        statuses[r[2] or "unknown"] = statuses.get(r[2] or "unknown", 0) + 1
    matrix = {str(p): {str(s): 0 for s in LEVELS} for p in LEVELS}
    would_forecast: List[Dict[str, Any]] = []
    would_drop: List[Dict[str, Any]] = []
    prod_levels: List[Optional[int]] = []
    shadow_levels: List[Optional[int]] = []
    prod_low = 0
    shadow_low = 0
    n_paired = 0
    for iso3, hz, _status, lik, lvl, rationale_json, _cost, _model in ok:
        p = prod.get(((iso3 or "").upper(), (hz or "").upper()))
        shadow_levels.append(lvl)
        if lik is not None and float(lik) <= LOW_LIKELIHOOD:
            shadow_low += 1
        if not p or p.get("level") is None or lvl is None:
            continue
        n_paired += 1
        p_lvl = int(p["level"])
        prod_levels.append(p_lvl)
        if p.get("likelihood") is not None and float(p["likelihood"]) <= LOW_LIKELIHOOD:
            prod_low += 1
        if p_lvl in LEVELS and int(lvl) in LEVELS:
            matrix[str(p_lvl)][str(int(lvl))] += 1
        try:
            bullets = json.loads(rationale_json or "[]")
        except (TypeError, ValueError):
            bullets = []
        item = {
            "iso3": iso3, "hazard_code": hz,
            "production_level": p_lvl, "production_likelihood": p.get("likelihood"),
            "shadow_level": int(lvl), "shadow_likelihood": lik,
            "shadow_rationale": [str(b) for b in (bullets or [])][:4],
        }
        if int(lvl) >= 1 and p_lvl == 0:
            would_forecast.append(item)
        elif p_lvl >= 1 and int(lvl) == 0:
            would_drop.append(item)
    key = lambda d: -(float(d.get("shadow_likelihood") or 0))  # noqa: E731
    would_forecast.sort(key=key)
    would_drop.sort(key=lambda d: -(float(d.get("production_likelihood") or 0)))
    agree = sum(matrix[str(lv)][str(lv)] for lv in LEVELS)
    out.update({
        "available": True,
        "model_id": rows[0][7],
        "n_rows": len(rows),
        "n_ok": len(ok),
        "n_paired": n_paired,
        "statuses": statuses,
        "production_levels": _level_counts(prod_levels),
        "shadow_levels": _level_counts(shadow_levels),
        "share_low_likelihood": {
            "threshold": LOW_LIKELIHOOD,
            "expected": EXPECTED_SHARE_LOW,
            "production": round(prod_low / n_paired, 3) if n_paired else None,
            "shadow": round(shadow_low / len(ok), 3) if ok else None,
        },
        "agreement_matrix": matrix,
        "exact_agreement": round(agree / n_paired, 3) if n_paired else None,
        "would_have_forecast": would_forecast,
        "production_only": would_drop,
        "cost_usd": round(sum(float(r[6] or 0) for r in rows), 4),
        "n_failures": sum(v for k, v in statuses.items() if k not in ("ok", "skipped_cap", "unfinished")),
        "n_skipped_cap": statuses.get("skipped_cap", 0),
        "n_unfinished": statuses.get("unfinished", 0),
    })
    return out


def markdown(rep: Dict[str, Any]) -> str:
    lines = ["## RC shadow arm beside production", ""]
    if not rep.get("available"):
        lines.append(f"Not available: {rep.get('reason')}.")
        return "\n".join(lines) + "\n"
    share = rep["share_low_likelihood"]
    lines += [
        f"Shadow model `{rep.get('model_id')}`: {rep['n_ok']} of {rep['n_rows']} assessments ok, "
        f"{rep['n_paired']} paired with production. Cost ${rep['cost_usd']:.2f}; "
        f"{rep['n_failures']} failed, {rep['n_skipped_cap']} skipped by the cap, "
        f"{rep['n_unfinished']} unfinished.",
        "",
        "| level | production | shadow |", "|---|--:|--:|",
    ]
    for lv in LEVELS:
        lines.append(f"| {lv} | {rep['production_levels'][str(lv)]} | {rep['shadow_levels'][str(lv)]} |")
    lines += [
        "",
        f"Share at likelihood <= {share['threshold']}: production {share['production']}, "
        f"shadow {share['shadow']} (expected about {share['expected']}).",
        "",
        "Agreement (rows production, columns shadow):", "",
        "| prod \\ shadow | 0 | 1 | 2 | 3 |", "|---|--:|--:|--:|--:|",
    ]
    for p in LEVELS:
        cells = " | ".join(str(rep["agreement_matrix"][str(p)][str(s)]) for s in LEVELS)
        lines.append(f"| {p} | {cells} |")
    lines += ["", f"Exact agreement: {rep['exact_agreement']}.", ""]
    for title, key in (("Would have forecast (shadow >= 1, production 0)", "would_have_forecast"),
                       ("Production only (production >= 1, shadow 0)", "production_only")):
        items = rep.get(key) or []
        lines.append(f"### {title}: {len(items)}")
        for it in items:
            lines.append(
                f"- {it['iso3']} {it['hazard_code']}: production L{it['production_level']} "
                f"({it['production_likelihood']}), shadow L{it['shadow_level']} "
                f"({it['shadow_likelihood']})" + (
                    "; " + " / ".join(it["shadow_rationale"][:2]) if it.get("shadow_rationale") else "")
            )
        lines.append("")
    return "\n".join(lines)
