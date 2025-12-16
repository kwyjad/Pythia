# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
from typing import Any, Dict

from pythia.db.schema import connect


def load_hs_triage_entry(run_id: str, iso3: str, hazard_code: str) -> Dict[str, Any]:
    """Load the latest HS triage entry for a given question/hazard."""

    con = connect(read_only=True)
    try:
        row = con.execute(
            """
            SELECT tier, triage_score, need_full_spd,
                   drivers_json, regime_shifts_json, data_quality_json, scenario_stub
            FROM hs_triage
            WHERE run_id = ? AND iso3 = ? AND hazard_code = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [run_id, iso3, hazard_code],
        ).fetchone()
        if not row:
            return {}
        return {
            "tier": row[0],
            "triage_score": row[1],
            "need_full_spd": bool(row[2]),
            "drivers": json.loads(row[3] or "[]"),
            "regime_shifts": json.loads(row[4] or "[]"),
            "data_quality": json.loads(row[5] or "{}"),
            "scenario_stub": row[6] or "",
        }
    finally:
        con.close()
