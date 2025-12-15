# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
import logging
from typing import Any, List

from pythia.db.schema import connect as pythia_connect

LOG = logging.getLogger(__name__)


def main() -> None:
    """
    One-off maintenance script to deactivate legacy questions:

    - Any question with hazard_code = 'ACO'.
    - Any question with pythia_metadata_json["source"] = "demo".
    """

    logging.basicConfig(level=logging.INFO)
    con = pythia_connect(read_only=False)

    LOG.info("Deactivating ACO questions in 'questions' table...")
    con.execute(
        """
        UPDATE questions
        SET status = 'inactive'
        WHERE UPPER(COALESCE(hazard_code, '')) = 'ACO'
          AND status = 'active'
        """
    )
    aco_count = con.execute(
        """
        SELECT COUNT(*) FROM questions
        WHERE UPPER(COALESCE(hazard_code, '')) = 'ACO'
          AND status = 'inactive'
        """
    ).fetchone()[0]
    LOG.info("ACO questions now inactive: %d", aco_count)

    LOG.info("Deactivating demo questions (pythia_metadata_json.source = 'demo')...")
    rows = con.execute(
        "SELECT question_id, pythia_metadata_json FROM questions WHERE status = 'active'"
    ).fetchall()

    demo_ids: List[str] = []
    for question_id, meta_raw in rows:
        if not meta_raw:
            continue
        try:
            meta: dict[str, Any] = json.loads(meta_raw)
        except Exception:
            continue
        if (meta.get("source") or "").lower() == "demo":
            demo_ids.append(question_id)

    if demo_ids:
        con.execute(
            "UPDATE questions SET status = 'inactive' WHERE question_id IN (%s)"
            % ",".join(["?"] * len(demo_ids)),
            demo_ids,
        )
    LOG.info("Demo questions deactivated: %d", len(demo_ids))

    con.close()


if __name__ == "__main__":
    main()
