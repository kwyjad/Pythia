# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import logging
from datetime import date

from resolver.db import duckdb_io

from pythia.tools.compute_calibration_pythia import (
    compute_calibration_pythia,
    _group_by_hazard_metric,
    _load_samples,
)


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())


def maybe_run_calibration(
    db_url: str | None = None,
    as_of: date | None = None,
    min_questions: int = 30,
) -> None:
    """Run calibration if enough resolved questions exist for any hazard/metric group."""

    if as_of is None:
        as_of = date.today()
    as_of_month = as_of.strftime("%Y-%m")

    conn = duckdb_io.get_db(db_url or duckdb_io.DEFAULT_DB_URL)
    try:
        samples = _load_samples(conn, as_of_month)
        groups = _group_by_hazard_metric(samples)

        eligible: dict[tuple[str, str], int] = {}
        for (hazard_code, metric), items in groups.items():
            question_keys = {s.question_key for s in items}
            n_questions = len(question_keys)
            if n_questions >= min_questions:
                eligible[(hazard_code, metric)] = n_questions

        if not eligible:
            LOGGER.info(
                "Auto-calibration: no hazard/metric groups with >=%d resolved questions (as_of_month=%s); skipping.",
                min_questions,
                as_of_month,
            )
            return

        LOGGER.info(
            "Auto-calibration: eligible groups (as_of_month=%s): %s",
            as_of_month,
            ", ".join(
                f"{hazard_code}/{metric} (n={n_questions})"
                for (hazard_code, metric), n_questions in sorted(eligible.items())
            ),
        )
    finally:
        duckdb_io.close_db(conn)

    LOGGER.info(
        "Auto-calibration: invoking compute_calibration_pythia(as_of=%s)", as_of.isoformat()
    )
    compute_calibration_pythia(db_url=db_url or duckdb_io.DEFAULT_DB_URL, as_of=as_of)
