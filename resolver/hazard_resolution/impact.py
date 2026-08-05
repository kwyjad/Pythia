# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Layer 2 orchestration: run the impact ladder over a month's triggers.

This is the wiring, not the logic. It fetches the ladder's sources once
per month, then for every TRIGGERED country-month builds candidates
(:mod:`candidates`), asks the deterministic reconciler for a verdict
(:mod:`reconcile`), and writes it (:mod:`resolutions`). No decision is
made here; every rule the answer depends on lives in the modules above,
which is what keeps the ladder auditable.

Both hazards share this path — a triggered cyclone month and a triggered
flood month walk the same ladder, differing only in which detector
triggered them.

Source failures are carried, not swallowed. If EM-DAT could not be
reached, the run's ``NO_DATA`` rows record that the top rung was
UNAVAILABLE rather than empty. Those are very different statements, and
only one of them means "nobody reported a figure".
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from resolver.hazard_resolution import candidates as cand_mod
from resolver.hazard_resolution import emdat as emdat_mod
from resolver.hazard_resolution import gdacs as gdacs_mod
from resolver.hazard_resolution import idmc_idu as idu_mod
from resolver.hazard_resolution import ifrc_go as go_mod
from resolver.hazard_resolution import reconcile as reconcile_mod
from resolver.hazard_resolution import resolutions as res_mod
from resolver.hazard_resolution.rulebook import Rulebook
from resolver.hazard_resolution.schema import ensure_haz_schema

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)


@dataclass
class LadderRun:
    """What one month's ladder pass did."""

    hazard: str
    ym: str
    cells: int = 0
    resolved_value: int = 0
    no_data: int = 0
    pending: int = 0
    flagged: int = 0
    lower_bound: int = 0
    provisional: int = 0
    frozen_skipped: int = 0
    fetches: dict[str, Any] = field(default_factory=dict)

    @property
    def unavailable_sources(self) -> list[str]:
        return sorted(
            name for name, meta in self.fetches.items() if not meta.get("ok", False)
        )


def national_population(
    con: "duckdb.DuckDBPyConnection", iso3: str, year: int
) -> float | None:
    """The country's population denominator for the sanity cap.

    Takes the most recent year at or before the target; falls back to the
    earliest available when the cell predates the table (a denominator
    from the wrong decade still bounds a figure usefully, and the
    alternative is no cap at all).
    """

    row = con.execute(
        """
        SELECT population FROM haz_raw_population
        WHERE iso3 = ? AND year <= ?
        ORDER BY year DESC LIMIT 1
        """,
        [iso3.upper(), int(year)],
    ).fetchone()
    if row is None:
        row = con.execute(
            """
            SELECT population FROM haz_raw_population
            WHERE iso3 = ? ORDER BY year ASC LIMIT 1
            """,
            [iso3.upper()],
        ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def fetch_ladder_sources(
    con: "duckdb.DuckDBPyConnection",
    ym: str,
    hazard: str,
    rulebook: Rulebook,
    *,
    skip_fetch: bool = False,
) -> dict[str, Any]:
    """Refresh every ladder source for one month. Never raises.

    Returns one entry per source describing whether it answered, so the
    caller can distinguish an empty rung from an unreachable one.
    """

    if skip_fetch:
        LOG.info("[impact] --skip-fetch: using the existing raw caches")
        return {
            name: {"ok": True, "skipped": True}
            for name in ("emdat", "ifrc_go", "idmc_idu", "gdacs")
        }

    outcomes = {
        "emdat": emdat_mod.fetch_emdat(con, ym, hazard, rulebook),
        "ifrc_go": go_mod.fetch_ifrc_go(con, ym, hazard, rulebook),
        "idmc_idu": idu_mod.fetch_idmc_idu(con, ym, hazard, rulebook),
        # GDACS is fetched here too because it supplies the sanity ceiling
        # for BOTH hazards — for floods it has already run as the detector,
        # and the raw cache makes the second call a no-op upsert.
        "gdacs": gdacs_mod.fetch_gdacs_events(con, ym, hazard, rulebook),
    }
    for name, outcome in outcomes.items():
        if not outcome.ok:
            LOG.warning(
                "[impact] ladder source %s unavailable for %s %s: %s",
                name, hazard, ym, outcome.error,
            )
    return {name: outcome.as_provenance() for name, outcome in outcomes.items()}


def resolve_triggered_cells(
    con: "duckdb.DuckDBPyConnection",
    *,
    ym: str,
    hazard: str,
    iso3s: list[str],
    rulebook: Rulebook,
    fetches: dict[str, Any] | None = None,
    dry_run: bool = False,
    today: dt.date | None = None,
) -> LadderRun:
    """Walk the ladder for every triggered cell in ``iso3s``."""

    ensure_haz_schema(con)
    run = LadderRun(hazard=hazard, ym=ym, fetches=fetches or {})
    year = int(ym.split("-")[0])
    unavailable = run.unavailable_sources

    for iso3 in sorted(iso3s):
        found = cand_mod.build_candidates(con, iso3, ym, hazard, rulebook)
        if not dry_run:
            cand_mod.write_candidates(con, found, iso3, ym, hazard)

        verdict = reconcile_mod.reconcile(
            iso3=iso3,
            ym=ym,
            hazard=hazard,
            candidates=found,
            rulebook=rulebook,
            national_population=national_population(con, iso3, year),
            today=today,
        )
        # A rung that could not be READ is not a rung that was empty. Record
        # the difference on the row itself so a NO_DATA can be re-litigated.
        if unavailable:
            verdict.provenance.setdefault("decision", {})[
                "sources_unavailable"
            ] = unavailable
        verdict.provenance["source_fetches"] = run.fetches

        run.cells += 1
        if verdict.status == reconcile_mod.STATUS_RESOLVED_VALUE:
            run.resolved_value += 1
            run.provisional += int(verdict.provisional)
        elif verdict.status == reconcile_mod.STATUS_PENDING:
            run.pending += 1
        else:
            run.no_data += 1
        run.flagged += int(verdict.flagged)
        run.lower_bound += int(verdict.lower_bound)

        if dry_run:
            continue
        outcome = res_mod.write_reconciliation(con, verdict, rulebook, today=today)
        if outcome == res_mod.WRITE_FROZEN_SKIP:
            run.frozen_skipped += 1

    LOG.info(
        "[impact] %s %s ladder: %d cells -> %d values (%d lower-bound), "
        "%d no-data, %d pending, %d flagged, %d provisional, %d frozen-skipped",
        hazard, ym, run.cells, run.resolved_value, run.lower_bound,
        run.no_data, run.pending, run.flagged, run.provisional, run.frozen_skipped,
    )
    if unavailable:
        LOG.warning(
            "[impact] %s %s: ladder ran with unavailable sources %s — NO_DATA "
            "rows from this run record that those rungs were unreadable, not empty",
            hazard, ym, ",".join(unavailable),
        )
    return run


def triggered_iso3s(
    con: "duckdb.DuckDBPyConnection", ym: str, hazard: str
) -> list[str]:
    """Countries whose trigger row for this month says a hazard occurred."""

    year, month = (int(p) for p in ym.split("-"))
    return [
        row[0]
        for row in con.execute(
            """
            SELECT iso3 FROM haz_triggers
            WHERE hazard = ? AND year = ? AND month = ? AND triggered
            ORDER BY iso3
            """,
            [hazard, year, month],
        ).fetchall()
    ]
