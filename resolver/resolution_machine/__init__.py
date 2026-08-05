# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""People-affected resolution machine.

For every country-month-hazard the machine produces either a credible
number of people affected, a credible zero, or an explicit "no data" —
always with full provenance.  Two layers:

- Layer 1 (detection): decide from physical/event sources whether any
  qualifying hazard occurred.  Nothing occurred + a silent ReliefWeb
  keyword sweep → resolve zero with evidence of absence.
- Layer 2 (impact): walk a fixed precedence ladder of reporting sources
  (EM-DAT → LLM-extracted ReliefWeb figures → IFRC GO → IDMC IDU lower
  bound) for a people-affected figure.  (Later phases.)

Phase 1 implements the cyclone detection path: IBTrACS triggers,
ReliefWeb silence checks, and RESOLVED_ZERO rows.

All thresholds live in ``rulebook.yaml`` (never hard-coded); every
resolution row carries provenance; resolutions freeze at month-end +
``freeze.days_after_month_end`` and are never reopened (later revisions
go to ``pa_resolution_revisions``).  See ``README.md`` in this package.
"""
