# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Hazard resolution machine: per country-month-hazard people-affected ground truth.

For every (iso3, year, month, hazard) cell this module produces exactly one of:

* ``RESOLVED_VALUE`` — a credible people-affected number with full provenance,
* ``RESOLVED_ZERO``  — a credible zero, backed by recorded evidence of absence,
* ``NO_DATA``        — an explicit gap, flagged for human review.

Behaviour is driven by ``rulebook.yaml`` (the single human-readable source of
truth for thresholds, source precedence and freeze policy — never hard-code
those values). Detection decides whether a qualifying hazard occurred; the
impact ladder then walks reporting sources for a stated figure. GDACS is used
for detection and as a plausibility ceiling only, never as a resolution value.
Reconciliation is deterministic — rules only, no LLM calls.

See ``README.md`` in this directory for the architecture overview and
``schema.py`` for the ``haz_*`` DuckDB tables.
"""

from resolver.hazard_resolution.rulebook import (  # noqa: F401
    Rulebook,
    RulebookError,
    load_rulebook,
)

__all__ = ["Rulebook", "RulebookError", "load_rulebook"]
