# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ENSO state and forecast module.

Provides cached ENSO conditions and probabilistic forecasts from the
IRI/CPC ENSO Quick Look page for injection into climate-sensitive hazard
prompts (TC, FL, DR, HW).  The heavy scraping is done offline via
:mod:`enso_module`; this package exposes lightweight readers that
load the cached JSON and return prompt-ready text.

Usage (from the Pythia pipeline)::

    from horizon_scanner.enso import get_enso_prompt_context, get_enso_state

    # For prompt injection — returns a ready-to-inject text block:
    context_text = get_enso_prompt_context()

    # For programmatic access — returns an ENSOForecast dataclass:
    forecast = get_enso_state()
"""

from horizon_scanner.enso.enso_module import get_enso_prompt_context, get_enso_state

__all__ = ["get_enso_prompt_context", "get_enso_state"]
