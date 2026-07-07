# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl — a parallel deep-research forecasting harness.

Sibyl produces an alternative, independent SPD forecast for the highest-
volatility affected/fatalities questions in each run. Unlike standard
Pythia — which reasons over batch-ingested structured feeds in DuckDB —
Sibyl's primary evidence channel is agentic open-web research performed on
demand by Claude Opus via the Anthropic API. The two tracks share exactly
one structured input by design: the historical base rate from the Resolver
DB (the outside view). See ``sibyl/DISCOVERY.md`` for the integration map.
"""

__all__ = ["config"]
