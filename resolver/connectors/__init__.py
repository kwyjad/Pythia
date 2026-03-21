# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Connector registry — the single place to register data sources."""

from __future__ import annotations

from typing import Sequence

from .protocol import CANONICAL_COLUMNS, Connector
from .acled import AcledConnector
from .gdacs import GdacsConnector
from .idmc import IdmcConnector
from .ifrc_montandon import IfrcMontandonConnector
from .views import ViewsConnector
from .conflictforecast import ConflictForecastOrgConnector
from .acled_cast import AcledCastConnector
from .fewsnet_ipc import FewsnetIpcConnector

# Add new connectors here.  The orchestrator (run_pipeline.py) iterates
# this registry to discover which sources to pull.
REGISTRY: dict[str, type] = {
    "acled": AcledConnector,
    "gdacs": GdacsConnector,
    "idmc": IdmcConnector,
    "ifrc_montandon": IfrcMontandonConnector,
    "fewsnet_ipc": FewsnetIpcConnector,
}

# Forecast connectors write to ``conflict_forecasts`` (not facts_resolved).
# They follow a different interface: fetch_forecasts() → DataFrame.
# Use ``fetch_conflict_forecasts.py`` to run these.
FORECAST_REGISTRY: dict[str, type] = {
    "views": ViewsConnector,
    "conflictforecast_org": ConflictForecastOrgConnector,
    "acled_cast": AcledCastConnector,
}


def discover_connectors(names: Sequence[str] | None = None) -> list[Connector]:
    """Instantiate connectors by name, or all registered connectors."""
    if names:
        unknown = [n for n in names if n not in REGISTRY]
        if unknown:
            raise KeyError(f"Unknown connector(s): {unknown}")
        return [REGISTRY[n]() for n in names]
    return [cls() for cls in REGISTRY.values()]


__all__ = [
    "CANONICAL_COLUMNS",
    "Connector",
    "REGISTRY",
    "FORECAST_REGISTRY",
    "discover_connectors",
]
