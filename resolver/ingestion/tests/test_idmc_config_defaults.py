# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Config loading tests for IDMC skeleton."""
from resolver.ingestion.idmc.config import load


def test_idmc_config_defaults_loads():
    cfg = load()
    assert cfg.enabled is True
    assert set(cfg.api.series) == {"stock", "flow"}
    assert "monthly_flow" in cfg.api.endpoints and "stock" in cfg.api.endpoints
    assert isinstance(cfg.field_aliases.value_flow, list)
