"""Config loading tests for IDMC skeleton."""
from resolver.ingestion.idmc.config import load


def test_idmc_config_defaults_loads():
    cfg = load()
    assert cfg.enabled is True
    assert set(cfg.api.series) == {"stock", "flow"}
    assert "monthly_flow" in cfg.api.endpoints and "stock" in cfg.api.endpoints
    assert isinstance(cfg.field_aliases.value_flow, list)
