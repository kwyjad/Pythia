# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from pathlib import Path

import yaml


def test_export_config_yaml_parses():
    cfg = Path("resolver/tools/export_config.yml")
    assert cfg.exists(), "export_config.yml missing"
    with cfg.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "export_config root must be a mapping"
    assert "metrics" in data and isinstance(
        data["metrics"], dict
    ), "export_config must define a metrics mapping"
