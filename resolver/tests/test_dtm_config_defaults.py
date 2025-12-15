# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import yaml
from pathlib import Path


def test_dtm_config_defaults_admin0_open_countries():
    cfg = yaml.safe_load(Path("resolver/ingestion/config/dtm.yml").read_text(encoding="utf-8"))
    api = cfg.get("api", {})
    assert api.get("countries", []) == [], "countries must be empty (discover all)"
    assert api.get("admin_levels") == ["admin0"], "admin_levels must default to admin0 only"
