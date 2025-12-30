# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from pathlib import Path


def test_no_d3_geo_import_in_web_map():
    map_file = (
        Path(__file__).resolve().parents[2]
        / "web"
        / "src"
        / "components"
        / "RiskIndexMap.tsx"
    )
    content = map_file.read_text()
    assert 'from "d3-geo"' not in content
    assert "from 'd3-geo'" not in content
