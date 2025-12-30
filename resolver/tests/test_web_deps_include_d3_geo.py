# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import json
from pathlib import Path


def test_web_dependencies_include_d3_geo():
    package_json = Path(__file__).resolve().parents[2] / "web" / "package.json"
    data = json.loads(package_json.read_text())
    deps = data.get("dependencies", {})
    assert "d3-geo" in deps
