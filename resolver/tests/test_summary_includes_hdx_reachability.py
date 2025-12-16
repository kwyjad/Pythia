# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from scripts.ci.summarize_connectors import render_summary_md


def test_summary_includes_hdx_reachability() -> None:
    markdown = render_summary_md(
        [],
        hdx_reachability={
            "dataset": "preliminary-internal-displacement-updates",
            "resource_url": "https://data.humdata.org/idus_view_flat.csv",
            "resource_status_code": 200,
        },
    )
    assert "## HDX Reachability" in markdown
    assert "idus_view_flat.csv" in markdown
