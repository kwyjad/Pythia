# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Contract test: source_priority.yml entries match canonical adapter slugs."""

from __future__ import annotations

from pathlib import Path

import yaml
import pytest

from resolver.transform.normalize import ADAPTER_REGISTRY

PRIORITY_PATH = (
    Path(__file__).resolve().parents[2]
    / "ingestion"
    / "config"
    / "source_priority.yml"
)


def _load_priority_entries() -> list[str]:
    """Load and lowercase the priority list from source_priority.yml."""
    with PRIORITY_PATH.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or []
    if isinstance(loaded, dict):
        loaded = loaded.get("priority", list(loaded.values()))
    return [str(item).strip().lower() for item in loaded if str(item).strip()]


class TestSourcePriorityContract:
    def test_priority_file_exists(self) -> None:
        assert PRIORITY_PATH.exists(), f"Missing {PRIORITY_PATH}"

    def test_priority_not_empty(self) -> None:
        entries = _load_priority_entries()
        assert len(entries) >= 2, "Expected at least 2 source priority entries"

    def test_active_sources_present(self) -> None:
        """Verify the three active canonical sources are listed."""
        entries = _load_priority_entries()
        assert "ifrc_go" in entries, "ifrc_go missing from source_priority.yml"
        assert "acled" in entries, "acled missing from source_priority.yml"
        assert "idmc" in entries, "idmc missing from source_priority.yml"

    def test_adapter_registry_covers_active_sources(self) -> None:
        """Every active source in priority list has a normalize adapter."""
        active = {"ifrc_go", "acled", "idmc"}
        registry_keys = set(ADAPTER_REGISTRY.keys())
        missing = active - registry_keys
        assert not missing, f"Adapters missing for active sources: {missing}"

    def test_no_stale_ifrc_entry(self) -> None:
        """Ensure priority list doesn't have bare 'ifrc' (should be 'ifrc_go')."""
        entries = _load_priority_entries()
        # 'ifrc' alone would not match the adapter's canonical_source='ifrc_go'
        # so it would get fallback rank in resolve_sources.py
        if "ifrc" in entries:
            pytest.fail(
                "source_priority.yml has 'ifrc' but the adapter produces "
                "'ifrc_go' as the source column. Use 'ifrc_go' instead."
            )
