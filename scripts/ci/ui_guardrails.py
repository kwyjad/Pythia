from __future__ import annotations

from pathlib import Path
import sys


def assert_condition(condition: bool, message: str) -> None:
    if condition:
        return
    print(f"UI guardrail failed: {message}")
    sys.exit(1)


def main() -> None:
    risk_map_path = Path("web/src/components/RiskIndexMap.tsx")
    risk_map_content = risk_map_path.read_text(encoding="utf-8")

    assert_condition(
        "from \"d3-geo\"" not in risk_map_content
        and "from 'd3-geo'" not in risk_map_content,
        "RiskIndexMap must not import d3-geo.",
    )
    assert_condition(
        "setAttribute(\"fill\", \"var(--risk-map-" not in risk_map_content,
        "RiskIndexMap must not set fill via var(--risk-map-...) attributes.",
    )

    world_svg_path = Path("web/public/maps/world.svg")
    world_svg_content = world_svg_path.read_text(encoding="utf-8")

    iso3_count = world_svg_content.count('data-iso3="')
    assert_condition(
        iso3_count >= 150,
        f"World SVG must contain at least 150 data-iso3 entries (found {iso3_count}).",
    )
    assert_condition(
        "viewBox=\"" in world_svg_content,
        "World SVG must include a viewBox attribute.",
    )

    print("UI guardrails passed.")


if __name__ == "__main__":
    main()
