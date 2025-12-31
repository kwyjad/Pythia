from __future__ import annotations

from pathlib import Path
import sys


def fail_guardrail(message: str, details: dict[str, object] | None = None) -> None:
    print(f"UI guardrail failed: {message}")
    if details:
        for key, value in details.items():
            print(f"- {key}: {value}")
    sys.exit(1)


def main() -> None:
    risk_map_path = Path("web/src/components/RiskIndexMap.tsx")
    risk_map_content = risk_map_path.read_text(encoding="utf-8")

    if (
        "from \"d3-geo\"" in risk_map_content
        or "from 'd3-geo'" in risk_map_content
    ):
        fail_guardrail(
            "RiskIndexMap must not import d3-geo.",
            {"file": str(risk_map_path)},
        )
    if "setAttribute(\"fill\", \"var(--risk-map-" in risk_map_content:
        fail_guardrail(
            "RiskIndexMap must not set fill via var(--risk-map-...) attributes.",
            {"file": str(risk_map_path)},
        )

    world_svg_path = Path("web/public/maps/world.svg")
    world_svg_content = world_svg_path.read_text(encoding="utf-8")

    iso3_count = world_svg_content.count('data-iso3="')
    has_view_box = "viewBox=\"" in world_svg_content
    if iso3_count < 150:
        fail_guardrail(
            "World SVG must contain at least 150 data-iso3 entries.",
            {"file": str(world_svg_path), "iso3_count": iso3_count},
        )
    if not has_view_box:
        fail_guardrail(
            "World SVG must include a viewBox attribute.",
            {"file": str(world_svg_path), "iso3_count": iso3_count},
        )

    print("UI guardrails passed.")


if __name__ == "__main__":
    main()
