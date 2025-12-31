from __future__ import annotations

from pathlib import Path
import re
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
    view_box_match = re.search(r'viewBox="([^"]+)"', world_svg_content)
    has_view_box = view_box_match is not None
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
    view_box_values = []
    if view_box_match:
        for value in re.split(r"[\s,]+", view_box_match.group(1).strip()):
            try:
                view_box_values.append(float(value))
            except ValueError:
                view_box_values = []
                break
    if len(view_box_values) != 4:
        fail_guardrail(
            "World SVG viewBox must contain four numeric values.",
            {"file": str(world_svg_path), "viewBox": view_box_match.group(1)},
        )
    view_box_width = view_box_values[2]
    view_box_height = view_box_values[3]
    path_matches = re.findall(r'd="([^"]+)"', world_svg_content)
    coord_matches: list[tuple[float, float]] = []
    for path_d in path_matches[:400]:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", path_d)
        for idx in range(0, len(numbers) - 1, 2):
            try:
                x_val = float(numbers[idx])
                y_val = float(numbers[idx + 1])
            except ValueError:
                continue
            coord_matches.append((x_val, y_val))
    if not coord_matches:
        fail_guardrail(
            "World SVG must include numeric path coordinates.",
            {"file": str(world_svg_path)},
        )
    xs, ys = zip(*coord_matches, strict=True)
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    span_x = max_x - min_x
    span_y = max_y - min_y
    ratio_x = span_x / view_box_width if view_box_width else 0
    ratio_y = span_y / view_box_height if view_box_height else 0
    if ratio_x < 0.5 or ratio_y < 0.5:
        fail_guardrail(
            "World SVG coordinate spread is implausibly narrow.",
            {
                "file": str(world_svg_path),
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "ratio_x": f"{ratio_x:.3f}",
                "ratio_y": f"{ratio_y:.3f}",
            },
        )

    print("UI guardrails passed.")


if __name__ == "__main__":
    main()
