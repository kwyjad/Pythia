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
    globals_path = Path("web/src/styles/globals.css")
    globals_content = globals_path.read_text(encoding="utf-8")
    if "bg-slate-950" in globals_content:
        fail_guardrail(
            "globals.css must not include bg-slate-950.",
            {"file": str(globals_path)},
        )
    if "bg-fredBg" in globals_content:
        tailwind_config_path = Path("web/tailwind.config.ts")
        tailwind_config_content = tailwind_config_path.read_text(encoding="utf-8")
        if "fredBg" not in tailwind_config_content:
            fail_guardrail(
                "Tailwind config must define fredBg when globals.css references bg-fredBg.",
                {"file": str(tailwind_config_path)},
            )

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
    per_path_bboxes: list[tuple[float, float]] = []
    for path_d in path_matches[:400]:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", path_d)
        coords: list[tuple[float, float]] = []
        for idx in range(0, len(numbers) - 1, 2):
            try:
                x_val = float(numbers[idx])
                y_val = float(numbers[idx + 1])
            except ValueError:
                continue
            coord_matches.append((x_val, y_val))
            coords.append((x_val, y_val))
        if coords:
            xs = [coord[0] for coord in coords]
            ys = [coord[1] for coord in coords]
            per_path_bboxes.append((max(xs) - min(xs), max(ys) - min(ys)))
    if not coord_matches:
        fail_guardrail(
            "World SVG must include numeric path coordinates.",
            {"file": str(world_svg_path)},
        )
    if not per_path_bboxes:
        fail_guardrail(
            "World SVG must include per-path coordinate data.",
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

    widths = sorted(width for width, _ in per_path_bboxes)
    heights = sorted(height for _, height in per_path_bboxes)

    def percentile(values: list[float], frac: float) -> float:
        idx = max(0, min(len(values) - 1, int(round(frac * (len(values) - 1)))))
        return values[idx]

    p50_width = percentile(widths, 0.5)
    p90_width = percentile(widths, 0.9)
    p50_height = percentile(heights, 0.5)
    p90_height = percentile(heights, 0.9)
    if p90_width < 20 or p90_height < 20:
        fail_guardrail(
            "World SVG per-path bboxes are too small.",
            {
                "file": str(world_svg_path),
                "p50_width": f"{p50_width:.2f}",
                "p90_width": f"{p90_width:.2f}",
                "p50_height": f"{p50_height:.2f}",
                "p90_height": f"{p90_height:.2f}",
            },
        )

    def find_iso3_path(iso3: str) -> str | None:
        pattern = re.compile(rf'data-iso3="{iso3}"[^>]*d="([^"]+)"')
        match = pattern.search(world_svg_content)
        if not match:
            return None
        return match.group(1)

    def bbox_center(path_d: str) -> tuple[float, float] | None:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", path_d)
        coords = []
        for idx in range(0, len(numbers) - 1, 2):
            try:
                x_val = float(numbers[idx])
                y_val = float(numbers[idx + 1])
            except ValueError:
                continue
            coords.append((x_val, y_val))
        if not coords:
            return None
        xs = [coord[0] for coord in coords]
        ys = [coord[1] for coord in coords]
        return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)

    afghanistan_path = find_iso3_path("AFG")
    australia_path = find_iso3_path("AUS")
    if not afghanistan_path or not australia_path:
        fail_guardrail(
            "World SVG must include sentinel ISO3 paths for AFG and AUS.",
            {"file": str(world_svg_path)},
        )
    afg_center = bbox_center(afghanistan_path)
    aus_center = bbox_center(australia_path)
    if afg_center is None or aus_center is None:
        fail_guardrail(
            "World SVG sentinel path bbox centers could not be computed.",
            {"file": str(world_svg_path)},
        )
    if not (600 <= afg_center[0] <= 800):
        fail_guardrail(
            "World SVG AFG center appears implausible.",
            {"file": str(world_svg_path), "afg_center": afg_center},
        )
    if not (300 <= aus_center[1] <= 450):
        fail_guardrail(
            "World SVG AUS center appears implausible.",
            {"file": str(world_svg_path), "aus_center": aus_center},
        )

    about_page_path = Path("web/src/app/about/page.tsx")
    ai_prompts_path = Path("web/src/app/about/AiPromptsSection.tsx")
    prompts_path = Path("forecaster/prompts.py")
    about_page_content = about_page_path.read_text(encoding="utf-8")
    ai_prompts_content = ai_prompts_path.read_text(encoding="utf-8")
    prompts_content = prompts_path.read_text(encoding="utf-8")
    ai_combined = "\n".join([about_page_content, ai_prompts_content])
    ai_markers = [
        "AI Prompts",
        "Web search",
        "Horizon scan",
        "Researcher",
        "Forecast (SPD)",
        "Scenario",
        "How forecast questions are constructed",
    ]
    missing_markers = [marker for marker in ai_markers if marker not in ai_combined]
    if missing_markers:
        fail_guardrail(
            "AI Prompts section missing required markers.",
            {
                "missing": ", ".join(missing_markers),
                "files": f"{about_page_path}, {ai_prompts_path}",
            },
        )

    prompt_markers = [
        "FORECASTING INSTRUCTIONS (Bayesian SPD)",
        "Final JSON output (IMPORTANT)",
        "REQUIRED OUTPUT FORMAT (use headings exactly as written)",
        "People affected (PA) buckets",
        "Conflict fatalities buckets",
    ]
    for marker in prompt_markers:
        if marker not in ai_prompts_content:
            fail_guardrail(
                "AI Prompts section missing prompt marker.",
                {"marker": marker, "file": str(ai_prompts_path)},
            )
        if marker not in prompts_content:
            fail_guardrail(
                "Prompt source missing required marker.",
                {"marker": marker, "file": str(prompts_path)},
            )

    print("UI guardrails passed.")


if __name__ == "__main__":
    main()
