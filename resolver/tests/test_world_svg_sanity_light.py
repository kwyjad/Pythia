import re
from pathlib import Path


def extract_numeric_tokens(path_data: str) -> list[float]:
    tokens = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", path_data)
    return [float(token) for token in tokens]


def test_world_svg_sanity_light() -> None:
    svg_path = Path("web/public/maps/world.svg")
    content = svg_path.read_text(encoding="utf-8")

    assert content.count('data-iso3="') >= 150
    assert 'viewBox="0 0 1000 500"' in content

    path_matches = re.findall(r'<path[^>]*d="([^"]+)"', content)
    assert path_matches

    sample = " ".join(path_matches[:50])
    tokens = extract_numeric_tokens(sample)
    assert tokens

    max_value = max(tokens)
    min_value = min(tokens)
    assert max_value >= 900 or (max_value - min_value) >= 800
