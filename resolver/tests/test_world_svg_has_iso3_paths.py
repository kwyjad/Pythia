import re
from pathlib import Path


def test_world_svg_has_iso3_paths() -> None:
    svg_path = Path("web/public/maps/world.svg")
    content = svg_path.read_text(encoding="utf-8")
    assert 'data-iso3="' in content
    assert content.count('data-iso3="') >= 150
    assert 'viewBox="' in content

    path_matches = re.findall(r'<path[^>]*data-iso3="[^"]+"[^>]*d="([^"]+)"', content)
    assert len(path_matches) >= 150

    long_paths = sum(1 for d in path_matches if len(d) > 200)
    assert long_paths >= 120

    sample = " ".join(path_matches[:200])
    tokens = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", sample)
    assert tokens
    max_value = max(float(token) for token in tokens)
    assert max_value >= 500
