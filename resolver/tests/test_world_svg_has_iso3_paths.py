from pathlib import Path


def test_world_svg_has_iso3_paths() -> None:
    svg_path = Path("web/public/maps/world.svg")
    content = svg_path.read_text(encoding="utf-8")
    assert 'data-iso3="' in content
    assert content.count('data-iso3="') >= 150
    assert 'viewBox="' in content
