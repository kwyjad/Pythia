from pathlib import Path


def test_web_map_has_no_d3_geo_import() -> None:
    map_path = Path("web/src/components/RiskIndexMap.tsx")
    content = map_path.read_text(encoding="utf-8")
    assert "d3-geo" not in content
