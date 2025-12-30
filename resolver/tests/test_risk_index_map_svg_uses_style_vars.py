from pathlib import Path


def test_risk_index_map_svg_uses_style_vars() -> None:
    content = Path("web/src/components/RiskIndexMap.tsx").read_text(encoding="utf-8")
    assert ".style.fill" in content or "style.setProperty(\"fill\"" in content
    assert "setAttribute(\"fill\", \"var(" not in content
