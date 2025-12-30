from pathlib import Path


def test_risk_index_map_has_debug_overlay():
    content = Path("web/src/components/RiskIndexMap.tsx").read_text()
    assert "debug_map" in content
    assert 'console.groupCollapsed("[RiskIndexMap] debug")' in content
    assert 'setProperty("fill"' in content
    assert '"important"' in content
