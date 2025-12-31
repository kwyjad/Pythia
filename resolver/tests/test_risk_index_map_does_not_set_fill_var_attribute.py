from pathlib import Path


def test_risk_index_map_does_not_set_fill_var_attribute() -> None:
    content = Path("web/src/components/RiskIndexMap.tsx").read_text(encoding="utf-8")
    assert "setAttribute(\"fill\"," not in content
    assert "setAttribute(\"fill\", \"var(--risk-map-" not in content
    assert (
        ".style.fill" in content
        or '.style.setProperty("fill"' in content
        or ".style.setProperty('fill'" in content
        or 'setProperty("fill"' in content
        or "setProperty('fill'" in content
    )
