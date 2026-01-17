from horizon_scanner.horizon_scanner import _build_hs_evidence_query


def test_hs_tail_trigger_query_contains_expected_tokens() -> None:
    query = _build_hs_evidence_query("Afghanistan", "AFG")

    assert "TAIL-UP" in query
    assert "TAIL-DOWN" in query
    assert "BASELINE" in query
    assert "last 120 days" in query
    assert any(code in query for code in ["ACE", "DI", "FL", "TC", "DR", "HW"])
