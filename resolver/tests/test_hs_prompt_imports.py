from horizon_scanner.prompts import build_hs_triage_prompt


def test_hs_triage_prompt_imports():
    prompt = build_hs_triage_prompt(
        country_name="Testland",
        iso3="TST",
        hazard_catalog={"ACE": "Conflict"},
        resolver_features={},
        model_info={},
        evidence_pack={"markdown": "Signal"},
    )

    assert isinstance(prompt, str)
    assert prompt
    assert "regime_change" in prompt
    assert "trigger_signals" in prompt
