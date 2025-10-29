from pathlib import Path

from resolver.tools.export_facts import DEFAULT_CONFIG, export_facts


def test_dtm_flows_export_semantics(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()

    fixture = Path(__file__).parent / "data" / "dtm_admin0_flows_sample.csv"
    csv_path = staging_dir / "dtm_displacement_admin0_flows.csv"
    csv_path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    out_dir = tmp_path / "exports"
    result = export_facts(
        inp=staging_dir,
        config_path=DEFAULT_CONFIG,
        out_dir=out_dir,
        write_db="0",
    )

    df = result.dataframe
    assert not df.empty
    flows = df[df["metric"] == "idp_displacement_new_dtm"].reset_index(drop=True)
    assert len(flows) == 3
    assert set(flows["semantics"]) == {"new"}
    assert set(flows["series_semantics"]) == {"new"}
    assert set(flows["iso3"]) == {"COD", "SDN"}
    assert set(flows["as_of_date"]) == {"2025-05-31", "2025-07-31"}

    cod_may = flows[(flows["iso3"] == "COD") & (flows["as_of_date"] == "2025-05-31")]
    assert len(cod_may) == 1
    assert cod_may.iloc[0]["value"] == "320"

    matched = result.report["matched_files"][0]
    assert matched["source"] == "dtm_displacement_admin0_flows"
    assert matched["rows_after_filters"] == 4
    assert matched["rows_after_dedupe"] == 3
