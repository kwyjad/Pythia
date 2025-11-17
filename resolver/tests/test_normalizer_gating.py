from pathlib import Path

import pandas as pd

from resolver.tools import freeze_snapshot


def _run_freeze(facts_path: Path, outdir: Path, month: str) -> Path:
    result = freeze_snapshot.freeze_snapshot(
        facts=facts_path,
        month=month,
        outdir=outdir,
        overwrite=True,
        write_db=False,
    )
    assert result.resolved_csv is not None
    return result.resolved_csv


def test_normalizer_gating(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESOLVER_WRITE_DB", "0")
    monkeypatch.setattr(freeze_snapshot, "run_validator", lambda _path: None)

    outdir = Path("snapshots")
    summary_path = Path("diagnostics") / "summary.md"

    acled_rows = [
        {
            "iso3": "BFA",
            "ym": "2024-01",
            "hazard_code": "CU",
            "hazard_label": "Conflict",
            "hazard_class": "conflict",
            "metric": "events",
            "value": "3",
            "as_of_date": "2024-01-15",
            "publication_date": "2024-01-20",
            "publisher": "ACLED",
            "source_type": "dataset",
        }
    ]
    acled_path = tmp_path / "acled.csv"
    pd.DataFrame(acled_rows).to_csv(acled_path, index=False)

    acled_resolved = _run_freeze(acled_path, outdir, "2024-01")
    acled_df = pd.read_csv(acled_resolved)
    assert set(acled_df["publisher"].unique()) == {"ACLED"}
    acled_month_df = pd.read_csv(acled_path.with_name("facts_for_month.csv"))
    assert set(acled_month_df["publisher"].unique()) == {"ACLED"}

    assert summary_path.exists()
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "Freeze snapshot — normalizer" in summary_text
    assert "applied=false reason=non-emdat facts" in summary_text

    summary_path.unlink()

    emdat_rows = [
        {
            "iso3": "PHL",
            "ym": "2024-02",
            "hazard_code": "TC",
            "hazard_label": "Tropical Cyclone",
            "hazard_class": "disaster",
            "metric": "affected",
            "value": "100",
            "as_of_date": "2024-02-15",
            "publication_date": "",
            "publisher": "",
            "source_type": "",
        }
    ]
    emdat_path = tmp_path / "emdat.csv"
    pd.DataFrame(emdat_rows).to_csv(emdat_path, index=False)

    _run_freeze(emdat_path, outdir, "2024-02")
    emdat_month_df = pd.read_csv(emdat_path.with_name("facts_for_month.csv"))
    assert "CRED / UCLouvain (EM-DAT)" in set(emdat_month_df["publisher"].unique())

    assert summary_path.exists()
    emdat_summary = summary_path.read_text(encoding="utf-8")
    assert "applied=true reason=emdat facts" in emdat_summary
