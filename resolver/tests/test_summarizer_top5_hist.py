from __future__ import annotations

from scripts.ci.summarize_connectors import build_markdown


def test_source_sample_top5(tmp_path):
    diagnostics_root = tmp_path / "diagnostics" / "ingestion"
    sample_path = diagnostics_root / "dtm" / "samples" / "admin0_head.csv"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(
        "CountryISO3,admin0Name\n"
        "SSD,Juba\n"
        "SSD,Juba\n"
        "UGA,Kampala\n"
        "ETH,Addis Ababa\n",
        encoding="utf-8",
    )

    markdown = build_markdown(
        [],
        diagnostics_root=diagnostics_root,
        staging_root=tmp_path / "resolver" / "staging",
    )

    lines = markdown.splitlines()
    iso_line = next(line for line in lines if "CountryISO3 top 5" in line)
    assert "SSD (2)" in iso_line
    assert "UGA (1)" in iso_line
    assert "ETH (1)" in iso_line
    admin_line = next(line for line in lines if "admin0Name top 5" in line)
    assert "Juba (2)" in admin_line
    assert "Kampala (1)" in admin_line
    assert "Addis Ababa (1)" in admin_line
