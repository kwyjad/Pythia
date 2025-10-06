from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from resolver.ingestion import reliefweb_client as rw


def _base_cfg() -> Dict[str, object]:
    return {
        "enable_pdfs": True,
        "enable_ocr": True,
        "min_text_chars_before_ocr": 1500,
        "preferred_pdf_titles": ["situation report", "flash update", "key figures"],
        "timeout_seconds": 5,
    }


def _base_fields(pdf_url: str) -> Dict[str, object]:
    return {
        "file": [
            {
                "mime_type": "application/pdf",
                "url": pdf_url,
                "name": "Situation Report",
                "date": {"created": "2025-08-01T00:00:00Z"},
                "page_count": 4,
            }
        ],
        "date": {"created": "2025-08-01T00:00:00Z", "changed": "2025-08-02T00:00:00Z"},
    }


@pytest.fixture(autouse=True)
def _reset_level_cache(tmp_path, monkeypatch):
    level_dir = tmp_path / "levels"
    monkeypatch.setattr(rw, "LEVEL_CACHE", level_dir)
    monkeypatch.setattr(rw, "LEVEL_CACHE_FILE", level_dir / "levels.json")
    monkeypatch.setattr(rw, "_LEVEL_STATE", None)
    level_dir.mkdir(parents=True, exist_ok=True)
    yield


@pytest.fixture
def _stub_pdf(monkeypatch):
    monkeypatch.setattr(rw, "_download_pdf", lambda session, url, timeout=30.0: b"stub")


def _rows_to_dicts(rows: List[List[str]]) -> List[Dict[str, str]]:
    return [dict(zip(rw.PDF_COLUMNS, row)) for row in rows]


def test_pdf_selection_heuristic():
    resources = [
        {
            "name": "Generic attachment",
            "date": {"created": "2025-01-01T00:00:00Z"},
            "page_count": 2,
        },
        {
            "name": "Somalia Situation Report",
            "date": {"created": "2025-02-01T00:00:00Z"},
            "page_count": 1,
        },
    ]
    preferred = ["flash update", "situation report"]
    best = rw.select_best_pdf(resources, preferred)
    assert best["name"] == "Somalia Situation Report"


def test_table_precedence_over_narrative(monkeypatch, _stub_pdf):
    cfg = _base_cfg()
    fields = _base_fields("https://example.org/table.pdf")
    text = """
Country | People in Need | People Affected
Somalia | 120,000 | 90,000

The report notes that 50,000 people in need remain in hard-to-reach areas.
"""
    monkeypatch.setattr(rw, "smart_extract", lambda content, min_chars: (text, {"method": "native"}))
    rows = rw.build_pdf_rows_for_item(
        cfg,
        session=None,  # unused by stub download
        fields=fields,
        hazard_code="FL",
        hazard_label="Flood",
        hazard_class="natural",
        iso_pairs=[("Somalia", "SOM")],
        source_name="OCHA",
        source_type="sitrep",
        report_title="Somalia SitRep",
        source_url="https://reliefweb.int/report/123",
    )
    data = _rows_to_dicts(rows)
    assert data[0]["value_level"] == "120000"
    assert data[0]["value"] == "120000"
    assert data[0]["extraction_layer"] == "table"
    assert data[0]["method_value"] == "reported"


def test_infographic_precedence_over_narrative(monkeypatch, _stub_pdf):
    cfg = _base_cfg()
    fields = _base_fields("https://example.org/infographic.pdf")
    text = """
KEY FIGURES
People Affected: 45k individuals

Narrative states 10,000 people affected in rural zones.
"""
    monkeypatch.setattr(rw, "smart_extract", lambda content, min_chars: (text, {"method": "hybrid"}))
    rows = rw.build_pdf_rows_for_item(
        cfg,
        session=None,
        fields=fields,
        hazard_code="FL",
        hazard_label="Flood",
        hazard_class="natural",
        iso_pairs=[("Somalia", "SOM")],
        source_name="OCHA",
        source_type="sitrep",
        report_title="Somalia Flash Update",
        source_url="https://reliefweb.int/report/456",
    )
    data = _rows_to_dicts(rows)
    assert data[0]["value_level"] == "45000"
    assert data[0]["extraction_layer"] == "infographic"
    assert "text_extraction=hybrid" in data[0]["method_details"]


def test_household_to_people_conversion(monkeypatch, _stub_pdf):
    cfg = _base_cfg()
    fields = _base_fields("https://example.org/hh.pdf")
    text = """
Country | Households Affected
Somalia | 1,000
"""
    monkeypatch.setattr(rw, "smart_extract", lambda *_: (text, {"method": "native"}))
    rw.PPH._loaded = True
    rw.PPH._table = {"SOM": rw.PPHEntry("SOM", 5.2, "Survey", 2024)}
    rows = rw.build_pdf_rows_for_item(
        cfg,
        session=None,
        fields=fields,
        hazard_code="FL",
        hazard_label="Flood",
        hazard_class="natural",
        iso_pairs=[("Somalia", "SOM")],
        source_name="OCHA",
        source_type="sitrep",
        report_title="Somalia SitRep",
        source_url="https://reliefweb.int/report/789",
    )
    data = _rows_to_dicts(rows)
    assert data[0]["value_level"] == str(int(1000 * 5.2))
    assert data[0]["method_value"] == "derived_from_households"
    assert "PPH=5.20" in data[0]["method_details"]
    rw.PPH._loaded = False
    rw.PPH._table = {}


def test_date_picking_coverage_period(monkeypatch, _stub_pdf):
    cfg = _base_cfg()
    fields = _base_fields("https://example.org/date.pdf")
    text = """
Reporting period: 01–31 Aug 2025
Country | People in Need
Somalia | 1,234
"""
    monkeypatch.setattr(rw, "smart_extract", lambda *_: (text, {"method": "native"}))
    rows = rw.build_pdf_rows_for_item(
        cfg,
        session=None,
        fields=fields,
        hazard_code="FL",
        hazard_label="Flood",
        hazard_class="natural",
        iso_pairs=[("Somalia", "SOM")],
        source_name="OCHA",
        source_type="sitrep",
        report_title="Somalia SitRep",
        source_url="https://reliefweb.int/report/101",
    )
    data = _rows_to_dicts(rows)
    assert data[0]["as_of_date"] == "2025-08-31"


def test_multi_country_split(monkeypatch, _stub_pdf):
    cfg = _base_cfg()
    fields = _base_fields("https://example.org/multi.pdf")
    text = """
Country | People in Need
Somalia | 1,000
Kenya | 2,000

Combined statement: 3,000 people in need across Somalia and Kenya.
"""
    monkeypatch.setattr(rw, "smart_extract", lambda *_: (text, {"method": "native"}))
    rows = rw.build_pdf_rows_for_item(
        cfg,
        session=None,
        fields=fields,
        hazard_code="FL",
        hazard_label="Flood",
        hazard_class="natural",
        iso_pairs=[("Somalia", "SOM"), ("Kenya", "KEN")],
        source_name="OCHA",
        source_type="sitrep",
        report_title="Regional SitRep",
        source_url="https://reliefweb.int/report/202",
    )
    data = _rows_to_dicts(rows)
    assert {row["iso3"]: row["value_level"] for row in data} == {"SOM": "1000", "KEN": "2000"}

    # Ambiguous aggregate only should yield no rows.
    text_ambiguous = "Overall, 5,000 people in need across Somalia and Kenya."
    monkeypatch.setattr(rw, "smart_extract", lambda *_: (text_ambiguous, {"method": "native"}))
    ambiguous_rows = rw.build_pdf_rows_for_item(
        cfg,
        session=None,
        fields=fields,
        hazard_code="FL",
        hazard_label="Flood",
        hazard_class="natural",
        iso_pairs=[("Somalia", "SOM"), ("Kenya", "KEN")],
        source_name="OCHA",
        source_type="sitrep",
        report_title="Regional SitRep",
        source_url="https://reliefweb.int/report/202",
    )
    assert ambiguous_rows == []


def test_event_id_deterministic(monkeypatch, _stub_pdf):
    cfg = _base_cfg()
    fields = _base_fields("https://example.org/event.pdf")
    text = """
Country | People in Need
Somalia | 2,222
"""

    monkeypatch.setattr(rw, "smart_extract", lambda *_: (text, {"method": "native"}))

    rows_first = rw.build_pdf_rows_for_item(
        cfg,
        session=None,
        fields=fields,
        hazard_code="FL",
        hazard_label="Flood",
        hazard_class="natural",
        iso_pairs=[("Somalia", "SOM")],
        source_name="OCHA",
        source_type="sitrep",
        report_title="Somalia SitRep",
        source_url="https://reliefweb.int/report/303",
    )

    # Reset cache to simulate a fresh run with identical inputs.
    rw._LEVEL_STATE = None
    if rw.LEVEL_CACHE_FILE.exists():
        rw.LEVEL_CACHE_FILE.unlink()

    rows_second = rw.build_pdf_rows_for_item(
        cfg,
        session=None,
        fields=fields,
        hazard_code="FL",
        hazard_label="Flood",
        hazard_class="natural",
        iso_pairs=[("Somalia", "SOM")],
        source_name="OCHA",
        source_type="sitrep",
        report_title="Somalia SitRep",
        source_url="https://reliefweb.int/report/303",
    )

    first_id = rows_first[0][0]
    second_id = rows_second[0][0]
    assert first_id == second_id


def test_delta_computation(monkeypatch, _stub_pdf):
    cfg = _base_cfg()
    fields = _base_fields("https://example.org/delta.pdf")

    text_july = """
Reporting period: 01–31 Jul 2025
Country | People in Need
Somalia | 1,000
"""
    text_aug = """
Reporting period: 01–31 Aug 2025
Country | People in Need
Somalia | 1,500
"""

    monkeypatch.setattr(rw, "smart_extract", lambda *_: (text_july, {"method": "native"}))
    july_rows = rw.build_pdf_rows_for_item(
        cfg,
        session=None,
        fields=fields,
        hazard_code="FL",
        hazard_label="Flood",
        hazard_class="natural",
        iso_pairs=[("Somalia", "SOM")],
        source_name="OCHA",
        source_type="sitrep",
        report_title="Somalia July SitRep",
        source_url="https://reliefweb.int/report/404",
    )
    monkeypatch.setattr(rw, "smart_extract", lambda *_: (text_aug, {"method": "native"}))
    aug_rows = rw.build_pdf_rows_for_item(
        cfg,
        session=None,
        fields=fields,
        hazard_code="FL",
        hazard_label="Flood",
        hazard_class="natural",
        iso_pairs=[("Somalia", "SOM")],
        source_name="OCHA",
        source_type="sitrep",
        report_title="Somalia Aug SitRep",
        source_url="https://reliefweb.int/report/405",
    )
    july = _rows_to_dicts(july_rows)[0]
    aug = _rows_to_dicts(aug_rows)[0]
    assert july["value"] == "1000"
    assert aug["value"] == "500"


def test_ocr_threshold_trigger(monkeypatch, _stub_pdf):
    cfg = _base_cfg()
    cfg["enable_ocr"] = True
    fields = _base_fields("https://example.org/ocr.pdf")
    text = """
KEY FIGURES
People in Need: 10k
"""

    def fake_extract(content, min_chars):
        assert min_chars == cfg["min_text_chars_before_ocr"]
        return text, {"method": "hybrid"}

    monkeypatch.setattr(rw, "smart_extract", fake_extract)
    rows = rw.build_pdf_rows_for_item(
        cfg,
        session=None,
        fields=fields,
        hazard_code="FL",
        hazard_label="Flood",
        hazard_class="natural",
        iso_pairs=[("Somalia", "SOM")],
        source_name="OCHA",
        source_type="sitrep",
        report_title="Somalia OCR SitRep",
        source_url="https://reliefweb.int/report/406",
    )
    data = _rows_to_dicts(rows)
    assert "text_extraction=hybrid" in data[0]["method_details"]


def test_manifest_and_tier(tmp_path, monkeypatch):
    out_dir = tmp_path / "staging"
    out_dir.mkdir()
    monkeypatch.setattr(rw, "STAGING", out_dir)
    monkeypatch.setattr(rw, "PDF_OUTPUT", out_dir / "reliefweb_pdf.csv")
    monkeypatch.setattr(rw, "PDF_CACHE", out_dir / "cache")
    monkeypatch.setattr(rw, "LEVEL_CACHE", out_dir / "levels")
    monkeypatch.setattr(rw, "LEVEL_CACHE_FILE", out_dir / "levels" / "levels.json")
    monkeypatch.setattr(rw, "_LEVEL_STATE", None)

    def fake_make_rows():
        api_rows = []
        pdf_row = [
            "event-1",
            "Somalia",
            "SOM",
            "FL",
            "Flood",
            "natural",
            "in_need",
            "new",
            "100",
            "persons",
            "100",
            "2025-08-31",
            "2025-08-01",
            "2025-08-31",
            "OCHA",
            "sitrep",
            "https://reliefweb.int/report/manifest",
            "https://reliefweb.int/resource/manifest.pdf",
            "Somalia SitRep",
            "definition",
            "pdf",
            "reported",
            "extracted_from_table; text_extraction=native",
            "table",
            "Somalia | 100",
            "2",
            "med",
            1,
            "2025-09-01T00:00:00Z",
        ]
        tracker = {"count": 0, "mode": "post", "persisted": False}
        return api_rows, [pdf_row], tracker

    monkeypatch.setattr(rw, "make_rows", fake_make_rows)
    monkeypatch.delenv("RESOLVER_SKIP_RELIEFWEB", raising=False)
    rw.main()

    pdf_path = out_dir / "reliefweb_pdf.csv"
    assert pdf_path.exists()
    pdf_df = pd.read_csv(pdf_path, dtype=str)
    assert pdf_df["tier"].tolist() == ["2"]

    manifest_path = pdf_path.with_suffix(".csv.meta.json")
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    assert payload["row_count"] == 1
