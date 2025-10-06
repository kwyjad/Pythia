from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import pytest

from resolver.tools.schema_validate import load_schema, validate_staging_csv

REPO_ROOT = Path(__file__).resolve().parents[2]
RESOLVER_ROOT = REPO_ROOT / "resolver"
TOOLS_DIR = RESOLVER_ROOT / "tools"
STAGING_DIR = RESOLVER_ROOT / "staging"

STAGING_FILES: List[Path] = sorted(STAGING_DIR.glob("*.csv"))

SCHEMA_PATH = TOOLS_DIR / "schema.yml"


@pytest.fixture(scope="session")
def schema() -> Dict[str, object]:
    return load_schema(SCHEMA_PATH)


@pytest.mark.parametrize(
    "csv_path",
    [pytest.param(path, id=path.name) for path in STAGING_FILES]
    if STAGING_FILES
    else [
        pytest.param(
            None,
            id="no-staging-files",
            marks=pytest.mark.skip(reason="resolver/staging is empty; run the ingestor first"),
        )
    ],
)
def test_staging_csv_matches_schema(csv_path: Path | None, schema: Dict[str, object]) -> None:
    if csv_path is None:
        pytest.skip("resolver/staging is empty; run resolver/ingestion/run_all_stubs.py")

    ok, errors = validate_staging_csv(csv_path, schema)
    if not ok:
        formatted = "\n  - " + "\n  - ".join(errors)
        pytest.fail(f"{csv_path.name} schema violations:{formatted}")


def test_validate_known_entity_with_temp_files(tmp_path: Path, schema: Dict[str, object]) -> None:
    entity = schema["entities"].get("staging.common", {})
    columns = entity.get("columns", [])
    required_names = [col["name"] for col in columns if col.get("required")]

    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    valid_path = valid_dir / "ifrc_go.csv"

    valid_row = {
        "event_id": "TEST-1",
        "country_name": "Testland",
        "iso3": "TST",
        "hazard_code": "TC",
        "hazard_label": "Test Cyclone",
        "hazard_class": "Cyclone",
        "metric": "affected",
        "value": "123",
        "unit": "persons",
        "as_of_date": "2025-01-02",
        "publication_date": "2025-01-03",
        "publisher": "Example Org",
        "source_type": "agency",
        "source_url": "https://example.org/report",
        "doc_title": "Synthetic event",
        "definition_text": "Synthetic definition",
        "method": "estimate",
        "confidence": "med",
        "revision": "1",
        "ingested_at": "2025-01-03",
    }

    with valid_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(required_names)
        writer.writerow([valid_row.get(name, "") for name in required_names])

    ok, errors = validate_staging_csv(valid_path, schema)
    assert ok, f"Expected valid CSV, got errors: {errors}"

    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()
    missing_path = missing_dir / "ifrc_go.csv"

    missing_headers = [name for name in required_names if name != "unit"]
    with missing_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(missing_headers)
        writer.writerow([valid_row.get(name, "") for name in missing_headers])

    ok, errors = validate_staging_csv(missing_path, schema)
    assert not ok
    assert any("unit" in err for err in errors)

    enum_dir = tmp_path / "invalid_enum"
    enum_dir.mkdir()
    enum_path = enum_dir / "ifrc_go.csv"

    with enum_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(required_names)
        bad_row = valid_row.copy()
        bad_row["metric"] = "invalid"
        writer.writerow([bad_row.get(name, "") for name in required_names])

    ok, errors = validate_staging_csv(enum_path, schema)
    assert not ok
    joined = " ".join(errors)
    assert "invalid enum" in joined
