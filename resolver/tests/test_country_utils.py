import csv
from pathlib import Path

from resolver.ingestion.utils.country_utils import (
    load_all_iso3,
    parse_countries_env,
    resolve_countries,
)


def _load_name_map() -> dict[str, str]:
    master_path = Path("resolver/ingestion/static/iso3_master.csv")
    mapping: dict[str, str] = {}
    with master_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code = str(row.get("admin0Pcode", "")).strip().upper()
            name = str(row.get("admin0Name", "")).strip().upper()
            if code and name and name not in mapping:
                mapping[name] = code
    return mapping


def test_parse_countries_env_parses_and_normalizes() -> None:
    master_codes = set(load_all_iso3())
    name_map = _load_name_map()
    resolved = parse_countries_env("afg, pak\nKenya", master_codes, name_map)
    assert resolved == ["AFG", "PAK", "KEN"]


def test_resolve_countries_env_override_wins_over_config() -> None:
    resolved = resolve_countries(["AFG"], "PAK,ETH")
    assert resolved == ["PAK", "ETH"]


def test_resolve_countries_empty_list_means_all() -> None:
    all_codes = load_all_iso3()
    resolved = resolve_countries([])
    assert resolved == all_codes
    assert len(resolved) == len(all_codes)
