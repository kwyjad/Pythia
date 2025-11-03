from resolver.ingestion.utils.country_utils import load_all_iso3, resolve_countries


def test_empty_list_means_all_countries() -> None:
    all_codes = load_all_iso3()
    resolved = resolve_countries([])
    assert resolved == all_codes
    assert len(resolved) == len(all_codes)
