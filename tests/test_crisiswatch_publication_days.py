# Pythia / Copyright (c) 2025 Kevin Wyjad
"""The CrisisWatch publication-day measurement: pure logic, no network."""

from scripts.ci import crisiswatch_publication_days as pd


def _cdx(per_url):
    def cdx(url, start, end, match_type):
        return [(ts, st) for ts, st in per_url.get((url, match_type), []) if start <= ts[:8] <= end]
    return cdx


def test_edition_slugs_cover_the_december_year_boundary():
    assert pd.edition_slugs(2025, 12) == [
        "crisisgroup.org/crisiswatch/december-trends-and-january-alerts-2025",
        "crisisgroup.org/crisiswatch/december-trends-and-january-alerts-2026",
    ]
    assert pd.edition_slugs(2026, 3) == [
        "crisisgroup.org/crisiswatch/march-trends-and-april-alerts-2026",
    ]


def test_bracket_from_the_main_page_and_the_edition_page():
    main = [("20260401120000", "200"), ("20260403120000", "200"), ("20260405120000", "200")]
    shown = {"20260401120000": (2026, 2), "20260403120000": (2026, 2), "20260405120000": (2026, 3)}
    cdx = _cdx({
        ("crisisgroup.org/crisiswatch", "exact"): main,
        ("crisisgroup.org/crisiswatch/march-trends-and-april-alerts-2026", "exact"): [("20260404090000", "200")],
    })
    row = pd.measure_edition(2026, 3, cdx=cdx, edition_of=shown.get)
    assert row.main_page_last_older == "20260403120000"
    assert row.main_page_first == "20260405120000"
    assert row.edition_page_first == "20260404090000"
    assert row.lower_bound_day == 3
    assert row.upper_bound_day == 4  # the earlier of the two routes


def test_a_late_edition_stops_the_schedule_change():
    late = pd.EditionDay(edition="2026-05", main_page_first="20260612000000")
    early = pd.EditionDay(edition="2026-06", main_page_first="20260702000000")
    v = pd.verdict([late, early], 9)
    assert v["ok"] is False and v["late_editions"] == ["2026-05"]
    assert "STOP" in pd.render([late, early], v)


def test_a_capture_two_months_on_counts_as_late_not_unknown():
    row = pd.EditionDay(edition="2026-05", main_page_first="20260702000000")
    assert row.upper_bound_day == 99
    assert pd.verdict([row], 9)["late_editions"] == ["2026-05"]


def test_nothing_captured_is_reported_unmeasured_not_ok_silently():
    row = pd.measure_edition(2026, 1, cdx=_cdx({}), edition_of=lambda ts: None)
    v = pd.verdict([row], 9)
    assert v["unmeasured_editions"] == ["2026-01"]
    assert "Not measured" in pd.render([row], v)


def test_reliefweb_repost_can_tighten_the_upper_bound():
    cdx = _cdx({("crisisgroup.org/crisiswatch", "exact"): [("20260715000000", "200")]})
    row = pd.measure_edition(
        2026, 6, cdx=cdx, edition_of=lambda ts: (2026, 6),
        reliefweb=lambda y, m: ("20260703", "CrisisWatch June 2026"),
    )
    assert row.main_page_first == "20260715000000"
    assert row.upper_bound_day == 3
    assert "2026-07-03" in pd.render([row], pd.verdict([row], 9))
