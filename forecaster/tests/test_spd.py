from __future__ import annotations

from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from forecaster.ensemble import _parse_spd_json, MemberOutput, EnsembleResult  # type: ignore
from forecaster.aggregate import aggregate_spd  # type: ignore
from forecaster.cli import _write_spd_ensemble_to_db, SPD_CLASS_BINS  # type: ignore


def test_parse_spd_json_basic():
    """Happy-path SPD parsing: minimal JSON, missing months filled, rows normalized."""
    text = """
    {
      "month_1": [0.1, 0.2, 0.3, 0.2, 0.2],
      "month_2": [0.0, 0.0, 0.0, 0.0, 1.0]
    }
    """
    out = _parse_spd_json(text)
    assert isinstance(out, dict)

    # We always materialise month_1..month_6
    expected_keys = {f"month_{i}" for i in range(1, 7)}
    assert set(out.keys()) == expected_keys

    # Each month is a length-5 prob vector that sums to ~1
    for key, vec in out.items():
        assert isinstance(vec, list)
        assert len(vec) == 5
        s = sum(vec)
        assert 0.99 <= s <= 1.01, f"{key} not normalised: sum={s}"

    # month_1 preserves the relative mass ordering
    m1 = out["month_1"]
    assert m1[2] == max(m1)  # bucket 3 has the highest weight


def test_parse_spd_json_failure_all_zero():
    """All-zero or empty SPDs should be treated as parse failure."""
    text = """
    {
      "month_1": [0, 0, 0, 0, 0],
      "month_2": [0, 0, 0, 0, 0]
    }
    """
    out = _parse_spd_json(text)
    assert out is None

    # Completely invalid JSON also returns None
    bad = "this is not json at all"
    assert _parse_spd_json(bad) is None


def test_aggregate_spd_shape_and_uniform_fallback():
    """aggregate_spd should normalise, respect evidence, and fall back to uniform."""
    # Member with a strong preference for bucket 1 in month_1 only
    spd_single = {"month_1": [1.0, 0.0, 0.0, 0.0, 0.0]}
    member = MemberOutput(
        name="m1",
        ok=True,
        parsed=spd_single,
        raw_text="",
    )
    ens = EnsembleResult(members=[member])

    spd_mean, expected, summary = aggregate_spd(ens)

    # month_1 is concentrated in bucket 1
    m1 = spd_mean["month_1"]
    assert len(m1) == 5
    assert 0.99 <= sum(m1) <= 1.01
    assert m1[0] == max(m1)

    # Months without explicit evidence should be (approximately) uniform
    for m_idx in range(2, 7):
        key = f"month_{m_idx}"
        v = spd_mean[key]
        assert len(v) == 5
        assert 0.99 <= sum(v) <= 1.01
        # All entries should be close to each other
        assert max(v) - min(v) < 0.05

    # Expected values are consistent with SPD and bucket centroids
    # (sanity: they are positive and ordered by month key)
    assert expected
    assert set(expected.keys()) == {f"month_{i}" for i in range(1, 7)}
    for val in expected.values():
        assert val > 0.0


@pytest.mark.db
def test_write_spd_ensemble_to_db_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Write a simple SPD to DuckDB and confirm row counts and normalisation."""
    # Use a private DuckDB file and override config lookup
    db_path = tmp_path / "pythia_spd_test.duckdb"
    db_url = f"duckdb:///{db_path}"
    monkeypatch.setattr("forecaster.cli._pythia_db_url_from_config", lambda: db_url)

    # Simple SPD: uniform across all buckets and months
    spd_main = {f"month_{i}": [1.0 / len(SPD_CLASS_BINS)] * len(SPD_CLASS_BINS) for i in range(1, 7)}

    question_id = "q-spd-test"
    _write_spd_ensemble_to_db(question_id=question_id, run_id="run-test", spd_main=spd_main)

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT horizon_m, class_bin, p FROM forecasts_ensemble WHERE question_id=? ORDER BY horizon_m, class_bin",
            [question_id],
        ).fetchall()
    finally:
        con.close()

    # 6 months × 5 buckets = 30 rows
    assert len(rows) == 6 * len(SPD_CLASS_BINS)

    # Per-horizon sum of probabilities should be ~1.0 and include all buckets
    by_h = {}
    for horizon_m, class_bin, p in rows:
        by_h.setdefault(horizon_m, []).append(float(p))
        assert class_bin in SPD_CLASS_BINS

    assert set(by_h.keys()) == set(range(1, 7))
    for h, probs in by_h.items():
        assert len(probs) == len(SPD_CLASS_BINS)
        s = sum(probs)
        assert 0.99 <= s <= 1.01, f"horizon {h} not normalised: sum={s}"
