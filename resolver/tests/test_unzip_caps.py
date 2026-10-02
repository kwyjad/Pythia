# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""A downloaded zip is read no further than its cap, whatever it claims.

A few hundred bytes can unpack to gigabytes, so a hostile or broken archive
must be refused before it fills a runner's memory, not after.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from resolver.hazard_resolution import drought_indicators


def _zip(name: str, payload: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, payload)
    return buf.getvalue()


def test_a_member_past_the_cap_is_refused(monkeypatch):
    monkeypatch.setattr(drought_indicators, "_MAX_UNZIPPED_BYTES", 1000)
    blob = _zip("hotspots.csv", b"a" * 50_000)
    assert len(blob) < 1000  # compresses far below the cap; the cap is on what it unpacks to
    with pytest.raises(ValueError, match="unpacks past"):
        drought_indicators._unzip_first_table(blob)


def test_a_member_under_the_cap_is_read():
    blob = _zip("hotspots.csv", b"asap0_id;date;hs_code\n1;2026-08-01;1\n")
    text, kind = drought_indicators._unzip_first_table(blob)
    assert kind == "text/csv"
    assert "hs_code" in text
