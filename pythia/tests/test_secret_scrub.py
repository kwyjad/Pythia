# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The shared credential scrubber, in Python and in DuckDB."""

from __future__ import annotations

import duckdb
import pytest

from pythia.secret_scrub import (
    COMBINED_PATTERN,
    REDACTED,
    SQL_PATTERNS,
    contains_secret,
    scrub_text,
)

SECRETS = [
    "AIzaSyA1234567890abcdefghijkLMNOP",
    "sk-proj-abcdefghijklmnopqrstuvwx",
    "sk-ant-api03-abcdefghijklmnopqrstu",
    "ghp_abcdefghijklmnopqrstuvwxyz012345",
    "github_pat_11ABCDEFG0123456789_abcdefghij",
    "BSAabcdefghijklmnopqrstuvwx",
    "xoxb-1234567890-abcdefghij",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop",
]

# Ordinary prose that a naive pattern would mangle. The release scrub runs
# over published prompts and answers, so a false positive rewrites them.
CLEAN = [
    "the risk-assessment-framework for flood risk in the basin",
    "a task-based approach to disk-space-management reviews",
    "monkey=12 and key: value appear in the prompt as data",
    "https://reliefweb.int/report/somalia/floods?page=2&sort=date",
    # A public URL from the live release's Sibyl traces: a short id under a
    # parameter that happens to be called Key.
    "https://www.adrc.asia/view_disaster_en.php?NationCode=156&Lang=en&Key=2801",
    "",
]


@pytest.mark.parametrize("secret", SECRETS)
def test_every_credential_shape_is_scrubbed(secret):
    text = f"error calling provider with {secret} in it"
    out = scrub_text(text, values=[])
    assert secret not in out
    assert REDACTED in out


def test_a_query_parameter_key_of_any_shape_is_scrubbed():
    out = scrub_text("GET https://api.ipcinfo.org/population?key=deadbeef0123456789abcdef01234567&start=2020", values=[])
    assert "deadbeef0123456789abcdef01234567" not in out
    assert "?key=<redacted>&start=2020" in out


def test_an_environment_secret_with_no_known_shape_is_scrubbed():
    out = scrub_text("token was zq9-unusual-credential-value", values=["zq9-unusual-credential-value"])
    assert "zq9-unusual-credential-value" not in out


@pytest.mark.parametrize("text", CLEAN)
def test_ordinary_text_is_left_alone(text):
    assert scrub_text(text, values=[]) == text
    assert not contains_secret(text, values=[])


def test_non_strings_pass_through():
    assert scrub_text(None) is None
    assert scrub_text(42) == 42


def test_duckdb_and_python_agree_on_every_fixture():
    """The release scrub uses SQL_PATTERNS; the writers use scrub_text."""

    con = duckdb.connect()
    fixtures = [f"x {s} y" for s in SECRETS] + ["u?key=abc123xyz&v=1"] + CLEAN
    for text in fixtures:
        sql_out = text
        for pattern, replacement in SQL_PATTERNS:
            sql_out = con.execute(
                "SELECT regexp_replace(?, ?, ?, 'g')", [sql_out, pattern, replacement]
            ).fetchone()[0]
        assert sql_out == scrub_text(text, values=[]), text
        hit = con.execute("SELECT regexp_matches(?, ?)", [text, COMBINED_PATTERN]).fetchone()[0]
        assert hit == contains_secret(text, values=[]), text
