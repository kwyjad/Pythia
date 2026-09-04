# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""A tier-0 connector that could not READ its source must not exit 0.

On 2026-09-03 ACLED authentication failed outright. ``acled_client.main()``
caught the exception, wrote a header-only CSV, printed ``rows=0 (error: ...)``
and returned False. The ``__main__`` shim discarded that bool, so the
process exited 0. ``run_connectors.py`` reads the exit code, so the FATAL
Phase 1 connectors step went green having written no tier-0 conflict rows at
all, and the run only died a step later, in the monthly-fatalities CLI, on a bare
traceback that named nothing.

"The source said nothing happened" and "we could not read the source" are
different facts. A deliberate skip and an empty window are answers and exit 0;
an unreachable source is not, and must be red.
"""

from __future__ import annotations

import pytest

from resolver.ingestion import acled_client, ifrc_go_client


# --- ACLED -------------------------------------------------------------------


def test_acled_unreadable_source_exits_non_zero(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("RESOLVER_SKIP_ACLED", raising=False)
    monkeypatch.setattr(acled_client, "OUT_PATH", tmp_path / "acled.csv")
    monkeypatch.setattr(acled_client, "_write_run_summary", lambda *a, **k: None)

    def boom():
        raise RuntimeError(
            "ACLED OAuth password grant returned HTTP 200 with a body that is not JSON"
        )

    monkeypatch.setattr(acled_client, "collect_rows", boom)

    assert acled_client.cli_main() == 1
    out = capsys.readouterr().out
    assert "::error title=ACLED could not be read::" in out
    assert "not an empty month" in out
    # The header-only CSV is still written, because downstream steps must not crash on
    # a missing file just because the pull failed.
    assert (tmp_path / "acled.csv").exists()


def test_acled_empty_window_exits_zero(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("RESOLVER_SKIP_ACLED", raising=False)
    monkeypatch.setattr(acled_client, "OUT_PATH", tmp_path / "acled.csv")
    monkeypatch.setattr(acled_client, "collect_rows", lambda: [])

    assert acled_client.cli_main() == 0
    assert "::error" not in capsys.readouterr().out


def test_acled_deliberate_skip_exits_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("RESOLVER_SKIP_ACLED", "1")
    monkeypatch.setattr(acled_client, "OUT_PATH", tmp_path / "acled.csv")

    assert acled_client.cli_main() == 0


def test_acled_success_exits_zero(monkeypatch, tmp_path):
    monkeypatch.delenv("RESOLVER_SKIP_ACLED", raising=False)
    monkeypatch.setattr(acled_client, "OUT_PATH", tmp_path / "acled.csv")
    monkeypatch.setattr(acled_client, "collect_rows", lambda: [["x"] * len(acled_client.CANONICAL_HEADERS)])
    monkeypatch.setattr(acled_client, "_write_rows", lambda rows, path: None)

    assert acled_client.cli_main() == 0


def test_acled_unread_state_does_not_leak_between_runs(monkeypatch, tmp_path):
    """A failed run must not make the next, healthy run red."""
    monkeypatch.delenv("RESOLVER_SKIP_ACLED", raising=False)
    monkeypatch.setattr(acled_client, "OUT_PATH", tmp_path / "acled.csv")
    monkeypatch.setattr(acled_client, "_write_run_summary", lambda *a, **k: None)

    def boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(acled_client, "collect_rows", boom)
    assert acled_client.cli_main() == 1

    monkeypatch.setattr(acled_client, "collect_rows", lambda: [])
    assert acled_client.cli_main() == 0


def test_acled_main_still_returns_false_on_skip(monkeypatch, tmp_path):
    """test_connectors_headers.py pins this contract; the shim must not change it."""
    monkeypatch.setenv("RESOLVER_SKIP_ACLED", "1")
    monkeypatch.setattr(acled_client, "OUT_PATH", tmp_path / "acled.csv")
    assert acled_client.main() is False


# --- IFRC GO -----------------------------------------------------------------


def test_ifrc_unreadable_source_exits_non_zero(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("RESOLVER_SKIP_IFRCGO", raising=False)
    monkeypatch.setattr(ifrc_go_client, "resolve_output_path", lambda p: tmp_path / "ifrc_go.csv")

    def boom():
        raise RuntimeError("GO API token rejected")

    monkeypatch.setattr(ifrc_go_client, "collect_rows", boom)

    assert ifrc_go_client.cli_main() == 1
    out = capsys.readouterr().out
    assert "::error title=IFRC GO could not be read::" in out
    assert (tmp_path / "ifrc_go.csv").exists()


def test_ifrc_empty_window_exits_zero(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("RESOLVER_SKIP_IFRCGO", raising=False)
    monkeypatch.setattr(ifrc_go_client, "resolve_output_path", lambda p: tmp_path / "ifrc_go.csv")
    monkeypatch.setattr(ifrc_go_client, "collect_rows", lambda: [])

    assert ifrc_go_client.cli_main() == 0
    assert "::error" not in capsys.readouterr().out


def test_ifrc_deliberate_skip_exits_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("RESOLVER_SKIP_IFRCGO", "1")
    monkeypatch.setattr(ifrc_go_client, "resolve_output_path", lambda p: tmp_path / "ifrc_go.csv")

    assert ifrc_go_client.cli_main() == 0
