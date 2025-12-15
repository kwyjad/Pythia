# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Regression tests for guarded logging initialisation."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from resolver.ingestion import dtm_client


@pytest.mark.usefixtures("monkeypatch")
def test_dtm_file_logging_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log_path = tmp_path / "dtm_client.log"
    monkeypatch.setattr(dtm_client, "DTM_CLIENT_LOG_PATH", log_path)
    # Remove pre-existing handlers to avoid leakage between tests
    for handler in list(dtm_client.LOG.handlers):
        if isinstance(handler, logging.FileHandler):
            dtm_client.LOG.removeHandler(handler)
            handler.close()
    dtm_client._FILE_LOGGING_INITIALIZED = False

    dtm_client._setup_file_logging()
    dtm_client._setup_file_logging()

    file_handlers = [
        handler for handler in dtm_client.LOG.handlers if isinstance(handler, logging.FileHandler)
    ]
    assert len(file_handlers) == 1
    assert Path(file_handlers[0].baseFilename) == log_path
    for handler in file_handlers:
        dtm_client.LOG.removeHandler(handler)
        handler.close()
