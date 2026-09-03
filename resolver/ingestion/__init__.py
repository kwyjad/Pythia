# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.


# Every ingestion connector runs as its own `python -m resolver.ingestion.<name>`
# subprocess (see scripts/ci/run_connectors.py), so installing the HTTP
# recorder in each of their mains would be ten edits and a standing invitation
# to forget the eleventh. Installing it here reaches all of them, including
# connectors written after this comment.
#
# It is a no-op unless PYTHIA_RUN_LOG_DIR is set, so an import outside CI costs
# one environment lookup, and a failure inside it is swallowed: a diagnostic
# must never be the reason a connector cannot start.
try:  # pragma: no cover - exercised through the connector subprocesses
    from resolver.diagnostics.http_recorder import maybe_install_from_env as _install

    _install()
except Exception:  # noqa: BLE001
    pass
