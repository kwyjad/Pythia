# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Collectors for the operational half of the Pythia debug bundle.

The twelve files the bundle emitted before this package confirmed that the
pipeline ran. They could not say why it ran badly: diagnosing the
2026-09-01 run needed workflow logs, provider batch objects, git history
and source files that lived outside the zip, and that nobody downstream of
the runner can reach. Each module here answers one of those questions from
inside the bundle.

Every collector is written to the same contract, because a diagnostics bug
must never take down a forecast pipeline:

* it never raises out of ``collect``;
* the caller wraps it anyway (``scripts.dump_pythia_debug_bundle._run_collector``),
  so a failure writes a stub file recording the error and the phase continues;
* anything that looks like a credential is redacted before it is written.
"""
