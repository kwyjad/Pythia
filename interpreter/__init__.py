# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Interpreter: one plain-language report per cycle, explaining Fred's output.

The model (Claude Opus 5, role ``interpreter``) explains; it never
calculates. All arithmetic arrives pre-computed in the input pack (the
Phase 1 deviation metrics and Phase 2 bundles); the model emits structured
JSON whose prose carries ``{{fig:...}}`` placeholders that the renderer
resolves against the pack — the model's own idea of a number is never
printed. Generation is fully automatic: no draft state, no approval gate.

Entry point: ``python -m interpreter.run`` (see run.py). Storage:
``interpretations`` table (pythia/db/schema.py, mirrored in store.py).
"""
