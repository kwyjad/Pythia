# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Route-group modules for the Pythia API.

Each module defines ``router = APIRouter()`` and holds the endpoint
functions moved verbatim out of ``pythia.api.app`` (July 2026
decomposition). Shared infrastructure lives in ``pythia.api.core``;
route modules must never import ``pythia.api.app``.
"""
