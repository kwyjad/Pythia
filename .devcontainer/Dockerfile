# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/devcontainers/python:3.11

# Optional: system deps (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Workdir is set by devcontainer.json; keep image lean
