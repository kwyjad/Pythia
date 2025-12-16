#!/usr/bin/env bash
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

# Common environment flags to keep CI runs side-effect free and quiet.
export DISABLE_GIT_PUSH=1
export RESOLVER_CI=1
export PYTHONWARNINGS=ignore
