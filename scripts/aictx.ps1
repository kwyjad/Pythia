# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

param([string]$Base = "origin/main")
python tools/context_pack.py --base $Base
Write-Host "Context pack created under .\context\"
