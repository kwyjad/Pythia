# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

Param(
  [string]$WheelDir = "tools/offline_wheels"
)

python -m pip install --upgrade pip
python -m pip install --no-index --find-links $WheelDir -r "$WheelDir/constraints-db.txt"
