# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""AI analysis bundle builders.

One self-describing zip per pipeline milestone, designed to be handed to an
AI analyst (Claude Code agent or a chat upload) so it can reconstruct how
forecasts were made and how they performed. The scored-forecast bundle
(`build_scored_forecast_bundle`) is built at the calibration terminus, where
the canonical DB contains forecast-time reasoning AND realized outcomes.
"""
