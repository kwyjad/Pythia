# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from scripts.dump_pythia_debug_bundle import main as debug_bundle_main


DEPRECATION_NOTE = (
    "[info] dump_forecast_run_debug is deprecated; use dump_pythia_debug_bundle instead."
)


def main() -> None:
    print(DEPRECATION_NOTE)
    debug_bundle_main()


if __name__ == "__main__":
    main()
