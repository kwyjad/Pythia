from __future__ import annotations

from scripts.dump_pythia_debug_bundle import main as debug_bundle_main


DEPRECATION_NOTE = (
    "[info] dump_first_forecast_prompt is deprecated; use dump_pythia_debug_bundle instead."
)


def main() -> None:
    print(DEPRECATION_NOTE)
    debug_bundle_main()


if __name__ == "__main__":
    main()
