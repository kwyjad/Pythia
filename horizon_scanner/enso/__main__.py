# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Entry point for ``python -m horizon_scanner.enso``.

``python -m horizon_scanner.enso.enso_module`` imports the package first
(which imports ``enso_module`` for its own re-exports) and then runs the
module a second time under ``__main__``, so runpy warns:

    RuntimeWarning: 'horizon_scanner.enso.enso_module' found in sys.modules
    after import of package 'horizon_scanner.enso', but prior to execution

The module is loaded twice, its module-level state exists twice, and the
step that backfilled 919 ONI rows ran under a warning. Invoking the package
runs this file instead, which imports the module exactly once.
"""

from __future__ import annotations

from horizon_scanner.enso.enso_module import main

if __name__ == "__main__":
    raise SystemExit(main())
