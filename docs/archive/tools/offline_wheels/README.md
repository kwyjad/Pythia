# Offline DuckDB Wheels

This directory stores pre-downloaded wheels that enable installing the resolver's
DB testing dependencies without requiring outbound internet access. Two usage
patterns are supported:

## Offline cache (recommended)

1. On a machine with internet connectivity, run `python scripts/download_db_wheels.py`
   to refresh the cached wheel set.
2. Commit the downloaded `*.whl` files along with this README so team members on
   restricted networks can install the dependencies directly from source
   control.
3. On an offline or behind-proxy machine, execute
   `scripts/install_db_extra_offline.sh` (or the PowerShell variant) to install
   from the cached wheels using `pip --no-index --find-links`.

## Proxy-assisted install (alternative)

If corporate policy allows proxy-based access, configure the `HTTP_PROXY` and
`HTTPS_PROXY` environment variables (or a `pip.conf`/`pip.ini` file) so pip can
reach PyPI directly. The repository README documents sample commands for Linux,
macOS, and Windows environments.

Keep this directory small: only DuckDB, pytest, httpx, and other DB-contract-test
requirements should be cached here. Refresh the wheels whenever versions bump in
CI to maintain parity between offline installs and automated jobs.
