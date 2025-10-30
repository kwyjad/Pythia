# Configuration Files

This repository keeps long-lived exporter and snapshot configuration in
`resolver/tools/export_config.yml`. The file is consumed by several fast
pytest suites, so a malformed edit can break test setup before anything runs.

## Guard rails

* `pytest -k export_config_yaml_valid` runs
  `resolver/tests/test_export_config_yaml_valid.py`, which verifies the YAML is
  parseable and still exposes a top-level `metrics` mapping.
* When editing configuration, run
  ``python -c "import yaml; yaml.safe_load(open('resolver/tools/export_config.yml'))"``
  to double-check parsing locally.

## Overlays for experimental work

Experimental or connector-specific additions should live in overlay files until
they are ready for the main config. For example, the stub IDMC metrics live in
`resolver/tools/export_config_idmc.overlay.yml` with the entries disabled by
default. Copy overlays into your local environment when you need to test them,
without risking the shared baseline.
