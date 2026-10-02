# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately
through GitHub: open the repository's **Security** tab and choose **Report a
vulnerability**. The report reaches the maintainer alone.

Say what you found, where, and how to reproduce it. A report that names a file
and a line, or a request and its response, is the quickest to act on. You
should hear back within a week. Please give a reasonable time for a fix before
describing the problem in public.

## What is public by design

Pythia publishes its work, so some things that look sensitive are meant to be
seen. None of these is a vulnerability in itself:

- **Forecasts and their inputs.** Questions, forecast distributions, scores,
  resolutions and calibration weights, on the dashboard and through the API.
- **Model transcripts.** The prompts sent to each model and the responses it
  gave, including the excerpts of source reports a prompt quotes.
- **The release database.** The DuckDB file attached to the
  `pythia-data-latest` release, which the API serves. It is built by
  `scripts/ci/build_release_db.py`, which leaves out licensed third-party rows
  and pipeline internals and scrubs credentials from every text column.

What would be a vulnerability: a credential in any published file, log or
artifact; licensed raw data (ACLED events, EM-DAT records, ACAPS narrative
products) reachable from the release or the API; a way to make the API read a
file, reach a network address or exhaust the host; script running in the
dashboard's origin; or a way for a pull request from a fork to influence what
the scheduled pipeline runs or publishes.

## Scope

The code in this repository, the GitHub Actions workflows in
`.github/workflows/`, the public API, and the dashboard. Third-party services
Pythia reads from are out of scope; report problems with them to their owners.
