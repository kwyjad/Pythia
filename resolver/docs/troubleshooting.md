# Troubleshooting Guide

This guide captures common issues encountered when running Resolver locally or in CI, with a focus on the ReliefWeb PDF pipeline.

## ReliefWeb API challenges

- **HTTP 202 (WAF challenge):** ReliefWeb occasionally returns a `202 Accepted` with JavaScript challenges. The client falls back to a header-only CSV. Re-run later or supply cached manifests. The connector logs `waf_challenge_detected=1` when this occurs.
- **Rate limits:** Respect backoff hints in `retry_after` headers; the runner automatically sleeps, but you can reduce concurrency by running `python resolver/ingestion/reliefweb_client.py --max-workers 1`.

## PDF downloads

- **Network disabled:** CI sets `RELIEFWEB_PDF_ALLOW_NETWORK=0`, so attachment downloads are skipped. Ensure mocks cover required fixtures before enabling the branch in tests.
- **Invalid URLs or empty files:** The connector records failures in the manifest with `status=error`. Delete `resolver/staging/.cache/reliefweb/pdf/` to retry downloads locally.

## OCR fallback

- **Unexpected OCR usage:** Decrease `RELIEFWEB_PDF_ENABLE_OCR` or raise `min_text_chars_before_ocr` in the config. OCR can be slower and may require language packs not installed in minimal containers.
- **OCR failures:** When OCR raises errors, the connector logs `extraction_method=ocr_failed` and skips the metric. Re-run with `RELIEFWEB_PDF_ENABLE_OCR=0` to avoid repeated failures.

## Date and metric anomalies

- **Mixed date formats:** The parser attempts multiple date patterns. Check connector logs for `date_parse_warning` messages and manually patch the PDF metadata if required.
- **Household conversion surprises:** Inspect the `pph_used` column and confirm the value exists in `avg_household_size.csv` or overrides. Supply a custom file via `RELIEFWEB_PPH_OVERRIDE_PATH` when testing scenario-specific multipliers.
- **Empty PDFs or images only:** Native extraction may return little text. Enable OCR for such cases, or mark the attachment as `pdf_skip=1` in the manifest to avoid repeated attempts.

## General debugging tips

- Inspect `resolver/logs/ingestion/*.log` for structured details (JSON) including selector scores and parsing diagnostics.
- Use `pytest -vv` on targeted tests (e.g., `pytest -vv resolver/tests/ingestion/test_reliefweb_pdf.py::test_household_conversion`) to reproduce parsing paths quickly.
- Confirm schema expectations with `pytest -q resolver/tests/test_staging_schema_all.py` after modifying connectors.
