# Analysis reports on the dashboard

Fred writes three kinds of report that look back at its own work. Scheduled Claude sessions write them as Markdown into Google Drive; the dashboard's **Performance Analysis** page (`/performance-analysis`) lists them, shows them, and offers each as a PDF and as Markdown.

| Type id | Display name | File name | Written |
|---|---|---|---|
| `skill` | Skill analysis | `fred_performance_<YYYY-MM>.md` | after each scoring round (11th) |
| `resolver` | Resolver pipeline review | `fred_resolver_review_<YYYY-MM>.md` | after each scoring round |
| `forecast` | Forecast pipeline review | `fred_forecast_review_<YYYY-MM-DD>__<fc_run_id>.md` | after each forecast chain |

Any of them may end `__rerun_<YYYY-MM-DD>.md`: a newer version of the same report, which never replaces the original. The page shows the newest and lists the others under "Earlier versions".

## Flow

```
Drive: Fred_archive/Performance_Reports/<YYYY>/*.md
  -> sync_analysis_reports.yml (daily 09:41 UTC; rclone remote fred_reports, drive.readonly)
       scripts/ci/analysis_reports.py build: parse, credential scan, PDF (WeasyPrint)
  -> GitHub release "analysis-reports": <id>.md, <id>.pdf, index.json
  -> API: /v1/analysis_reports, /{id}/content, /{id}/download?format=md|pdf
  -> page: /performance-analysis and /performance-analysis/<id>
```

The API never reads Drive and holds no Drive credential. The `analysis-reports` release is separate from the data release (`pythia-data-latest`); nothing here touches the canonical DB.

## What the sync reads from a report

Each report starts with a `# title`, a `| Field | Value |` table and a `## 1. Summary` section.

- **Chain**: the first header value matching `(\d{4}-\d{2}-\d{2})__(scoring|forecast)__(?:(fc_\d+)__)?gh(\d+)`, e.g. `pythia_2026-11-11__scoring__gh43009722469.zip` or `2026-10-13__forecast__fc_1791866322__gh38355281121`. With none, the report is still indexed under its file-name month and the job logs a warning.
- **Dry run**: `DRY RUN` anywhere before `## 1.`.
- **Template version** and **Prepared**: the table rows of those names (the first date in Prepared).
- **Summary**: the first paragraph under `## 1. Summary`, Markdown stripped, cut at a sentence end below 320 characters.
- **Verdict** (forecast reviews only): `worked with faults`, `failed` or `worked` in the summary's first sentence.

Files under `template/`, and files whose names match none of the patterns, are skipped and named in the job summary.

## index.json

```json
{
  "schema_version": 1,
  "generated_at": "2026-11-26T09:41:00+00:00",
  "source": "Fred_archive/Performance_Reports",
  "reports": [
    {
      "report_id": "fred_forecast_review_2026-10-13__fc_1791866322",
      "type": "forecast", "display_name": "Forecast pipeline review",
      "period": "2026-10", "file_date": "2026-10-13", "file_fc_run_id": "fc_1791866322",
      "rerun_of": null, "rerun_date": null,
      "title": "...", "chain": {"date": "...", "kind": "forecast", "fc_run_id": "...",
                                "publish_run_id": "...", "folder_name": "..."},
      "dry_run": false, "template_version": "1.0 (2026-11-24)", "prepared": "2026-11-25",
      "summary": "...", "verdict": "worked with faults",
      "sha256": "...", "bytes": 43019, "drive_modified": "...",
      "md_asset": "<id>.md", "pdf_asset": "<id>.pdf", "removed_from_drive": false
    }
  ]
}
```

`pdf_asset` is null when the PDF failed to render. A report that has gone from Drive stays in the index with `removed_from_drive: true`; assets are never deleted.

## Upload rules

- Only assets whose `sha256` changed since the current `index.json` are uploaded; `index.json` goes last, so it never names an asset that is not there yet.
- A report whose text matches a credential in the job's environment is blocked (`scan_for_secrets`, the cycle bundle's scan), not uploaded, and named in the job summary.
- A PDF that fails is logged and the report is indexed with `pdf: null`.

## The page

Months newest first; the latest open. Each month has a **scoring round** (skill analysis and resolver review; a missing one shows "Not yet written") and **forecast runs** (one block per run). A production forecast run since 1 October 2026 with no review shows "Review pending"; the API finds those runs in `forecasts_ensemble` (test runs never get a placeholder). Filters (type, year, dry runs) live in the URL: `types=skill,forecast`, `year=2026`, `dry=0`.

## Settings

| Setting | Where | Default |
|---|---|---|
| `RCLONE_REPORTS_CONF_B64` | repository secret | (none: the sync says so and exits green). Setup: docs/SECRETS.md, "Analysis reports" |
| `ANALYSIS_REPORTS_TAG` | workflow env | `analysis-reports` |
| `PYTHIA_ANALYSIS_REPORTS_TAG` | API env (Render) | `analysis-reports` |
| `PYTHIA_DATA_REPO` | API env (Render) | already set for the data release |

## Force a rebuild

**Actions → Sync Analysis Reports → Run workflow**, tick `force_rebuild`. Every PDF is rendered again and uploaded; Markdown is uploaded only where it changed.
