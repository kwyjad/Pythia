const API_BASE =
  process.env.NEXT_PUBLIC_PYTHIA_API_BASE ?? "http://localhost:8000/v1";

const CSV_DOWNLOAD_URL = `${API_BASE}/downloads/forecasts.csv`;
const XLSX_DOWNLOAD_URL = `${API_BASE}/downloads/forecasts.xlsx`;
const TOTAL_COSTS_URL = `${API_BASE}/downloads/total_costs.csv`;
const MONTHLY_COSTS_URL = `${API_BASE}/downloads/monthly_costs.csv`;
const RUN_COSTS_URL = `${API_BASE}/downloads/run_costs.csv`;
const TRIAGE_DOWNLOAD_URL = `${API_BASE}/downloads/triage.csv`;
const SCORES_ENSEMBLE_MEAN_URL = `${API_BASE}/downloads/scores_ensemble_mean.csv`;
const SCORES_ENSEMBLE_BAYESMC_URL = `${API_BASE}/downloads/scores_ensemble_bayesmc.csv`;
const SCORES_MODEL_URL = `${API_BASE}/downloads/scores_model.csv`;

const DownloadsPage = () => {
  return (
    <div className="space-y-6">
      <section className="space-y-2">
        <h1 className="text-3xl font-semibold">Downloads</h1>
        <p className="text-sm text-fred-text">
          Export forecast outputs for offline analysis and QA.
        </p>
      </section>

      <section className="rounded-lg border border-fred-secondary bg-fred-surface p-6">
        <div className="space-y-2">
          <h2 className="text-lg font-semibold">
            Forecast SPD &amp; EIV export
          </h2>
          <p className="text-sm text-fred-text">
            One row per ISO3, hazard, model, and forecast month. Includes
            regime-change probability, direction, magnitude, and score columns.
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <a
              href={CSV_DOWNLOAD_URL}
              className="inline-flex items-center rounded-md bg-fred-secondary px-4 py-2 text-sm font-semibold text-white hover:opacity-90"
            >
              Download .csv
            </a>
            <a
              href={XLSX_DOWNLOAD_URL}
              className="text-xs text-fred-primary underline underline-offset-4 hover:text-fred-secondary"
            >
              Excel (if available; otherwise downloads CSV)
            </a>
          </div>
        </div>
      </section>

      <section className="rounded-lg border border-fred-secondary bg-fred-surface p-6">
        <div className="space-y-2">
          <h2 className="text-lg font-semibold">Performance scores</h2>
          <p className="text-sm text-fred-text">
            Scoring data (Brier, Log Loss, CRPS) for ensemble and individual
            models. Per-question exports include full 6-month SPD forecasts,
            EIV, resolutions, and triage data.
          </p>
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <a
              href={SCORES_ENSEMBLE_MEAN_URL}
              className="inline-flex items-center rounded-md bg-fred-secondary px-4 py-2 font-semibold text-white hover:opacity-90"
            >
              Ensemble Mean scores (CSV)
            </a>
            <a
              className="text-fred-primary underline underline-offset-4 hover:text-fred-secondary"
              href={SCORES_ENSEMBLE_BAYESMC_URL}
            >
              Ensemble BayesMC scores (CSV)
            </a>
            <a
              className="text-fred-primary underline underline-offset-4 hover:text-fred-secondary"
              href={SCORES_MODEL_URL}
            >
              Model summary scores (CSV)
            </a>
          </div>
        </div>
      </section>

      <section className="rounded-lg border border-fred-secondary bg-fred-surface p-6">
        <div className="space-y-2">
          <h2 className="text-lg font-semibold">Cost exports</h2>
          <p className="text-sm text-fred-text">
            Tidy CSVs for total, monthly, and run-level cost summaries.
          </p>
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <a
              href={TOTAL_COSTS_URL}
              className="inline-flex items-center rounded-md bg-fred-secondary px-4 py-2 font-semibold text-white hover:opacity-90"
            >
              Total costs (CSV)
            </a>
            <a
              className="text-fred-primary underline underline-offset-4 hover:text-fred-secondary"
              href={MONTHLY_COSTS_URL}
            >
              Monthly costs (CSV)
            </a>
            <a
              className="text-fred-primary underline underline-offset-4 hover:text-fred-secondary"
              href={RUN_COSTS_URL}
            >
              Run costs (CSV)
            </a>
          </div>
        </div>
      </section>

      <section className="rounded-lg border border-fred-secondary bg-fred-surface p-6">
        <div className="space-y-2">
          <h2 className="text-lg font-semibold">Run triage results</h2>
          <p className="text-sm text-fred-text">
            One row per run × country with HS triage score, tier, and model.
          </p>
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <a
              href={TRIAGE_DOWNLOAD_URL}
              className="inline-flex items-center rounded-md bg-fred-secondary px-4 py-2 font-semibold text-white hover:opacity-90"
            >
              Run triage results (CSV)
            </a>
          </div>
        </div>
      </section>
    </div>
  );
};

export default DownloadsPage;
