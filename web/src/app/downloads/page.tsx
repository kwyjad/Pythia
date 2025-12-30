const API_BASE =
  process.env.NEXT_PUBLIC_PYTHIA_API_BASE ?? "http://localhost:8000/v1";

const CSV_DOWNLOAD_URL = `${API_BASE}/downloads/forecasts.csv`;
const XLSX_DOWNLOAD_URL = `${API_BASE}/downloads/forecasts.xlsx`;
const TOTAL_COSTS_URL = `${API_BASE}/downloads/total_costs.csv`;
const MONTHLY_COSTS_URL = `${API_BASE}/downloads/monthly_costs.csv`;
const RUN_COSTS_URL = `${API_BASE}/downloads/run_costs.csv`;

const DownloadsPage = () => {
  return (
    <div className="space-y-6">
      <section className="space-y-2">
        <h1 className="text-3xl font-semibold text-white">Downloads</h1>
        <p className="text-sm text-slate-400">
          Export forecast outputs for offline analysis and QA.
        </p>
      </section>

      <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
        <div className="space-y-2">
          <h2 className="text-lg font-semibold text-white">
            Forecast SPD &amp; EIV export
          </h2>
          <p className="text-sm text-slate-400">
            One row per ISO3, hazard, model, and forecast month.
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <a
              href={CSV_DOWNLOAD_URL}
              className="inline-flex items-center rounded-md bg-indigo-500 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-400"
            >
              Download .csv
            </a>
            <a
              href={XLSX_DOWNLOAD_URL}
              className="text-xs text-slate-300 underline underline-offset-4 hover:text-white"
            >
              Excel (if available; otherwise downloads CSV)
            </a>
          </div>
        </div>
      </section>

      <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
        <div className="space-y-2">
          <h2 className="text-lg font-semibold text-white">Cost exports</h2>
          <p className="text-sm text-slate-400">
            Tidy CSVs for total, monthly, and run-level cost summaries.
          </p>
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <a
              href={TOTAL_COSTS_URL}
              className="inline-flex items-center rounded-md bg-indigo-500 px-4 py-2 font-semibold text-white hover:bg-indigo-400"
            >
              Total costs (CSV)
            </a>
            <a className="text-slate-300 underline underline-offset-4 hover:text-white" href={MONTHLY_COSTS_URL}>
              Monthly costs (CSV)
            </a>
            <a className="text-slate-300 underline underline-offset-4 hover:text-white" href={RUN_COSTS_URL}>
              Run costs (CSV)
            </a>
          </div>
        </div>
      </section>
    </div>
  );
};

export default DownloadsPage;
