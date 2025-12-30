const API_BASE =
  process.env.NEXT_PUBLIC_PYTHIA_API_BASE ?? "http://localhost:8000/v1";

const DOWNLOAD_URL = `${API_BASE}/downloads/forecasts.xlsx`;

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
            Forecast SPD &amp; EIV export (Excel)
          </h2>
          <p className="text-sm text-slate-400">
            One row per ISO3, hazard, model, and forecast month.
          </p>
          <a
            href={DOWNLOAD_URL}
            className="inline-flex items-center rounded-md bg-indigo-500 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-400"
          >
            Download .xlsx
          </a>
        </div>
      </section>
    </div>
  );
};

export default DownloadsPage;
