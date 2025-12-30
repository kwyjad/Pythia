import { apiGet } from "../../lib/api";

type CostsRow = {
  grain: string;
  row_type: string;
  year: number | null;
  month: number | null;
  run_id: string | null;
  model: string | null;
  phase: string | null;
  total_cost_usd: number | null;
  n_questions: number | null;
  avg_cost_per_question: number | null;
  median_cost_per_question: number | null;
  n_countries: number | null;
  avg_cost_per_country: number | null;
  median_cost_per_country: number | null;
};

type CostsResponse = {
  tables: {
    summary: CostsRow[];
    by_model: CostsRow[];
    by_phase: CostsRow[];
  };
};

type LatencyRow = {
  run_id: string | null;
  year: number | null;
  month: number | null;
  model: string | null;
  phase: string | null;
  n_calls: number | null;
  p50_elapsed_ms: number | null;
  p90_elapsed_ms: number | null;
};

type LatenciesResponse = {
  rows: LatencyRow[];
};

const numberFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 2,
});

const formatNumber = (value: number | null | undefined) => {
  if (value === null || value === undefined) return "—";
  return numberFormatter.format(value);
};

const formatCurrency = (value: number | null | undefined) => {
  if (value === null || value === undefined) return "—";
  return `$${numberFormatter.format(value)}`;
};

const formatMonth = (year: number | null, month: number | null) => {
  if (!year || !month) return "—";
  return `${year}-${String(month).padStart(2, "0")}`;
};

const renderCostsTable = (rows: CostsRow[], emptyMessage: string) => {
  return (
    <div className="overflow-x-auto rounded-lg border border-slate-800">
      <table className="w-full table-auto border-collapse text-sm">
        <thead className="bg-slate-900 text-slate-300">
          <tr>
            <th className="px-3 py-2 text-left">Year-Month</th>
            <th className="px-3 py-2 text-left">Run</th>
            <th className="px-3 py-2 text-left">Model</th>
            <th className="px-3 py-2 text-left">Phase</th>
            <th className="px-3 py-2 text-right">Total cost (USD)</th>
            <th className="px-3 py-2 text-right">Questions</th>
            <th className="px-3 py-2 text-right">Avg/Q</th>
            <th className="px-3 py-2 text-right">Median/Q</th>
            <th className="px-3 py-2 text-right">Countries</th>
            <th className="px-3 py-2 text-right">Avg/Country</th>
            <th className="px-3 py-2 text-right">Median/Country</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800 text-slate-200">
          {rows.length === 0 ? (
            <tr>
              <td className="px-3 py-3 text-slate-400" colSpan={11}>
                {emptyMessage}
              </td>
            </tr>
          ) : (
            rows.map((row, idx) => (
              <tr
                key={`${row.grain}-${row.row_type}-${row.run_id ?? "none"}-${
                  row.model ?? "none"
                }-${row.phase ?? "none"}-${idx}`}
              >
                <td className="px-3 py-2">{formatMonth(row.year, row.month)}</td>
                <td className="px-3 py-2">{row.run_id ?? "—"}</td>
                <td className="px-3 py-2">{row.model ?? "—"}</td>
                <td className="px-3 py-2">{row.phase ?? "—"}</td>
                <td className="px-3 py-2 text-right">
                  {formatCurrency(row.total_cost_usd)}
                </td>
                <td className="px-3 py-2 text-right">
                  {formatNumber(row.n_questions ?? undefined)}
                </td>
                <td className="px-3 py-2 text-right">
                  {formatCurrency(row.avg_cost_per_question)}
                </td>
                <td className="px-3 py-2 text-right">
                  {formatCurrency(row.median_cost_per_question)}
                </td>
                <td className="px-3 py-2 text-right">
                  {formatNumber(row.n_countries ?? undefined)}
                </td>
                <td className="px-3 py-2 text-right">
                  {formatCurrency(row.avg_cost_per_country)}
                </td>
                <td className="px-3 py-2 text-right">
                  {formatCurrency(row.median_cost_per_country)}
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
};

const renderLatencyTable = (rows: LatencyRow[], emptyMessage: string) => {
  return (
    <div className="overflow-x-auto rounded-lg border border-slate-800">
      <table className="w-full table-auto border-collapse text-sm">
        <thead className="bg-slate-900 text-slate-300">
          <tr>
            <th className="px-3 py-2 text-left">Year-Month</th>
            <th className="px-3 py-2 text-left">Run</th>
            <th className="px-3 py-2 text-left">Model</th>
            <th className="px-3 py-2 text-left">Phase</th>
            <th className="px-3 py-2 text-right">Calls</th>
            <th className="px-3 py-2 text-right">P50 (ms)</th>
            <th className="px-3 py-2 text-right">P90 (ms)</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800 text-slate-200">
          {rows.length === 0 ? (
            <tr>
              <td className="px-3 py-3 text-slate-400" colSpan={7}>
                {emptyMessage}
              </td>
            </tr>
          ) : (
            rows.map((row, idx) => (
              <tr
                key={`${row.run_id ?? "none"}-${row.model ?? "none"}-${
                  row.phase ?? "none"
                }-${idx}`}
              >
                <td className="px-3 py-2">{formatMonth(row.year, row.month)}</td>
                <td className="px-3 py-2">{row.run_id ?? "—"}</td>
                <td className="px-3 py-2">{row.model ?? "—"}</td>
                <td className="px-3 py-2">{row.phase ?? "—"}</td>
                <td className="px-3 py-2 text-right">{formatNumber(row.n_calls)}</td>
                <td className="px-3 py-2 text-right">
                  {formatNumber(row.p50_elapsed_ms)}
                </td>
                <td className="px-3 py-2 text-right">
                  {formatNumber(row.p90_elapsed_ms)}
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
};

const safeGet = async <T,>(path: string, fallback: T): Promise<T> => {
  try {
    return await apiGet<T>(path);
  } catch (error) {
    console.warn(`Failed to load ${path}:`, error);
    return fallback;
  }
};

const CostsPage = async () => {
  const [total, monthly, runs, latencies] = await Promise.all([
    safeGet<CostsResponse>("/costs/total", {
      tables: { summary: [], by_model: [], by_phase: [] },
    }),
    safeGet<CostsResponse>("/costs/monthly", {
      tables: { summary: [], by_model: [], by_phase: [] },
    }),
    safeGet<CostsResponse>("/costs/runs", {
      tables: { summary: [], by_model: [], by_phase: [] },
    }),
    safeGet<LatenciesResponse>("/costs/latencies", { rows: [] }),
  ]);

  return (
    <div className="space-y-8">
      <section className="space-y-2">
        <h1 className="text-3xl font-semibold text-white">Costs</h1>
        <p className="text-sm text-slate-400">
          Review LLM spend and latency across total, monthly, and run-level
          aggregates.
        </p>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-white">Total costs</h2>
        {renderCostsTable(total.tables.summary, "No total cost summary available.")}
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-2">
            <h3 className="text-sm font-semibold text-slate-200">By model</h3>
            {renderCostsTable(total.tables.by_model, "No model breakdown available.")}
          </div>
          <div className="space-y-2">
            <h3 className="text-sm font-semibold text-slate-200">By phase</h3>
            {renderCostsTable(total.tables.by_phase, "No phase breakdown available.")}
          </div>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-white">Monthly costs</h2>
        {renderCostsTable(monthly.tables.summary, "No monthly summary available.")}
        <details className="space-y-3 rounded-lg border border-slate-800 bg-slate-900/40 p-4">
          <summary className="cursor-pointer text-sm font-semibold text-slate-200">
            View monthly breakdowns by model and phase
          </summary>
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-slate-200">By model</h3>
              {renderCostsTable(monthly.tables.by_model, "No monthly model data.")}
            </div>
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-slate-200">By phase</h3>
              {renderCostsTable(monthly.tables.by_phase, "No monthly phase data.")}
            </div>
          </div>
        </details>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-white">Run costs</h2>
        {renderCostsTable(runs.tables.summary, "No run summary available.")}
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-2">
            <h3 className="text-sm font-semibold text-slate-200">By model</h3>
            {renderCostsTable(runs.tables.by_model, "No run model breakdown available.")}
          </div>
          <div className="space-y-2">
            <h3 className="text-sm font-semibold text-slate-200">By phase</h3>
            {renderCostsTable(runs.tables.by_phase, "No run phase breakdown available.")}
          </div>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-white">Run latencies</h2>
        {renderLatencyTable(latencies.rows, "No latency data available.")}
      </section>
    </div>
  );
};

export default CostsPage;
