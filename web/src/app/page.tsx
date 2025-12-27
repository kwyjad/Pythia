import Link from "next/link";

import KpiCard from "../components/KpiCard";
import { apiGet } from "../lib/api";
import type {
  DiagnosticsSummaryResponse,
  RiskIndexResponse,
  VersionResponse
} from "../lib/types";

const getCurrentMonth = () => {
  const now = new Date();
  const year = now.getUTCFullYear();
  const month = String(now.getUTCMonth() + 1).padStart(2, "0");
  return `${year}-${month}`;
};

const OverviewPage = async () => {
  const [version, diagnostics, riskIndex] = await Promise.all([
    apiGet<VersionResponse>("/version"),
    apiGet<DiagnosticsSummaryResponse>("/diagnostics/summary"),
    apiGet<RiskIndexResponse>("/risk_index", {
      metric: "PA",
      target_month: getCurrentMonth(),
      horizon_m: 1,
      normalize: true
    })
  ]);

  const activeQuestions = diagnostics.questions_by_status.find(
    (row) => row.status?.toLowerCase() === "active"
  );

  return (
    <div className="space-y-8">
      <section className="space-y-2">
        <h1 className="text-3xl font-semibold text-white">Overview</h1>
        <p className="text-sm text-slate-400">
          Last updated:{" "}
          <span className="text-slate-200">
            {version.latest_hs_created_at ?? "Unknown"}
          </span>
        </p>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <KpiCard label="Active questions" value={activeQuestions?.n ?? 0} />
        <KpiCard
          label="Questions with forecasts"
          value={diagnostics.questions_with_forecasts}
        />
      </section>

      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-white">Risk index</h2>
          <span className="text-sm text-slate-400">
            Metric {riskIndex.metric} • {riskIndex.target_month}
          </span>
        </div>
        <div className="overflow-x-auto rounded-lg border border-slate-800">
          <table>
            <thead>
              <tr>
                <th>ISO3</th>
                <th>Expected value</th>
                <th>Per capita</th>
              </tr>
            </thead>
            <tbody>
              {riskIndex.rows.map((row) => (
                <tr key={`${row.iso3}-${row.horizon_m}`}>
                  <td>
                    <Link href={`/countries/${row.iso3}`}>{row.iso3}</Link>
                  </td>
                  <td>{row.expected_value?.toFixed(2) ?? "-"}</td>
                  <td>{row.per_capita?.toFixed(6) ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};

export default OverviewPage;
