import Link from "next/link";

import KpiCard from "../components/KpiCard";
import { apiGet } from "../lib/api";
import type {
  DiagnosticsSummaryResponse,
  RiskIndexResponse,
  VersionResponse,
} from "../lib/types";

export default async function OverviewPage() {
  const [version, diagnostics] = await Promise.all([
    apiGet<VersionResponse>("/version"),
    apiGet<DiagnosticsSummaryResponse>("/diagnostics/summary"),
  ]);
  let riskIndex: RiskIndexResponse | null = null;
  try {
    riskIndex = await apiGet<RiskIndexResponse>("/risk_index", {
      metric: "PA",
      horizon_m: 1,
      normalize: true,
    });
  } catch (error) {
    console.warn("Risk index unavailable:", error);
  }

  const activeQuestions =
    diagnostics.questions_by_status?.find(
      (row) => (row.status ?? "").toLowerCase() === "active"
    ) ?? null;

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
          value={diagnostics.questions_with_forecasts ?? 0}
        />
      </section>

      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-white">Risk index</h2>
          {riskIndex ? (
            <span className="text-sm text-slate-400">
              Metric {riskIndex.metric} • {riskIndex.target_month}
            </span>
          ) : null}
        </div>

        {riskIndex ? (
          <div className="overflow-x-auto rounded-lg border border-slate-800">
            <table className="w-full border-collapse text-sm">
              <thead className="bg-slate-900 text-slate-300">
                <tr>
                  <th className="px-3 py-2 text-left">ISO3</th>
                  <th className="px-3 py-2 text-right">Expected value</th>
                  <th className="px-3 py-2 text-right">Per capita</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {riskIndex.rows?.map((row) => (
                  <tr key={`${row.iso3}-${row.horizon_m}`} className="text-slate-200">
                    <td className="px-3 py-2">
                      <Link className="underline underline-offset-2" href={`/countries/${row.iso3}`}>
                        {row.iso3}
                      </Link>
                    </td>
                    <td className="px-3 py-2 text-right">
                      {typeof row.expected_value === "number"
                        ? row.expected_value.toFixed(2)
                        : "-"}
                    </td>
                    <td className="px-3 py-2 text-right">
                      {typeof row.per_capita === "number"
                        ? row.per_capita.toFixed(6)
                        : "-"}
                    </td>
                  </tr>
                ))}
                {(!riskIndex.rows || riskIndex.rows.length === 0) && (
                  <tr>
                    <td className="px-3 py-3 text-slate-400" colSpan={3}>
                      No rows returned for {riskIndex.target_month}.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
            Risk index unavailable. Please try again later.
          </div>
        )}
      </section>
    </div>
  );
}
