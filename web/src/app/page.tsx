import KpiCard from "../components/KpiCard";
import RiskIndexPanel from "../components/RiskIndexPanel";
import { apiGet } from "../lib/api";
import type {
  CountriesResponse,
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
  let countries: CountriesResponse | null = null;
  try {
    riskIndex = await apiGet<RiskIndexResponse>("/risk_index", {
      metric: "PA",
      horizon_m: 6,
      normalize: true,
    });
  } catch (error) {
    console.warn("Risk index unavailable:", error);
  }
  try {
    countries = await apiGet<CountriesResponse>("/countries");
  } catch (error) {
    console.warn("Countries unavailable:", error);
  }

  const activeQuestions =
    diagnostics.questions_by_status?.find(
      (row) => (row.status ?? "").toLowerCase() === "active"
    ) ?? null;

  const kpiAside = (
    <div className="space-y-4" data-testid="risk-index-kpi-panel">
      <KpiCard label="Active questions" value={activeQuestions?.n ?? 0} />
      <KpiCard
        label="Questions with forecasts"
        value={diagnostics.questions_with_forecasts ?? 0}
      />
    </div>
  );

  return (
    <div className="space-y-6">
      <section className="space-y-2">
        <h1 className="text-3xl font-semibold text-white">Risk Index</h1>
        <p className="text-sm text-slate-400">
          Last updated:{" "}
          <span className="text-slate-200">
            {version.latest_hs_created_at ?? "Unknown"}
          </span>
        </p>
      </section>

      <section className="space-y-4">
        {riskIndex ? (
          <RiskIndexPanel
            aside={kpiAside}
            countriesRows={countries?.rows ?? []}
            initialResponse={riskIndex}
            mapHeightClassName="h-[520px] md:h-[600px] lg:h-[720px]"
          />
        ) : (
          <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
            Risk index unavailable (API error).
          </div>
        )}
      </section>
    </div>
  );
}
