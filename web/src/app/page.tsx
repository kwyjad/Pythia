import RiskIndexPanel from "../components/RiskIndexPanel";
import { apiGet } from "../lib/api";
import type {
  CountriesResponse,
  DiagnosticsKpiScopesResponse,
  RiskIndexResponse,
  VersionResponse,
} from "../lib/types";

export const dynamic = "force-dynamic";
export const revalidate = 0;

const formatLastUpdated = (timestamp: string | null | undefined) => {
  if (!timestamp) return "Unknown";
  const core = timestamp.slice(0, 19);
  if (core.length === 19 && core.includes("T")) {
    const [date, time] = core.split("T");
    if (date && time) {
      return `Date: ${date}  Time ${time}`;
    }
  }
  return timestamp;
};

export default async function OverviewPage() {
  if (process.env.NODE_ENV !== "production") {
    console.log("[page] dynamic=force-dynamic", { route: "/" });
  }
  const [version, kpiScopes] = await Promise.all([
    apiGet<VersionResponse>("/version"),
    apiGet<DiagnosticsKpiScopesResponse>("/diagnostics/kpi_scopes", {
      metric_scope: "PA",
    }),
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

  return (
    <div className="space-y-6">
      <section className="space-y-2">
        <h1 className="text-3xl font-semibold">Risk Index</h1>
        <p className="text-sm text-fred-text">
          Last updated:{" "}
          <span className="text-fred-text font-medium">
            {formatLastUpdated(version.latest_hs_created_at)}
          </span>
        </p>
      </section>

      <section className="space-y-4">
        {riskIndex ? (
          <RiskIndexPanel
            countriesRows={countries?.rows ?? []}
            initialResponse={riskIndex}
            kpiScopes={kpiScopes}
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
