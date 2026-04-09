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

export default async function OverviewPage({
  searchParams,
}: {
  searchParams: { [key: string]: string | string[] | undefined };
}) {
  const includeTest = searchParams?.include_test === "true";

  if (process.env.NODE_ENV !== "production") {
    console.log("[page] dynamic=force-dynamic", { route: "/" });
  }
  const [version, kpiScopes] = await Promise.all([
    apiGet<VersionResponse>("/version"),
    apiGet<DiagnosticsKpiScopesResponse>("/diagnostics/kpi_scopes", {
      metric_scope: "PA",
      include_test: includeTest || undefined,
    }),
  ]);
  let riskIndex: RiskIndexResponse | null = null;
  let countries: CountriesResponse | null = null;
  try {
    riskIndex = await apiGet<RiskIndexResponse>("/risk_index", {
      metric: "PA",
      horizon_m: 6,
      normalize: true,
      include_test: includeTest || undefined,
    });
  } catch (error) {
    console.warn("Risk index unavailable:", error);
  }
  try {
    countries = await apiGet<CountriesResponse>("/countries", {
      metric_scope: "PA",
      ...(kpiScopes.selected_month
        ? { year_month: kpiScopes.selected_month }
        : {}),
      ...(kpiScopes.selected_run_id
        ? { forecaster_run_id: kpiScopes.selected_run_id }
        : {}),
      include_test: includeTest || undefined,
    });
  } catch (error) {
    console.warn("Countries unavailable:", error);
  }

  return (
    <div className="space-y-6">
      <section className="space-y-2">
        <h1 className="text-3xl font-semibold">
          Run Results
        </h1>
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
            key={String(includeTest)}
            countriesRows={countries?.rows ?? []}
            initialResponse={riskIndex}
            kpiScopes={kpiScopes}
            mapHeightClassName="h-[360px] sm:h-[420px] md:h-[520px] lg:h-[720px]"
            includeTest={includeTest}
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
