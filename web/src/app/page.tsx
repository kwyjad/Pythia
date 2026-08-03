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
  // These two were the only unguarded calls on this page, and /version returns
  // 503 whenever the API's DB sync is misconfigured or wedged — so one degraded
  // dependency threw the whole route and the dashboard rendered as a blank
  // error page. A blank page is indistinguishable from "there is no data",
  // which is exactly how a publish/sync lag gets misread as data loss. Degrade
  // visibly instead: keep rendering, and say what is unavailable.
  const degraded: string[] = [];

  // include_test keeps "Last updated"/"Latest forecast scan" consistent
  // with the Test toggle: with Test OFF they reflect production data only.
  const [versionResult, kpiScopesResult] = await Promise.allSettled([
    apiGet<VersionResponse>("/version", {
      include_test: includeTest || undefined,
    }),
    apiGet<DiagnosticsKpiScopesResponse>("/diagnostics/kpi_scopes", {
      metric_scope: "PA",
      include_test: includeTest || undefined,
    }),
  ]);

  let version: VersionResponse = {};
  if (versionResult.status === "fulfilled") {
    version = versionResult.value;
  } else {
    console.warn("Version unavailable:", versionResult.reason);
    degraded.push("run freshness");
  }

  // Every field is optional/nullable downstream, so the empty shape is safe:
  // the month and run selectors render empty and the panel falls back to its
  // month-window branch rather than crashing.
  let kpiScopes: DiagnosticsKpiScopesResponse = {
    available_months: [],
    selected_month: null,
    scopes: {},
  };
  if (kpiScopesResult.status === "fulfilled") {
    kpiScopes = kpiScopesResult.value;
  } else {
    console.warn("KPI scopes unavailable:", kpiScopesResult.reason);
    degraded.push("run selection");
  }
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
    degraded.push("country rows");
  }

  return (
    <div className="space-y-6">
      {degraded.length > 0 && (
        <div
          role="status"
          className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100"
        >
          <span className="font-medium">Live data unavailable</span> — the API
          did not answer for: {degraded.join(", ")}. What you see below may be
          incomplete or out of date. This usually means the API is still syncing
          a freshly published database; check <code>/v1/health</code> for
          sync status.
        </div>
      )}
      <section className="space-y-2">
        <h1 className="text-3xl font-semibold">
          Run Results
        </h1>
        <p className="text-sm text-fred-text">
          Last updated:{" "}
          <span className="text-fred-text font-medium">
            {formatLastUpdated(
              version.latest_data_at ?? version.latest_hs_created_at
            )}
          </span>
        </p>
        <p className="text-xs text-fred-muted">
          Latest forecast scan:{" "}
          {formatLastUpdated(version.latest_hs_created_at)}
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
