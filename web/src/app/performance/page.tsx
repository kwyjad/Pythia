import { apiGet } from "../../lib/api";
import type { PerformanceScoresResponse } from "../../lib/types";
import PerformancePanel from "./PerformancePanel";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export default async function PerformancePage() {
  let data: PerformanceScoresResponse = { summary_rows: [], run_rows: [] };
  let loadError: string | null = null;

  try {
    data = await apiGet<PerformanceScoresResponse>("/performance/scores");
  } catch (error) {
    loadError = "Unable to load performance scores right now.";
    console.warn("Failed to load performance scores:", error);
  }

  return (
    <div className="space-y-6">
      <section className="space-y-2">
        <h1 className="text-3xl font-semibold">Forecasting Performance</h1>
        <p className="text-sm text-fred-text">
          Scoring metrics (Brier, Log Loss, CRPS) across models, hazards, and
          runs. Lower scores indicate better calibration.
        </p>
      </section>

      <div className="w-full bg-yellow-400 px-4 py-3 text-center text-sm text-black">
        NOTE: These are preliminary scores from early runs with Fred changing
        resolution sources for climate hazards midstream. Climate hazard
        baselines and resolution data is spotty. Performance will be
        artificially poor until resolution issues can be resolved.
      </div>

      {loadError ? (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
          {loadError}
        </div>
      ) : (
        <PerformancePanel initialData={data} />
      )}
    </div>
  );
}
