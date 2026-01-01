import { Suspense } from "react";

import CostsClient, {
  CostsResponse,
  LatenciesResponse,
  RunRuntimesResponse,
} from "./CostsClient";
import { apiGet } from "../../lib/api";

export const dynamic = "force-dynamic";
export const revalidate = 0;

const safeGet = async <T,>(path: string, fallback: T): Promise<T> => {
  try {
    return await apiGet<T>(path);
  } catch (error) {
    console.warn(`Failed to load ${path}:`, error);
    return fallback;
  }
};

const CostsPage = async () => {
  if (process.env.NODE_ENV !== "production") {
    console.log("[page] dynamic=force-dynamic", { route: "/costs" });
  }
  const [total, monthly, runs, latencies, runRuntimes] = await Promise.all([
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
    safeGet<RunRuntimesResponse>("/costs/run_runtimes", { rows: [] }),
  ]);

  return (
    <Suspense fallback={<div />}>
      <CostsClient
        total={total.tables}
        monthly={monthly.tables}
        runs={runs.tables}
        latencies={latencies.rows}
        runRuntimes={runRuntimes.rows}
      />
    </Suspense>
  );
};

export default CostsPage;
