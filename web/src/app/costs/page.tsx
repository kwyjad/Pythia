import { Suspense } from "react";

import CostsClient, {
  CostsResponse,
  LatenciesResponse,
  RunRuntimesResponse,
} from "./CostsClient";
import { apiGet } from "../../lib/api";

export const dynamic = "force-dynamic";
export const revalidate = 0;

type QueryParams = Record<string, string | number | boolean | null | undefined>;

const safeGet = async <T,>(path: string, fallback: T, params?: QueryParams): Promise<T> => {
  try {
    return await apiGet<T>(path, params);
  } catch (error) {
    console.warn(`Failed to load ${path}:`, error);
    return fallback;
  }
};

const CostsPage = async ({
  searchParams,
}: {
  searchParams: { [key: string]: string | string[] | undefined };
}) => {
  const includeTest = searchParams?.include_test === "true";
  const testParam: QueryParams = includeTest ? { include_test: true } : {};

  if (process.env.NODE_ENV !== "production") {
    console.log("[page] dynamic=force-dynamic", { route: "/costs" });
  }
  const [total, monthly, runs, latencies, runRuntimes] = await Promise.all([
    safeGet<CostsResponse>("/costs/total", {
      tables: { summary: [], by_model: [], by_phase: [] },
    }, testParam),
    safeGet<CostsResponse>("/costs/monthly", {
      tables: { summary: [], by_model: [], by_phase: [] },
    }, testParam),
    safeGet<CostsResponse>("/costs/runs", {
      tables: { summary: [], by_model: [], by_phase: [] },
    }, testParam),
    safeGet<LatenciesResponse>("/costs/latencies", { rows: [] }, testParam),
    safeGet<RunRuntimesResponse>("/costs/run_runtimes", { rows: [] }, testParam),
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
