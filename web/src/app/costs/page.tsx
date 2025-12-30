import CostsClient, {
  CostsResponse,
  LatenciesResponse,
} from "./CostsClient";
import { apiGet } from "../../lib/api";

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
    <CostsClient
      total={total.tables}
      monthly={monthly.tables}
      runs={runs.tables}
      latencies={latencies.rows}
    />
  );
};

export default CostsPage;
