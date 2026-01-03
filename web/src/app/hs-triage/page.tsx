import { apiGet } from "../../lib/api";
import HsTriageClient from "./HsTriageClient";

export const metadata = {
  title: "HS Triage",
};

type HsRunRow = {
  run_id: string;
  triage_date?: string | null;
  countries_triaged?: number | null;
};

type HsRunsResponse = {
  rows: HsRunRow[];
};

export default async function HsTriagePage() {
  let runs: HsRunRow[] = [];
  let loadError: string | null = null;

  try {
    const data = await apiGet<HsRunsResponse>("/hs_runs", { limit: 50 });
    runs = data.rows ?? [];
  } catch (error) {
    loadError = error instanceof Error ? error.message : "Failed to load HS runs.";
  }

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold">HS Triage</h1>
        <p className="text-sm text-fred-text">
          Inspect HS triage scores per run, including individual call scores.
        </p>
      </header>
      <HsTriageClient
        initialRuns={runs}
        initialRunId={runs[0]?.run_id ?? ""}
        initialError={loadError}
      />
    </div>
  );
}
