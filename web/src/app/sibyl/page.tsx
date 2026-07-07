import { apiGet } from "../../lib/api";
import type {
  SibylQuestionsResponse,
  SibylRunsResponse,
  SibylSummaryResponse,
} from "../../lib/types";
import SibylClient from "./SibylClient";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export default async function SibylPage({
  searchParams,
}: {
  searchParams: { [key: string]: string | string[] | undefined };
}) {
  const includeTest = searchParams?.include_test === "true";
  const sibylRunId =
    typeof searchParams?.sibyl_run_id === "string"
      ? searchParams.sibyl_run_id
      : undefined;

  let summary: SibylSummaryResponse = { run: null, questions: [] };
  let questions: SibylQuestionsResponse = { sibyl_run_id: null, rows: [] };
  let runs: SibylRunsResponse = { rows: [] };
  let loadError: string | null = null;

  try {
    [summary, questions, runs] = await Promise.all([
      apiGet<SibylSummaryResponse>("/sibyl/summary", {
        include_test: includeTest || undefined,
        sibyl_run_id: sibylRunId,
      }),
      apiGet<SibylQuestionsResponse>("/sibyl/questions", {
        include_test: includeTest || undefined,
        sibyl_run_id: sibylRunId,
      }),
      apiGet<SibylRunsResponse>("/sibyl/runs", {
        include_test: includeTest || undefined,
      }),
    ]);
  } catch (error) {
    loadError = "Unable to load Sibyl data right now.";
    console.warn("Failed to load Sibyl data:", error);
  }

  return (
    <div className="space-y-6">
      <section className="space-y-2">
        <h1 className="text-3xl font-semibold">Sibyl — Deep-Research Track</h1>
        <p className="text-sm text-fred-text">
          A parallel oracle beside the standard pipeline: for the ten most
          volatile affected/fatalities questions of each run, Claude Opus
          researches the open web in independent agentic trials and produces
          its own probability distribution. The only shared input with the
          standard track is the Resolver base rate. Track-vs-track divergence
          (Jensen–Shannon) highlights where deep research disagrees with the
          structured-data ensemble.
        </p>
      </section>

      {loadError ? (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
          {loadError}
        </div>
      ) : (
        <SibylClient
          summary={summary}
          questions={questions}
          runs={runs}
          includeTest={includeTest}
        />
      )}
    </div>
  );
}
