import { apiGet } from "../../../lib/api";
import type { QuestionBundleResponse } from "../../../lib/types";

type QuestionPageProps = {
  params: { questionId: string };
  searchParams?: { hs_run_id?: string };
};

const QuestionPage = async ({ params, searchParams }: QuestionPageProps) => {
  const bundle = await apiGet<QuestionBundleResponse>("/question_bundle", {
    question_id: params.questionId,
    hs_run_id: searchParams?.hs_run_id
  });

  const question = bundle.question as Record<string, unknown> | undefined;

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-2xl font-semibold text-white">
          {question?.wording ?? "Question detail"}
        </h1>
        <p className="text-sm text-slate-400">
          {question?.iso3 ?? ""} • {question?.hazard_code ?? ""} •{" "}
          {question?.metric ?? ""} • {question?.target_month ?? ""}
        </p>
      </section>

      <section className="space-y-4">
        <div>
          <h2 className="text-lg font-semibold text-white">HS</h2>
          <pre className="mt-2 whitespace-pre-wrap rounded-lg border border-slate-800 bg-slate-900/60 p-4 text-xs text-slate-200">
            {JSON.stringify(bundle.hs ?? null, null, 2)}
          </pre>
        </div>
        <div>
          <h2 className="text-lg font-semibold text-white">Forecast</h2>
          <pre className="mt-2 whitespace-pre-wrap rounded-lg border border-slate-800 bg-slate-900/60 p-4 text-xs text-slate-200">
            {JSON.stringify(bundle.forecast ?? null, null, 2)}
          </pre>
        </div>
        <div>
          <h2 className="text-lg font-semibold text-white">Context</h2>
          <pre className="mt-2 whitespace-pre-wrap rounded-lg border border-slate-800 bg-slate-900/60 p-4 text-xs text-slate-200">
            {JSON.stringify(bundle.context ?? null, null, 2)}
          </pre>
        </div>
        <div>
          <h2 className="text-lg font-semibold text-white">LLM calls</h2>
          <pre className="mt-2 whitespace-pre-wrap rounded-lg border border-slate-800 bg-slate-900/60 p-4 text-xs text-slate-200">
            {JSON.stringify(bundle.llm_calls ?? null, null, 2)}
          </pre>
        </div>
      </section>
    </div>
  );
};

export default QuestionPage;
