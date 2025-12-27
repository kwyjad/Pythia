import { apiGet } from "../../../lib/api";
import type { QuestionBundleResponse } from "../../../lib/types";

type QuestionPageProps = {
  params: { questionId: string };
};

const QuestionPage = async ({ params }: QuestionPageProps) => {
  const bundle = await apiGet<QuestionBundleResponse>("/question_bundle", {
    question_id: params.questionId
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
          <h2 className="text-lg font-semibold text-white">HS scenarios</h2>
          <pre className="mt-2 whitespace-pre-wrap rounded-lg border border-slate-800 bg-slate-900/60 p-4 text-xs text-slate-200">
            {JSON.stringify(bundle.hs_scenarios ?? null, null, 2)}
          </pre>
        </div>
        <div>
          <h2 className="text-lg font-semibold text-white">Research</h2>
          <pre className="mt-2 whitespace-pre-wrap rounded-lg border border-slate-800 bg-slate-900/60 p-4 text-xs text-slate-200">
            {JSON.stringify(bundle.research ?? null, null, 2)}
          </pre>
        </div>
        <div>
          <h2 className="text-lg font-semibold text-white">Ensemble forecast</h2>
          <pre className="mt-2 whitespace-pre-wrap rounded-lg border border-slate-800 bg-slate-900/60 p-4 text-xs text-slate-200">
            {JSON.stringify(bundle.ensemble ?? null, null, 2)}
          </pre>
        </div>
        <div>
          <h2 className="text-lg font-semibold text-white">Per-model forecasts</h2>
          <pre className="mt-2 whitespace-pre-wrap rounded-lg border border-slate-800 bg-slate-900/60 p-4 text-xs text-slate-200">
            {JSON.stringify(bundle.per_model ?? null, null, 2)}
          </pre>
        </div>
      </section>
    </div>
  );
};

export default QuestionPage;
