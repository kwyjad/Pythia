import { apiGet } from "../../../lib/api";
import type { QuestionBundleResponse } from "../../../lib/types";
import QuestionDetailView from "./QuestionDetailView";

type QuestionPageProps = {
  params: { questionId: string };
  searchParams?: { hs_run_id?: string };
};

const QuestionPage = async ({ params, searchParams }: QuestionPageProps) => {
  const bundle = await apiGet<QuestionBundleResponse>("/question_bundle", {
    question_id: params.questionId,
    hs_run_id: searchParams?.hs_run_id,
    include_llm_calls: true,
    include_transcripts: false,
    limit_llm_calls: 200
  });

  return <QuestionDetailView bundle={bundle} />;
};

export default QuestionPage;
