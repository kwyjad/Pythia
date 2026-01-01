import { notFound } from "next/navigation";

import { apiGet } from "../../../lib/api";
import type { QuestionBundleResponse } from "../../../lib/types";
import QuestionDetailView from "./QuestionDetailView";

type QuestionPageProps = {
  params: { questionId: string };
  searchParams?: { hs_run_id?: string };
};

const isNotFoundError = (error: unknown) =>
  error instanceof Error && error.message.includes("(404)");

const QuestionPage = async ({ params, searchParams }: QuestionPageProps) => {
  let bundle: QuestionBundleResponse;
  const fetchBundle = (hsRunId?: string) =>
    apiGet<QuestionBundleResponse>("/question_bundle", {
      question_id: params.questionId,
      hs_run_id: hsRunId,
      include_llm_calls: true,
      include_transcripts: false,
      limit_llm_calls: 200
    });
  try {
    bundle = await fetchBundle(searchParams?.hs_run_id);
  } catch (error) {
    if (isNotFoundError(error) && searchParams?.hs_run_id) {
      try {
        bundle = await fetchBundle();
      } catch (retryError) {
        if (isNotFoundError(retryError)) {
          notFound();
        }
        throw retryError;
      }
    } else if (isNotFoundError(error)) {
      notFound();
    } else {
      throw error;
    }
  }

  return <QuestionDetailView bundle={bundle} />;
};

export default QuestionPage;
