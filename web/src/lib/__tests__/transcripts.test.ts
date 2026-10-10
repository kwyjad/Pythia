import { describe, expect, it, vi } from "vitest";

import { MAX_IDS_PER_REQUEST, callIdsToFetch, createTranscriptLoader } from "../transcripts";

const rows = [{ call_id: "a" }, { call_id: "b" }, { call_id: "c" }];

describe("callIdsToFetch", () => {
  it("asks for one row of a deduplicated prompt-only stage", () => {
    expect(callIdsToFetch({ dedup: true, promptOnly: true }, rows)).toEqual(["a"]);
  });

  it("asks for every row where responses are shown", () => {
    expect(callIdsToFetch({ dedup: true }, rows)).toEqual(["a", "b", "c"]);
    expect(callIdsToFetch({}, rows)).toEqual(["a", "b", "c"]);
  });

  it("skips rows with no call id", () => {
    expect(callIdsToFetch({}, [{ call_id: "" }, {}, { call_id: "x" }])).toEqual(["x"]);
  });
});

describe("createTranscriptLoader", () => {
  it("fetches an id once and serves it from memory after", async () => {
    const fetcher = vi.fn(async (ids: string[]) =>
      Object.fromEntries(ids.map((id) => [id, { prompt_text: `p-${id}`, response_text: "" }])),
    );
    const load = createTranscriptLoader(fetcher);
    expect((await load(["a"])).a.prompt_text).toBe("p-a");
    await load(["a"]);
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it("splits a long list into requests the API accepts", async () => {
    const fetcher = vi.fn(async () => ({}));
    const load = createTranscriptLoader(fetcher);
    const ids = Array.from({ length: MAX_IDS_PER_REQUEST + 1 }, (_, i) => `id${i}`);
    await load(ids);
    expect(fetcher).toHaveBeenCalledTimes(2);
  });
});
