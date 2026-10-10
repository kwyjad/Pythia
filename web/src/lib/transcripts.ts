// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
//
// Prompt and response text for the question page, fetched only when a reader
// opens a stage. The page used to receive every transcript with the bundle;
// most visits read none of it, and that bundle ran the API out of memory.

export type TranscriptText = { prompt_text: string; response_text: string };

export type TranscriptStage = { dedup?: boolean; promptOnly?: boolean };

// Matches LLM_CALL_TEXT_MAX_IDS in pythia/api/routes/questions.py.
export const MAX_IDS_PER_REQUEST = 25;

const idOf = (row: Record<string, unknown>): string | null =>
  typeof row.call_id === "string" && row.call_id ? row.call_id : null;

/**
 * The call ids whose text a stage actually shows. A deduplicated stage shows
 * the first row's prompt once; a prompt-only stage shows no responses. So a
 * deduplicated prompt-only stage (the SPD forecast) needs one row, not every
 * ensemble member's prompt and response.
 */
export const callIdsToFetch = (
  stage: TranscriptStage,
  rows: Record<string, unknown>[],
): string[] => {
  const shown = stage.dedup && rows.length > 1 && stage.promptOnly ? rows.slice(0, 1) : rows;
  const ids = shown.map(idOf).filter((id): id is string => id !== null);
  return Array.from(new Set(ids));
};

export type TextFetcher = (ids: string[]) => Promise<Record<string, TranscriptText>>;

/** Wrap a fetcher so an id already loaded is never asked for again. */
export const createTranscriptLoader = (fetcher: TextFetcher) => {
  const cache = new Map<string, TranscriptText>();
  return async (ids: string[]): Promise<Record<string, TranscriptText>> => {
    const missing = ids.filter((id) => !cache.has(id));
    for (let i = 0; i < missing.length; i += MAX_IDS_PER_REQUEST) {
      const batch = missing.slice(i, i + MAX_IDS_PER_REQUEST);
      const got = await fetcher(batch);
      batch.forEach((id) => {
        cache.set(id, got[id] ?? { prompt_text: "", response_text: "" });
      });
    }
    const out: Record<string, TranscriptText> = {};
    ids.forEach((id) => {
      const hit = cache.get(id);
      if (hit) out[id] = hit;
    });
    return out;
  };
};
