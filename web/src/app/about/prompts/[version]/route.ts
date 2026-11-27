// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

import { extractListedVersionPrompts } from "../../../../lib/prompt_extractor";

export const dynamic = "force-dynamic";

/** The prompts of one archived version, as JSON. Only versions listed in
 *  docs/prompts/versions.json are served; anything else is 404. */
export async function GET(
  _request: Request,
  { params }: { params: { version: string } },
): Promise<Response> {
  const prompts = await extractListedVersionPrompts(params.version);
  if (prompts === null) {
    return new Response("Not found", { status: 404 });
  }
  return Response.json(prompts, {
    headers: { "Cache-Control": "public, max-age=86400" },
  });
}
