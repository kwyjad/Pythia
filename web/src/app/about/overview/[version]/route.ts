// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

import { loadListedOverview } from "../../../../lib/overview_loader";

export const dynamic = "force-dynamic";

/** One archived version of the system overview, as Markdown. Only versions
 *  listed in docs/overview/versions.json are served; anything else is 404. */
export async function GET(
  _request: Request,
  { params }: { params: { version: string } },
): Promise<Response> {
  const markdown = await loadListedOverview(params.version);
  if (markdown === null) {
    return new Response("Not found", { status: 404 });
  }
  return new Response(markdown, {
    status: 200,
    headers: {
      "Content-Type": "text/markdown; charset=utf-8",
      // An archived version never changes once written.
      "Cache-Control": "public, max-age=86400",
    },
  });
}
