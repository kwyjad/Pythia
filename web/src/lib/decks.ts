// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

import { safeHref } from "./safe_href";

/**
 * The Fred slide decks offered on the About page.
 *
 * `.github/workflows/publish_decks.yml` copies docs/decks/*.pptx to Drive and
 * uploads decks.json to the pythia-data-latest release; this module reads it.
 * A deck with no Drive copy has `viewUrl` null and a direct `downloadUrl`.
 */
export interface Deck {
  number: number;
  title: string;
  viewUrl: string | null;
  downloadUrl: string | null;
  lastChecked: string | null;
  sizeBytes: number | null;
}

export const DECKS_JSON_URL =
  process.env.PYTHIA_DECKS_JSON_URL ??
  "https://github.com/kwyjad/pythia/releases/download/pythia-data-latest/decks.json";

/** The decks in decks.json, in deck order. Anything malformed is dropped. */
export function parseDecks(raw: unknown): Deck[] {
  const list =
    raw && typeof raw === "object" && Array.isArray((raw as { decks?: unknown }).decks)
      ? ((raw as { decks: unknown[] }).decks as Record<string, unknown>[])
      : [];
  const out: Deck[] = [];
  for (const d of list) {
    if (!d || typeof d !== "object") continue;
    const number = typeof d.number === "number" ? d.number : NaN;
    const title = typeof d.title === "string" ? d.title.trim() : "";
    const viewUrl = safeHref(typeof d.view_url === "string" ? d.view_url : null);
    const downloadUrl = safeHref(typeof d.download_url === "string" ? d.download_url : null);
    if (!Number.isFinite(number) || !title || (!viewUrl && !downloadUrl)) continue;
    out.push({
      number,
      title,
      viewUrl,
      downloadUrl,
      lastChecked: typeof d.last_checked === "string" ? d.last_checked : null,
      sizeBytes: typeof d.size_bytes === "number" ? d.size_bytes : null,
    });
  }
  return out.sort((a, b) => a.number - b.number);
}

/** "2.3 MB" from a byte count. */
export function formatSize(bytes: number | null): string | null {
  if (bytes == null || !Number.isFinite(bytes) || bytes <= 0) return null;
  if (bytes < 1_000_000) return `${Math.max(1, Math.round(bytes / 1000))} KB`;
  return `${(bytes / 1_000_000).toFixed(1)} MB`;
}

/** Fetch decks.json; an unreachable or malformed file gives no decks. */
export async function loadDecks(url: string = DECKS_JSON_URL): Promise<Deck[]> {
  try {
    const res = await fetch(url, {
      next: { revalidate: 3600 },
      signal: AbortSignal.timeout(5000),
    } as RequestInit);
    if (!res.ok) return [];
    return parseDecks(await res.json());
  } catch {
    return [];
  }
}
