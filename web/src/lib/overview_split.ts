// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

/** The heading that opens the change log in docs/fred_overview.md. The deck
 *  sync check (scripts/decks/check_deck_sync.py) looks for the same text. */
export const CHANGELOG_HEADING = "Changes in this Version";

const CHANGELOG_LINE = /^##\s+Changes in this Version\s*$/;

export interface SplitOverview {
  /** Everything except the change log, in its original order. */
  main: string;
  /** The change log's body, without its heading line; null when absent. */
  changelog: string | null;
}

/**
 * Split an overview into its main text and its change log. The change log
 * runs from its `##` heading to the next `##` heading or the end, wherever it
 * sits: the current document puts it last, archived versions put it first.
 * A document with no change log heading is returned whole.
 */
export function splitOverview(md: string): SplitOverview {
  const lines = md.split(/\r?\n/);
  const start = lines.findIndex((l) => CHANGELOG_LINE.test(l.trim()));
  if (start < 0) return { main: md, changelog: null };
  let end = lines.length;
  for (let i = start + 1; i < lines.length; i += 1) {
    if (/^##\s/.test(lines[i]) && !/^###/.test(lines[i])) {
      end = i;
      break;
    }
  }
  const main = [...lines.slice(0, start), ...lines.slice(end)].join("\n").trim();
  const changelog = lines.slice(start + 1, end).join("\n").trim();
  return { main, changelog };
}
