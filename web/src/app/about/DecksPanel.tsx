// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

import type { Deck } from "../../lib/decks";
import { formatSize } from "../../lib/decks";

const LINK =
  "font-medium text-fred-primary underline underline-offset-2 hover:text-fred-text";

export default function DecksPanel({ decks }: { decks: Deck[] }) {
  return (
    <section
      aria-labelledby="slide-decks"
      className="rounded-md border border-fred-border bg-fred-surface p-4 shadow-fredCard"
    >
      <h2 id="slide-decks" className="text-xl font-semibold">
        Slide decks
      </h2>
      <p className="mt-1 text-sm text-fred-text">
        Eight short decks that walk through Fred, one part at a time.
      </p>
      {decks.length === 0 ? (
        <p className="mt-3 text-sm text-fred-text" data-testid="decks-unavailable">
          The slide decks are not available yet. Please check back later.
        </p>
      ) : (
        <ul className="mt-3 divide-y divide-fred-border">
          {decks.map((d) => {
            const size = formatSize(d.sizeBytes);
            return (
              <li
                key={d.number}
                className="flex flex-wrap items-baseline gap-x-4 gap-y-1 py-2 text-sm"
              >
                <span className="min-w-[18rem] flex-1 text-fred-text">
                  <span className="mr-2 text-fred-muted">{d.number}</span>
                  {d.title}
                </span>
                {d.viewUrl && (
                  <a href={d.viewUrl} target="_blank" rel="noopener noreferrer" className={LINK}>
                    View
                  </a>
                )}
                {d.downloadUrl && (
                  <a href={d.downloadUrl} rel="noopener noreferrer" className={LINK}>
                    Download{size ? ` (${size})` : ""}
                  </a>
                )}
                {d.lastChecked && (
                  <span className="text-xs text-fred-muted">
                    Checked {d.lastChecked}
                  </span>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
