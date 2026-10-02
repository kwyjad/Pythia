// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

/**
 * Return `value` when it is an http(s) URL, else null.
 *
 * Several links on the dashboard come from model output (Sibyl's sources,
 * markdown in reports). A `javascript:` or `data:` href written there would
 * run in the dashboard's origin when clicked, so anything that is not a plain
 * web address is rendered as text instead of a link.
 */
export function safeHref(value: string | null | undefined): string | null {
  if (!value) return null;
  const trimmed = value.trim();
  try {
    const url = new URL(trimmed);
    return url.protocol === "http:" || url.protocol === "https:" ? trimmed : null;
  } catch {
    return null;
  }
}
