// Pythia
// Copyright (c) 2025 Kevin Wyjad
// Licensed under the Pythia Non-Commercial Public License v1.0.
// See the LICENSE file in the project root for details.

export default function Loading() {
  return (
    <div className="space-y-6" aria-busy="true" aria-label="Loading reports">
      <div className="h-9 w-64 animate-pulse rounded-md bg-fred-border/60" />
      <div className="h-16 max-w-3xl animate-pulse rounded-md bg-fred-border/60" />
      {[0, 1, 2].map((i) => (
        <div key={i} className="h-28 animate-pulse rounded-md bg-fred-border/60" />
      ))}
    </div>
  );
}
