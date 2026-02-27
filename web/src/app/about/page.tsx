import { readFile } from "node:fs/promises";
import path from "node:path";

import {
  extractAllPrompts,
  extractAllVersionedPrompts,
} from "../../lib/prompt_extractor";
import { renderSimpleMarkdown } from "../../lib/simple_markdown";
import AiPromptsSection from "./AiPromptsSection";

const WELCOME_MD = `## Welcome!

Fred is an experimental humanitarian impact forecasting system. Its objective is to test the effectiveness of LLM as horizon scanning and forecasting agents in the humanitarian space. Fred started operation in December 2025. Its first "stable" run (i.e. with a standard set of models and processes) was in January 2026. Fred runs monthly and will update with new forecasts at the start of each month.

To repeat: FRED IS AN EXPERIMENTAL SYSTEM. Do not use Fred's forecasts for anything, ever, for any reason, except pure entertainment. Consult an astrologer instead. Assume Fred's forecasts are rubbish. Whether or not they are any good is what we are here to figure out. Even if for some reason Fred's forecasts are not rubbish, they certainly don't foresee the future.

View at your own risk, and don't even think about using Fred's outputs. Beyond this, don't expect stability. Everything about Fred could change at any moment. In fact, it will change, that's the only sure thing around here. It is an experimental system. Question resolution is a major weak spot and this will negatively affect forecasting skill scoring.

## Code and Contact

- Fred's code (almost entirely python) is open for non-profit or research use - not commercial. You can access the code at [github.com/kwyjad/Pythia](https://github.com/kwyjad/Pythia). Be aware that the code is a vibe-coded mess, and again, use at your own risk (Note: In GitHub Fred is called Pythia).
- If you are interested in talking or collaborating, so am I. Contact me on [LinkedIn](https://www.linkedin.com/in/kevinwyjad/).
`;

async function loadOverviewMarkdown(): Promise<string | null> {
  const candidates = [
    path.join(process.cwd(), "..", "docs", "fred_overview.md"),
    path.join(process.cwd(), "docs", "fred_overview.md"),
  ];

  for (const p of candidates) {
    try {
      return await readFile(p, "utf-8");
    } catch {
      // keep trying
    }
  }
  return null;
}

export const metadata = {
  title: "About",
};

export default async function AboutPage() {
  const [overviewMd, prompts, versionedData] = await Promise.all([
    loadOverviewMarkdown(),
    extractAllPrompts(),
    extractAllVersionedPrompts(),
  ]);

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold">About</h1>
        <p className="text-sm text-fred-text">
          About Fred and how the system works.
        </p>
      </header>
      <article className="max-w-none">
        {renderSimpleMarkdown(WELCOME_MD)}
        {overviewMd && renderSimpleMarkdown(overviewMd)}
        <AiPromptsSection
          currentPrompts={prompts}
          versions={versionedData.versions}
          versionedPrompts={versionedData.prompts}
        />
      </article>
    </div>
  );
}
