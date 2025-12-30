export type ScenarioBlock = { type: "h2" | "h3" | "p" | "li"; text: string };

const H2_HEADINGS = ["Context", "Humanitarian Needs", "Operational Impacts"];
const H3_HEADINGS = [
  "WASH",
  "Health",
  "Nutrition",
  "Protection",
  "Education",
  "Shelter",
  "FoodSecurity",
];

const normalizeHeading = (value: string): string => value.trim();

export const formatScenario = (text: string): ScenarioBlock[] => {
  const normalized = text.replace(/\r\n/g, "\n");
  const withInjectedBreaks = normalized
    .replace(/Context\s*-\s*•/gi, "Context\n•")
    .replace(/Humanitarian Needs\s*-\s*/gi, "Humanitarian Needs\n")
    .replace(/Operational Impacts\s*-\s*•/gi, "Operational Impacts\n•")
    .replace(/\s-\s*•\s*/g, "\n• ");

  const lines = withInjectedBreaks
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const blocks: ScenarioBlock[] = [];

  lines.forEach((line) => {
    const cleanLine = normalizeHeading(line.replace(/\s+/g, " "));
    if (H2_HEADINGS.includes(cleanLine)) {
      blocks.push({ type: "h2", text: cleanLine });
      return;
    }

    const h3Match = H3_HEADINGS.find((heading) =>
      cleanLine.toLowerCase().startsWith(`${heading.toLowerCase()}:`)
    );
    if (h3Match) {
      blocks.push({ type: "h3", text: h3Match });
      const remainder = cleanLine.slice(h3Match.length + 1).trim();
      if (remainder) {
        blocks.push({ type: "p", text: remainder });
      }
      return;
    }

    if (cleanLine.startsWith("•") || cleanLine.startsWith("-")) {
      const bullet = cleanLine.replace(/^[•-]\s?/, "").trim();
      if (bullet) {
        blocks.push({ type: "li", text: bullet });
      }
      return;
    }

    const colonMatch = H3_HEADINGS.find((heading) =>
      cleanLine.toLowerCase().includes(`${heading.toLowerCase()}:`)
    );
    if (colonMatch) {
      const [before, after] = cleanLine.split(/:\s*/, 2);
      blocks.push({ type: "h3", text: before.trim() || colonMatch });
      if (after) {
        blocks.push({ type: "p", text: after.trim() });
      }
      return;
    }

    blocks.push({ type: "p", text: cleanLine });
  });

  return blocks;
};
