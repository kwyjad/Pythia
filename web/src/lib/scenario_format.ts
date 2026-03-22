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
    // Strip markdown heading markers (## / ###) and bold markers (**)
    const stripped = cleanLine.replace(/^#+\s*/, "").replace(/\*\*/g, "").trim();

    if (H2_HEADINGS.includes(stripped)) {
      blocks.push({ type: "h2", text: stripped });
      return;
    }

    const h3Match = H3_HEADINGS.find((heading) =>
      stripped.toLowerCase().startsWith(`${heading.toLowerCase()}:`)
    );
    if (h3Match) {
      blocks.push({ type: "h3", text: h3Match });
      const remainder = stripped.slice(h3Match.length + 1).trim();
      if (remainder) {
        blocks.push({ type: "p", text: remainder });
      }
      return;
    }

    // Detect markdown headings that aren't in our known lists
    if (/^#{2}\s+/.test(cleanLine)) {
      blocks.push({ type: "h2", text: stripped });
      return;
    }
    if (/^#{3}\s+/.test(cleanLine)) {
      blocks.push({ type: "h3", text: stripped });
      return;
    }

    if (cleanLine.startsWith("•") || cleanLine.startsWith("-")) {
      const bullet = cleanLine.replace(/^[•-]\s?/, "").trim();
      // Check if this "bullet" is actually a sector heading like "- Health:"
      const sectorMatch = H3_HEADINGS.find((heading) =>
        bullet.toLowerCase().startsWith(`${heading.toLowerCase()}:`)
      );
      if (sectorMatch) {
        blocks.push({ type: "h3", text: sectorMatch });
        const remainder = bullet.slice(sectorMatch.length + 1).trim();
        if (remainder) {
          blocks.push({ type: "p", text: remainder });
        }
        return;
      }
      if (bullet) {
        blocks.push({ type: "li", text: bullet });
      }
      return;
    }

    const colonMatch = H3_HEADINGS.find((heading) =>
      stripped.toLowerCase().includes(`${heading.toLowerCase()}:`)
    );
    if (colonMatch) {
      const [before, after] = stripped.split(/:\s*/, 2);
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
