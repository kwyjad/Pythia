import React from "react";

const INLINE_TOKEN_REGEX = /(\*\*\*[^*]+\*\*\*|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\)|https?:\/\/\S+)/g;
const TABLE_ROW_REGEX = /^\|.+\|$/;
const TABLE_SEP_REGEX = /^\|[\s:-]+\|$/;
const HEADING_REGEX = /^(#{1,4})\s+(.*)$/;
const ORDERED_LIST_REGEX = /^\d+[.)]\s+(.*)$/;
const UNORDERED_LIST_REGEX = /^-\s+(.*)$/;

function renderInline(text: string): React.ReactNode[] {
  const parts = text.split(INLINE_TOKEN_REGEX).filter((part) => part !== "");
  let inlineKey = 0;

  return parts.map((part) => {
    if (part.startsWith("**")) {
      const content = part.replace(/\*+/g, "");
      inlineKey += 1;
      return <strong key={`bold-${inlineKey}`}>{content}</strong>;
    }

    const linkMatch = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (linkMatch) {
      inlineKey += 1;
      return (
        <a
          key={`mdlink-${inlineKey}`}
          className="text-fred-primary underline underline-offset-2 hover:text-fred-secondary"
          href={linkMatch[2]}
          rel="noreferrer"
          target="_blank"
        >
          {linkMatch[1]}
        </a>
      );
    }

    if (part.startsWith("http://") || part.startsWith("https://")) {
      inlineKey += 1;
      return (
        <a
          key={`link-${inlineKey}`}
          className="text-fred-primary underline underline-offset-2 hover:text-fred-secondary"
          href={part}
          rel="noreferrer"
          target="_blank"
        >
          {part}
        </a>
      );
    }

    inlineKey += 1;
    return <React.Fragment key={`text-${inlineKey}`}>{part}</React.Fragment>;
  });
}

export function renderSimpleMarkdown(md: string): React.ReactNode {
  const lines = md.split(/\r?\n/);
  const elements: React.ReactNode[] = [];
  let paragraphLines: string[] = [];
  let listType: "ul" | "ol" | null = null;
  let listItems: React.ReactNode[][] = [];
  let tableLines: string[] = [];
  let blockKey = 0;

  const flushTable = () => {
    if (tableLines.length < 2) {
      tableLines = [];
      return;
    }
    const parseRow = (line: string) =>
      line.split("|").slice(1, -1).map((c) => c.trim());

    const header = parseRow(tableLines[0]);
    const bodyStart = TABLE_SEP_REGEX.test(tableLines[1]) ? 2 : 1;
    const bodyRows = tableLines.slice(bodyStart).map(parseRow);

    blockKey += 1;
    elements.push(
      <div key={`tw-${blockKey}`} className="my-4 overflow-x-auto">
        <table className="min-w-full border-collapse text-sm text-fred-text">
          {bodyStart === 2 && (
            <thead>
              <tr>
                {header.map((cell, i) => (
                  <th
                    key={`th-${blockKey}-${i}`}
                    className="border border-fred-secondary/40 bg-fred-surface px-3 py-2 text-left font-semibold"
                  >
                    {renderInline(cell)}
                  </th>
                ))}
              </tr>
            </thead>
          )}
          <tbody>
            {bodyRows.map((row, ri) => (
              <tr key={`tr-${blockKey}-${ri}`}>
                {row.map((cell, ci) => (
                  <td
                    key={`td-${blockKey}-${ri}-${ci}`}
                    className="border border-fred-secondary/40 px-3 py-2"
                  >
                    {renderInline(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
    tableLines = [];
  };

  const flushParagraph = () => {
    if (paragraphLines.length === 0) {
      return;
    }
    const text = paragraphLines.join(" ").trim();
    if (text) {
      blockKey += 1;
      elements.push(
        <p key={`p-${blockKey}`} className="my-4 leading-relaxed text-fred-text">
          {renderInline(text)}
        </p>
      );
    }
    paragraphLines = [];
  };

  const flushList = () => {
    if (!listType || listItems.length === 0) {
      listType = null;
      listItems = [];
      return;
    }

    blockKey += 1;
    const Tag = listType;
    const listClass =
      listType === "ol"
        ? "my-4 list-decimal pl-6 space-y-1"
        : "my-4 list-disc pl-6 space-y-1";
    elements.push(
      <Tag key={`list-${blockKey}`} className={listClass}>
        {listItems.map((item, index) => (
          <li key={`li-${blockKey}-${index}`} className="text-fred-text">
            {item}
          </li>
        ))}
      </Tag>
    );
    listType = null;
    listItems = [];
  };

  lines.forEach((line) => {
    const trimmed = line.trim();

    if (trimmed === "") {
      flushParagraph();
      flushList();
      flushTable();
      return;
    }

    if (trimmed === "---") {
      flushParagraph();
      flushList();
      flushTable();
      return;
    }

    if (TABLE_ROW_REGEX.test(trimmed)) {
      flushParagraph();
      flushList();
      tableLines.push(trimmed);
      return;
    }

    if (tableLines.length > 0) {
      flushTable();
    }

    const headingMatch = trimmed.match(HEADING_REGEX);
    if (headingMatch) {
      flushParagraph();
      flushList();
      const level = headingMatch[1].length;
      const content = headingMatch[2];
      blockKey += 1;
      if (level === 1) {
        elements.push(
          <h1 key={`h1-${blockKey}`} className="mt-10 mb-3 text-2xl font-semibold">
            {renderInline(content)}
          </h1>
        );
      } else if (level === 2) {
        elements.push(
          <h2 key={`h2-${blockKey}`} className="mt-8 mb-3 text-xl font-semibold">
            {renderInline(content)}
          </h2>
        );
      } else if (level === 3) {
        elements.push(
          <h3 key={`h3-${blockKey}`} className="mt-6 mb-2 text-lg font-semibold">
            {renderInline(content)}
          </h3>
        );
      } else {
        elements.push(
          <h4 key={`h4-${blockKey}`} className="mt-4 mb-2 text-base font-semibold text-fred-text">
            {renderInline(content)}
          </h4>
        );
      }
      return;
    }

    const unorderedMatch = trimmed.match(UNORDERED_LIST_REGEX);
    if (unorderedMatch) {
      flushParagraph();
      if (listType && listType !== "ul") {
        flushList();
      }
      listType = "ul";
      listItems.push(renderInline(unorderedMatch[1]));
      return;
    }

    const orderedMatch = trimmed.match(ORDERED_LIST_REGEX);
    if (orderedMatch) {
      flushParagraph();
      if (listType && listType !== "ol") {
        flushList();
      }
      listType = "ol";
      listItems.push(renderInline(orderedMatch[1]));
      return;
    }

    if (listType) {
      flushList();
    }
    paragraphLines.push(trimmed);
  });

  flushParagraph();
  flushList();
  flushTable();

  return <>{elements}</>;
}
