import React from "react";

import { safeHref } from "./safe_href";

// Order matters: bold-italic before bold before italic; code spans are
// matched as one token so nothing inside them is read as emphasis.
const INLINE_TOKEN_REGEX =
  /(`[^`]+`|\*\*\*[^*]+\*\*\*|\*\*[^*]+\*\*|(?<![\w*])\*[^*\s][^*]*\*(?![\w*])|(?<![\w])_[^_\s][^_]*_(?![\w])|\[[^\]]+\]\([^)]+\)|https?:\/\/[^\s)|]+)/g;
const TABLE_ROW_REGEX = /^\|.+\|$/;
const TABLE_SEP_REGEX = /^\|[\s:|-]+\|$/;
const HEADING_REGEX = /^(#{1,4})\s+(.*)$/;
const ORDERED_LIST_REGEX = /^(\s*)\d+[.)]\s+(.*)$/;
const UNORDERED_LIST_REGEX = /^(\s*)-\s+(.*)$/;

function renderInline(text: string): React.ReactNode[] {
  const parts = text.split(INLINE_TOKEN_REGEX).filter((part) => part !== "");
  let inlineKey = 0;

  return parts.map((part) => {
    if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) {
      inlineKey += 1;
      return (
        <code
          key={`code-${inlineKey}`}
          className="rounded bg-fred-bg px-1 py-0.5 font-mono text-[0.9em] text-fred-text"
        >
          {part.slice(1, -1)}
        </code>
      );
    }

    if (part.startsWith("***") && part.endsWith("***") && part.length > 6) {
      inlineKey += 1;
      return (
        <strong key={`bi-${inlineKey}`}>
          <em>{part.slice(3, -3)}</em>
        </strong>
      );
    }

    if (part.startsWith("**")) {
      const content = part.replace(/\*+/g, "");
      inlineKey += 1;
      return <strong key={`bold-${inlineKey}`}>{content}</strong>;
    }

    if (/^\*[^*\s][^*]*\*$/.test(part) || /^_[^_\s][^_]*_$/.test(part)) {
      inlineKey += 1;
      return <em key={`em-${inlineKey}`}>{part.slice(1, -1)}</em>;
    }

    const linkMatch = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (linkMatch) {
      inlineKey += 1;
      // The markdown is model output: a `javascript:` target renders as
      // its label, never as a link.
      const href = safeHref(linkMatch[2]);
      if (!href) {
        return <span key={`mdlink-${inlineKey}`}>{linkMatch[1]}</span>;
      }
      return (
        <a
          key={`mdlink-${inlineKey}`}
          className="text-fred-primary underline underline-offset-2 hover:text-fred-secondary"
          href={href}
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

type ListItem = { indent: number; ordered: boolean; content: React.ReactNode[] };
type ListNode = { ordered: boolean; items: { content: React.ReactNode[]; children: ListNode[] }[] };

/** Nest flat (indent, type) items: deeper indent opens a list inside the
 *  item above it; a change of type at the same depth starts a new list. */
export function buildListTree(items: ListItem[]): ListNode[] {
  const roots: ListNode[] = [];
  const stack: { indent: number; node: ListNode }[] = [];
  items.forEach((item) => {
    while (stack.length && item.indent < stack[stack.length - 1].indent) {
      stack.pop();
    }
    let top = stack[stack.length - 1];
    if (top && item.indent === top.indent && top.node.ordered !== item.ordered) {
      stack.pop();
      top = stack[stack.length - 1];
    }
    if (!top || item.indent > top.indent) {
      const node: ListNode = { ordered: item.ordered, items: [] };
      const parent = top?.node.items[top.node.items.length - 1];
      if (parent) {
        parent.children.push(node);
      } else {
        roots.push(node);
      }
      stack.push({ indent: item.indent, node });
      top = stack[stack.length - 1];
    }
    top.node.items.push({ content: item.content, children: [] });
  });
  return roots;
}

function renderList(nodes: ListNode[], key: string): React.ReactNode {
  return (
    <React.Fragment key={key}>
      {nodes.map((node, ni) => {
        const Tag = node.ordered ? "ol" : "ul";
        const cls = node.ordered
          ? "my-2 list-decimal pl-6 space-y-1"
          : "my-2 list-disc pl-6 space-y-1";
        return (
          <Tag key={`${key}-${ni}`} className={cls}>
            {node.items.map((item, ii) => (
              <li key={`${key}-${ni}-${ii}`} className="text-fred-text">
                {item.content}
                {item.children.length > 0 && renderList(item.children, `${key}-${ni}-${ii}-c`)}
              </li>
            ))}
          </Tag>
        );
      })}
    </React.Fragment>
  );
}

/** The plain text of a heading line (markdown stripped), for a contents list. */
function headingText(content: string): string {
  return content
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*+([^*]+)\*+/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .trim();
}

/** Every level-2 heading, with the id renderSimpleMarkdown gives it when
 *  called with `{ headingIds: true }`. */
export function extractHeadings(md: string): { id: string; text: string }[] {
  const out: { id: string; text: string }[] = [];
  md.split(/\r?\n/).forEach((line) => {
    const m = line.trim().match(HEADING_REGEX);
    if (m && m[1].length === 2) {
      out.push({ id: `sec-${out.length + 1}`, text: headingText(m[2]) });
    }
  });
  return out;
}

export function renderSimpleMarkdown(
  md: string,
  options: { headingIds?: boolean } = {}
): React.ReactNode {
  let h2Count = 0;
  const lines = md.split(/\r?\n/);
  const elements: React.ReactNode[] = [];
  let paragraphLines: string[] = [];
  let listItems: ListItem[] = [];
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
    if (listItems.length === 0) {
      return;
    }
    blockKey += 1;
    elements.push(renderList(buildListTree(listItems), `list-${blockKey}`));
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
          <h2
            key={`h2-${blockKey}`}
            id={options.headingIds ? `sec-${(h2Count += 1)}` : undefined}
            className="mt-8 mb-3 scroll-mt-20 text-xl font-semibold"
          >
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

    const listMatch = line.match(UNORDERED_LIST_REGEX) ?? line.match(ORDERED_LIST_REGEX);
    if (listMatch) {
      flushParagraph();
      listItems.push({
        indent: listMatch[1].replace(/\t/g, "    ").length,
        ordered: !UNORDERED_LIST_REGEX.test(line),
        content: renderInline(listMatch[2]),
      });
      return;
    }

    // A wrapped line indented under a list item continues that item.
    if (listItems.length > 0 && line.length - line.trimStart().length > 0) {
      const last = listItems[listItems.length - 1];
      last.content = [...last.content, " ", ...renderInline(trimmed)];
      return;
    }

    if (listItems.length > 0) {
      flushList();
    }
    paragraphLines.push(trimmed);
  });

  flushParagraph();
  flushList();
  flushTable();

  return <>{elements}</>;
}
