import React from "react";

const INLINE_TOKEN_REGEX = /(\*\*\*[^*]+\*\*\*|\*\*[^*]+\*\*|https?:\/\/\S+)/g;
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

    if (part.startsWith("http://") || part.startsWith("https://")) {
      inlineKey += 1;
      return (
        <a
          key={`link-${inlineKey}`}
          className="text-sky-300 underline underline-offset-2 hover:text-sky-200"
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
  let blockKey = 0;

  const flushParagraph = () => {
    if (paragraphLines.length === 0) {
      return;
    }
    const text = paragraphLines.join(" ").trim();
    if (text) {
      blockKey += 1;
      elements.push(
        <p key={`p-${blockKey}`} className="my-4 leading-relaxed text-slate-200">
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
          <li key={`li-${blockKey}-${index}`} className="text-slate-200">
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
      return;
    }

    if (trimmed === "---") {
      flushParagraph();
      flushList();
      return;
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
          <h1 key={`h1-${blockKey}`} className="mt-10 mb-3 text-2xl font-semibold text-white">
            {renderInline(content)}
          </h1>
        );
      } else if (level === 2) {
        elements.push(
          <h2 key={`h2-${blockKey}`} className="mt-8 mb-3 text-xl font-semibold text-white">
            {renderInline(content)}
          </h2>
        );
      } else if (level === 3) {
        elements.push(
          <h3 key={`h3-${blockKey}`} className="mt-6 mb-2 text-lg font-semibold text-white">
            {renderInline(content)}
          </h3>
        );
      } else {
        elements.push(
          <h4 key={`h4-${blockKey}`} className="mt-4 mb-2 text-base font-semibold text-white">
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

  return <>{elements}</>;
}
