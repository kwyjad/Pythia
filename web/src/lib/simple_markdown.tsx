import React from "react";

const INLINE_TOKEN_REGEX = /(\*\*\*[^*]+\*\*\*|\*\*[^*]+\*\*|https?:\/\/\S+)/g;
const HEADING_REGEX = /^(#{1,3})\s+(.*)$/;
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
        <a key={`link-${inlineKey}`} href={part} rel="noreferrer" target="_blank">
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
        <p key={`p-${blockKey}`}>{renderInline(text)}</p>
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
    elements.push(
      <Tag key={`list-${blockKey}`}>
        {listItems.map((item, index) => (
          <li key={`li-${blockKey}-${index}`}>{item}</li>
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
      blockKey += 1;
      elements.push(<hr key={`hr-${blockKey}`} />);
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
          <h1 key={`h1-${blockKey}`}>{renderInline(content)}</h1>
        );
      } else if (level === 2) {
        elements.push(
          <h2 key={`h2-${blockKey}`}>{renderInline(content)}</h2>
        );
      } else {
        elements.push(
          <h3 key={`h3-${blockKey}`}>{renderInline(content)}</h3>
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
