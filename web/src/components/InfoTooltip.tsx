"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { MouseEvent } from "react";

type InfoTooltipProps = {
  text: string;
};

type Position = {
  top: number;
  left: number;
};

const TOOLTIP_OFFSET = 8;
const TOOLTIP_WIDTH = 320;

export default function InfoTooltip({ text }: InfoTooltipProps) {
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState<Position | null>(null);

  const updatePosition = () => {
    if (!triggerRef.current) {
      return;
    }
    const rect = triggerRef.current.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    const left = Math.min(
      Math.max(TOOLTIP_OFFSET, rect.left),
      Math.max(TOOLTIP_OFFSET, viewportWidth - TOOLTIP_WIDTH - TOOLTIP_OFFSET)
    );
    const top = rect.bottom + TOOLTIP_OFFSET;
    setPosition({ top, left });
  };

  useEffect(() => {
    if (!open) {
      return;
    }
    updatePosition();
    const handleScroll = () => updatePosition();
    const handleResize = () => updatePosition();
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node | null;
      if (triggerRef.current?.contains(target) || tooltipRef.current?.contains(target)) {
        return;
      }
      setOpen(false);
    };
    window.addEventListener("scroll", handleScroll, true);
    window.addEventListener("resize", handleResize);
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      window.removeEventListener("scroll", handleScroll, true);
      window.removeEventListener("resize", handleResize);
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [open]);

  const handleMouseDown = (event: MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const handleClick = (event: MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();
    setOpen((prev) => !prev);
  };

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className="rounded-full border border-slate-500 px-1 text-xs text-slate-300"
        onMouseDown={handleMouseDown}
        onClick={handleClick}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
      >
        ?
      </button>
      {open && position
        ? createPortal(
            <div
              ref={tooltipRef}
              className="z-50 max-w-xs rounded border border-slate-600 bg-slate-950 px-2 py-1 text-xs text-slate-200 shadow-lg"
              style={{
                position: "fixed",
                top: position.top,
                left: position.left,
                width: TOOLTIP_WIDTH,
              }}
              role="tooltip"
            >
              {text}
            </div>,
            document.body
          )
        : null}
    </>
  );
}
