"use client";
import { useEffect, useMemo, useRef, useState } from "react";

import { classifyJenks, jenksBreaks } from "../lib/jenks";
import type { CountriesRow, RiskIndexRow, RiskView } from "../lib/types";

type RiskIndexMapProps = {
  riskRows: RiskIndexRow[];
  countriesRows: CountriesRow[];
  view: RiskView;
};

type DebugSample = {
  dLen: number;
  fillAttr: string | null;
  styleAttr: string | null;
  computedFill: string;
  computedOpacity: string;
  computedDisplay: string;
};

type DebugInfo = {
  svgLoaded: boolean;
  viewBox: string | null;
  widthAttr: string | null;
  heightAttr: string | null;
  taggedNodes: number;
  targetPaths: number;
  medianBboxWidth: number | null;
  medianBboxHeight: number | null;
  firstPathSample: DebugSample | null;
  lastApplyStatus: string;
};

const eivFormatter = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 0,
});

const perCapitaFormatter = new Intl.NumberFormat(undefined, {
  style: "percent",
  minimumFractionDigits: 3,
  maximumFractionDigits: 8,
});

const formatValueLabel = (value: number, isPerCapita: boolean) =>
  isPerCapita ? perCapitaFormatter.format(value) : eivFormatter.format(value);

export default function RiskIndexMap({
  riskRows,
  countriesRows,
  view,
}: RiskIndexMapProps) {
  const [svgText, setSvgText] = useState<string>("");
  const [svgWarnings, setSvgWarnings] = useState<string[]>([]);
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    title: string;
    value: string;
  } | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const debugEnabled =
    typeof window !== "undefined" &&
    (new URLSearchParams(window.location.search).get("debug_map") === "1" ||
      window.localStorage.getItem("pythia_debug_map") === "1");

  useEffect(() => {
    let active = true;
    fetch("/maps/world.svg")
      .then((res) => res.text())
      .then((data) => {
        if (!active) return;
        setSvgText(data);
      })
      .catch((error) => {
        console.warn("Failed to load world map:", error);
      });
    return () => {
      active = false;
    };
  }, []);

  const isPerCapita = view === "PA_PC" || view === "FATALITIES_PC";

  const hasQuestionsIso3 = useMemo(() => {
    const set = new Set<string>();
    countriesRows.forEach((row) => {
      const iso3 = (row.iso3 ?? "").toUpperCase();
      if (iso3) {
        set.add(iso3);
      }
    });
    return set;
  }, [countriesRows]);

  const valueByIso3 = useMemo(() => {
    const map = new Map<string, number>();
    riskRows.forEach((row) => {
      const iso3 = (row.iso3 ?? "").toUpperCase();
      const value = isPerCapita ? row.total_pc : row.total;
      if (iso3 && typeof value === "number" && Number.isFinite(value)) {
        map.set(iso3, value);
      }
    });
    return map;
  }, [riskRows, isPerCapita]);

  const values = useMemo(
    () => Array.from(valueByIso3.values()).filter((value) => Number.isFinite(value)),
    [valueByIso3]
  );

  const breaks = useMemo(() => jenksBreaks(values, 5), [values]);

  useEffect(() => {
    if (!svgText) {
      setTooltip(null);
      setSvgWarnings([]);
      if (debugEnabled) {
        setDebugInfo({
          svgLoaded: false,
          viewBox: null,
          widthAttr: null,
          heightAttr: null,
          taggedNodes: 0,
          targetPaths: 0,
          medianBboxWidth: null,
          medianBboxHeight: null,
          firstPathSample: null,
          lastApplyStatus: "no svg",
        });
      } else {
        setDebugInfo(null);
      }
      return;
    }
    const container = containerRef.current;
    if (!container) return;
    const svgEl = container.querySelector("svg");
    if (svgEl) {
      svgEl.setAttribute("width", "100%");
      svgEl.setAttribute("height", "100%");
      svgEl.setAttribute("preserveAspectRatio", "xMidYMid meet");
    }
    const iso3Elements = Array.from(
      container.querySelectorAll<SVGElement>("[data-iso3]")
    );
    const paths = iso3Elements.flatMap((el) => {
      if (el.tagName.toLowerCase() === "path") {
        return [el as SVGPathElement];
      }
      return Array.from(el.querySelectorAll<SVGPathElement>("path"));
    });
    const warnings: string[] = [];
    const css = getComputedStyle(document.documentElement);
    const palette = {
      c1: css.getPropertyValue("--risk-map-c1").trim() || "#7fd3dd",
      c2: css.getPropertyValue("--risk-map-c2").trim() || "#4bb5c4",
      c3: css.getPropertyValue("--risk-map-c3").trim() || "#1e92a3",
      c4: css.getPropertyValue("--risk-map-c4").trim() || "#0f7183",
      c5: css.getPropertyValue("--risk-map-c5").trim() || "#0b5563",
      noEiv: css.getPropertyValue("--risk-map-no-eiv").trim() || "#6b7280",
      noQ: css.getPropertyValue("--risk-map-no-questions").trim() || "#cbd5f5",
    };
    const colorScale = [
      palette.c1,
      palette.c2,
      palette.c3,
      palette.c4,
      palette.c5,
    ];
    if (iso3Elements.length < 150) {
      warnings.push(
        "World map asset invalid: expected 150+ country paths with data-iso3. Check web/public/maps/world.svg."
      );
    }
    const sampledLengths = paths
      .slice(0, 20)
      .map((path) => (path.getAttribute("d") || "").length)
      .filter((length) => length > 0)
      .sort((a, b) => a - b);
    if (sampledLengths.length) {
      const mid = Math.floor(sampledLengths.length / 2);
      const median =
        sampledLengths.length % 2 === 0
          ? (sampledLengths[mid - 1] + sampledLengths[mid]) / 2
          : sampledLengths[mid];
      if (median < 150) {
        warnings.push(
          "World map asset looks non-polygonal (paths too small). Replace web/public/maps/world.svg with real country outlines."
        );
      }
    }
    setSvgWarnings(warnings);
    const normalizeVisibility = (path: SVGPathElement) => {
      path.style.setProperty("display", "inline", "important");
      path.style.setProperty("visibility", "visible", "important");
      path.style.setProperty("opacity", "1", "important");
      path.style.setProperty("stroke", "rgba(148,163,184,0.55)", "important");
      path.style.setProperty("stroke-width", "0.6", "important");
      path.style.setProperty("vector-effect", "non-scaling-stroke", "important");
    };
    const listeners: Array<() => void> = [];
    let lastApplyStatus = "ok";
    let debugSample: DebugSample[] = [];
    let medianBboxWidth: number | null = null;
    let medianBboxHeight: number | null = null;
    try {
      iso3Elements.forEach((element) => {
        const targets =
          element.tagName.toLowerCase() === "path"
            ? [element as SVGPathElement]
            : Array.from(element.querySelectorAll<SVGPathElement>("path"));
        targets.forEach((path) => normalizeVisibility(path));
      });
    } catch (error) {
      lastApplyStatus = `failed: ${
        error instanceof Error ? error.message : "unknown error"
      }`;
    }
    iso3Elements.forEach((element) => {
      const iso3 = (element.getAttribute("data-iso3") || "").toUpperCase();
      if (!iso3) {
        return;
      }
      const targets =
        element.tagName.toLowerCase() === "path"
          ? [element as SVGPathElement]
          : Array.from(element.querySelectorAll<SVGPathElement>("path"));
      if (!targets.length) {
        return;
      }
      const value = valueByIso3.get(iso3);
      let fillColor = palette.noQ;
      if (typeof value === "number" && Number.isFinite(value)) {
        const classIndex = classifyJenks(value, breaks);
        fillColor = colorScale[classIndex] ?? palette.c1;
      } else if (hasQuestionsIso3.has(iso3)) {
        fillColor = palette.noEiv;
      }
      targets.forEach((path) => {
        normalizeVisibility(path);
        path.style.setProperty("fill", fillColor, "important");
        path.style.setProperty("stroke", "rgba(148,163,184,0.55)", "important");
        path.style.setProperty("stroke-width", "0.6", "important");
        path.style.setProperty("vector-effect", "non-scaling-stroke", "important");
      });

      const handleMouseMove = (event: MouseEvent) => {
        const target = event.currentTarget as SVGElement | null;
        const name =
          target?.getAttribute("data-name")?.trim() || iso3.toUpperCase();
        let valueLabel = "No forecasts";
        if (typeof value === "number") {
          valueLabel = formatValueLabel(value, isPerCapita);
        } else if (!hasQuestionsIso3.has(iso3)) {
          valueLabel = "No questions";
        }
        const rect = container.getBoundingClientRect();
        setTooltip({
          x: event.clientX - rect.left + 12,
          y: event.clientY - rect.top + 12,
          title: name,
          value: valueLabel,
        });
      };

      const handleMouseLeave = () => setTooltip(null);

      targets.forEach((path) => {
        path.addEventListener("mousemove", handleMouseMove);
        path.addEventListener("mouseleave", handleMouseLeave);
      });

      listeners.push(() => {
        targets.forEach((path) => {
          path.removeEventListener("mousemove", handleMouseMove);
          path.removeEventListener("mouseleave", handleMouseLeave);
        });
      });
    });
    if (debugEnabled) {
      const sampleTargets = paths.slice(0, 10);
      debugSample = sampleTargets.map((path) => ({
        dLen: path.getAttribute("d")?.length ?? 0,
        fillAttr: path.getAttribute("fill"),
        styleAttr: path.getAttribute("style"),
        computedFill: getComputedStyle(path).fill,
        computedOpacity: getComputedStyle(path).opacity,
        computedDisplay: getComputedStyle(path).display,
      }));
      const bboxSamples: Array<{ width: number; height: number }> = [];
      sampleTargets.forEach((path) => {
        try {
          const bbox = path.getBBox();
          if (Number.isFinite(bbox.width) && Number.isFinite(bbox.height)) {
            bboxSamples.push({ width: bbox.width, height: bbox.height });
          }
        } catch {
          // getBBox can throw if element isn't rendered.
        }
      });
      if (bboxSamples.length) {
        const widths = bboxSamples
          .map((sample) => sample.width)
          .sort((a, b) => a - b);
        const heights = bboxSamples
          .map((sample) => sample.height)
          .sort((a, b) => a - b);
        const mid = Math.floor(widths.length / 2);
        const pickMedian = (list: number[]) =>
          list.length % 2 === 0
            ? (list[mid - 1] + list[mid]) / 2
            : list[mid];
        medianBboxWidth = pickMedian(widths);
        medianBboxHeight = pickMedian(heights);
      }
      const meta = {
        svgLoaded: Boolean(svgEl),
        viewBox: svgEl?.getAttribute("viewBox") ?? null,
        widthAttr: svgEl?.getAttribute("width") ?? null,
        heightAttr: svgEl?.getAttribute("height") ?? null,
        taggedNodes: iso3Elements.length,
        targetPaths: paths.length,
        medianBboxWidth,
        medianBboxHeight,
        lastApplyStatus,
      };
      console.groupCollapsed("[RiskIndexMap] debug");
      console.log(meta);
      console.table(debugSample);
      console.groupEnd();
      setDebugInfo({
        ...meta,
        firstPathSample: debugSample[0] ?? null,
      });
    } else {
      setDebugInfo(null);
    }

    return () => {
      listeners.forEach((cleanup) => cleanup());
    };
  }, [svgText, valueByIso3, breaks, hasQuestionsIso3, isPerCapita, debugEnabled]);

  return (
    <div className="relative w-full rounded-lg border border-slate-800 bg-slate-950/30 p-4">
      <div className="text-sm font-semibold text-slate-200">World overview</div>
      <p className="mt-1 text-xs text-slate-400">
        Jenks breaks calculated from the selected risk values.
      </p>
      {svgWarnings.length ? (
        <div className="mt-3 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
          {svgWarnings.map((warning) => (
            <div key={warning}>{warning}</div>
          ))}
        </div>
      ) : null}
      {debugEnabled && debugInfo ? (
        <div className="mt-3 rounded-md border border-slate-700/60 bg-slate-950/30 px-3 py-2 text-xs text-slate-200">
          <div className="font-semibold text-slate-100">Map debug</div>
          <div className="text-slate-400">
            Enable via ?debug_map=1 or localStorage pythia_debug_map=1
          </div>
          <div className="mt-2 grid gap-1 text-slate-200">
            <div>SVG loaded: {debugInfo.svgLoaded ? "yes" : "no"}</div>
            <div>viewBox: {debugInfo.viewBox ?? "none"}</div>
            <div>
              tagged nodes: {debugInfo.taggedNodes} | target paths:{" "}
              {debugInfo.targetPaths}
            </div>
            <div>
              median bbox (w×h):{" "}
              {debugInfo.medianBboxWidth && debugInfo.medianBboxHeight
                ? `${debugInfo.medianBboxWidth.toFixed(
                    2
                  )}×${debugInfo.medianBboxHeight.toFixed(2)}`
                : "n/a"}
            </div>
            <div>
              first path computed fill:{" "}
              {debugInfo.firstPathSample?.computedFill ?? "n/a"} | opacity:{" "}
              {debugInfo.firstPathSample?.computedOpacity ?? "n/a"} | display:{" "}
              {debugInfo.firstPathSample?.computedDisplay ?? "n/a"}
            </div>
            <div>last apply pass: {debugInfo.lastApplyStatus}</div>
          </div>
        </div>
      ) : null}
      <div ref={containerRef} className="relative mt-3 h-[360px] w-full">
        <div
          aria-label="Risk index world map"
          className="h-full w-full [&_svg]:h-full [&_svg]:w-full"
          dangerouslySetInnerHTML={{ __html: svgText }}
        />
        {tooltip ? (
          <div
            className="pointer-events-none absolute z-10 rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-100 shadow-lg"
            style={{ left: tooltip.x, top: tooltip.y }}
          >
            <div className="font-semibold">{tooltip.title}</div>
            <div className="text-slate-300">{tooltip.value}</div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
