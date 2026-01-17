"use client";
import { memo, useEffect, useMemo, useRef, useState } from "react";

import { classifyJenks, jenksBreaks } from "../lib/jenks";
import type { CountriesRow, RiskIndexRow, RiskView } from "../lib/types";

type RiskIndexMapProps = {
  riskRows: RiskIndexRow[];
  countriesRows: CountriesRow[];
  view: RiskView;
  heightClassName?: string;
};

const SENTINEL_ISO3 = ["AFG", "AUS"] as const;

type DebugSample = {
  dLen: number;
  fillAttr: string | null;
  styleAttr: string | null;
  computedFill: string;
  computedOpacity: string;
  computedDisplay: string;
};

type FetchDiagnostics = {
  fetchUrl: string;
  fetchOk: boolean | null;
  fetchStatus: number | null;
  fetchContentType: string | null;
  fetchBytes: number | null;
  fetchError: string | null;
};

type BBoxDiagnostics = {
  x: number;
  y: number;
  width: number;
  height: number;
  maxX: number;
  maxY: number;
};

type SamplePathBboxStats = {
  count: number;
  wMin: number | null;
  wP50: number | null;
  wP90: number | null;
  wMax: number | null;
  hMin: number | null;
  hP50: number | null;
  hP90: number | null;
  hMax: number | null;
  tinyCount: number;
};

type SampleComputedStyle = {
  computedFill: string;
  computedStroke: string;
  computedOpacity: string;
  computedDisplay: string;
  computedVisibility: string;
};

type SentinelSvgCenter = {
  found: boolean;
  cx: number | null;
  cy: number | null;
  w: number | null;
  h: number | null;
};

type DebugInfo = {
  svgLoaded: boolean;
  originalViewBox: string | null;
  finalViewBox: string | null;
  preserveAspectRatio: string | null;
  widthAttr: string | null;
  heightAttr: string | null;
  taggedNodes: number;
  targetPaths: number;
  svgHasPathsWithD: boolean;
  firstIso3: string | null;
  requestedHeightClassName: string | null;
  medianBboxWidth: number | null;
  medianBboxHeight: number | null;
  firstPathSample: DebugSample | null;
  samplePathBboxStats: SamplePathBboxStats | null;
  sampleComputedStyle: SampleComputedStyle | null;
  sentinelSvgCenters: Record<string, SentinelSvgCenter>;
  lastApplyStatus: string;
  contentBBox: BBoxDiagnostics | null;
  coverageW: number | null;
  coverageH: number | null;
  autoFitApplied: boolean;
  autoFitViewBox: string | null;
  palette: Record<string, string>;
  svgNodeVersion: number;
  domReplaceCount: number;
  svgIsConnected: boolean;
  containerClientRect: { w: number; h: number } | null;
  svgClientRect: { w: number; h: number } | null;
  svgComputed: { display: string; visibility: string; opacity: string } | null;
  ancestorOpacityProduct: number | null;
  matchCounts: {
    riskRows: number;
    valueByIso3: number;
    iso3Elements: number;
    matchedIso3: number;
    unmatchedIso3: number;
  };
};

const eivFormatter = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 0,
});

const parseViewBox = (viewBox: string | null) => {
  if (!viewBox) return null;
  const parts = viewBox
    .trim()
    .split(/[\s,]+/)
    .map((value) => Number(value));
  if (parts.length !== 4 || parts.some((value) => Number.isNaN(value))) {
    return null;
  }
  const [vbX, vbY, vbW, vbH] = parts;
  return { vbX, vbY, vbW, vbH };
};

const SvgMarkup = memo(function SvgMarkup({ svgText }: { svgText: string }) {
  const html = useMemo(() => ({ __html: svgText }), [svgText]);
  return (
    <div
      aria-label="Forecast index world map"
      className="h-full w-full [&_svg]:h-full [&_svg]:w-full"
      dangerouslySetInnerHTML={html}
    />
  );
});

export default function RiskIndexMap({
  riskRows,
  countriesRows,
  view,
  heightClassName,
}: RiskIndexMapProps) {
  const [svgText, setSvgText] = useState<string>("");
  const [svgWarnings, setSvgWarnings] = useState<string[]>([]);
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null);
  const [fetchDiagnostics, setFetchDiagnostics] = useState<FetchDiagnostics>({
    fetchUrl: "/maps/world.svg",
    fetchOk: null,
    fetchStatus: null,
    fetchContentType: null,
    fetchBytes: null,
    fetchError: null,
  });
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    title: string;
    value: string;
  } | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const svgHostRef = useRef<HTMLDivElement | null>(null);
  const svgNodeRef = useRef<SVGSVGElement | null>(null);
  const domReplaceCountRef = useRef<number>(0);
  const svgNodeVersionRef = useRef<number>(0);
  const resolvedHeightClassName = heightClassName ?? "h-[360px]";
  const perCapitaDigits = view === "FATALITIES_PC" ? 5 : 2;
  const perCapitaFormatter = useMemo(
    () =>
      new Intl.NumberFormat(undefined, {
        style: "percent",
        minimumFractionDigits: perCapitaDigits,
        maximumFractionDigits: perCapitaDigits,
      }),
    [perCapitaDigits]
  );
  const [debugEnabled] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    const debugQuery =
      new URLSearchParams(window.location.search).get("debug_map") === "1";
    const stored = window.localStorage.getItem("pythia_debug_map_deep");
    const legacy = window.localStorage.getItem("pythia_debug_map");
    return debugQuery || stored === "1" || legacy === "1";
  });

  useEffect(() => {
    let active = true;
    fetch("/maps/world.svg")
      .then(async (res) => {
        const contentType = res.headers.get("content-type");
        const text = await res.text();
        return {
          ok: res.ok,
          status: res.status,
          contentType,
          text,
        };
      })
      .then((data) => {
        if (!active) return;
        setFetchDiagnostics({
          fetchUrl: "/maps/world.svg",
          fetchOk: data.ok,
          fetchStatus: data.status,
          fetchContentType: data.contentType,
          fetchBytes: data.text.length,
          fetchError: data.ok ? null : "HTTP error",
        });
        setSvgText(data.text);
      })
      .catch((error) => {
        console.warn("Failed to load world map:", error);
        if (!active) return;
        setFetchDiagnostics({
          fetchUrl: "/maps/world.svg",
          fetchOk: false,
          fetchStatus: null,
          fetchContentType: null,
          fetchBytes: null,
          fetchError: error instanceof Error ? error.message : "unknown error",
        });
      });
    return () => {
      active = false;
    };
  }, []);

  const isPerCapita = view === "PA_PC" || view === "FATALITIES_PC";
  const formatValueLabel = (value: number, isPerCapitaValue: boolean) =>
    isPerCapitaValue
      ? perCapitaFormatter.format(value)
      : eivFormatter.format(value);

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

  const rcByIso3 = useMemo(() => {
    const map = new Map<string, number>();
    countriesRows.forEach((row) => {
      const iso3 = (row.iso3 ?? "").toUpperCase();
      const level = row.highest_rc_level;
      if (iso3 && (level === 1 || level === 2 || level === 3)) {
        map.set(iso3, level);
      }
    });
    return map;
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
    const css = getComputedStyle(document.documentElement);
    const palette = {
      c1: css.getPropertyValue("--risk-map-c1").trim() || "#7fd3dd",
      c2: css.getPropertyValue("--risk-map-c2").trim() || "#4bb5c4",
      c3: css.getPropertyValue("--risk-map-c3").trim() || "#1e92a3",
      c4: css.getPropertyValue("--risk-map-c4").trim() || "#0f7183",
      c5: css.getPropertyValue("--risk-map-c5").trim() || "#0b5563",
      noEiv: css.getPropertyValue("--risk-map-no-eiv").trim() || "#6b7280",
      noQ: css.getPropertyValue("--risk-map-no-questions").trim() || "#cbd5f5",
      triagedNoForecast:
        css.getPropertyValue("--risk-map-no-eiv").trim() || "#6b7280",
      notTriaged:
        css.getPropertyValue("--risk-map-no-questions").trim() || "#cbd5f5",
    };
    if (!svgText) {
      setTooltip(null);
      setSvgWarnings([]);
      const emptySentinelCenters = Object.fromEntries(
        SENTINEL_ISO3.map((iso3) => [
          iso3,
          {
            found: false,
            cx: null,
            cy: null,
            w: null,
            h: null,
          },
        ])
      );
      setDebugInfo({
        svgLoaded: false,
        originalViewBox: null,
        finalViewBox: null,
        preserveAspectRatio: null,
        widthAttr: null,
        heightAttr: null,
        taggedNodes: 0,
        targetPaths: 0,
        svgHasPathsWithD: false,
        firstIso3: null,
        requestedHeightClassName: resolvedHeightClassName,
        medianBboxWidth: null,
        medianBboxHeight: null,
        firstPathSample: null,
        samplePathBboxStats: null,
        sampleComputedStyle: null,
        sentinelSvgCenters: emptySentinelCenters,
        lastApplyStatus: "no svg",
        contentBBox: null,
        coverageW: null,
        coverageH: null,
        autoFitApplied: false,
        autoFitViewBox: null,
        palette,
        svgNodeVersion: svgNodeVersionRef.current,
        domReplaceCount: domReplaceCountRef.current,
        svgIsConnected: false,
        containerClientRect: null,
        svgClientRect: null,
        svgComputed: null,
        ancestorOpacityProduct: null,
        matchCounts: {
          riskRows: riskRows.length,
          valueByIso3: valueByIso3.size,
          iso3Elements: 0,
          matchedIso3: 0,
          unmatchedIso3: 0,
        },
      });
      return;
    }
    const container = containerRef.current;
    const svgHost = svgHostRef.current;
    if (!container || !svgHost) return;
    const svgEl = svgHost.querySelector("svg");
    if (svgEl) {
      svgEl.setAttribute("width", "100%");
      svgEl.setAttribute("height", "100%");
      svgEl.setAttribute("preserveAspectRatio", "xMidYMid meet");
    }
    if (svgEl !== svgNodeRef.current) {
      svgNodeRef.current = svgEl;
      svgNodeVersionRef.current += 1;
    }
    const originalViewBox = svgEl?.getAttribute("viewBox") ?? null;
    let contentBBox: BBoxDiagnostics | null = null;
    let coverageW: number | null = null;
    let coverageH: number | null = null;
    let autoFitApplied = false;
    let autoFitViewBox: string | null = null;
    const iso3Elements = Array.from(
      svgHost.querySelectorAll<SVGElement>("[data-iso3]")
    );
    const iso3ElementMap = new Map<string, SVGElement[]>();
    iso3Elements.forEach((element) => {
      const iso3 = (element.getAttribute("data-iso3") || "").toUpperCase();
      if (!iso3) {
        return;
      }
      const existing = iso3ElementMap.get(iso3) ?? [];
      existing.push(element);
      iso3ElementMap.set(iso3, existing);
    });
    const paths = iso3Elements.flatMap((el) => {
      if (el.tagName.toLowerCase() === "path") {
        return [el as SVGPathElement];
      }
      return Array.from(el.querySelectorAll<SVGPathElement>("path"));
    });
    const warnings: string[] = [];
    const colorScale = [
      palette.c1,
      palette.c2,
      palette.c3,
      palette.c4,
      palette.c5,
    ];
    if (svgEl) {
      const existingOverlay = svgEl.querySelector("#rc-overlay");
      if (existingOverlay && existingOverlay.parentNode) {
        existingOverlay.parentNode.removeChild(existingOverlay);
      }
      if (rcByIso3.size > 0) {
        const overlay = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "g"
        );
        overlay.setAttribute("id", "rc-overlay");
        svgEl.appendChild(overlay);
        const rcColors: Record<number, string> = {
          1: "#facc15",
          2: "#fb923c",
          3: "#ef4444",
        };
        rcByIso3.forEach((level, iso3) => {
          const elements = iso3ElementMap.get(iso3) ?? [];
          if (!elements.length) {
            return;
          }
          let minX: number | null = null;
          let minY: number | null = null;
          let maxX: number | null = null;
          let maxY: number | null = null;
          elements.forEach((element) => {
            const targetPaths =
              element.tagName.toLowerCase() === "path"
                ? [element as SVGPathElement]
                : Array.from(element.querySelectorAll<SVGPathElement>("path"));
            targetPaths.forEach((path) => {
              try {
                const bbox = path.getBBox();
                if (
                  Number.isFinite(bbox.x) &&
                  Number.isFinite(bbox.y) &&
                  Number.isFinite(bbox.width) &&
                  Number.isFinite(bbox.height)
                ) {
                  const nextMinX = bbox.x;
                  const nextMinY = bbox.y;
                  const nextMaxX = bbox.x + bbox.width;
                  const nextMaxY = bbox.y + bbox.height;
                  minX = minX == null ? nextMinX : Math.min(minX, nextMinX);
                  minY = minY == null ? nextMinY : Math.min(minY, nextMinY);
                  maxX = maxX == null ? nextMaxX : Math.max(maxX, nextMaxX);
                  maxY = maxY == null ? nextMaxY : Math.max(maxY, nextMaxY);
                }
              } catch {
                // getBBox can throw if element isn't rendered.
              }
            });
          });
          if (minX == null || minY == null || maxX == null || maxY == null) {
            return;
          }
          const width = maxX - minX;
          const height = maxY - minY;
          if (!Number.isFinite(width) || !Number.isFinite(height)) {
            return;
          }
          const radius = Math.max(2, Math.min(8, Math.min(width, height) * 0.2));
          const circle = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "circle"
          );
          circle.setAttribute("cx", String(minX + width / 2));
          circle.setAttribute("cy", String(minY + height / 2));
          circle.setAttribute("r", String(radius));
          circle.setAttribute("fill", rcColors[level] ?? rcColors[1]);
          circle.setAttribute("stroke", "rgba(15, 23, 42, 0.75)");
          circle.setAttribute("stroke-width", "0.8");
          circle.setAttribute("vector-effect", "non-scaling-stroke");
          circle.setAttribute("opacity", "0.9");
          const title = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "title"
          );
          title.textContent = `${iso3} • RC Level ${level}`;
          circle.appendChild(title);
          overlay.appendChild(circle);
        });
      }
    }
    if (iso3Elements.length < 150) {
      warnings.push(
        "World map asset invalid: expected 150+ country paths with data-iso3. Check web/public/maps/world.svg."
      );
    }
    const firstPath = svgHost.querySelector("path");
    const firstPathDLen = (firstPath?.getAttribute("d") || "").length;
    const svgHasPathsWithD = firstPathDLen > 20;
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
    let samplePathBboxStats: SamplePathBboxStats | null = null;
    let sampleComputedStyle: SampleComputedStyle | null = null;
    const sentinelSvgCenters: Record<string, SentinelSvgCenter> =
      Object.fromEntries(
        SENTINEL_ISO3.map((iso3) => [
          iso3,
          { found: false, cx: null, cy: null, w: null, h: null },
        ])
      );
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
      const parsedViewBox = parseViewBox(originalViewBox);
      if (svgEl) {
        try {
          const bbox = svgEl.getBBox();
          if (
            Number.isFinite(bbox.x) &&
            Number.isFinite(bbox.y) &&
            Number.isFinite(bbox.width) &&
            Number.isFinite(bbox.height)
          ) {
            contentBBox = {
              x: bbox.x,
              y: bbox.y,
              width: bbox.width,
              height: bbox.height,
              maxX: bbox.x + bbox.width,
              maxY: bbox.y + bbox.height,
            };
          }
        } catch {
          // getBBox can throw if the SVG is not rendered yet.
        }
      }
      if (contentBBox && parsedViewBox) {
        coverageW = contentBBox.width / parsedViewBox.vbW;
        coverageH = contentBBox.height / parsedViewBox.vbH;
        if (coverageW < 0.35 || coverageH < 0.35) {
          const pad = Math.max(
            2,
            Math.min(contentBBox.width, contentBBox.height) * 0.05
          );
          const nextViewBox = [
            contentBBox.x - pad,
            contentBBox.y - pad,
            contentBBox.width + pad * 2,
            contentBBox.height + pad * 2,
          ];
          svgEl?.setAttribute("viewBox", nextViewBox.join(" "));
          autoFitApplied = true;
          autoFitViewBox = nextViewBox.join(" ");
        }
      }
      const samplePaths = paths.slice(0, 50);
      const bboxSamples: Array<{ width: number; height: number }> = [];
      let tinyCount = 0;
      samplePaths.forEach((path) => {
        try {
          const bbox = path.getBBox();
          if (Number.isFinite(bbox.width) && Number.isFinite(bbox.height)) {
            bboxSamples.push({ width: bbox.width, height: bbox.height });
            if (bbox.width < 2 && bbox.height < 2) {
              tinyCount += 1;
            }
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
        const pickPercentile = (values: number[], percentile: number) => {
          const idx = Math.max(
            0,
            Math.min(values.length - 1, Math.round((values.length - 1) * percentile))
          );
          return values[idx];
        };
        samplePathBboxStats = {
          count: bboxSamples.length,
          wMin: widths[0] ?? null,
          wP50: pickPercentile(widths, 0.5),
          wP90: pickPercentile(widths, 0.9),
          wMax: widths[widths.length - 1] ?? null,
          hMin: heights[0] ?? null,
          hP50: pickPercentile(heights, 0.5),
          hP90: pickPercentile(heights, 0.9),
          hMax: heights[heights.length - 1] ?? null,
          tinyCount,
        };
        if (samplePathBboxStats.wP50 !== null) {
          medianBboxWidth = samplePathBboxStats.wP50;
        }
        if (samplePathBboxStats.hP50 !== null) {
          medianBboxHeight = samplePathBboxStats.hP50;
        }
        const tinyRatio = tinyCount / bboxSamples.length;
        if (
          tinyRatio > 0.7 ||
          (samplePathBboxStats.wP90 !== null && samplePathBboxStats.wP90 < 10) ||
          (samplePathBboxStats.hP90 !== null && samplePathBboxStats.hP90 < 10)
        ) {
          warnings.push(
            "World SVG appears to contain mostly microscopic shapes; replace world-countries-iso3.geojson and regenerate world.svg."
          );
        }
      }
      const sampleStyleTarget = samplePaths[0];
      if (sampleStyleTarget) {
        const computed = getComputedStyle(sampleStyleTarget);
        sampleComputedStyle = {
          computedFill: computed.fill,
          computedStroke: computed.stroke,
          computedOpacity: computed.opacity,
          computedDisplay: computed.display,
          computedVisibility: computed.visibility,
        };
      }
      SENTINEL_ISO3.forEach((iso3) => {
        const element = iso3Elements.find(
          (node) => (node.getAttribute("data-iso3") || "").toUpperCase() === iso3
        );
        if (!element) {
          return;
        }
        const path =
          element.tagName.toLowerCase() === "path"
            ? (element as SVGPathElement)
            : element.querySelector<SVGPathElement>("path");
        if (!path) {
          return;
        }
        try {
          const bbox = path.getBBox();
          if (
            Number.isFinite(bbox.width) &&
            Number.isFinite(bbox.height) &&
            Number.isFinite(bbox.x) &&
            Number.isFinite(bbox.y)
          ) {
            sentinelSvgCenters[iso3] = {
              found: true,
              cx: bbox.x + bbox.width / 2,
              cy: bbox.y + bbox.height / 2,
              w: bbox.width,
              h: bbox.height,
            };
          }
        } catch {
          // getBBox can throw if element isn't rendered.
        }
      });
    }
    setSvgWarnings(warnings);
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
      const meta = {
        svgLoaded: Boolean(svgEl),
        originalViewBox,
        finalViewBox: svgEl?.getAttribute("viewBox") ?? null,
        preserveAspectRatio: svgEl?.getAttribute("preserveAspectRatio") ?? null,
        widthAttr: svgEl?.getAttribute("width") ?? null,
        heightAttr: svgEl?.getAttribute("height") ?? null,
        taggedNodes: iso3Elements.length,
        targetPaths: paths.length,
        medianBboxWidth,
        medianBboxHeight,
        lastApplyStatus,
        contentBBox,
        coverageW,
        coverageH,
        autoFitApplied,
        autoFitViewBox,
        bboxStatsPresent: Boolean(samplePathBboxStats),
        samplePathBboxStats,
        sampleComputedStyle,
        sentinelSvgCenters,
      };
      console.groupCollapsed("[RiskIndexMap] debug");
      console.log(meta);
      console.table(debugSample);
      console.groupEnd();
    }
    const iso3Set = new Set(
      iso3Elements
        .map((element) => (element.getAttribute("data-iso3") || "").toUpperCase())
        .filter((iso3) => iso3.length > 0)
    );
    let matchedIso3 = 0;
    iso3Set.forEach((iso3) => {
      if (valueByIso3.has(iso3)) {
        matchedIso3 += 1;
      }
    });
    const unmatchedIso3 = Math.max(iso3Set.size - matchedIso3, 0);
    const containerRect = container.getBoundingClientRect();
    const svgRect = svgEl?.getBoundingClientRect();
    const svgComputedStyle = svgEl ? getComputedStyle(svgEl) : null;
    let ancestorOpacityProduct: number | null = null;
    if (svgEl) {
      let product = 1;
      let node: Element | null = svgEl;
      while (node) {
        const opacityValue = parseFloat(getComputedStyle(node).opacity);
        if (!Number.isNaN(opacityValue)) {
          product *= opacityValue;
        }
        if (node === container) {
          break;
        }
        node = node.parentElement;
      }
      ancestorOpacityProduct = product;
    }
    let observer: MutationObserver | null = null;
    if (svgHost && debugEnabled) {
      observer = new MutationObserver((mutations) => {
        const sawSvgMutation = mutations.some((mutation) => {
          if (mutation.type !== "childList") return false;
          const removedSvg = Array.from(mutation.removedNodes).some((node) =>
            node instanceof SVGSVGElement ||
            (node instanceof Element && node.querySelector("svg"))
          );
          const addedSvg = Array.from(mutation.addedNodes).some((node) =>
            node instanceof SVGSVGElement ||
            (node instanceof Element && node.querySelector("svg"))
          );
          return removedSvg || addedSvg;
        });
        if (sawSvgMutation) {
          domReplaceCountRef.current += 1;
          setDebugInfo((prev) =>
            prev
              ? {
                  ...prev,
                  domReplaceCount: domReplaceCountRef.current,
                  svgNodeVersion: svgNodeVersionRef.current,
                  svgIsConnected: Boolean(svgNodeRef.current?.isConnected),
                }
              : prev
          );
        }
      });
      observer.observe(svgHost, { childList: true, subtree: true });
    }
    const firstIso3 =
      iso3Elements[0]?.getAttribute("data-iso3")?.toUpperCase() ?? null;
    setDebugInfo({
      svgLoaded: Boolean(svgEl),
      originalViewBox,
      finalViewBox: svgEl?.getAttribute("viewBox") ?? null,
      preserveAspectRatio: svgEl?.getAttribute("preserveAspectRatio") ?? null,
      widthAttr: svgEl?.getAttribute("width") ?? null,
      heightAttr: svgEl?.getAttribute("height") ?? null,
      taggedNodes: iso3Elements.length,
      targetPaths: paths.length,
      svgHasPathsWithD,
      firstIso3,
      requestedHeightClassName: resolvedHeightClassName,
      medianBboxWidth,
      medianBboxHeight,
      firstPathSample: debugSample[0] ?? null,
      samplePathBboxStats,
      sampleComputedStyle,
      sentinelSvgCenters,
      lastApplyStatus,
      contentBBox,
      coverageW,
      coverageH,
      autoFitApplied,
      autoFitViewBox,
      palette,
      svgNodeVersion: svgNodeVersionRef.current,
      domReplaceCount: domReplaceCountRef.current,
      svgIsConnected: Boolean(svgEl?.isConnected),
      containerClientRect: {
        w: containerRect.width,
        h: containerRect.height,
      },
      svgClientRect: svgRect
        ? { w: svgRect.width, h: svgRect.height }
        : null,
      svgComputed: svgComputedStyle
        ? {
            display: svgComputedStyle.display,
            visibility: svgComputedStyle.visibility,
            opacity: svgComputedStyle.opacity,
          }
        : null,
      ancestorOpacityProduct,
      matchCounts: {
        riskRows: riskRows.length,
        valueByIso3: valueByIso3.size,
        iso3Elements: iso3Set.size,
        matchedIso3,
        unmatchedIso3,
      },
    });

    return () => {
      listeners.forEach((cleanup) => cleanup());
      observer?.disconnect();
    };
  }, [
    svgText,
    valueByIso3,
    breaks,
    hasQuestionsIso3,
    isPerCapita,
    perCapitaFormatter,
    debugEnabled,
    riskRows,
    rcByIso3,
    resolvedHeightClassName,
  ]);

  return (
    <div className="relative w-full rounded-lg border border-fred-secondary bg-slate-950/30 p-4">
      {svgWarnings.length ? (
        <div className="mt-3 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
          {svgWarnings.map((warning) => (
            <div key={warning}>{warning}</div>
          ))}
        </div>
      ) : null}
      <div
        ref={containerRef}
        data-testid="risk-index-map-container"
        className={`relative mt-3 w-full ${resolvedHeightClassName}`}
      >
        <div ref={svgHostRef} className="h-full w-full">
          <SvgMarkup svgText={svgText} />
        </div>
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
      <div className="absolute bottom-3 right-3 rounded-md border border-fred-secondary bg-fred-surface/90 px-3 py-2 text-[11px] text-fred-text shadow-fredCard sm:text-xs">
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-1">
          {[
            { label: "Very low", color: "var(--risk-map-c1)" },
            { label: "Low", color: "var(--risk-map-c2)" },
            { label: "Moderate", color: "var(--risk-map-c3)" },
            { label: "High", color: "var(--risk-map-c4)" },
            { label: "Very high", color: "var(--risk-map-c5)" },
            { label: "Triaged but no forecast", color: "var(--risk-map-no-eiv)" },
            { label: "Not triaged", color: "var(--risk-map-no-questions)" },
          ].map((item) => (
            <div key={item.label} className="flex items-center gap-2">
              <span
                className="h-3 w-3 rounded-sm border border-fred-border"
                style={{ backgroundColor: item.color }}
              />
              <span>{item.label}</span>
            </div>
          ))}
        </div>
      </div>
      {debugEnabled && debugInfo ? (
        <>
          {debugInfo.autoFitApplied &&
          debugInfo.coverageW !== null &&
          debugInfo.coverageH !== null ? (
            <div className="mt-3 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
              Map geometry occupies only{" "}
              {(debugInfo.coverageW * 100).toFixed(1)}%×
              {(debugInfo.coverageH * 100).toFixed(1)}% of viewBox; auto-fit
              applied. This usually means the SVG asset is malformed and should
              be regenerated.
            </div>
          ) : null}
          <details className="mt-3 rounded-md border border-slate-700/60 bg-slate-950/30 px-3 py-2 text-[11px] text-slate-200 sm:text-xs">
            <summary className="cursor-pointer text-slate-100">
              Map debug — SVG loaded: {debugInfo.svgLoaded ? "yes" : "no"} |
              iso3 nodes: {debugInfo.taggedNodes} | matched iso3s:{" "}
              {debugInfo.matchCounts.matchedIso3} | container:{" "}
              {debugInfo.containerClientRect
                ? `${debugInfo.containerClientRect.w.toFixed(
                    0
                  )}×${debugInfo.containerClientRect.h.toFixed(0)}`
                : "n/a"}{" "}
              | fetch:{" "}
              {fetchDiagnostics.fetchOk === null
                ? "n/a"
                : fetchDiagnostics.fetchOk
                ? "ok"
                : "error"}
            </summary>
          </details>
        </>
      ) : null}
    </div>
  );
}
