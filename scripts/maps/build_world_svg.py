#!/usr/bin/env python3
import json
import sys
from pathlib import Path

authored_by = "Generated from Natural Earth admin-0 GeoJSON"

ROOT = Path(__file__).resolve().parents[2]
GEOJSON_PATH = ROOT / "web" / "public" / "maps" / "world-countries-iso3.geojson"
SVG_PATH = ROOT / "web" / "public" / "maps" / "world.svg"

VIEWBOX_WIDTH = 1000
VIEWBOX_HEIGHT = 500
MIN_ISO3_FEATURES = 150


def project(lon: float, lat: float) -> tuple[float, float]:
    x = (lon + 180.0) / 360.0 * VIEWBOX_WIDTH
    y = (90.0 - lat) / 180.0 * VIEWBOX_HEIGHT
    return x, y


def fmt(value: float) -> str:
    text = f"{value:.2f}"
    text = text.rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    return text


def iter_outer_rings(geometry: dict) -> list[list[tuple[float, float]]]:
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates") or []
    rings: list[list[tuple[float, float]]] = []
    if geom_type == "Polygon":
        if coords:
            rings.append([(float(lon), float(lat)) for lon, lat in coords[0]])
        return rings
    if geom_type == "MultiPolygon":
        for polygon in coords:
            if polygon:
                rings.append([(float(lon), float(lat)) for lon, lat in polygon[0]])
    return rings


def get_iso3(properties: dict) -> str | None:
    iso3 = properties.get("iso3") or properties.get("ISO_A3") or properties.get("ADM0_A3")
    if not iso3 or iso3 == "-99":
        return None
    return str(iso3).strip()


def get_name(properties: dict) -> str | None:
    name = properties.get("name") or properties.get("NAME")
    if not name:
        return None
    return str(name).strip()


def build_paths(data: dict) -> list[str]:
    features = data.get("features", [])
    entries: list[tuple[str, str | None, dict]] = []
    for feature in features:
        properties = feature.get("properties") or {}
        iso3 = get_iso3(properties)
        if not iso3:
            continue
        name = get_name(properties)
        entries.append((iso3, name, feature))

    entries.sort(key=lambda item: (item[0], item[1] or ""))

    paths: list[str] = []
    for iso3, name, feature in entries:
        geometry = feature.get("geometry") or {}
        for ring in iter_outer_rings(geometry):
            if not ring:
                continue
            commands: list[str] = []
            for index, (lon, lat) in enumerate(ring):
                x, y = project(lon, lat)
                cmd = "M" if index == 0 else "L"
                commands.append(f"{cmd} {fmt(x)} {fmt(y)}")
            commands.append("Z")
            attrs = [f'data-iso3="{iso3}"']
            if name:
                attrs.append(f'data-name="{name}"')
            path_d = " ".join(commands)
            attrs.append(f'd="{path_d}"')
            paths.append(f"  <path {' '.join(attrs)} />")
    return paths


def validate_geojson_iso3(data: dict) -> None:
    features = data.get("features", [])
    iso3_features = 0
    for feature in features:
        properties = feature.get("properties") or {}
        if get_iso3(properties):
            iso3_features += 1
    if iso3_features < MIN_ISO3_FEATURES:
        print(
            "GeoJSON validation failed: not enough ISO3-tagged features.",
        )
        print(f"ISO3 features: {iso3_features} (minimum {MIN_ISO3_FEATURES})")
        sys.exit(1)


def validate_geojson_bounds(data: dict) -> None:
    min_lon = None
    max_lon = None
    min_lat = None
    max_lat = None
    for feature in data.get("features", []):
        geometry = feature.get("geometry") or {}
        for ring in iter_outer_rings(geometry):
            for lon, lat in ring:
                min_lon = lon if min_lon is None else min(min_lon, lon)
                max_lon = lon if max_lon is None else max(max_lon, lon)
                min_lat = lat if min_lat is None else min(min_lat, lat)
                max_lat = lat if max_lat is None else max(max_lat, lat)
    if None in (min_lon, max_lon, min_lat, max_lat):
        print("GeoJSON validation failed: no coordinates found.")
        sys.exit(1)
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat
    if lon_span < 300 or lat_span < 120:
        print(
            "GeoJSON validation failed: coordinate span too small.",
        )
        print(
            f"Longitude span: {lon_span:.2f} ({min_lon:.2f} to {max_lon:.2f})"
        )
        print(
            f"Latitude span: {lat_span:.2f} ({min_lat:.2f} to {max_lat:.2f})"
        )
        sys.exit(1)


def main() -> None:
    data = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    validate_geojson_bounds(data)
    validate_geojson_iso3(data)
    paths = build_paths(data)
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEWBOX_WIDTH} {VIEWBOX_HEIGHT}">',
        f"  <!-- {authored_by} -->",
        *paths,
        "</svg>",
    ]
    SVG_PATH.write_text("\n".join(svg_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
