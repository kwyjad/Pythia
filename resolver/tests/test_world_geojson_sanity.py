import json
from pathlib import Path


def iter_outer_ring_positions(geometry):
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if geom_type == "Polygon":
        ring = coords[0] if coords else []
        for lon, lat in ring:
            yield lon, lat
        return
    if geom_type == "MultiPolygon":
        for polygon in coords:
            ring = polygon[0] if polygon else []
            for lon, lat in ring:
                yield lon, lat


def has_iso3(properties):
    if not isinstance(properties, dict):
        return False
    iso3 = properties.get("iso3") or properties.get("ISO_A3") or properties.get("ADM0_A3") or properties.get("adm0_a3")
    return bool(iso3)


def test_world_geojson_sanity():
    path = Path(__file__).resolve().parents[2] / "web" / "public" / "maps" / "world-countries-iso3.geojson"
    with path.open() as handle:
        data = json.load(handle)

    assert data.get("type") == "FeatureCollection"
    features = data.get("features", [])
    assert len(features) > 150

    polygon_features = [
        feature
        for feature in features
        if (feature.get("geometry") or {}).get("type") in {"Polygon", "MultiPolygon"}
    ]
    assert len(polygon_features) >= 100

    iso3_features = [feature for feature in features if has_iso3(feature.get("properties") or {})]
    assert len(iso3_features) >= 150

    latitudes = set()
    for feature in polygon_features[:50]:
        geometry = feature.get("geometry") or {}
        for _, lat in iter_outer_ring_positions(geometry):
            if isinstance(lat, (float, int)):
                latitudes.add(round(lat, 4))

    assert len(latitudes) > 200
